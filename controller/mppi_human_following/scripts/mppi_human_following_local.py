#!/usr/bin/env python3
import rclpy
import numba
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist, PolygonStamped, Point32, Pose2D, PoseStamped, Point
from std_msgs.msg import Header
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
import numpy as np
import time
import math
import numba
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
import copy
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from scout_msgs.msg import ScoutRCState

# Information about your GPU
gpu = cuda.get_current_device()
max_threads_per_block = gpu.MAX_THREADS_PER_BLOCK
max_square_block_dim = (int(gpu.MAX_BLOCK_DIM_X**0.5), int(gpu.MAX_BLOCK_DIM_X**0.5))
max_blocks = gpu.MAX_GRID_DIM_X
max_rec_blocks = rec_max_control_rollouts = int(1e5) # Though theoretically limited by max_blocks on GPU
rec_min_control_rollouts = 100


class Config:
  
  """ Configurations that are typically fixed throughout execution. """
  
  def __init__(self, 
               T=5, # Horizon (s)
               dt=0.1, # Length of each step (s)
               num_control_rollouts=1024, # Number of control sequences
               num_vis_state_rollouts=20, # Number of visualization rollouts
               seed=1):
    
    self.seed = seed
    self.T = T
    self.dt = dt
    self.num_steps = int(T/dt)
    self.max_threads_per_block = max_threads_per_block # save just in case

    assert T > 0
    assert dt > 0
    assert T > dt
    assert self.num_steps > 0
    
    # Number of control rollouts are currently limited by the number of blocks
    self.num_control_rollouts = num_control_rollouts
    if self.num_control_rollouts > rec_max_control_rollouts:
      self.num_control_rollouts = rec_max_control_rollouts
      print("MPPI Config: Clip num_control_rollouts to be recommended max number of {}. (Max={})".format(
        rec_max_control_rollouts, max_blocks))
    elif self.num_control_rollouts < rec_min_control_rollouts:
      self.num_control_rollouts = rec_min_control_rollouts
      print("MPPI Config: Clip num_control_rollouts to be recommended min number of {}. (Recommended max={})".format(
        rec_min_control_rollouts, rec_max_control_rollouts))
    
    # For visualizing state rollouts
    self.num_vis_state_rollouts = num_vis_state_rollouts
    self.num_vis_state_rollouts = min([self.num_vis_state_rollouts, self.num_control_rollouts])
    self.num_vis_state_rollouts = max([1, self.num_vis_state_rollouts])


DEFAULT_OBS_WEIGHT  = 1e5
DEFAULT_CBF_WEIGHT = 1e5
DEFAULT_DIST_WEIGHT = 10
DEFAULT_ANGLE_WEIGHT = 20

# Stage costs (device function)
@cuda.jit('float32(float32, float32)', device=True, inline=True)
def stage_cost(dist2, dist_weight):
  return dist_weight*dist2 # squared term makes the robot move faster

# Terminal costs (device function)
@cuda.jit('float32(float32, boolean)', device=True, inline=True)
def term_cost(dist2, goal_reached):
  return (1-np.float32(goal_reached))*dist2



class MPPI_Numba(object):
  
  """ 
  Implementation of Information theoretic MPPI by Williams et. al. 
  Alg 2. in https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf


  Planner object that initializes GPU memory and runs MPPI on GPU via numba. 
  
  Typical workflow: 
    1. Initialize object with config that allows pre-initialization of GPU memory
    2. reset()
    3. setup(mppi_params) based on problem instance
    4. solve(), which returns optimized control sequence
    5. get_state_rollout() for visualization
    6. shift_and_update(next_state, optimal_u_sequence, num_shifts=1)
    7. Repeat from 2 if params have changed
  """

  def __init__(self, cfg):

    # Fixed configs
    self.cfg = cfg
    self.T = cfg.T
    self.dt = cfg.dt
    self.num_steps = cfg.num_steps
    self.num_control_rollouts = cfg.num_control_rollouts

    self.num_vis_state_rollouts = cfg.num_vis_state_rollouts
    self.seed = cfg.seed

    # Basic info 
    self.max_threads_per_block = cfg.max_threads_per_block

    # Initialize reuseable device variables
    self.noise_samples_d = None
    self.u_cur_d = None
    self.u_prev_d = None
    self.costs_d = None
    self.weights_d = None
    self.rng_states_d = None
    self.state_rollout_batch_d = None # For visualization only. Otherwise, inefficient

    # Other task specific params
    self.device_var_initialized = False
    self.reset()

    
  def reset(self):
    # Other task specific params
    self.u_seq0 = np.zeros((self.num_steps, 3), dtype=np.float32)
    self.params = None
    self.params_set = False

    self.u_prev_d = None
    
    # Initialize all fixed-size device variables ahead of time. (Do not change in the lifetime of MPPI object)
    self.init_device_vars_before_solving()


  def init_device_vars_before_solving(self):

    if not self.device_var_initialized:
      t0 = time.time()
      
      self.noise_samples_d = cuda.device_array((self.num_control_rollouts, self.num_steps, 3), dtype=np.float32) # to be sampled collaboratively via GPU
      self.u_cur_d = cuda.to_device(self.u_seq0) 
      self.u_prev_d = cuda.to_device(self.u_seq0) 
      self.costs_d = cuda.device_array((self.num_control_rollouts), dtype=np.float32)
      self.weights_d = cuda.device_array((self.num_control_rollouts), dtype=np.float32)
      self.rng_states_d = create_xoroshiro128p_states(self.num_control_rollouts*self.num_steps, seed=self.seed)
      
      self.state_rollout_batch_d = cuda.device_array((self.num_vis_state_rollouts, self.num_steps+1, 3), dtype=np.float32)
      
      self.device_var_initialized = True
      print("MPPI planner has initialized GPU memory after {} s".format(time.time()-t0))


  def setup(self, params):
    # These tend to change (e.g., current robot position, the map) after each step
    self.set_params(params)


  def set_params(self, params):
    self.params = copy.deepcopy(params)
    self.params_set = True


  def check_solve_conditions(self):
    if not self.params_set:
      print("MPPI parameters are not set. Cannot solve")
      return False
    if not self.device_var_initialized:
      print("Device variables not initialized. Cannot solve.")
      return False
    return True

  def solve(self):
    """Entry point for different algoritims"""
    
    if not self.check_solve_conditions():
      print("MPPI solve condition not met. Cannot solve. Return")
      return
    
    return self.solve_with_nominal_dynamics()


  def move_mppi_task_vars_to_device(self):
    vrange_x_d = cuda.to_device(self.params['vrange_x'].astype(np.float32))
    vrange_y_d = cuda.to_device(self.params['vrange_y'].astype(np.float32))
    wrange_d = cuda.to_device(self.params['wrange'].astype(np.float32))
    xgoal_d = cuda.to_device(self.params['xgoal'].astype(np.float32))
    goal_tolerance_d = np.float32(self.params['goal_tolerance'])
    angle_tolerance_d = np.float32(self.params['angle_tolerance'])
    lambdweight_d = np.float32(self.params['lambdweight'])
    u_std_d = cuda.to_device(self.params['u_std'].astype(np.float32))
    x0_d = cuda.to_device(self.params['x0'].astype(np.float32))
    human_d = cuda.to_device(self.params['human'].astype(np.float32))


    safety_length = np.float32(self.params['safety_length'])
    safety_width = np.float32(self.params['safety_width'])
    alpha_cbf = np.float32(self.params['alpha_cbf'])

    dt_d = np.float32(self.params['dt'])

    obs_cost_d = np.float32(DEFAULT_OBS_WEIGHT if 'obs_penalty' not in self.params 
                                     else self.params['obs_penalty'])
    return vrange_x_d, vrange_y_d, wrange_d, xgoal_d, \
           goal_tolerance_d, angle_tolerance_d, lambdweight_d, \
           u_std_d, x0_d, human_d, safety_length, safety_width, alpha_cbf, dt_d, obs_cost_d

  def _moving_average_filter(self, xx: np.ndarray, window_size: int):
        b = np.ones(window_size)/window_size
        xx_mean = np.convolve(xx, b, mode="same")
        n_conv = math.ceil(window_size/2)
        xx_mean[0] *= window_size/n_conv
        for i in range(1, n_conv):
            xx_mean[i] *= window_size/(i+n_conv)
            xx_mean[-i] *= window_size/(i + n_conv - (window_size % 2)) 
        return xx_mean


  def solve_with_nominal_dynamics(self):
    """
    Launch GPU kernels that use nominal dynamics but adjsuts cost function based on worst-case linear speed.
    """
    
    vrange_x_d,vrange_y_d, wrange_d, xgoal_d, goal_tolerance_d,angle_tolerance_d, lambdweight_d, \
           u_std_d, x0_d, human_d, safety_length, safety_width, alpha_cbf, dt_d, obs_cost_d = self.move_mppi_task_vars_to_device()
  
    # Weight for distance cost
    dist_weight = DEFAULT_DIST_WEIGHT if 'dist_weight' not in self.params else self.params['dist_weight']
    cbf_weight = DEFAULT_CBF_WEIGHT if 'cbf_weight' not in self.params else self.params['cbf_weight']
    obs_weight = DEFAULT_OBS_WEIGHT if 'obs_weight' not in self.params else self.params['obs_weight']
    angle_weight = DEFAULT_ANGLE_WEIGHT if 'angle_weight' not in self.params else self.params['angle_weight']

    # Optimization loop
    for k in range(self.params['num_opt']):

      # Sample control noise
      self.sample_noise_numba[self.num_control_rollouts, self.num_steps](
            self.rng_states_d, u_std_d, self.noise_samples_d)
      
      # Rollout and compute mean or cvar
      self.rollout_numba[self.num_control_rollouts, 1](
        vrange_x_d,
        vrange_y_d,
        wrange_d,
        xgoal_d,
        obs_cost_d, 
        goal_tolerance_d,
        angle_tolerance_d,
        lambdweight_d,
        u_std_d,
        x0_d,
        human_d,
        safety_length,
        safety_width,
        alpha_cbf,  
        dt_d,
        dist_weight,
        angle_weight,
        cbf_weight,
        obs_weight,
        self.noise_samples_d,
        self.u_cur_d,

        # results
        self.costs_d
      )
      # apply moving average filter for smoothing input sequence
      self.u_prev_d = self.u_cur_d

      # Compute cost and update the optimal control on device
      self.update_useq_numba[1, 32](
        lambdweight_d, 
        self.costs_d, 
        self.noise_samples_d, 
        self.weights_d, 
        vrange_x_d,
        vrange_y_d,
        wrange_d,
        self.u_cur_d
      )

    return self.u_cur_d.copy_to_host()


  def update(self, x0, human, xgoal):
    self.params["x0"] = x0.copy()
    self.params["human"] = human.copy()
    self.params["xgoal"] = xgoal.copy()
  


  def shift(self, u_cur, num_shifts=1):
    
    self.shift_optimal_control_sequence(u_cur, num_shifts)



  def shift_optimal_control_sequence(self, u_cur, num_shifts=1):
    u_cur_shifted = u_cur.copy()
    u_cur_shifted[:-num_shifts] = u_cur_shifted[num_shifts:]
    self.u_cur_d = cuda.to_device(u_cur_shifted.astype(np.float32))


  def get_state_rollout(self):
    """
    Generate state sequences based on the current optimal control sequence.
    """

    assert self.params_set, "MPPI parameters are not set"

    if not self.device_var_initialized:
      print("Device variables not initialized. Cannot run mppi.")
      return
    
    # Move things to GPU
    vrange_x_d = cuda.to_device(self.params['vrange_x'].astype(np.float32))
    vrange_y_d = cuda.to_device(self.params['vrange_y'].astype(np.float32))
    wrange_d = cuda.to_device(self.params['wrange'].astype(np.float32))
    x0_d = cuda.to_device(self.params['x0'].astype(np.float32))
    human_d = cuda.to_device(self.params['human'].astype(np.float32))
    dt_d = np.float32(self.params['dt'])

    self.get_state_rollout_across_control_noise[self.num_vis_state_rollouts, 1](
        self.state_rollout_batch_d, # where to store results
        x0_d,
        dt_d,
        self.noise_samples_d,
        vrange_x_d,
        vrange_y_d,
        wrange_d,
        self.u_prev_d,
        self.u_cur_d,
        )
    
    return self.state_rollout_batch_d.copy_to_host()
  
  """GPU kernels from here on"""
  
  @staticmethod
  @cuda.jit(fastmath=True)
  def rollout_numba(
          vrange_x_d,
          vrange_y_d, 
          wrange_d, 
          xgoal_d, 
          obs_cost_d, 
          goal_tolerance_d,
          angle_tolerance_d, 
          lambdweight_d, 
          u_std_d, 
          x0_d,
          human_d,
          safety_length,
          safety_width,
          alpha_cbf, 
          dt_d,
          dist_weight_d,
          angle_weight_d,
          cbf_weight_d,
          obs_weight_d,
          noise_samples_d,
          u_cur_d,
          costs_d):
    """
    There should only be one thread running in each block, where each block handles a single sampled control sequence.
    """


    # Get block id and thread id
    bid = cuda.blockIdx.x   # index of block
    tid = cuda.threadIdx.x  # index of thread within a block

    costs_d[bid] = 0.0

    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    x_curr = cuda.local.array(6, numba.float32)
    x_curr_human = cuda.local.array(6, numba.float32)

    for i in range(6): 
      x_curr[i] = x0_d[i]
      x_curr_human[i]=human_d[i]

    timesteps = len(u_cur_d)
    goal_reached = False
    angle_goal_reached = False
    vel_x_goal_reached = False
    vel_y_goal_reached = False
    goal_tolerance_d2 = goal_tolerance_d*goal_tolerance_d
    angle_tolerance_d2 = angle_tolerance_d*angle_tolerance_d
    dist_to_goal2 = 1e9
    angle_to_goal2 = 1e9
    v_x_nom = v_y_nom = v_x_noisy = v_y_noisy = w_nom = w_noisy = 0.0
    r = 0.7


    # Initialize transformed robot coordinates list
    transformed_robot_x = cuda.local.array(3, numba.float32)  # Array for 3 x-coordinates
    transformed_robot_y = cuda.local.array(3, numba.float32)  # Array for 3 y-coordinates

    # printed=False
    for t in range(timesteps):
      # Nominal noisy control
      v_x_nom = u_cur_d[t, 0] + noise_samples_d[bid, t, 0]
      v_y_nom = u_cur_d[t, 1] + noise_samples_d[bid, t, 1]
      w_nom = u_cur_d[t, 2] + noise_samples_d[bid, t, 2]

      v_x_noisy = max(vrange_x_d[0], min(vrange_x_d[1], v_x_nom))
      v_y_noisy = max(vrange_y_d[0], min(vrange_y_d[1], v_y_nom))
      w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))

      delta_x =  x_curr_human[0] - x_curr[0]
      delta_y =  x_curr_human[1] -  x_curr[1]
      delta_theta = x_curr_human[2] -  x_curr[2]

      x_robot_human = - delta_x * math.cos(delta_theta) - delta_y * math.sin(delta_theta)
      y_robot_human = delta_x * math.sin(delta_theta) -  delta_y * math.cos(delta_theta)

      pre_equation_value = x_robot_human**2  + y_robot_human**2 - r**2

      # Forward simulate
      x_curr[0] += dt_d*(v_x_noisy * math.cos(x_curr[2]) - v_y_noisy * math.sin(x_curr[2]))
      x_curr[1] += dt_d*(v_x_noisy * math.sin(x_curr[2]) + v_y_noisy * math.cos(x_curr[2]))
      x_curr[2] += dt_d*w_noisy

      delta_x =  x_curr_human[0] - x_curr[0]
      delta_y =  x_curr_human[1] -  x_curr[1]
      delta_theta = x_curr_human[2] -  x_curr[2]

      x_robot_human = - delta_x * math.cos(delta_theta) - delta_y * math.sin(delta_theta)
      y_robot_human = delta_x * math.sin(delta_theta) -  delta_y * math.cos(delta_theta)
      current_equation_value = x_robot_human**2  + y_robot_human**2 - r**2

      cbf=alpha_cbf*(pre_equation_value) - current_equation_value

      if cbf > 0.0:
        cbf_cost=1e4
        costs_d[bid] += stage_cost(cbf_cost, cbf_weight_d)


      # Following
      dist_to_goal2 = (xgoal_d[0]- x_curr[0])**2 +(xgoal_d[1]- x_curr[1])**2
      # costs_d[bid]+= stage_cost(dist_to_goal2, dist_weight_d)

      dist_to_goalx2 = (xgoal_d[0]- x_curr[0])**2 
      costs_d[bid]+= stage_cost(dist_to_goalx2, 70.0)

      dist_to_goaly2 = (xgoal_d[1]- x_curr[1])**2
      costs_d[bid]+= stage_cost(dist_to_goaly2, 50.0)

      angle_to_goal2 = (xgoal_d[2]-x_curr[2])**2 
      costs_d[bid]+= stage_cost(angle_to_goal2, 50.0)

      # if math.sqrt(dist_to_goal2)<0.1:
      #    v=0.0

      vel_x_to_goal2 = (x_curr_human[3]-v_x_noisy)**2 
      costs_d[bid]+= stage_cost(vel_x_to_goal2, 0.3)

      # vel_y_to_goal2 = (x_curr_human[4]-x_curr[4])**2 
      # costs_d[bid]+= stage_cost(vel_y_to_goal2, 0.5)

      if dist_to_goal2<= goal_tolerance_d2 and angle_to_goal2 <= angle_tolerance_d2:
        goal_reached = True
        angle_goal_reached = True
        break
                     
    # Accumulate terminal cost 
    costs_d[bid] += term_cost(dist_to_goal2, goal_reached)
    costs_d[bid] += term_cost(angle_to_goal2, angle_goal_reached)
    # costs_d[bid] += term_cost(vel_x_to_goal2, vel_x_goal_reached)
    # costs_d[bid] += term_cost(vel_y_to_goal2, vel_y_goal_reached)

    for t in range(timesteps):
      costs_d[bid] += lambdweight_d*(
              (u_cur_d[t,0]/(u_std_d[0]))*noise_samples_d[bid, t,0] + (u_cur_d[t,1]/(u_std_d[1]))*noise_samples_d[bid, t, 1]+ (u_cur_d[t,2]/(u_std_d[2]))*noise_samples_d[bid, t, 2])

      
  @staticmethod
  @cuda.jit(fastmath=True)
  def update_useq_numba(
        lambdweight_d,
        costs_d,
        noise_samples_d,
        weights_d,
        vrange_x_d,
        vrange_y_d,
        wrange_d,
        u_cur_d):
    """
    GPU kernel that updates the optimal control sequence based on previously evaluated cost values.
    Assume that the function is invoked as update_useq_numba[1, NUM_THREADS], with one block and multiple threads.
    """

    tid = cuda.threadIdx.x
    num_threads = cuda.blockDim.x
    numel = len(noise_samples_d)
    gap = int(math.ceil(numel / num_threads))

    # Find the minimum value via reduction
    starti = min(tid*gap, numel)
    endi = min(starti+gap, numel)
    if starti<numel:
      weights_d[starti] = costs_d[starti]
    for i in range(starti, endi):
      weights_d[starti] = min(weights_d[starti], costs_d[i])
    cuda.syncthreads()

    s = gap
    while s < numel:
      if (starti % (2 * s) == 0) and ((starti + s) < numel):
        # Stride by `s` and add
        weights_d[starti] = min(weights_d[starti], weights_d[starti + s])
      s *= 2
      cuda.syncthreads()

    beta = weights_d[0]
    
    # Compute weight
    for i in range(starti, endi):
      weights_d[i] = math.exp(-1./lambdweight_d*(costs_d[i]-beta))
    cuda.syncthreads()

    # Normalize
    # Reuse costs_d array
    for i in range(starti, endi):
      costs_d[i] = weights_d[i]
    cuda.syncthreads()
    for i in range(starti+1, endi):
      costs_d[starti] += costs_d[i]
    cuda.syncthreads()
    s = gap
    while s < numel:
      if (starti % (2 * s) == 0) and ((starti + s) < numel):
        # Stride by `s` and add
        costs_d[starti] += costs_d[starti + s]
      s *= 2
      cuda.syncthreads()

    for i in range(starti, endi):
      weights_d[i] /= costs_d[0]
    cuda.syncthreads()
    
    # update the u_cur_d
    timesteps = len(u_cur_d)
    for t in range(timesteps):
      for i in range(starti, endi):
        cuda.atomic.add(u_cur_d, (t, 0), weights_d[i]*noise_samples_d[i, t, 0])
        cuda.atomic.add(u_cur_d, (t, 1), weights_d[i]*noise_samples_d[i, t, 1])
        cuda.atomic.add(u_cur_d, (t, 2), weights_d[i]*noise_samples_d[i, t, 2])
    cuda.syncthreads()

    # Blocks crop the control together
    tgap = int(math.ceil(timesteps / num_threads))
    starti = min(tid*tgap, timesteps)
    endi = min(starti+tgap, timesteps)


    for ti in range(starti, endi):
      u_cur_d[ti, 0] = max(vrange_x_d[0], min(vrange_x_d[1], u_cur_d[ti, 0]))
      u_cur_d[ti, 1] = max(vrange_y_d[0], min(vrange_y_d[1], u_cur_d[ti, 1]))
      u_cur_d[ti, 2] = max(wrange_d[0], min(wrange_d[1], u_cur_d[ti, 2]))



  @staticmethod
  @cuda.jit(fastmath=True)
  def get_state_rollout_across_control_noise(
          state_rollout_batch_d, # where to store results
          x0_d, 
          dt_d,
          noise_samples_d,
          vrange_x_d,
          vrange_y_d,
          wrange_d,
          u_prev_d,
          u_cur_d):
    """
    Do a fixed number of rollouts for visualization across blocks.
    Assume kernel is launched as get_state_rollout_across_control_noise[num_blocks, 1]
    The block with id 0 will always visualize the best control sequence. Other blocks will visualize random samples.
    """
    
    # Use block id
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    timesteps = len(u_cur_d)


    if bid==0:
      # Visualize the current best 
      # Explicit unicycle update and map lookup
      # From here on we assume grid is properly padded so map lookup remains valid
      x_curr = cuda.local.array(6, numba.float32)
      for i in range(6): 
        x_curr[i] = x0_d[i]
        state_rollout_batch_d[bid,0,i] = x0_d[i]
      
      for t in range(timesteps):
        # Nominal noisy control
        v_x_nom = u_cur_d[t, 0]
        v_y_nom = u_cur_d[t, 1]
        w_nom = u_cur_d[t, 2]

        # Forward simulate
        x_curr[0] += dt_d*(v_x_nom * math.cos(x_curr[2]) - v_y_nom * math.sin(x_curr[2]))
        x_curr[1] += dt_d*(v_x_nom * math.sin(x_curr[2]) + v_y_nom * math.cos(x_curr[2]))
        x_curr[2] += dt_d*w_nom

        # Save state
        state_rollout_batch_d[bid,t+1,0] = x_curr[0]
        state_rollout_batch_d[bid,t+1,1] = x_curr[1]
        state_rollout_batch_d[bid,t+1,2] = x_curr[2]



    else:
      
      # Explicit unicycle update and map lookup
      # From here on we assume grid is properly padded so map lookup remains valid
      x_curr = cuda.local.array(6, numba.float32)
      for i in range(6): 
        x_curr[i] = x0_d[i]
        state_rollout_batch_d[bid,0,i] = x0_d[i]

      
      for t in range(timesteps):
        # Nominal noisy control
        v_x_nom = u_prev_d[t, 0] + noise_samples_d[bid, t, 0]
        v_y_nom = u_prev_d[t, 1] + noise_samples_d[bid, t, 1]
        w_nom = u_prev_d[t, 2] + noise_samples_d[bid, t, 2]

        v_x_noisy = max(vrange_x_d[0], min(vrange_x_d[1], v_x_nom))
        v_y_noisy = max(vrange_y_d[0], min(vrange_y_d[1], v_y_nom))
        w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))

        # # Nominal noisy control
        x_curr[0] += dt_d*(v_x_noisy * math.cos(x_curr[2]) - v_y_noisy * math.sin(x_curr[2]))
        x_curr[1] += dt_d*(v_x_noisy * math.sin(x_curr[2]) + v_y_noisy * math.cos(x_curr[2]))
        x_curr[2] += dt_d*w_noisy
        
        # Forward simulate
        x_curr[0] += dt_d*(v_x_nom * math.cos(x_curr[2]) - v_y_nom * math.sin(x_curr[2]))
        x_curr[1] += dt_d*(v_x_nom * math.sin(x_curr[2]) + v_y_nom * math.cos(x_curr[2]))
        x_curr[2] += dt_d*w_nom

        # Save state
        state_rollout_batch_d[bid,t+1,0] = x_curr[0]
        state_rollout_batch_d[bid,t+1,1] = x_curr[1]
        state_rollout_batch_d[bid,t+1,2] = x_curr[2]


  @staticmethod
  @cuda.jit(fastmath=True)
  def sample_noise_numba(rng_states, u_std_d, noise_samples_d):
    """
    Should be invoked as sample_noise_numba[NUM_U_SAMPLES, NUM_THREADS].
    noise_samples_d.shape is assumed to be (num_rollouts, time_steps, 2)
    Assume each thread corresponds to one time step
    For consistency, each block samples a sequence, and threads (not too many) work together over num_steps.
    This will not work if time steps are more than max_threads_per_block (usually 1024)
    """
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    abs_thread_id = cuda.grid(1)

    noise_samples_d[block_id, thread_id, 0] = u_std_d[0]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)
    noise_samples_d[block_id, thread_id, 1] = u_std_d[1]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)
    noise_samples_d[block_id, thread_id, 2] = u_std_d[2]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)
        


class LocalPlanningNode(Node):
    def __init__(self):
        # Initialize the ROS2 Node with the name 'mppi_human_following'
        super().__init__('mppi_human_following')

        # QoS (Quality of Service) Profile for message communication
        qos_profile = QoSProfile(depth=1)

        # Create subscriptions to listen to topics
        # Subscribing to 'occupancy_grid_topic' to receive the map data
        self.human_sub = self.create_subscription(Odometry,'/human_robot_pos', self.human_callback,5)
        self.robot_sub = self.create_subscription(Odometry,'/odom', self.odom_callback,5)
        self.rc_sub = self.create_subscription(ScoutRCState,'/scout_rc_state', self.rc_callback,5)

        # Create publishers to send data to topics
        # Publisher for sending velocity commands to 'cmd_vel' topic
        self.pub_vel = self.create_publisher(Twist, 'cmd_vel', qos_profile)

        # Publisher for sending local path to 'local_path' topic
        self.local_path_publisher = self.create_publisher(Path, '/local_path', qos_profile)

        # Publisher for sending robot footprint data to 'robot_footprint' topic
        # self.pub_footprint = self.create_publisher(PolygonStamped, 'robot_footprint', qos_profile)

        # Publisher for sending the sampled path to 'sampled_path' topic
        self.sampled_path_publisher = self.create_publisher(MarkerArray, '/sampled_path', qos_profile)

        # Timer for periodic tasks, triggers the timer_callback function every 0.001 seconds
        self.timer = self.create_timer(0.001, self.timer_callback)

        # Declare and get parameters for robot configuration
        self.base_link_frame = self.declare_parameter('base_link_frame', 'base_link').get_parameter_value().string_value

        self.min_vel_x = self.declare_parameter('min_vel_x', -1.5).get_parameter_value().double_value
        self.max_vel_x = self.declare_parameter('max_vel_x', 2.5).get_parameter_value().double_value
        self.min_vel_y = self.declare_parameter('min_vel_y', -1.5).get_parameter_value().double_value
        self.max_vel_y = self.declare_parameter('max_vel_y', 1.5).get_parameter_value().double_value
        self.min_ang_z = self.declare_parameter('min_ang_z', -1.0).get_parameter_value().double_value
        self.max_ang_z = self.declare_parameter('max_ang_z', 1.0).get_parameter_value().double_value


        self.dist_weight = self.declare_parameter('dist_weight', 70.0).get_parameter_value().double_value
        self.angle_weight = self.declare_parameter('angle_weight', 30.0).get_parameter_value().double_value
        self.cbf_weight = self.declare_parameter('cbf_weight', 100000.0).get_parameter_value().double_value
        self.obs_weight = self.declare_parameter('obs_weight', 100000.0).get_parameter_value().double_value
        self.lambdweight = self.declare_parameter('lambda_weight', 0.1).get_parameter_value().double_value

        self.safety_length = self.declare_parameter('safety_length', 1.5).get_parameter_value().double_value
        self.safety_width = self.declare_parameter('safety_width', 1.0).get_parameter_value().double_value
        self.alpha_cbf = self.declare_parameter('alpha_cbf', 0.5).get_parameter_value().double_value

        self.distance_start = self.declare_parameter('distance_start', 0.5).get_parameter_value().double_value
        
        # Initialize state variables (goal and initial position)
        self.xgoal = np.zeros(3)  # Goal state (x, y, theta)
        self.x0 = np.zeros(6)     # Initial state

        self.human_position = None
        self.twist = Twist()
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.ang_z = 0.0

        self.local_x = -1.7
        self.local_y = 0.0

        # MPPI (Model Predictive Path Integral) configuration
        self.cfg = Config(T=3.0, dt=0.1, num_control_rollouts=1000, num_vis_state_rollouts=10, seed=1)
        self.mppi_params = self.create_mppi_params()
        
        # Initialize MPPI planner with the provided configuration and parameters
        self.mppi_planner = MPPI_Numba(self.cfg)
        self.mppi_planner.setup(self.mppi_params)



    def create_mppi_params(self):
        """Helper function to define the parameters for MPPI (Model Predictive Path Integral) planner. To respond more quickly, increase the desired weight."""
        return {
            'dt': self.cfg.dt,                    # Time step for control
            'x0': self.x0,                        # Initial state
            # 'human': self.human_position,       # Initial human state
            'xgoal': self.xgoal,                  # Goal state
            'goal_tolerance': 0.15,               # Tolerance for goal position
            'angle_tolerance': 0.2,               # Tolerance for goal angle
            'dist_weight': self.dist_weight,      # Weight for distance-to-goal cost
            'angle_weight': self.angle_weight,    # Weight for goal-to-goal angle cost
            'cbf_weight': self.cbf_weight,        # Weight for cbf cost
            'obs_weight': self.obs_weight,        # Weight for obs cost
            'lambdweight': self.lambdweight,  # Temperature parameter in MPPI
            'num_opt': 1,                    # Number of optimization steps
            'u_std': np.array([0.1, 0.1, 0.05]),  # Noise standard deviation for control input
            'vrange_x': np.array([self.min_vel_x, self.max_vel_x]),   # Linear x vel range
            'vrange_y': np.array([self.min_vel_y, self.max_vel_y]),   # Linear y vel range
            'wrange': np.array([self.min_ang_z, self.max_ang_z]),    # Angular vel range
            'safety_length': self.safety_length,
            'safety_width': self.safety_width,
            'alpha_cbf': self.alpha_cbf
        }


    def quaternion_to_yaw(self, quaternion):
        """Converts a quaternion to a yaw angle (in radians)."""
        return R.from_quat(quaternion).as_euler('xyz', degrees=False)[2]

    def human_callback(self, msg):
        # theta_human = self.quaternion_to_yaw([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        # directions = [0, np.pi/2, np.pi, 3*np.pi/2]
        # closest = min(directions, key=lambda x: abs((theta_human - x + np.pi) % (2*np.pi) - np.pi))
        # print(f"Raw theta: {theta_human:.2f}, Snapped to: {closest:.2f} rad")
        # theta_human = closest  # Gán lại góc đã được lượng tử hóa
        theta_human=0.0
        self.human_position=np.array([msg.pose.pose.position.x,msg.pose.pose.position.y,theta_human,msg.twist.twist.linear.x,msg.twist.twist.linear.y, msg.twist.twist.angular.z])

    def rc_callback(self, msg):
        if msg.swa==0:
          self.local_x = 0.4
          self.local_y = 0.9
        elif msg.swa==3:
          self.local_x = -1.5
          self.local_y = 0.0
        elif msg.swa==2:
          self.local_x = 0.4
          self.local_y = -0.9


    def odom_callback(self, msg):
        self.vel_x = msg.twist.twist.linear.x
        self.vel_y = msg.twist.twist.linear.y
        self.ang_z = msg.twist.twist.angular.z
        # Update the MPPI planner with the new goal and control sequence
        self.x0=np.array([0.0, 0.0, 0.0, self.vel_x, self.vel_y, self.ang_z])
        

    def publish_path(self, x, y, publisher, base):
        """Publish the path (a series of poses) to a given publisher."""
        path_msg = Path()
        path_msg.header.frame_id = base  # Set the appropriate frame ID (e.g., 'base_link')
        length = len(x)
        for idx in range(length):
            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = float(x[idx])
            pose_stamped.pose.position.y = float(y[idx])
            path_msg.poses.append(pose_stamped)
        publisher.publish(path_msg)
    
    def sampled_path(self, x, y, base):
        """Publish the sampled path (markers for visualization) to a given publisher."""
        marker_array = MarkerArray()        
        id_path = 0
        num_path = len(x[0])
        for idx in range(num_path):
            marker = Marker()
            marker.header.frame_id = base  # Set the appropriate frame ID
            marker.ns = 'robot_paths'
            marker.id = id_path
            id_path += 1
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.01
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()
            length_path = len(x)
            for i in range(length_path):
                point = Point()
                point.x = float(x[i][idx])
                point.y = float(y[i][idx])
                marker.points.append(point)
            marker_array.markers.append(marker)
        self.sampled_path_publisher.publish(marker_array)

    def timer_callback(self):
        """Timer callback function that is executed periodically (every 0.001 seconds)."""
        if self.human_position is None:
            self.get_logger().info("Please open the human's position node")
            return


        rotated_x = self.human_position[0] + (self.local_x * np.cos(self.human_position[2]) - self.local_y * np.sin(self.human_position[2]))
        rotated_y = self.human_position[1] + (self.local_x * np.sin(self.human_position[2]) + self.local_y * np.cos(self.human_position[2]))
        self.xgoal = np.array([rotated_x, rotated_y,self.human_position[2]])
       
        self.mppi_planner.update(self.x0, self.human_position ,self.xgoal)
        useq = self.mppi_planner.solve()
        u_curr = useq[0]     

        self.twist.linear.x = float(u_curr[0])
        self.twist.linear.y = float(u_curr[1])
        self.twist.angular.z = float(u_curr[2])
        self.pub_vel.publish(self.twist)

        # self.rollout_states_vis = self.mppi_planner.get_state_rollout()
        # self.publish_path(self.rollout_states_vis[0,:,0], self.rollout_states_vis[0,:,1], self.local_path_publisher, "base_link")
        # self.sampled_path(self.rollout_states_vis[:,:,0].T,self.rollout_states_vis[:,:,1].T,"base_link")

        self.mppi_planner.shift(useq, num_shifts=1)

def main(args=None):
    rclpy.init(args=args)
    planning_node = LocalPlanningNode()
    try:
        rclpy.spin(planning_node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        planning_node.get_logger().info('Shutting down LocalPlanningNode...')
    finally:
        planning_node.destroy_node()
        print("Closing local_planning...")
        rclpy.shutdown()


if __name__ == '__main__':
    main()
