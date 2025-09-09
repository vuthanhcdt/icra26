/*
 * Copyright (C) 2020 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <string>

#include <gz/plugin/Register.hh>

#include <gz/common/Profiler.hh>

#include <gz/sim/components/Actor.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Util.hh>

#include "ControlActor.hh"

using namespace gz;
using namespace sim;
using namespace systems;

/// \brief Private ControlActor data class.
class gz::sim::systems::ControlActorPrivate
{
  /// \brief Entity for the actor.
  public: Entity actorEntity{kNullEntity};

  /// \brief Velocity of the actor
  public: double velocity{0.8};

  /// \brief Current robot position
  public: Entity robotEntity{kNullEntity};

  /// \brief Minimum distance in meters to keep away from target.
  public: double minDistance{0.3};


  /// \brief Velocity of the animation dislocation on the X axis, in m/s.
  /// Used to coordinate translational motion with the actor's feet.
  /// TODO(louise) Automatically calculate it from the root node's first and
  /// last keyframes
  public: double animationXVel{2.0};

  /// \brief Time of the last update.
  public: std::chrono::steady_clock::duration lastUpdate{0};

  /// \brief True if currently following
  public: bool following{true};

  /// \brief Last received velocity command.
  public: gz::msgs::Twist cmdVelMsg;
  // Message pose vector
  public: gz::msgs::Pose_V poseVMsgGlobal;
  public: gz::msgs::Pose_V poseVMsgLocal;

  public: std::string topicCmdVel{"cmd_vel_actor"};
  public: std::string topicActorPose{"pose_follow_actor"};
  public: std::string robotname{"scout_mini"};
  public: std::string topicActorMode{"mode_position_actor"};
  public: std::string FrameID{"odom"};
  public: std::string ChildFrameID{"human_link"};
  public: std::string FrameLocalID{"base_link"};
  public: std::string ChildFrameLocalID{"human_local_link"};
  public: std::string topicActorLocalPose{"human_local_pose"};
  public: std::string topicActorGlobalPose{"human_global_pose"};
  public: bool followPosition{false};  
  public: bool publishActorPositionGlobal{true};  
  public: bool publishActorPositionLocal{true}; 
  public: bool publishActorTFGlobal{true}; 
  public: bool publishActorTFLocal{true}; 
  
  public: std::chrono::steady_clock::time_point lastCmdVelTime;

  public: double cmdVelTimeoutSec{0.5};

  public: math::Pose3d followPose{math::Pose3d::Zero};

  public: std::chrono::steady_clock::time_point lastPosePubTime;

  public: double posePubHz{120.0};  // mặc định 10Hz


};

//////////////////////////////////////////////////
ControlActor::ControlActor() :
  System(), dataPtr(std::make_unique<ControlActorPrivate>())
{
}

//////////////////////////////////////////////////
ControlActor::~ControlActor() = default;

//////////////////////////////////////////////////
void ControlActor::Configure(const Entity &_entity,
    const std::shared_ptr<const sdf::Element> &_sdf,
    EntityComponentManager &_ecm,
    EventManager &/*_eventMgr*/)
{
  this->dataPtr->actorEntity = _entity;

  auto actorComp = _ecm.Component<components::Actor>(_entity);
  if (!actorComp)
  {
    gzerr << "Entity [" << _entity << "] is not an actor." << std::endl;
    return;
  }

  if (_sdf->HasElement("min_distance"))
    this->dataPtr->minDistance = _sdf->Get<double>("min_distance");

  if (_sdf->HasElement("animation_x_vel"))
    this->dataPtr->animationXVel = _sdf->Get<double>("animation_x_vel");

  std::string animationName;

  // If animation not provided, use first one from SDF
  if (!_sdf->HasElement("animation"))
  {
    if (actorComp->Data().AnimationCount() < 1)
    {
      gzerr << "Actor SDF doesn't have any animations." << std::endl;
      return;
    }

    animationName = actorComp->Data().AnimationByIndex(0)->Name();
  }
  else
  {
    animationName = _sdf->Get<std::string>("animation");
  }

  if (animationName.empty())
  {
    gzerr << "Can't find actor's animation name." << std::endl;
    return;
  }

  auto animationNameComp = _ecm.Component<components::AnimationName>(_entity);
  if (nullptr == animationNameComp)
  {
    _ecm.CreateComponent(_entity, components::AnimationName(animationName));
  }
  else
  {
    *animationNameComp = components::AnimationName(animationName);
  }
  // Mark as a one-time-change so that the change is propagated to the GUI
  _ecm.SetChanged(_entity,
      components::AnimationName::typeId, ComponentState::OneTimeChange);

  // Set custom animation time from this plugin
  auto animTimeComp = _ecm.Component<components::AnimationTime>(_entity);
  if (nullptr == animTimeComp)
  {
    _ecm.CreateComponent(_entity, components::AnimationTime());
  }

  math::Pose3d initialPose;
  auto poseComp = _ecm.Component<components::Pose>(_entity);
  if (nullptr == poseComp)
  {
    _ecm.CreateComponent(_entity, components::Pose(
        math::Pose3d::Zero));
  }
  else
  {
    initialPose = poseComp->Data();

    // We'll be setting the actor's X/Y pose with respect to the world. So we
    // zero the current values.
    auto newPose = initialPose;
    newPose.Pos().X(0);
    newPose.Pos().Y(0);
    *poseComp = components::Pose(newPose);
  }

  // Having a trajectory pose prevents the actor from moving with the
  // SDF script
  auto trajPoseComp = _ecm.Component<components::TrajectoryPose>(_entity);
  if (nullptr == trajPoseComp)
  {
    // Leave Z to the pose component, control only 2D with Trajectory
    initialPose.Pos().Z(0);
    _ecm.CreateComponent(_entity, components::TrajectoryPose(initialPose));
  }


  // Read topic_actor_pose from SDF
  if (_sdf->HasElement("topic_actor_global_pose"))
  {
    this->dataPtr->topicActorGlobalPose = _sdf->Get<std::string>("topic_actor_global_pose");
  }

  // Read topic_actor_pose from SDF
  if (_sdf->HasElement("topic_actor_local_pose"))
  {
    this->dataPtr->topicActorLocalPose = _sdf->Get<std::string>("topic_actor_local_pose");
  }

  // Create publisher using topicActorPose
  this->poseGlobalPub = this->node.Advertise<gz::msgs::Pose>(this->dataPtr->topicActorGlobalPose);
  this->poseLocalPub = this->node.Advertise<gz::msgs::Pose>(this->dataPtr->topicActorLocalPose);
  this->poseVPubGLobal = this->node.Advertise<gz::msgs::Pose_V>("/tf");
  this->poseVPubLocal = this->node.Advertise<gz::msgs::Pose_V>("/tf");

  // if (_sdf->HasElement("follow_position"))
  // {
  //   this->dataPtr->followPosition = _sdf->Get<bool>("follow_position");
  // }

  if (_sdf->HasElement("publish_actor_position_global"))
  {
    this->dataPtr->publishActorPositionGlobal = _sdf->Get<bool>("publish_actor_position_global");
  }


  if (_sdf->HasElement("publish_actor_position_local"))
  {
    this->dataPtr->publishActorPositionLocal = _sdf->Get<bool>("publish_actor_position_local");
  }


  if (_sdf->HasElement("publish_actor_tf_global"))
  {
    this->dataPtr->publishActorTFGlobal = _sdf->Get<bool>("publish_actor_tf_global");
  }

  if (_sdf->HasElement("publish_actor_tf_local"))
  {
    this->dataPtr->publishActorTFLocal = _sdf->Get<bool>("publish_actor_tf_local");
  }

  if (_sdf->HasElement("cmd_vel_timeout"))
  {
    this->dataPtr->cmdVelTimeoutSec = _sdf->Get<double>("cmd_vel_timeout");
  }

  // Read topic_cmd_vel from SDF
  if (_sdf->HasElement("topic_cmd_vel"))
  {
    this->dataPtr->topicCmdVel = _sdf->Get<std::string>("topic_cmd_vel");
  }

  // Read topic mode actor from SDF
  if (_sdf->HasElement("topic_mode_actor"))
  {
    this->dataPtr->topicActorMode = _sdf->Get<std::string>("topic_mode_actor");
  }

  // Read frame name from SDF
  if (_sdf->HasElement("frame_id"))
  {
    this->dataPtr->FrameID = _sdf->Get<std::string>("frame_id");
  }

  // Read child frame name from SDF
  if (_sdf->HasElement("child_frame_id"))
  {
    this->dataPtr->ChildFrameID = _sdf->Get<std::string>("child_frame_id");
  }

    // Read frame name from SDF
  if (_sdf->HasElement("frame_local_id"))
  {
    this->dataPtr->FrameLocalID = _sdf->Get<std::string>("frame_local_id");
  }

  // Read child frame name from SDF
  if (_sdf->HasElement("child_frame_local_id"))
  {
    this->dataPtr->ChildFrameLocalID = _sdf->Get<std::string>("child_frame_local_id");
  }



  // Read topic_pose_follow_actor from SDF
  if (_sdf->HasElement("topic_pose_follow_actor"))
  {
    this->dataPtr->topicActorPose = _sdf->Get<std::string>("topic_pose_follow_actor");
  }

  if (_sdf->HasElement("pose_pub_hz"))
  {
    this->dataPtr->posePubHz = _sdf->Get<double>("pose_pub_hz");
  }

  if (_sdf->HasElement("robot_name"))
  {
    this->dataPtr->robotname = _sdf->Get<std::string>("robot_name");
  }

  // if (this->dataPtr->publishActorPositionLocal)
  // {
  //   auto robotName = _sdf->Get<std::string>("robot_name");
  //   this->dataPtr->robotEntity = _ecm.EntityByComponents(components::Name(robotName));
  //   if (kNullEntity == this->dataPtr->robotEntity)
  //   {
  //     gzerr << "Failed to find target entity [" << robotName << "]"
  //           << std::endl;
  //     return;
  //   }
  // }

  this->node.Subscribe(this->dataPtr->topicActorMode, &ControlActor::OnMode, this);
  this->node.Subscribe(this->dataPtr->topicCmdVel, &ControlActor::OnCmdVel, this);
  this->node.Subscribe(this->dataPtr->topicActorPose, &ControlActor::OnFollowPose, this);



}

//////////////////////////////////////////////////
void ControlActor::PreUpdate(const UpdateInfo &_info,
    EntityComponentManager &_ecm)
{
  GZ_PROFILE("ControlActor::PreUpdate");

  if (_info.paused)
    return;

  if (this->dataPtr->publishActorPositionLocal)
  {
    this->dataPtr->robotEntity = _ecm.EntityByComponents(components::Name(this->dataPtr->robotname));
    if (kNullEntity == this->dataPtr->robotEntity)
    {
      gzerr << "Failed to find target entity [" << this->dataPtr->robotname << "]"
            << std::endl;
      return;
    }
  }

    
  // TODO(louise) Throttle this system

  // Time delta
  std::chrono::duration<double> dtDuration = _info.simTime -
      this->dataPtr->lastUpdate;
  double dt = dtDuration.count();

  this->dataPtr->lastUpdate = _info.simTime;

  // Current world pose
  auto trajPoseComp = _ecm.Component<components::TrajectoryPose>(
      this->dataPtr->actorEntity);
  auto actorPose = trajPoseComp->Data();
  auto initialPose = actorPose;

  // Current robot
  auto robotPose = _ecm.Component<components::Pose>(this->dataPtr->robotEntity)->Data();
  // Tính vị trí actor tương đối với robot (trong khung robot)
  math::Pose3d relativePose = robotPose.Inverse() * actorPose;

  // gzmsg << "Actor Pose [Local to Robot]: Position=("
  //       << relativePose.Pos().X() << ", "
  //       << relativePose.Pos().Y() << ", "
  //       << relativePose.Pos().Z() << "), Rotation (Yaw)="
  //       << relativePose.Rot().Yaw() << std::endl;      

  this->ActorPose(actorPose, relativePose,_info);

  auto now = std::chrono::steady_clock::now();
  double timeSinceLastCmd = std::chrono::duration<double>(std::chrono::steady_clock::now() - this->dataPtr->lastCmdVelTime).count();

  if (timeSinceLastCmd > this->dataPtr->cmdVelTimeoutSec)
  {
    // Reset velocity to zero if timeout exceeded
    this->dataPtr->cmdVelMsg.mutable_linear()->set_x(0);
    this->dataPtr->cmdVelMsg.mutable_linear()->set_y(0);
    this->dataPtr->cmdVelMsg.mutable_linear()->set_z(0);

    this->dataPtr->cmdVelMsg.mutable_angular()->set_x(0);
    this->dataPtr->cmdVelMsg.mutable_angular()->set_y(0);
    this->dataPtr->cmdVelMsg.mutable_angular()->set_z(0);
  }


  if (this->dataPtr->followPosition)
  {

    // Direction to target
    auto dir = this->dataPtr->followPose.Pos() - actorPose.Pos();

    // auto dir = targetPose.Pos() - actorPose.Pos();
    dir.Z(0);

    // Stop if too close to target
    if (dir.Length() <= this->dataPtr->minDistance)
    {
      return;
    }

    if (!this->dataPtr->following)
    {
      this->dataPtr->following = true;
    }

    dir.Normalize();

    // Towards target
    math::Angle yaw = atan2(dir.Y(), dir.X());
    yaw.Normalize();

    actorPose.Pos() += dir * this->dataPtr->velocity * dt;
    actorPose.Pos().Z(0);
    actorPose.Rot() = math::Quaterniond(0, 0, yaw.Radian());

  }
  else
  {
    auto linearVel = this->dataPtr->cmdVelMsg.linear();
    auto angularVel = this->dataPtr->cmdVelMsg.angular();

    double yaw = actorPose.Rot().Yaw();

    double worldVx = cos(yaw) * linearVel.x() - sin(yaw) * linearVel.y();
    double worldVy = sin(yaw) * linearVel.x() + cos(yaw) * linearVel.y();

    actorPose.Pos().X() += worldVx * dt;
    actorPose.Pos().Y() += worldVy * dt;
    actorPose.Pos().Z(0);

    double newYaw = yaw + angularVel.z() * dt;
    actorPose.Rot() = math::Quaterniond(0, 0, newYaw);
    
  }

  // Distance traveled is used to coordinate motion with the walking
  // animation
  double distanceTraveled = (actorPose.Pos() - initialPose.Pos()).Length();

  // Update actor root pose
  *trajPoseComp = components::TrajectoryPose(actorPose);
  // Mark as a one-time-change so that the change is propagated to the GUI
  _ecm.SetChanged(this->dataPtr->actorEntity,
      components::TrajectoryPose::typeId, ComponentState::OneTimeChange);


  // Update actor bone trajectories based on animation time
  auto animTimeComp = _ecm.Component<components::AnimationTime>(
      this->dataPtr->actorEntity);

  auto animTime = animTimeComp->Data() +
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(distanceTraveled *
    this->dataPtr->animationXVel));
  *animTimeComp = components::AnimationTime(animTime);
  // Mark as a one-time-change so that the change is propagated to the GUI
  _ecm.SetChanged(this->dataPtr->actorEntity,
      components::AnimationTime::typeId, ComponentState::OneTimeChange);
}

void ControlActor::ActorPose(const math::Pose3d &_pose, const math::Pose3d &_relativepose, const UpdateInfo &_info)
{
  gz::msgs::Pose msg;
  gz::msgs::Pose local_msg;
  auto now = std::chrono::steady_clock::now();
  double dt = std::chrono::duration<double>(now - this->dataPtr->lastPosePubTime).count();
  double interval = 1.0 / this->dataPtr->posePubHz;

  if (dt < interval)
    return;  // chưa đến lúc publish

  this->dataPtr->lastPosePubTime = now;

  auto *position = msg.mutable_position();
  position->set_x(_pose.Pos().X());
  position->set_y(_pose.Pos().Y());
  position->set_z(_pose.Pos().Z());

  auto *orientation = msg.mutable_orientation();
  orientation->set_x(_pose.Rot().X());
  orientation->set_y(_pose.Rot().Y());
  orientation->set_z(_pose.Rot().Z());
  orientation->set_w(_pose.Rot().W());
  

  // Relative pose (local to robot)
  auto *local_pos = local_msg.mutable_position();
  local_pos->set_x(_relativepose.Pos().X());
  local_pos->set_y(_relativepose.Pos().Y());
  local_pos->set_z(_relativepose.Pos().Z());

  auto *local_rot = local_msg.mutable_orientation();
  local_rot->set_x(_relativepose.Rot().X());
  local_rot->set_y(_relativepose.Rot().Y());
  local_rot->set_z(_relativepose.Rot().Z());
  local_rot->set_w(_relativepose.Rot().W());


  if (this->dataPtr->publishActorPositionGlobal)
    this->poseGlobalPub.Publish(msg);

  if (this->dataPtr->publishActorPositionLocal)
    this->poseLocalPub.Publish(local_msg);

  if (this->dataPtr->publishActorTFGlobal)
  {
    this->poseVMsgGlobal.Clear();
    auto *poseMsgGlobal = this->poseVMsgGlobal.add_pose();
    *poseMsgGlobal = msg;
    poseMsgGlobal->mutable_position()->set_z(1.0);
    auto header = poseMsgGlobal->mutable_header();
    
    header->mutable_stamp()->CopyFrom(convert<msgs::Time>(_info.simTime));

    auto frame = header->add_data();
    frame->set_key("frame_id");
    frame->add_value(this->dataPtr->FrameID); 

    auto childFrame = header->add_data();
    childFrame->set_key("child_frame_id");
    childFrame->add_value(this->dataPtr->ChildFrameID); 

    this->poseVPubGLobal.Publish(this->poseVMsgGlobal);
  }

  if (this->dataPtr->publishActorTFLocal)
  {
    this->poseVMsgGlobal.Clear();
    auto *poseMsgGlobal = this->poseVMsgGlobal.add_pose();
    *poseMsgGlobal = local_msg;
    poseMsgGlobal->mutable_position()->set_z(1.0);
    auto header = poseMsgGlobal->mutable_header();
    
    header->mutable_stamp()->CopyFrom(convert<msgs::Time>(_info.simTime));

    auto frame = header->add_data();
    frame->set_key("frame_id");
    frame->add_value(this->dataPtr->FrameLocalID); 

    auto childFrame = header->add_data();
    childFrame->set_key("child_frame_id");
    childFrame->add_value(this->dataPtr->ChildFrameLocalID); 

    this->poseVPubGLobal.Publish(this->poseVMsgGlobal);
  }



}


void ControlActor::OnCmdVel(const gz::msgs::Twist &_msg)
{
  this->dataPtr->cmdVelMsg = _msg;
  
  if (this->dataPtr->followPosition)
  {
    this->dataPtr->velocity = this->dataPtr->cmdVelMsg.linear().x();
  }

  this->dataPtr->lastCmdVelTime = std::chrono::steady_clock::now();
}


void ControlActor::OnMode(const gz::msgs::Boolean &_msg)
{
  this->dataPtr->followPosition =_msg.data();
}


void ControlActor::OnFollowPose(const gz::msgs::Pose &_msg)
{
  this->dataPtr->followPose = math::Pose3d(
      math::Vector3d(_msg.position().x(), _msg.position().y(), _msg.position().z()),
      math::Quaterniond(_msg.orientation().w(), _msg.orientation().x(), _msg.orientation().y(), _msg.orientation().z())
  );

}




GZ_ADD_PLUGIN(ControlActor, System,
  ControlActor::ISystemConfigure,
  ControlActor::ISystemPreUpdate
)

GZ_ADD_PLUGIN_ALIAS(ControlActor, "gz::sim::systems::ControlActor")


