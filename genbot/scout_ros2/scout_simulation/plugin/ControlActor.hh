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
#ifndef GZ_SIM_SYSTEMS_CONTROLACTOR_HH_
#define GZ_SIM_SYSTEMS_CONTROLACTOR_HH_

#include <memory>
#include <gz/sim/config.hh>
#include <gz/sim/System.hh>

#include <gz/transport/Node.hh>
#include <gz/msgs/pose.pb.h>  // Gazebo pose protobuf message
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/twist.pb.h>
#include <gz/msgs/header.pb.h>
#include <gz/msgs/boolean.pb.h>

namespace gz
{
namespace sim
{
// Inline bracket to help doxygen filtering.
inline namespace GZ_SIM_VERSION_NAMESPACE {
namespace systems
{
  // Forward declarations.
  class ControlActorPrivate;

  /// \class ControlActor ControlActor.hh gz/sim/systems/ControlActor.hh
  /// \brief Make an actor follow a target entity in the world.
  ///
  /// ## System Parameters
  ///
  /// - `<target>`: Name of entity to follow.
  ///
  /// - `<min_distance>`: Distance in meters to keep from target's origin.
  ///
  /// - `<max_distance>`: Distance in meters from target's origin when to stop
  ///   following. When the actor is back within range it starts following
  ///   again.
  ///
  /// - `<velocity>`: Actor's velocity in m/s
  ///
  /// - `<animation>`: Actor's animation to play. If empty, the first animation
  ///   in the model will be used.
  ///
  /// - `<animation_x_vel>`: Velocity of the animation on the X axis. Used to
  ///   coordinate translational motion with the actor's animation.
  class ControlActor:
    public System,
    public ISystemConfigure,
    public ISystemPreUpdate
  {
    /// \brief Constructor
    public: explicit ControlActor();

    /// \brief Destructor
    public: ~ControlActor() override;

    // Documentation inherited
    public: void Configure(const Entity &_entity,
                           const std::shared_ptr<const sdf::Element> &_sdf,
                           EntityComponentManager &_ecm,
                           EventManager &_eventMgr) final;

    /// Documentation inherited
    public: void PreUpdate(const UpdateInfo &_info,
                           EntityComponentManager &_ecm) final;

    /// \brief Private data pointer.
    private: std::unique_ptr<ControlActorPrivate> dataPtr;
    private: void ActorPose(const math::Pose3d &_pose, const math::Pose3d &_relativepose, const UpdateInfo &_info);
    private: gz::transport::Node node;
    private: gz::transport::Node::Publisher poseGlobalPub;
    private: gz::transport::Node::Publisher poseLocalPub;
    private: gz::transport::Node::Publisher poseVPubGLobal;
    private: gz::transport::Node::Publisher poseVPubLocal;
    // Message pose vector
    private: gz::msgs::Pose_V poseVMsgGlobal;
    private: gz::msgs::Pose_V poseVMsgLocal;
    private: void OnCmdVel(const gz::msgs::Twist &_msg);
    private: void OnMode(const gz::msgs::Boolean &_msg);
    private: void OnFollowPose(const gz::msgs::Pose &_msg);

  };
  }
}
}
}
#endif
