''' Script to follow a given trajectory.
'''

#!/usr/bin/env python

import sys
import rospy as ros

from actionlib import SimpleActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, \
                             FollowJointTrajectoryGoal, FollowJointTrajectoryResult

import pickle
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_file', help='File to load and run')
    args = parser.parse_args()

    ros.init_node('follow_trajectory')
    time.sleep(1)

    # action = ros.resolve_name('~follow_joint_trajectory')
    action = '/effort_joint_trajectory_controller/follow_joint_trajectory'
    client = SimpleActionClient(action, FollowJointTrajectoryAction)
    ros.loginfo("move_to_start: Waiting for '" + action + "' action to come up")
    client.wait_for_server()

    # Open saved trajectory
    with open(args.traj_file, 'rb') as f:
        data = pickle.load(f)
        planned_path = data['robot_traj']
    joint_link_names = [
        'panda_joint1',
        'panda_joint2',
        'panda_joint3',
        'panda_joint4',
        'panda_joint5',
        'panda_joint6',
        'panda_joint7',
    ]
    
    # Get current pose
    topic = ros.resolve_name('franka_state_controller/joint_states')
    ros.loginfo("move_to_start: Waiting for message on topic '" + topic + "'")
    joint_state = ros.wait_for_message(topic, JointState)
    initial_pose = joint_state.position

    # Add all trajectory points to the goal trajectory.
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = joint_link_names
    total_duration = 0.0
    for q_i in planned_path:
        # Find delta_pose from previous pose
        delta_pose = [q_i[j] - initial_pose[j] for j in range(7)]
        max_movement = max((abs(delta_pose[j]) for j in range(7)))

        point = JointTrajectoryPoint()
        # Use either the time to move the furthest joint with 'max_dq' or 500ms,
            # whatever is greater
        interval_time = max(max_movement / ros.get_param('~max_dq', 0.2), 0.2)
        total_duration +=interval_time
        point.time_from_start = ros.Duration.from_sec(
            total_duration
        )

        point.positions = q_i.tolist()
        point.velocities = [dq_i/interval_time for dq_i in delta_pose]

        goal.trajectory.points.append(point)
        # Update initial pose to current waypoint
        initial_pose = q_i
    
    # Set the velocity of the last state to be 0.0
    goal.trajectory.points[-1].velocities = [0.0]*7

    goal.goal_time_tolerance = ros.Duration.from_sec(total_duration)

    ros.loginfo('Sending trajectory Goal to move to a current config')
    client.send_goal_and_wait(goal)

    result = client.get_result()
    if result.error_code != FollowJointTrajectoryResult.SUCCESSFUL:
        ros.logerr('move_to_start: Movement was not successful: ' + {
            FollowJointTrajectoryResult.INVALID_GOAL:
            """
            The joint pose you want to move to is invalid (e.g. unreachable, singularity...).
            Is the 'joint_pose' reachable?
            """,

            FollowJointTrajectoryResult.INVALID_JOINTS:
            """
            The joint pose you specified is for different joints than the joint trajectory controller
            is claiming. Does you 'joint_pose' include all 7 joints of the robot?
            """,

            FollowJointTrajectoryResult.PATH_TOLERANCE_VIOLATED:
            """
            During the motion the robot deviated from the planned path too much. Is something blocking
            the robot?
            """,

            FollowJointTrajectoryResult.GOAL_TOLERANCE_VIOLATED:
            """
            After the motion the robot deviated from the desired goal pose too much. Probably the robot
            didn't reach the joint_pose properly
            """,
        }[result.error_code])
    else:
        ros.loginfo('Successfully moved into target pose')
