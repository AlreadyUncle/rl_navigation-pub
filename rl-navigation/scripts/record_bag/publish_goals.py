# use with roslaunch teb_local_planner_tutorials robot_diff_drive_in_stage.launch

import rospy
import numpy as np
import actionlib
from actionlib_msgs.msg import *

from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist, Pose, Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int8
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionGoal


rospy.init_node('publish_goal_node')
rospy.loginfo("Setting Goal Publisher...")

move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)


#goal_publisher = rospy.Publisher('/move_base/goal', MoveBaseActionGoal, queue_size = 1) ##Set queue size
goal = MoveBaseGoal()
goal.target_pose.pose.position.x = 5.0
goal.target_pose.pose.position.y = 5.0
theta = (2*np.random.rand() - 1)*np.pi
goal_orientation = quaternion_from_euler(0, 0, theta)
goal.target_pose.pose.orientation.x = goal_orientation[0]
goal.target_pose.pose.orientation.y = goal_orientation[1]
goal.target_pose.pose.orientation.z = goal_orientation[2]
goal.target_pose.pose.orientation.w = goal_orientation[3]
goal.target_pose.header.frame_id = 'map'
goal.target_pose.header.stamp = rospy.Time.now()



move_base.send_goal(goal)

