#!/usr/bin/env python
# encoding: utf-8
import rospy
import os
import threading
from geometry_msgs.msg import Twist
from time import sleep
from autopilot_common import *
from dynamic_reconfigure.server import Server
from dynamic_reconfigure.client import Client
from arm_autopilot.cfg import AutoPilotPIDConfig
import cv2 as cv
import numpy as np


class LineDetect:
	def __init__(self):
    	rospy.on_shutdown(self.cancel)
    	rospy.init_node("LineDetect", anonymous=False)
    	self.ros_ctrl = ROSCtrl()
    	self.color = color_follow()
    	self.hsv_yaml = HSVYaml()
    	self.dyn_update = False
    	self.Calibration = False
    	self.select_flags = False
    	self.gripper_state = False
    	self.location_state = False
    	self.Track_state = 'identify'
    	self.windows_name = 'frame'
    	self.placement_completed = False
    	self.path_taken = []
    	self.path_index = 0
    	self.color.target_color_name = 'red'
    	#self.color_name_list = ['red', 'green', 'blue', 'yellow']
    	self.color_name_list = ['red', 'blue']
    	self.hsv_value = ()
    	self.color_cfg_src = self.index = self.cols = self.rows = 0
    	self.Mouse_XY = (0, 0)
    	self.Roi_init = ()
    	Server(AutoPilotPIDConfig, self.dyn_cfg_callback)
    	self.dyn_client = Client("LineDetect", timeout=60)
    	self.scale = 1000.0
    	self.FollowLinePID = (30.0, 0.0, 60.0)
    	self.linear = 0.08
    	self.PID_init()
    	self.joints_init = [90, 120, 0, 0, 90, 30]
    	#	self.color.color_hsv_list[self.color_name_list[i]] = self.hsv_yaml.read_hsv(self.color_name_list[i])
    	cv.namedWindow(self.windows_name, cv.WINDOW_AUTOSIZE)
    	cv.setMouseCallback(self.windows_name, self.onMouse, 0)
    	self.ros_ctrl.pubArm(self.joints_init)
    	#added 12/4
    	self.turn_history = []

#the process function handles inputs, colors and logs turns
	def process(self, rgb_img, action):
    	if action == 32 or self.ros_ctrl.joy_action==2:
        	self.Track_state = 'tracking'
        	self.Calibration = False
        	self.dyn_update = True
        	self.ros_ctrl.pubArm(self.joints_init)
    	elif action == ord('r') or action == ord('R'): self.Reset()
    	elif action == ord('q') or action == ord('Q'): self.cancel()
    	elif action == ord('c') or action == ord('C'):
        	self.Calibration = not self.Calibration
        	self.dyn_update = True
    	elif action == ord('i') or action == ord('I'):
        	self.Track_state = "identify"
        	self.Calibration = False
        	self.dyn_update = True
    	elif action == ord('f') or action == ord('F'):
        	color_index = self.color_name_list.index(self.color.target_color_name)
        	if color_index >= 3: color_index = 0
        	else: color_index += 1
        	self.color.target_color_name = self.color_name_list[color_index]
        	#self.hsv_value = self.hsv_yaml.read_hsv(self.color.target_color_name)
        	self.dyn_update = True
    	if self.Track_state == 'init':
        	cv.setMouseCallback(self.windows_name, self.onMouse, 0)
        	if self.select_flags == True:
            	cv.line(rgb_img, self.cols, self.rows, (255, 0, 0), 2)
            	cv.rectangle(rgb_img, self.cols, self.rows, (0, 255, 0), 2)
            	if self.Roi_init[0] != self.Roi_init[2] and self.Roi_init[1] != self.Roi_init[3]:
                	rgb_img, self.hsv_value = self.color.Roi_hsv(rgb_img, self.Roi_init)
                	self.color.color_hsv_list[self.color.target_color_name] = self.hsv_value
                	#self.hsv_yaml.write_hsv(self.color.target_color_name, self.hsv_value)
                	self.dyn_update = True
            	else: self.Track_state = 'init'
    	if self.Track_state != 'init' and len(self.hsv_value) != 0:
        	if self.Calibration:
            	self.color.msg_box = {}
            	self.color.line_follow(rgb_img, self.color.target_color_name, self.hsv_value)
            	#added dec 04
            	if self.Track_state == 'tracking' and len(self.color.binary) != 0:
                	intersection_type = self.detect_intersection(self.color.binary)
                	if intersection_type:
                    	rospy.loginfo(f"Intersection detected: {intersection_type}")
                    	self.handle_intersection(intersection_type)
                	else:
                    	rospy.loginfo("Moving straight on the line.")
                    	if len(self.path_taken) == 0 or self.path_taken[-1] != "Straight":
                        	self.path_taken.append("Straight")
        	else:
            	for i in range(len(self.color_name_list)):
                	threading.Thread(target=self.color.line_follow,
                                 	args=(rgb_img, self.color_name_list[i], self.color.color_hsv_list[self.color_name_list[i]],)).start()
    	if self.Track_state == 'tracking' and len(self.color.msg_circle) != 0 and \
            	not self.ros_ctrl.Joy_active and not self.gripper_state:
        	for i in self.color.msg_circle.keys():
            	if i == self.color.target_color_name and not self.location_state:
                	threading.Thread(target=self.execute, args=(self.color.msg_circle[self.color.target_color_name],)).start()
        	for i in self.color.msg_box.keys():
            	if i != self.color.target_color_name and len(self.color.msg_box) != 0 and len(self.color.msg_box[i]) != 0:
                	(point_x, point_y), circle_r = cv.minEnclosingCircle(self.color.msg_box[i])
                	print("circle_r: ", circle_r)
                	if (circle_r > 45 and circle_r < 100):
                    	#print("circle_r: ", circle_r)
                    	print("point_x, point_y: ", point_x, " ", point_y)
                    	threading.Thread(target=self.Wrecker, args=(point_x, point_y,)).start()
                	#Added Nov 21
                	if (self.placement_completed == False):
                    	if (circle_r > 150):
                        	print("point_x, point_y: ", point_x, " ", point_y)
                        	#threading.Thread(target=self.arm_dropper, args=(point_x, point_y,)).start()
                        	self.arm_dropper()
                        	sleep(2)
                   	 
            	else:
                	self.index += 1
                	if self.index >= 20: self.location_state = False
    	else:
        	if self.ros_ctrl.RobotRun_status == True: self.ros_ctrl.pubVel(0, 0)
    	if self.dyn_update == True: self.dyn_cfg_update()
    	return self.color.binary

	#the wrecker function controls the arm and picks up the object at an x and y point
	def Wrecker(self, point_x, point_y):
    	self.index = 0
    	self.location_state = True
    	self.placement_completed = False
    	if self.ros_ctrl.Buzzer_state == True: self.ros_ctrl.pubBuzzer(False)
    	if abs(point_x - 320) < 40: point_x = 320 #10
    	if abs(point_y - 400) < 40: point_y = 400
    	if abs(point_x - 320) < 10 and abs(point_y - 400) < 10: #10 20
        	#sleep(0.3)
        	if self.ros_ctrl.RobotRun_status == True: self.ros_ctrl.pubVel(0, 0)
        	self.gripper_state = True
        	sleep(0.3)
        	np_array = np.array([linear([320, 90], [343.5, 95])])
        	pos1 = np.dot(np_array, np.array([point_x, 1])).squeeze().tolist()
        	joints = [pos1, 7.0, 60.0, 38.0, 90]
        	# joints = [pos1, 8.0, 49.0, 46.0, 90]
        	# print ("joints: ", joints, type(joints))
        	if len(joints) != 0:
            	#print(point_x)
            	#print(point_y)
            	self.arm_gripper(joints)
        	self.color.msg_box = {}
        	self.color.msg_circle = {}
        	sleep(2.5)
        	self.gripper_state = False
        	self.location_state = False
    	else: self.robot_location(point_x, point_y)

	def robot_location(self, point_x, point_y):
    	[y, x] = self.PID_controller.update([(point_x - 320) / 10.0, (point_y - 400) / 10.0])
    	# print ("point_x: {}, point_y: {}".format(point_x, point_y))
    	#print ("x: {},y: {}".format(x, y))#15
    	if x >= 0.10: x = 0.10 #0.10
    	elif x <= -0.10: x = -0.10
    	if y >= 0.10: y = 0.10
    	elif y <= -0.10: y = -0.10
    	self.ros_ctrl.pubVel(x, y)
    	self.ros_ctrl.RobotRun_status = True

	def arm_gripper(self, joints):
    	joints.append(30)
    	self.ros_ctrl.pubArm(joints, run_time=8000)
    	sleep(2)
    	self.ros_ctrl.pubArm([], id=6, angle=132, run_time=1000)
    	sleep(2)
    	self.ros_ctrl.pubArm([90, 120, 0, 0, 90, 134], run_time=2000)
    	sleep(1)
    	self.ros_ctrl.pubVel(0.0,0.0,1.0)
    	sleep(3.6)
    	self.ros_ctrl.pubVel(0.0,0.0,0.0)
    	self.go_back_to_start()
   	 
   	 
	def arm_dropper(self):
    	rospy.loginfo("Placing the block...")
    	self.ros_ctrl.pubVel(0.2, 0.0, 0.0)
    	sleep(1)
    	self.ros_ctrl.pubVel(0.0, 0.0, 0.0)
    	sleep(1)
    	self.ros_ctrl.pubArm([90, 25, 20, 60, 88, 132], run_time=2000)
    	sleep(2)
    	self.ros_ctrl.pubArm([], id=6, angle=30, run_time=1000)  # Open gripper to release the block
    	sleep(2)
    	self.ros_ctrl.pubArm([90, 120, 0, 0, 90, 30], run_time=2000)
    	sleep(1)
    	self.placement_completed = True  
    	rospy.loginfo("Block placed successfully!")
    	self.ros_ctrl.pubVel(0.0,0.0,1.0)
    	sleep(3.6)
    	self.ros_ctrl.pubVel(0.0,0.0,0.0)
  	 
    	
	def execute(self, circle):
    	#the execute function controls the robots movement based on the circle's position
    	self.index = 0
    	if len(circle) == 0: self.ros_ctrl.pubVel(0, 0)
    	else:
        	if self.ros_ctrl.warning > 10:
            	rospy.loginfo("Obstacles ahead !!!")
            	self.ros_ctrl.pubVel(0, 0)
            	self.ros_ctrl.pubBuzzer(True)
            	self.ros_ctrl.Buzzer_state = True
        	else:
            	[z_Pid, _] = self.PID_controller.update([(circle[0] - 320) / 16, 0])
            	if self.ros_ctrl.img_flip == True: z = -z_Pid
            	else: z = z_Pid
            	x = self.linear
            	if self.ros_ctrl.Buzzer_state == True: self.ros_ctrl.pubBuzzer(False)
            	self.ros_ctrl.pubVel(x, 0, z=z)
            	# rospy.loginfo("point_x: {},linear: {}, z_Pid: {}".format(circle[0], x, z))
        	self.ros_ctrl.RobotRun_status = True

	def dyn_cfg_update(self):
#the dyn_cfg_update function updates the parameters of the dynamic configurations for the robots color and calibration settings
    	hsv = self.color.color_hsv_list[self.color.target_color_name]
    	params = {'Calibration': self.Calibration,
              	'Color': self.color.target_color_name,
              	'Hmin': hsv[0][0], 'Hmax': hsv[1][0],
              	'Smin': hsv[0][1], 'Smax': hsv[1][1],
              	'Vmin': hsv[0][2], 'Vmax': hsv[1][2]}
    	self.dyn_client.update_configuration(params)
    	self.dyn_update = False

	def dyn_cfg_callback(self, config, level)
    	self.scale = config['scale']
    	self.linear = .08#config['linear']
    	self.ros_ctrl.LaserAngle = config['LaserAngle']
    	self.ros_ctrl.ResponseDist = config['ResponseDist']
    	self.FollowLinePID = (config['Kp'], config['Ki'], config['Kd'])
    	if self.Track_state != 'mouse':
        	self.hsv_value = (
            	(config['Hmin'], config['Smin'], config['Vmin']),
            	(config['Hmax'], config['Smax'], config['Vmax']))
    	else:self.Track_state = 'identify'
    	self.Calibration = config["Calibration"]
    	color_cfg = config["Color"]
    	if self.color_cfg_src != color_cfg:
        	#self.hsv_value = self.hsv_yaml.read_hsv(self.color.target_color_name)
        	# print ("self.color_cfg_src: {},color_cfg: {}".format(self.color_cfg_src, color_cfg))
        	self.dyn_update = True
        	self.color_cfg_src = color_cfg
    	#else:
        	#self.hsv_yaml.write_hsv(self.color.target_color_name, self.hsv_value)
    	self.color.color_hsv_list[self.color.target_color_name] = self.hsv_value
    	self.color.target_color_name = self.color_name_list[color_cfg]
    	self.PID_init()
    	return config

	def putText_img(self, frame):
    	if self.Calibration: cv.putText(frame, "Calibration", (500, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 200), 1)
    	cv.putText(frame, self.color.target_color_name, (300, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 200), 1)
    	msg_index = len(self.color.msg_box.keys())
    	if msg_index != 0:
        	for i in self.color.msg_box.keys():
            	try: self.color.add_box(i)
            	except Exception as e: print ("e: ", e)
    	self.ros_ctrl.pubImg(frame)
    	return frame

	def onMouse(self, event, x, y, flags, param):
    	if x > 640 or y > 480: return
    	if event == 1:
        	self.Track_state = 'init'
        	self.select_flags = True
        	self.Calibration  = True
        	self.Mouse_XY = (x, y)
    	if event == 4:
        	self.select_flags = False
        	self.Track_state = 'mouse'
    	if self.select_flags == True:
        	self.cols = min(self.Mouse_XY[0], x), min(self.Mouse_XY[1], y)
        	self.rows = max(self.Mouse_XY[0], x), max(self.Mouse_XY[1], y)
        	self.Roi_init = (self.cols[0], self.cols[1], self.rows[0], self.rows[1])

	def Reset(self):
    	self.PID_init()
    	self.color.binary = ()
    	self.color.msg_box = {}
    	self.Track_state = 'init'
    	self.color.msg_circle = {}
    	self.gripper_state = False
    	self.ros_ctrl.Joy_active = False
    	self.Mouse_XY = (0, 0)
    	self.ros_ctrl.pubVel(0, 0)
    	self.ros_ctrl.pubBuzzer(False)
    	rospy.loginfo("Reset success!!!")

	def PID_init(self):
    	self.PID_controller = simplePID(
        	[0, 0],
        	[self.FollowLinePID[0] / (self.scale), self.FollowLinePID[0] / (self.scale)],
        	[self.FollowLinePID[1] / (self.scale), self.FollowLinePID[1] / (self.scale)],
        	[self.FollowLinePID[2] / (self.scale), self.FollowLinePID[2] / (self.scale)])

	def cancel(self):
    	self.Reset()
    	self.ros_ctrl.cancel()
    	print("Shutting down this node.")
   	 
	def image_callback(self, msg):
    	frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    	self.binary_image = self.process_frame(frame)
    	intersection = self.detect_intersection(self.binary_image)
    	if intersection:
        	rospy.loinfo(f"Detected Intersection: {intersection}")
    
	def process_frame(self, frame):
    	hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    	lower_red1 = np.array([0, 120, 70])
    	upper_red1 = np.array([10, 255, 255])
    	lower_red2 = np.array([170, 120, 70])
    	upper_red2 = np.array([180, 255, 255])
    	mask1 = cv.inRange(hsv_frame, lower_red1, upper_red1)
    	mask2 = cv.inRange(hsv_frame, lower_red2, upper_red2)
    	red_mask = cv.bitwise_or(mask1, mask2)
    	return red_mask
    
	def detect_intersection(self, binary_image):
    	height, width = binary_image.shape
    	mid_row, mid_col = height // 2, width // 2
	#added 12/5
    
    	left_region = binary_image[0:480, 0,240]
    	right_region = binary_image[0:480, 400:640]
    	center_region = binary_image[0:480, 300:360]
    	#added 12/4
    	#left_region = binary_image[mid_row - 10:mid_row + 10, :mid_col]
    	#right_region = binary_image[mid_row - 10:mid_row + 10, mid_col:]
    	#center_region = binary_image[mid_row:, mid_col - 10:mid_col + 10]

    
    	left_pixels = cv.countNonZero(left_region)
    	right_pixels = cv.countNonZero(right_region)
    	center_pixels = cv.countNonZero(center_region)

    
    	threshold = 15

    	if center_pixels > threshold:
        	if left_pixels > threshold and right_pixels > threshold:
            	print("Cross intersection")
            	return "Cross_intersection"
        	elif left_pixels > threshold:
            	self.path_taken.append("Left")
            	print("left turn")
            	return "Left_turn"
        	elif right_pixels > threshold:
            	self.path_taken.append("Right")
            	print("Right turn")
            	return "Right_turn"
        	else:
            	print("T-intersection")
            	return "T_intersection"
    	elif center_pixels > threshold and left_pixels <= threshold and right_pixels <= threshold:
        	print("Straight")
        	return "Straight"
    	return None
   	 
	def handle_intersection(self, intersection_type):
    	rospy.loginfo(f"Handling intersection: {intersection_type}")
    	available_paths = {
        	0: ["Straight", "Left", "Right"],  # First run priority
        	1: ["Left", "Right", "Straight"],  # Second run priority
        	2: ["Right", "Straight", "Left"]   # Third run priority
    	}

    	# Determine current priority based on the path index
    	current_priority = available_paths[self.path_index % len(available_paths)]

    	# Handle specific intersection types
    	if intersection_type == "Left_turn":
        	if "Left" in current_priority and "Left" not in self.path_taken:
            	self.path_taken.append("Left")
            	self.turn_left()
        	else:
            	rospy.loginfo("Skipping Left turn, taking next priority.")
            	self.go_straight()  # Default to Straight or next available path

    	elif intersection_type == "Right_turn":
        	if "Right" in current_priority and "Right" not in self.path_taken:
            	self.path_taken.append("Right")
            	self.turn_right()
        	else:
            	rospy.loginfo("Skipping Right turn, taking next priority.")
            	self.go_straight()

    	elif intersection_type in ["T_intersection", "Cross_intersection"]:
        	# Try all paths in priority order for the current run
        	for direction in current_priority:
            	if direction not in self.path_taken:
                	self.path_taken.append(direction)
                	print("direction: ", direction)
                	#print("self.path_taken: ", self.path_taken)
                	if direction == "Straight":
                    	self.go_straight()
                	elif direction == "Left":
                    	self.turn_left()
                	elif direction == "Right":
                    	self.turn_right()
                	return
        	rospy.loginfo("All paths explored for this intersection. Defaulting to Straight.")
        	self.go_straight()  # Default if all options are exhausted

    	else:
        	rospy.loginfo("No valid intersection detected.")
 
    	#added 12/4
	def display_path_taken(self):
    	rospy.loginfo(f"Turn History: {self.path_taken}")
   	 
    	#added 12/04
	def go_back_to_start(self):
    	rospy.loginfo("Returning to the starting point...")
    	reversed_history = self.path_taken[::-1]

    	for turn in reversed_history:
        	rospy.loginfo(f"Reversing action: {turn}")
        	if turn == "Left":
            	self.turn_right()
        	elif turn == "Right":
            	self.turn_left()
        	elif turn == "Straight":
            	self.go_straight()
        	rospy.sleep(0.5)

    	rospy.loginfo("Returned to the starting point!")
    	self.path_index += 1  # Increment path index
    	self.path_taken = []  # Clear path for the next run

   	 
	def turn_left(self):
    	rospy.loginfo("Turning left...")
    	self.ros_ctrl.pubVel(0.0, 0.0, 1.0)
    	rospy.sleep(1.5)
    	self.ros_ctrl.pubVel(0.0, 0.0, 0.0)
    	rospy.loginfo("Left turn complete.")

	def turn_right(self):
    	rospy.loginfo("Turning right...")
    	self.ros_ctrl.pubVel(0.0, 0.0, -1.0)
    	rospy.sleep(1.5)
    	self.ros_ctrl.pubVel(0.0, 0.0, 0.0)
    	rospy.loginfo("Right turn complete.")

	def go_straight(self):
    	rospy.loginfo("Moving straight...")
    	self.ros_ctrl.pubVel(0.2, 0.0, 0.0)
    	rospy.sleep(2)
    	self.ros_ctrl.pubVel(0.0, 0.0, 0.0)
    	rospy.loginfo("Straight movement complete.")
   	 
   	 
    


if __name__ == '__main__':
	line_detect = LineDetect()
	capture = cv.VideoCapture(0)
	cv_edition = cv.__version__
	if cv_edition[0] == '3': capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'XVID'))
	else: capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))
	capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
	capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
	while capture.isOpened():
    	start = time.time()
    	ret, frame = capture.read()
    	action = cv.waitKey(10) & 0xFF
    	if line_detect.ros_ctrl.img_flip == True: frame = cv.flip(frame, 1)
    	line_detect.color.frame = frame
    	binary = line_detect.process(frame, action)
    	end = time.time()
    	fps = 1 / (end - start)
    	text = "FPS : " + str(int(fps))
    	cv.putText(frame, text, (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 200), 1)
    	frame = line_detect.putText_img(frame)
    	if len(binary) != 0: cv.imshow(line_detect.windows_name, ManyImgs(1, ([frame, binary])))
    	else: cv.imshow(line_detect.windows_name, frame)
    	if action == ord('q') or action == 113: break
	capture.release()
	cv.destroyAllWindows()

