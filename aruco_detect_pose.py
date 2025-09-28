#!/usr/bin/env python
  
# from __future__ import print_function 
import cv2 # OpenCV contrib library 4.6
import numpy as np # Numpy v <2.0
from scipy.spatial.transform import Rotation as R # Standar SciPy library
import math # Math library included with Python
 
# Dictionary used to generate printed ArUco marker
aruco_name = "DICT_7X7_1000"
 
# ArUco dictionaries built into OpenCV. Substitute as desired.
ARUCO_DICT = {
  "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
  "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
  "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
  "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
  "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
  "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
  "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
  "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
  "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
  "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
  "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
  "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
  "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
  "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
  "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
  "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
  "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}
 
# Width of ArUco marker in meters 
marker_width = 0.05
 
# Calibration parameters yaml file
calibration_params = 'calibration.yaml'
 
def euler_from_quaternion(x, y, z, w):
  """ Convert a quaternion into euler angles (roll, pitch, yaw) """
  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + y * y)
  roll_x = math.atan2(t0, t1)
      
  t2 = +2.0 * (w * y - z * x)
  t2 = +1.0 if t2 > +1.0 else t2
  t2 = -1.0 if t2 < -1.0 else t2
  pitch_y = math.asin(t2)
      
  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  yaw_z = math.atan2(t3, t4)
      
  return roll_x, pitch_y, yaw_z # in radians
 
def main():
  """  Main method of the program.  """
  # Check that we have a valid ArUco marker
  if ARUCO_DICT.get(aruco_name, None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(
      args["type"]))
    sys.exit(0)
 
  # Load camera parameters from saved file
  cv_file = cv2.FileStorage(
    calibration_params, cv2.FILE_STORAGE_READ) 
  mtx = cv_file.getNode('K').mat()
  dst = cv_file.getNode('D').mat()
  cv_file.release()
     
  # Load ArUco dictionary
  print("[INFO] detecting '{}' markers...".format(
    aruco_name))
  this_aruco_dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_name])
  this_aruco_parameters = cv2.aruco.DetectorParameters_create()
   
  # Start video stream
  cap = cv2.VideoCapture(0)
   
  while(True):
  
    # Capture frame, method returns bool and frame
    ret, frame = cap.read()  
    # Detect ArUco markers in frame
    (corners, marker_ids, rejected) = cv2.aruco.detectMarkers(
      frame, this_aruco_dictionary, parameters=this_aruco_parameters)
       
    # Check if ArUco marker detected
    if marker_ids is not None:
 
      # Draw square around detected markers in frame
      cv2.aruco.drawDetectedMarkers(frame, corners, marker_ids)
       
      # Get rotation and translation vectors
      rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        marker_width,
        mtx,
        dst)
         
      # Get pose for ArUco marker with respect to camera
      # x-axis points right y-axis points down z-axis points out of camera
      for i, marker_id in enumerate(marker_ids):
       
        # Save translation
        translate_x = tvecs[i][0][0]
        translate_y = tvecs[i][0][1]
        translate_z = tvecs[i][0][2]
 
        # Save rotation 
        rotation_matrix = np.eye(4)
        rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
        r = R.from_matrix(rotation_matrix[0:3, 0:3])
        quat = r.as_quat()   
         
        # Assign Quaternion values
        rotate_x = quat[0] 
        rotate_y = quat[1] 
        rotate_z = quat[2] 
        rotate_w = quat[3] 
         
        # Euler angles in radians
        roll_x, pitch_y, yaw_z = euler_from_quaternion(rotate_x, rotate_y, rotate_z, rotate_w)
         
        roll_x = math.degrees(roll_x)
        pitch_y = math.degrees(pitch_y)
        yaw_z = math.degrees(yaw_z)
        print("translate_x: {}".format(translate_x))
        print("translate_y: {}".format(translate_y))
        print("translate_z: {}".format(translate_z))
        print("roll_x: {}".format(roll_x))
        print("pitch_y: {}".format(pitch_y))
        print("yaw_z: {}".format(yaw_z))
        print()
         
        # Draw axes on marker
        cv2.drawFrameAxes(frame, mtx, dst, rvecs[i], tvecs[i], 0.05)
     
    # Display resulting frame
    cv2.imshow('frame',frame)
          
    # Quit on "q" press on keyboard, 
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  
  # Close video stream
  cap.release()
  cv2.destroyAllWindows()
   
if __name__ == '__main__':
  print(__doc__)
  main()