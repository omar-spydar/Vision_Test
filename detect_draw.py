#!/usr/bin/env python

import numpy as np # Numpy v <2.0
print("numpy: ", np.__version__)
import cv2 # OpenCV contrib library 4.6
from scipy.spatial.transform import Rotation as R # Standard SciPy library
import math # Math library included with Python

# List of ArUco dictionaries to detect
# aruco_names = ["DICT_7X7_1000"]
aruco_names = ["DICT_ARUCO_ORIGINAL", "DICT_7X7_1000"]

# ArUco dictionaries built into OpenCV
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
marker_width = 0.15

# Calibration parameters yaml file
calibration_params = 'calibration_f.yaml'

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
    """ Main method of the program. """
    # Check that all specified ArUco dictionaries are valid
    for name in aruco_names:
        if ARUCO_DICT.get(name, None) is None:
            print("[INFO] ArUCo tag of '{}' is not supported".format(name))
            sys.exit(0)

    # Load camera parameters from saved file
    cv_file = cv2.FileStorage(calibration_params, cv2.FILE_STORAGE_READ)
    mtx = cv_file.getNode('K').mat()
    dst = cv_file.getNode('D').mat()
    cv_file.release()
    
    # Load ArUco dictionaries and parameters
    print("[INFO] detecting markers from {}...".format(", ".join(aruco_names)))
    aruco_dictionaries = [(name, cv2.aruco.Dictionary_get(ARUCO_DICT[name])) for name in aruco_names]
    aruco_parameters = cv2.aruco.DetectorParameters_create()
    
    # Start video stream
    # cap = cv2.VideoCapture(0)
    rtsp_url = "rtsp://admin:111111@10.0.0.84:554/cam/realmonitor?channel=2&subtype=0"
    iphone_url = 'http://10.0.0.156:4747/video'
    cap = cv2.VideoCapture(rtsp_url)
    
    while(True):
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Initialize lists to store all detected markers
        all_corners = []
        all_marker_ids = []
        all_rejected = []
        all_dict_names = []
        
        # Detect markers for each dictionary
        for dict_name, dictionary in aruco_dictionaries:
            (corners, marker_ids, rejected) = cv2.aruco.detectMarkers(
                frame, dictionary, parameters=aruco_parameters)
            
            if marker_ids is not None:
                all_corners.extend(corners)
                all_marker_ids.extend(marker_ids)
                all_rejected.extend(rejected)
                # Associate dictionary name with each marker ID
                all_dict_names.extend([dict_name] * len(marker_ids))
        
        # Check if any markers were detected
        if all_marker_ids:
            # Draw squares around detected markers
            cv2.aruco.drawDetectedMarkers(frame, all_corners, np.array(all_marker_ids))
            
            # Get rotation and translation vectors
            rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
                all_corners, marker_width, mtx, dst)
            
            # Process each detected marker
            for i, (marker_id, dict_name) in enumerate(zip(all_marker_ids, all_dict_names)):
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
                
                # Print dictionary name and marker ID in console
                print(f"Dictionary: {dict_name}, Marker ID: {marker_id}")
                print("translate_x: {}".format(translate_x))
                print("translate_y: {}".format(translate_y))
                print("translate_z: {}".format(translate_z))
                print("roll_x: {}".format(roll_x))
                print("pitch_y: {}".format(pitch_y))
                print("yaw_z: {}".format(yaw_z))
                print()
                
                # Draw dictionary name and marker ID on frame
                # Use top-left corner of the marker for text placement
                top_left = tuple(map(int, all_corners[i][0][0]))
                text = f"{dict_name} ID:{marker_id[0]}"
                cv2.putText(
                    frame,
                    text,
                    (top_left[0], top_left[1] - 20),  # Slightly above the marker
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # Font scale
                    (0, 255, 0),  # Green color
                    1,  # Thickness
                    cv2.LINE_AA
                )
                
                # Calculate right and left sides of the marker for text placement
                corners = all_corners[i][0]  # Get the four corners of the marker
                right_side_x = int(max(corners[:, 0])) + 10  # Right side of the box
                left_side_x = int(min(corners[:, 0])) - 60   # Left side of the box
                top_y = int(min(corners[:, 1]))+10  # Top of the box for vertical alignment
                
                # Translation text on the right side
                cv2.putText(
                    frame,
                    f"x={translate_x:.2f}",
                    (right_side_x, top_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),  # Blue color
                    1,
                    cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    f"y={translate_y:.2f}",
                    (right_side_x, top_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    f"z={translate_z:.2f}",
                    (right_side_x, top_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA
                )
                
                # Rotation text on the left side
                cv2.putText(
                    frame,
                    f"x={roll_x:.2f}",
                    (left_side_x, top_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),  # Red color
                    1,
                    cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    f"y={pitch_y:.2f}",
                    (left_side_x, top_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    f"z={yaw_z:.2f}",
                    (left_side_x, top_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )
                
                # Draw axes on marker
                cv2.drawFrameAxes(frame, mtx, dst, rvecs[i], tvecs[i], 0.05)
        
        # Display resulting frame
        cv2.imshow('frame', frame)
        
        # Quit on "q" press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Close video stream
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print(__doc__)
    main()