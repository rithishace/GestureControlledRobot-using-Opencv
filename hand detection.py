# -*- coding: utf-8 -*-
"""
Created on Sun May  2 13:32:27 2021

@author: rithish
"""

#importing the necessary libraries

import cv2
import numpy as np
from sklearn.metrics import pairwise
import sim                  #copeliasim library
import sys


# Declaring global variables that we will use throughout the function

background = None
v0=2
accumulated_weight = 0.5

# Manually setting up our Region Of Interest (ROI) for grabbing the hand.

roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600

sim.simxFinish(-1) # just in case, close all opened connections

clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5)

if clientID!=-1:  #check if client connection successful
    print ('Connected to remote API server')
    
else:
    print ('Connection not successful')
    sys.exit('Could not connect')


#retrieve motor  handles
errorCode,left_motor_handle=sim.simxGetObjectHandle(clientID,'LM',sim.simx_opmode_oneshot_wait)
errorCode,right_motor_handle=sim.simxGetObjectHandle(clientID,'RM',sim.simx_opmode_oneshot_wait)
# set ROBOT DIRECTIONS 
def forward():
    v0=2
    errorCode=sim.simxSetJointTargetVelocity(clientID,left_motor_handle,v0, sim.simx_opmode_streaming)
    errorCode=sim.simxSetJointTargetVelocity(clientID,right_motor_handle,v0, sim.simx_opmode_streaming)

    
def left():
    v0=2
    errorCode=sim.simxSetJointTargetVelocity(clientID,left_motor_handle,0, sim.simx_opmode_streaming)
    errorCode=sim.simxSetJointTargetVelocity(clientID,right_motor_handle,v0, sim.simx_opmode_streaming)

    
def right():
    v0=2
    
    errorCode=sim.simxSetJointTargetVelocity(clientID,left_motor_handle,v0, sim.simx_opmode_streaming)
    errorCode=sim.simxSetJointTargetVelocity(clientID,right_motor_handle,0, sim.simx_opmode_streaming)

def backward():
    v0=-2
    errorCode=sim.simxSetJointTargetVelocity(clientID,left_motor_handle,v0, sim.simx_opmode_streaming)
    errorCode=sim.simxSetJointTargetVelocity(clientID,right_motor_handle,v0, sim.simx_opmode_streaming)

def stop():
    v0=0
    errorCode=sim.simxSetJointTargetVelocity(clientID,left_motor_handle,v0, sim.simx_opmode_streaming)
    errorCode=sim.simxSetJointTargetVelocity(clientID,right_motor_handle,v0, sim.simx_opmode_streaming)

#Finding the average background value

"""The following function is used to calculate the weighted sum 
of input image src and the accumulator dst so that dst becomes a 
running average of a frame sequence"""

def calc_accum_avg(frame, accumulated_weight):
    '''
    Given a frame and a previous accumulated weight, computed
    the weighted average of the image passed in.
    '''
    
    # Grab the background
    global background
    
    # For first time, create the background from a copy of the frame.
    if background is None:
        background = frame.copy().astype("float")
        return None

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(frame, background, accumulated_weight)
    
    
    #Segment the hand region in frame

def segment(frame, threshold=25):
    global background
    
    # Calculates the Absolute Differentce between the backgroud and the passed in frame
    diff = cv2.absdiff(background.astype("uint8"), frame)

    # Apply a threshold to the image so we can grab the foreground
    # We only need the threshold, so we will throw away the first item in the tuple with an underscore _
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Grab the external contours form the image
    # Again, only grabbing what we need here and throwing away the rest
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list is 0, then we didn't grab any contours!
    if len(contours) == 0:
        return None
    else:
        # Given the way we are using the program, the largest external contour should be the hand (largest by area)
        # This will be our segment
        hand_segment = max(contours, key=cv2.contourArea)
        
        # Return both the hand segment and the thresholded hand image
        return (thresholded, hand_segment)
    
    
def count_fingers(thresholded, hand_segment):
    
    
            # Calculated the convex hull of the hand segment
    conv_hull = cv2.convexHull(hand_segment)
    
    # Now the convex hull will have at least 4 most outward points, on the top, bottom, left, and right.
    # Let's grab those points by using argmin and argmax. Keep in mind, this would require reading the documentation
    # And understanding the general array shape returned by the conv hull.

    # Find the top, bottom, left , and right.
    # Then make sure they are in tuple format
    top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

    # In theory, the center of the hand is half way between the top and bottom and halfway between left and right
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2

    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    
    # Calculate the Euclidean Distance between the center of the hand and the left, right, top, and bottom.
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
    
    # Grab the largest distance
    max_distance = distance.max()
    
    # Create a circle with 90% radius of the max euclidean distance
    radius = int(0.8 * max_distance)
    circumference = (2 * np.pi * radius)

    # Not grab an ROI of only that circle
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    
    # draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
    
    
    # Using bit-wise AND with the cirle ROI as a mask.
    # This then returns the cut out obtained using the mask on the thresholded hand image.
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # Grab contours in circle ROI
    contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Finger count starts at 0
    count = 0

    # loop through the contours to see if we count any more fingers.
    for cnt in contours:
        
        # Bounding box of countour
        (x, y, w, h) = cv2.boundingRect(cnt)

        # Increment count of fingers based on two conditions:
        
        # 1. Contour region is not the very bottom of hand area (the wrist)
        out_of_wrist = ((cY + (cY * 0.25)) > (y + h))
        
        # 2. Number of points along the contour does not exceed 25% of the circumference of the circular ROI (otherwise we're counting points off the hand)
        limit_points = ((circumference * 0.25) > cnt.shape[0])
        
        
        if  out_of_wrist and limit_points:
            count += 1

    return count

#Camera access
cam = cv2.VideoCapture(0)

# Intialize a frame count
num_frames = 0

# keep looping, until camera is open state
while cam.isOpened():
    # get the current frame
    ret, frame = cam.read()

    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)

    # clone the frame
    frame_copy = frame.copy()

    # Grab the ROI from the frame
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]

    # Apply grayscale and blur to ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    fingers=0
    # For the first 30 frames we will calculate the average of the background.
    # We will tell the user while this is happening
    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,0), 2)
            cv2.imshow("Show Ur Gesture",frame_copy)
            
    else:
        # now that we have the background, we can segment the hand.
        
        # segment the hand region
        hand = segment(gray)

        # First check if we were able to actually detect a hand
        if hand is not None:
            
            # unpack
            thresholded, hand_segment = hand

            # Draw contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0),1)

            # Count the fingers
            fingers = count_fingers(thresholded, hand_segment)
            directions=['halted','forward','left','right','backward','error']
            if fingers not in [0,1,2,3,4]:
                fingers=0
                
            # Display count
            cv2.putText(frame_copy, directions[fingers], (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

            # Also display the thresholded image
            cv2.imshow("Thesholded", thresholded)

    # Draw ROI Rectangle on frame copy
    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)
    # increment the number of frames for tracking
    num_frames += 1
    # Display the frame with segmented hand
    cv2.imshow("Show Ur Gesture", frame_copy)
    # Close windows with Esc
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k==ord('b'):
        num_frames=0
    else:
        pass
    if fingers==1:
        forward()
    elif fingers==2:
        left()
    elif fingers==3:
        right()
    elif fingers==4:
        backward()
    else:
        stop()
# Release the camera ,destroy all the windows and stop simulation
returnCode=sim.simxStopSimulation(clientID,sim.simx_opmode_oneshot)
cam.release()
cv2.destroyAllWindows()