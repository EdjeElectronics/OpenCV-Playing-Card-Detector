############## PiCamera video stream creator ###############
#
# Author: Evan Juras  (heavily copying from Adrian Rosebrock)
# Date: 9/5/17
# Description: Defines the PiVideoStream object, which controls
# acquisition of frames from the PiCamera. The object uses multi-threading to
# aquire camera frames in a separate thread from the main program. This allows
# the main thread to grab the most recent camera frame without having to take 
# it directly from the camera feed, reducing I/O time, which slightly improves 
# framerate.
#
# See http://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
# for a full explanation of the source code.

# Import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import cv2


class PiVideoStream:
    """Camera object"""
    def __init__(self,resolution=(640,480),framerate=30):
		# Initialize the camera and the camera image stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera,size=resolution)
        self.stream = self.camera.capture_continuous(
            self.rawCapture, format = "bgr", use_video_port = True)

		# Create a variable to store the camera frame and to control
		# when the camera is stopped
        self.frame = []
        self.stopped = False

    def start(self):
		# Start the thread to read frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
		# Keep looping indefinitely until the thread is stopped
        for f in self.stream:
			# Grab the frame from the stream and clear the stream
			# in preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)

			# If the camera is stopped, stop the thread and close
			# the camera resources
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()

    def read(self):
		# Return the most recent frame
        return self.frame

    def stop(self):
		# Indicate that the camera and thread should be stopped
        self.stopped = True