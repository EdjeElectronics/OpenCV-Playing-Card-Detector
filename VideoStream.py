############## Camera video stream creator ###############
#
# Author: Evan Juras  (heavily copying from Adrian Rosebrock)
# Date: 1/12/18
# Description: Defines the VideoStream object, which controls
# acquisition of frames from a PiCamera or USB camera. The object uses
# multi-threading to aquire camera frames in a separate thread from the main
# program. This allows the main thread to grab the most recent camera frame
# without having to take it directly from the camera feed, reducing I/O time,
# which slightly improves framerate.
#
# When using this with a USB Camera on a desktop or laptop, the framerate tends
# to be too fast. The Card Detector program still works, but it is intended
# for the lower processing power of the Raspberry Pi.
#
# See the following web pages for a full explanation of the source code:
# https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
# https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

# Import the necessary packages
from threading import Thread
import cv2


class VideoStream:
    """Camera object"""
    def __init__(self, resolution=(640,480),framerate=30,PiOrUSB=1,src=0):

        # Create a variable to indicate if it's a USB camera or PiCamera.
        # PiOrUSB = 1 will use PiCamera. PiOrUSB = 2 will use USB camera.
        self.PiOrUSB = PiOrUSB

        if self.PiOrUSB == 1: # PiCamera
            # Import packages from picamera library
            from picamera.array import PiRGBArray
            from picamera import PiCamera

            # Initialize the PiCamera and the camera image stream
            self.camera = PiCamera()
            self.camera.resolution = resolution
            self.camera.framerate = framerate
            self.rawCapture = PiRGBArray(self.camera,size=resolution)
            self.stream = self.camera.capture_continuous(
                self.rawCapture, format = "bgr", use_video_port = True)

            # Initialize variable to store the camera frame
            self.frame = []

        if self.PiOrUSB == 2: # USB camera
            # Initialize the USB camera and the camera image stream
            self.stream = cv2.VideoCapture(src)
            ret = self.stream.set(3,resolution[0])
            ret = self.stream.set(4,resolution[1])
            #ret = self.stream.set(5,framerate) #Doesn't seem to do anything so it's commented out

            # Read first frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

	# Create a variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread to read frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):

        if self.PiOrUSB == 1: # PiCamera
            
            # Keep looping indefinitely until the thread is stopped
            for f in self.stream:
                # Grab the frame from the stream and clear the stream
                # in preparation for the next frame
                self.frame = f.array
                self.rawCapture.truncate(0)

                if self.stopped:
                    # Close camera resources
                    self.stream.close()
                    self.rawCapture.close()
                    self.camera.close()

        if self.PiOrUSB == 2: # USB camera

            # Keep looping indefinitely until the thread is stopped
            while True:
                # If the camera is stopped, stop the thread
                if self.stopped:
                    # Close camera resources
                    self.stream.release()
                    return

                # Otherwise, grab the next frame from the stream
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
		# Return the most recent frame
        return self.frame

    def stop(self):
		# Indicate that the camera and thread should be stopped
        self.stopped = True
