# OpenCV-Playing-Card-Detector
This is a Python program that uses OpenCV to detect and identify playing cards from a PiCamera video feed on a Raspberry Pi. Check out the YouTube video that describes what it does and how it works:

https://www.youtube.com/watch?v=m-QPjO-2IkA


## Files
CardDetector.py contains the main script

Cards.py has classes and functions that are used by CardDetector.py

PiVideoStream.py creates a video stream from the PiCamera, and is used by CardDetector.py

Rank_Suit_Isolator.py is a standalone script that can be used to isolate the rank and suit from a set of cards to create train images

Card_Imgs contains all the train images of the card ranks and suits

## Dependencies
OpenCV-Python 3.2.0 and numpy 1.8.2:

See https://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/
for how to build and install OpenCV-Python on the Raspberry Pi


picamera library
```
sudo apt-get update
sudo apt-get install python-picamera python3-picamera
```

## Usage
Download this repository to a directory and run CardDetector.py from that directory. Cards need to be placed on a dark background for the detector to work.

The card detector will work best if you use isolated rank and suit images generated from your own cards. To do this, run Rank_Suit_Isolator.py to take pictures of your cards. It will ask you to take a picture of an Ace, then a Two, and so on. Then, it will ask you to take a picture of one card from each of the suits (Spades, Diamonds, Clubs, Hearts). As you take pictures of the cards, the script will automatically isolate the rank or suit and save them in the Card_Imgs directory (overwriting the existing images).
