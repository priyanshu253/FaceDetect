# Simple Face Detection with OpenCV

### Haar Cascade

 Face detection examples with haar cascade classifier algorithm (Face, eyes, mouth, other objects etc.). Cascade Classifier Training http://docs.opencv.org/3.1.0/dc/d88/tutorial_traincascade.html

**Requirements**
* OpenCV 3.x Version
* Python 2.7 and above Version


**Install Lib**

        sudo apt-get update
        
        sudo apt-get upgrade

        sudo apt-get install python-opencv
        
        sudo apt-get install python-dev libjpeg-dev libfreetype6-dev zlib1g-dev
        
        * symbolic links

        sudo ln -s /usr/lib/`uname -i`-linux-gnu/libfreetype.so /usr/lib/
        sudo ln -s /usr/lib/`uname -i`-linux-gnu/libjpeg.so /usr/lib/
        sudo ln -s /usr/lib/`uname -i`-linux-gnu/libz.so /usr/lib/
        
        sudo pip install Pillow


**Test OpenCV**

python -c "\\\
import cv2\
print(cv2.\_\_version\_\_)"


**Steps**

Step 1: Get Traning Data (Images) under 'dataSet' directory
        
        File Name: XXXXXX_UNIQUEID_IMAGENO.jpg
        
Step 2: Run trainer.py
        
        This will generate trainedModel.yml in project directory.
        
        Traning data-set is loaded by default from dataSet directory.
        
        You may provide your own data-set path.
        
        trainer.py --dataset_path "/home/someDataDirectory"

Step 3: Run detectFaces.py
        
        This will load a new window with live feed of your webcam.
        
        If unable to load webcam try chaning device no. 0 to 1 or 2 etc. at line 14.
        
        eg: g_objWebCam = g_objOpenCV2.VideoCapture(0) to g_objWebCam = g_objOpenCV2.VideoCapture(1)
        
        and try running the module again.
        
***Note:***

      Add unique id to each image dataSet and update if-else statement in detectFaces.py
        
Enjoy :)
