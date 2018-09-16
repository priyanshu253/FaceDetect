# Import OpenCV2
import cv2 as g_objOpenCV2
# Import OS for file path
import os as g_objOS
# Import Numpy
import numpy as g_objNumpy
# Import Python Image Library
from PIL import Image
# Import arg Parser
import argparse as g_objArgParse
# Import Sys
import sys

# Create Local Binary Patterns for face recognization
g_objRecognizer = g_objOpenCV2.face.createLBPHFaceRecognizer()

# For face detection, Use prebuilt frontal face training model,
g_objDetector = g_objOpenCV2.CascadeClassifier("haarcascade_frontalface_default.xml");

# Create method to get the images and label data
def getImagesAndLabels(prm_sDirectoryPathForDataSet):

    # Get all file path
    v_lImagePaths = [g_objOS.path.join(prm_sDirectoryPathForDataSet,f) for f in g_objOS.listdir(prm_sDirectoryPathForDataSet)] 

    # Initialize Face Samples
    v_arrFaceSamples=[]

    # Initialize Ids
    v_arrIds = []

    # Loop all the file path
    for v_sImagePath in v_lImagePaths:

        # Open image and convert it to grayscale
        v_objGreyScaleImage = Image.open(v_sImagePath).convert('L')

        # GreyScale image to Numpy array
        v_arrNumpyImage = g_objNumpy.array(v_objGreyScaleImage,'uint8')

        # Get ImageId from filename
        v_iId = int(g_objOS.path.split(v_sImagePath)[-1].split("_")[1])
        print("FileName: %s | ID: %d" % (v_sImagePath, v_iId))

        # Get the face from the training images
        v_objFaces = g_objDetector.detectMultiScale(v_arrNumpyImage)

        # Loop for each face, append to their respective ID
        for (x,y,w,h) in v_objFaces:

            # Append the Image to Face Samples
            v_arrFaceSamples.append(v_arrNumpyImage[y:y+h,x:x+w])

            # Append the ID to IDs
            v_arrIds.append(v_iId)

    # Return the Sample Faces and IDs Arrays
    return v_arrFaceSamples,v_arrIds

# Main method to get Data Set Directory Path
def main(prm_args):
	v_objParser = g_objArgParse.ArgumentParser(description='Face Trainer')
	v_objParser.add_argument('-s', '--dataset_path', default='dataSet', type=str, help='Dataset Path')

	prm_args = v_objParser.parse_args()
	v_sDirectoryPathForDataSet = prm_args.dataset_path

	# Get the faces and IDs
	faces,ids = getImagesAndLabels(v_sDirectoryPathForDataSet)

	# Train the model using the faces and IDs
	g_objRecognizer.train(faces, g_objNumpy.array(ids))

	# Save the model into trainedModel.yml
	g_objRecognizer.save('trainedModel.yml')


if __name__ == '__main__':
    from sys import argv
    try:
        main(argv)
    except KeyboardInterrupt:
        pass
    sys.exit()