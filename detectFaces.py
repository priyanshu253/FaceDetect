# Import OpenCV2
import cv2 as g_objOpenCV2

# Create Face Recognizer object
g_objFaceRecognizer = g_objOpenCV2.face.createLBPHFaceRecognizer()

# Load Trained Model
g_objFaceRecognizer.load("trainedModel.yml")

# Create Detect Face Object
g_objDetect = g_objOpenCV2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize Webcam, Default webcam in most of notebooks (0)
g_objWebCam = g_objOpenCV2.VideoCapture(0)

# Initialize font Object
g_objFont = g_objOpenCV2.FONT_HERSHEY_SIMPLEX

while True:

	# Read Image from Webcam
    v_bCheck, v_objImage = g_objWebCam.read()
    
    # Convert Image to Gray
    v_objGrayImage = g_objOpenCV2.cvtColor(v_objImage, g_objOpenCV2.COLOR_BGR2GRAY)
    
    # Detect all Faces in Image
    v_objFaces = g_objDetect.detectMultiScale(v_objGrayImage, scaleFactor=1.2, minNeighbors=5)
    
    # Loop all the Faces Detected
    for (x,y,w,h) in v_objFaces:
    	
    	# Draw Rectangle around the Detected Face
        g_objOpenCV2.rectangle(v_objImage, (x,y), (x+w,y+h), (0,0,255), 2)
        
        # Predict Face using Trained Model
        # Get ID of face detected and Confidence Level (Lower is Better)
        v_iID, v_fConfidenceLevel = g_objFaceRecognizer.predict(v_objGrayImage[y:y+h, x:x+w])
        print("Confidence Level: %f | ID: %d" % (v_fConfidenceLevel, v_iID))
        
        if(v_fConfidenceLevel < 45):
            if(v_iID == 1):
                v_iID = "Tom Cruise"
            elif(v_iID == 2):
                v_iID = "Priyanka Chopra"
            else:
                v_iID = "Not on System"
        else:
        	# If 
            v_iID = "Low Confidence Level!"

        g_objOpenCV2.putText(v_objImage, str(v_iID + " " + str(v_fConfidenceLevel)), (x,y+h), g_objFont, 4, (255,255,255), 2, g_objOpenCV2.LINE_AA)
	
    #Resize Frame.
    v_iImageWidth=int(v_objImage.shape[1])
    v_iImageHeight=int(v_objImage.shape[0])
    v_objResizedImage = g_objOpenCV2.resize(v_objImage, (v_iImageWidth, v_iImageHeight))
	
    g_objOpenCV2.imshow("Window Screen", v_objResizedImage)
    if g_objOpenCV2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Webcam
g_objWebCam.release()
# Destroy All Windows Of OpenCV2
g_objOpenCV2.destroyAllWindows()
