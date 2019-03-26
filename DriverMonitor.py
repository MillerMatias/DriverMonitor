from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

#Specify the subjects to be recognized
subjects = ["", "Matias", "Ella"]

#Facedetection function
def FaceDetection(img):
    
    #Load the images in grayscale.
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(grayImg, 1.2, 5)

    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]
    
    return grayImg[y: y + h, x: x + w], faces[0]

#Prepping the training-data from the training-data folder (eg. the images that we train on)
def PrepareTrainingData(data_folder_path):
    
    #Here we specify the directory to find faces and labels in our training-data pictures.
    dirs = sorted(os.listdir(data_folder_path))
    
    faces = []
    labels = []
    
    #Loop through each directory and read the images within them
    for dir_name in dirs:
        
        if not dir_name.startswith('s'):
            continue

        label = int(dir_name.replace('s', ''))
        
        subject_dir_path = data_folder_path + '/' + dir_name
        
        subject_images_names = os.listdir(subject_dir_path)
        
        
        for image_name in subject_images_names:

            if image_name.startswith('.'):
                continue

            image_path = subject_dir_path + '/' + image_name
            image = cv2.imread(image_path)

            #cv2.imshow('Training on image...', image)
            cv2.waitKey(1)

            face, rect = FaceDetection(image)

            if face is not None:
                faces.append(face)
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
        
    return faces, labels
#Here we use our prepped training data to see how many faces and labels we can make out of them.
print("Preparing data...")
faces, labels = PrepareTrainingData('training-data')
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create() 
face_recognizer.train(faces, np.array(labels))

#Function that draws the rectangle around the detected face.
def DrawRectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (230, 255, 0), 2)
    
#Drawing the label.
def DrawText(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (230, 255, 0), 2)

#Function that does the actual prediction of the face in real-time video.
def predict(video):
    #make a copy of the image as we don't want to chang original image
    #detect face from the image
    face, rect = FaceDetection(video)
    
    if face is None:
        
        cv2.putText(video, "CANT FIND FACE!!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2)
        return cv2.resize(video, (0, 0), fx = 0.7, fy = 0.7)
 
    label = face_recognizer.predict(face)
    
    label_text = subjects[label[0]]
    
    DrawRectangle(video, rect)
    DrawText(video, label_text, rect[0], rect[1] - 5)
    
    return cv2.resize(video, (0, 0), fx = 0.7, fy = 0.7)

#Caulcating the EAR.
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinatesq
    C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
    return ear

#Main function where we basically count the EAR, determine wether your eyes are open or not - and determine our drowsiness levels.
def main():
    
    #Define the treshholds and frames for the "eyes open/eyes closed function"
    EYE_AR_THRESH = 0.16

    #Define treshsholds for drowsiness alert
    DROWZINESS_THRESH = 0.25

    #Define the frame amount for the drowsiness alert
    SEVERE_TIREDNESS = 30
    MODERATE_DROWSINESS = 20
    MILD_DROWSINESS = 10

    #counter for how long you've kept your eyes closed.
    COUNTER = 0

    #Initalize dlibs face detector, using the 68 face landmarks file
    print("Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
 
    # Take the index of both eyes
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    # Start the video stream thread
    print("Starting video stream thread...")
    print("Press q to quit the program...")

    #start the videocapture
    cap = cv2.VideoCapture(0)
       
    # loop over frames from the video capture
    while True:

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    	# detect faces in the grayscale frame
        rects = detector(gray, 0)

    	# loop over the face detections
        for rect in rects:
    		# determine the facial landmarks for the face region, then
    		# convert the facial landmark (x, y)-coordinates to a NumPy
    		# array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
    
    		# extract the left and right eye coordinates, then use the
    		# coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
    
    		# average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
    
    		# compute the convex hull for the left and right eye, then
    		# visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (230, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (230, 255, 0), 1)
    
    		# check to see if the eye aspect ratio is below the blink
    		# threshold, and if so, increment the blink frame counter
         
    			# if the eyes were closed for a sufficient number of
    			# then increment the total number of blinks
            if ear < DROWZINESS_THRESH:
                COUNTER += 1

            #Specifying drowsniess levels.
                if COUNTER >= SEVERE_TIREDNESS and ear <= 0.13:
                    cv2.putText(frame, "SEVERE TIREDNESS/SLEEPING", (30, 80),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 252), 2)

                elif COUNTER >= MODERATE_DROWSINESS and ear <= 0.17:
                    cv2.putText(frame, "Moderate Drowsiness", (30, 80),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 171, 0), 2)

                elif COUNTER >= MILD_DROWSINESS and ear <= 0.25:
                    cv2.putText(frame, "Mild Drowsiness", (30, 80),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (230, 255, 0), 2)

            else:
                cv2.putText(frame, "Wide awake", (30, 80),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (17, 255, 0), 2)
                
                COUNTER = 0

            #Specifying when eyes are open, and when closed
            if ear < EYE_AR_THRESH:
                cv2.putText(frame, "EYES CLOSED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 255, 0), 2)
                
            else: 
                cv2.putText(frame, "EYES OPEN", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 255, 0), 2)
                

    		# Draw the EAR text
            cv2.putText(frame, "Eye Aspect Ratio: {:.2f}".format(ear), (30, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 255, 0), 2)
     
        #Predict the face
        recognize = predict(frame)

        cv2.imshow("Frame", recognize)
        key = cv2.waitKey(1) & 0xFF
     
    	# if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    cap.release()

main()