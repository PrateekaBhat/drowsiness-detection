
#Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame #For playing sound
import dlib
import time
import cv2
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd

#Initialize Pygame and load music
pygame.mixer.init()
pygame.mixer.music.load('audio/drowsinessAlert.WAV')

# Minimum threshold of eye aspect ratio below which alarm is triggerd
EYE_ASPECT_RATIO_THRESHOLD = 0.3

# Minimum threshold of mouth aspect ratio above which alarm is triggered
MOUTH_ASPECT_RATIO_THRESHOLD = 0.75
# Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
EYE_ASPECT_RATIO_CONSEC_FRAMES = 50

# Counts no. of consecutuve frames below threshold value
COUNTER = 0

#all eye  and mouth aspect ratio with time
ear_list=[]
total_ear=[]
mar_list=[]
total_mar=[]
timestamp=[]
total_timestamp=[]

#Load face cascade which will be used to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

#This function calculates and return eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2*C)
    return ear

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

#Load face detector and predictor, uses dlib shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


#Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

#Start webcam video capture
video_capture = cv2.VideoCapture(0)

#Give some time for camera to initialize(not required)
time.sleep(2)

yawns = 0
yawn_status = False


while(True):
    #Read each frame and flip it, and convert to grayscale
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_yawn_status = yawn_status


    #Detect facial points through detector function
    faces = detector(gray, 0)

    #Detect faces through haarcascade_frontalface_default.xml
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Draw rectangle around each face detected
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    #Detect facial points
    for face in faces:

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        #Get array of coordinates of leftEye and rightEye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mstart:mend]

        #Calculate aspect ratio of both eyes
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2
        EAR = "%.2f" % eyeAspectRatio
        # live datawrite in csv
        ear_list.append(EAR)
        timestamp.append(dt.datetime.now().strftime('%H:%M:%S'))

        #Use hull to remove convex contour discrepencies and draw eye shape around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)
        cv2.putText(frame, "Press q to exit", (150, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        output_text = "Yawn Count: " + str(yawns)
        cv2.putText(frame, output_text, (200, 150),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 127), 2)

        MAR = mouth_aspect_ratio(mouth)
        mar_list.append(MAR)

        #Detect if eye aspect ratio is less than threshold
        if eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD:
            COUNTER += 1
            #If no. of frames is greater than threshold frames,
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                cv2.imwrite('output/outputImage.jpg', frame)
                pygame.mixer.music.play(-1)
                cv2.putText(frame, "You are Drowsy", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        else:
            pygame.mixer.music.stop()
            COUNTER = 0

        if MAR > MOUTH_ASPECT_RATIO_THRESHOLD:
            yawn_status = True
            cv2.putText(frame, "You are yawning", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            pygame.mixer.music.play()
        else:
            yawn_status = False
            pygame.mixer.music.stop()

        if prev_yawn_status == True and yawn_status == False:
            yawns += 1
        # total data collection for plotting
    for i in ear_list:
        total_ear.append(i)
    for i in mar_list:
            total_mar.append(i)
    for i in timestamp:
        total_timestamp.append(i)

    #Show video feed
    cv2.imshow('Drowsiness Detection', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

a = total_ear
b = total_mar
c = total_timestamp

df = pd.DataFrame({"EAR": a, "MAR": b, "TIME": c})
df.to_csv("webcam_output.csv", index=False)
df=pd.read_csv("webcam_output.csv")

df.plot(x='TIME', y=['EAR', 'MAR'], color=['purple', 'green'])
#plt.xticks(rotation=45, ha='right')

plt.subplots_adjust(bottom=0.30)
plt.title('Eye and Mouth Aspect Ratio calculation over time using webcam')
plt.xlabel('Time')
plt.ylabel('EAR & MAR')
plt.gcf().autofmt_xdate()
plt.savefig('output/Ratio.png')

#Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv2.destroyAllWindows()
