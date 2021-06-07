import os

import cv2
authenticaton_flag = False

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0

# names related to ids: example ==> Prateeka: id=1,  etc
names = ['None', 'Prateeka', 'Roopa', 'Pravalikha']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:

    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        cv2.putText(img, "Press q to exit", (150, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(img, "Authentication successful", (200, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 127), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
            authenticaton_flag = True
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(img, "Authentication failed", (200, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            authenticaton_flag = False

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
    cv2.imshow('Login', img)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

if authenticaton_flag == True:
    os.system("python drowsiness_detect.py")