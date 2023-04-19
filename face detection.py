import cv2
cascade_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

while True:

    #capture frame by frame

    ret, frame = cap.read()

    frame = cv2.cvtColor(frame,0)
                                #multiscale to easily detect face recognition.
    detections = cascade_classifier.detectMultiScale(frame)

    if(len(detections) > 1):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
          # for (x,y,w,h) in detections:
    # 	frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #when everthing done, release the capture

