import cv2

capture = cv2.VideoCapture(0)
face_haar_cascade = cv2.CascadeClassifier('../opencv2/haarcascade_frontalface_default.xml')

while True:
    ret, image = capture.read()
    if not ret:
        break
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv2.imshow('Face Detection', image)
        if cv2.waitKey(30) & 0xFF == 27:
            break
capture.release()
cv2.destroyAllWindows()
