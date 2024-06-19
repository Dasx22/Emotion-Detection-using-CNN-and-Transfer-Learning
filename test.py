from keras.models import load_model
import cv2
import numpy as np

# Initialising the face classifier with the Haarcascade Model for face detection
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

classifier = load_model(r'Custom_CNN_model.keras')
# classifier = load_model(r'Final_Resnet50_Best_model.keras')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)       # '0' signifies video captured through itegrated webcam; we can also specify the path of a video stored  

# Loop for live video capture
while True:
    _, frame = cap.read()       # numpy array representing the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)      # returns the coordinates of the box enclosing faces
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)      # 2 is the thickness of the box
        # Region of Interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)
            
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else: 
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)       # here, 1 is the font scale factor and 2 is the font thickness
    
    cv2.imshow("Emotion Detector", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Release the video capture and destroy all OpenCV windows  
cap.release()
cv2.destroyAllWindows()

