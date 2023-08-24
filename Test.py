import numpy as np
import cv2
import pickle

width = 640
height = 480
threshold = 0.65

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# load trained model and read bytes
pickle_in = open("numPredictor.p","rb")
model = pickle.load(pickle_in)

# Preprocessing the image
def preProcessing(img):
    # Convert to grayscale
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    # Normalizing the color channel values to make it easier for testing
    img = img/255
    return img

while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    cv2.imshow("Processed Image", img)
    # reshape before sending to predictor
    img = img.reshape(1,32,32,1)
    #predict
    classIndex = int(model.predict_classes(img))
    prediction = model.predict(img)
    probVal = np.amax(prediction)

    if probVal > threshold:
        cv2.putText(imgOriginal,"Number is "+ str(classIndex) + " with probability " +str(probVal*100),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

    cv2.imshow("Original Image", imgOriginal)

    # For breaking, press q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

