import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle

# Settings
path = 'numberData'
testRatio = 0.2
validationRatio = 0.2
imageDimensions = (32,32,3)

images = []
classNo = []

myList = os.listdir(path)
print(myList)
print("Total No of Classes Detected ",len(myList))
noOfClasses = len(myList)
print("Importing Classes ......")
for x in range (0,noOfClasses):
    # each file into a list
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        # reading each image and resizing for computation 
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg,(32,32))
        # adding image onto a list
        images.append(curImg)
        # adding folder name as the label/class value
        classNo.append(x)
    print(x,end= " ")
print("  ")
print("Total Images in Images List = ",len(images))
print("Total IDS in classNo List= ",len(classNo))

# convert it into numpy array
images = np.array(images)
classNo = np.array(classNo)

# splitting the data into training, testing and validation | X is image, Y is label
X_train, X_test, y_train, y_test = train_test_split(images,classNo,test_size = testRatio) 
X_train, X_validation, y_train, y_validation = train_test_split(X_train,y_train,test_size = validationRatio) 
print("Training data size ",X_train.shape)
print("Testing data size ",X_test.shape)
print("Validation data size ",X_validation.shape)

numOfSamples = []
for x in range(0,noOfClasses):
    numOfSamples.append(len(np.where(y_train==x)[0]))
print(numOfSamples)

# bar graph for the data split
plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses),numOfSamples)
plt.title("No of Images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()

# Preprocessing the image
def preProcessing(img):
    # Convert to grayscale
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    # Normalizing the color channel values to make it easier for testing
    img = img/255
    return img

# Map to run a function over list or elements
X_train = np.array(list(map(preProcessing,X_train)))
X_test = np.array(list(map(preProcessing,X_test)))
X_validation = np.array(list(map(preProcessing,X_validation)))

# Reshaping the images, to add a depth of 1 for our CNN
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

# We augment the image to make the dataset more generic
dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
# Statistics
dataGen.fit(X_train)

# Encoding the matrices
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

def numModel():

    # Based on lenet model
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNodes = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape=(32,32,1),activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(noOfNodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model = numModel()
print(model.summary())

# training params
batchSizeVal= 50
epochsVal = 10
stepsPerEpochVal = 2000

history = model.fit_generator(dataGen.flow(X_train,y_train,batch_size=batchSizeVal),steps_per_epoch=stepsPerEpochVal,epochs=epochsVal,validation_data=(X_validation,y_validation),shuffle=1)

# Results plot
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

# evaluation
score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy =', score[1])

# for saving the model as a pickle object
# pickle_out= open("numPredictor.p", "wb")
# pickle.dump(model,pickle_out)
# pickle_out.close()

# for saving the model as h5
model.save('numPredictor.h5')