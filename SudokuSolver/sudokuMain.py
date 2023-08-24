import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To remove tensor flow warning
from util import *
import BTSolver1

# Path to the image we want to process and setting a square image size since sudoku puzzles are squares
pathImage = "SudokuEx/Puzzle5.jpg"
heightImg = 450
widthImg = 450

# Initializing model
model = intializePredectionModel()

# Preparing the image, reading, resizing and thresholding and grayscaling it
img = cv2.imread(pathImage) # read
img = cv2.resize(img, (widthImg, heightImg)) # resize
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8) # for testing
imgThreshold = preProcess(img) # threshold

# Finding all the contours
imgContours = img.copy() # copying
imgBigContour = img.copy() # copying
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find all the contours using external method for outer contour
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # to draw all the contours

# Finding the puzzle's corner points by taking the biggest Contours
biggest, maxArea = biggestContour(contours)
if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 10) # Draw the biggest contour
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]) # for warping, preparation
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

    # Splitting the image into 81 boxes to run a predictor on it to find the digit using pre-trained model
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    numbers = getPrediction(boxes, model)
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1) # Blank spaces are replaced with 0

    # Finding the solution
    board = np.array_split(numbers, 9) # Converting in row arrays to match the solver format
    print(board)
    try:
        BTSolver1.solve(board)
    except:
        pass
    print(np.matrix(board))
    # Converting the solution into displayable numbers in the appropriate positions
    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers = flatList*posArray
    imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers)

    # Final board with solutions
    pts2 = np.float32(biggest) 
    pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) 
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  
    # Image ready to be overlayed
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    # Combining them
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    # Drawing the grids
    imgDetectedDigits = drawGrid(imgDetectedDigits)
    imgSolvedDigits = drawGrid(imgSolvedDigits)


    imageArray = ([img, imgThreshold, imgContours, imgBigContour], [imgDetectedDigits, imgSolvedDigits, imgInvWarpColored, inv_perspective])
    stackedImage = stackImages(imageArray, 1)
    cv2.imshow('The process', stackedImage)
    cv2.imshow('Solution', inv_perspective)

else:
    print("No Sudoku Found")

cv2.waitKey(0)


