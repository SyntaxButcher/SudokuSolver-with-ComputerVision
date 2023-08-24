# SudokuSolver with ComputerVision
 
##Objective:

Develop a computer program that can detect Sudoku puzzles from images and solve them.

**Techniques Used:**

**Computer Vision (OpenCV):** Detects the Sudoku grid from a given image.
Backtracking Algorithm: Traditional algorithm for solving Sudoku puzzles. It attempts to fill the grid sequentially and if a problem is detected, it backtracks to the previous step and tries a different number.
Convolutional Neural Networks (CNNs): Trained on the MNIST dataset to recognize handwritten digits. It helps to interpret numbers in the detected Sudoku grid.
Implementation:

**Preprocessing:** The input image is preprocessed using OpenCV to detect contours. The biggest contour, which should represent the Sudoku grid, is isolated.
Digit Recognition: The isolated Sudoku grid is divided into individual cells (boxes) which are then passed through the trained CNN to recognize numbers.
Sudoku Solving: The recognized numbers are arranged into a 9x9 array and passed to the backtracking algorithm to find a solution.
Overlaying Solution: Once a solution is found, the answer is overlaid on the original image to show the solved Sudoku.

**Results & Findings:**

The backtracking algorithm efficiently solved Sudoku puzzles.
The CNN model, trained on the MNIST dataset, was reasonably accurate but had challenges distinguishing between numbers like 7 and 1, or 9 and 1, especially with varying handwriting styles.
The combined system effectively detected Sudoku puzzles from images, recognized the numbers, solved the puzzles, and overlaid the solutions.

**Challenges:**

Training a robust model for digit recognition: Recognizing handwritten digits can be challenging, especially with various handwriting styles.
Dealing with package versions and dependencies when integrating different technologies like OpenCV and TensorFlow.
Handling real-world variations in Sudoku puzzles, such as grid distortions, shadows, or non-standard fonts.
