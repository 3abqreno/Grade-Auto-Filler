# Grade Auto Filler

## Overview

The **Grade Auto Filler** is an innovative image processing project comprising two integral modules: the Bubble Sheet Solver and Grid Pattern/Numbers Detection. Leveraging advanced techniques such as Hough Transform, morphology filters, Support Vector Machines (SVM), OCR (Optical Character Recognition), and machine learning, this project addresses the automated processing of images with precision and efficiency.

## Modules

### Bubble Sheet Solver

- **Techniques Employed:**
  - Morphological Filters
- **Functionality:**
  - Detects and analyzes bubble sheets commonly used in surveys and exams.
  - Automates the grading process, reducing manual effort and potential errors.
  -Id detection
  - given answers can output grades easily in grade sheet

### Grid Pattern/Numbers Detection

- **Techniques Utilized:**
  - Hough Transform
  - Image processing techniques for grid pattern and numbers detection.
  - Support Vector Machines (SVM)
  - Optical Character Recognition (OCR)

- **Features:**
  - Detects grid patterns and numeric data in images.
  - Utilizes OCR to extract numeric information from 
  - Outputs detected grid to Excel 


# Bubble Sheet Solver

The Bubble Sheet Solver module is designed to automate the detection and grading of bubble sheets commonly used in surveys and exams. Here's a brief overview of how it works:

## Overview

The Bubble Sheet Solver utilizes advanced image processing techniques, including Hough Transform, morphology filters, and Support Vector Machines (SVM), to analyze and grade bubble sheets. The process involves the following key steps:

1. **Preprocessing:**
    - **Contour Detection and Perspective Correction:**
     - Find the contours of the paper to identify its boundaries.
     - Utilize `cv2.warpPerspective` to obtain the correct perspective, ensuring accurate analysis.

   - **Noise Reduction and Thresholding:**
     - Apply filters and blur techniques to remove noise from the image.
     - Perform thresholding to convert the image into a binary format, simplifying subsequent analysis.

   ![](/BubbleSheet/Docs/pre.png)

2. **Vertical Line Separation:**

     - Employ morphology operations to make a mask.
     - This step enhances the grading process by providing a clearer separation of individual responses.
        
     ![](/BubbleSheet/Docs/mask.png) ![](/BubbleSheet/Docs/lines.png)
3. **Answer Extraction:**
   - Utilize `cv2.findContours` to identify bubble contours and count the number of lit pixels to determine whether a bubble was selected.
   - Read answers from a text file and compare them with the detected selections.
   - Write the extracted answers to an Excel sheet for further analysis and reporting.
## Accuracy

In testing, the Bubble Sheet Solver has demonstrated an impressive accuracy rate of more than 95%.

## Dynamic

One notable feature of the Bubble Sheet Solver is its dynamic nature. It doesn't require prior knowledge of the number of questions or the number of answers per question; it autonomously detects these parameters during the analysis. This flexibility makes it well-suited for various scenarios without the need for manual configuration.

# Grade Sheet Module

The Grade Sheet Module is designed to detect the grid patterns and numbers in a sheet. This section provides an overview of how the module works and how to get started with it.

## Overview

The Grade Sheet Module focuses on identifying grid patterns and numbers in sheets. It utilizes image processing techniques, OCR (Optical Character Recognition),Hough transform, and machine learning components to achieve accurate results.



1. **input image:**
    -   
   ![Input Image](/grid/Docs/start.jpg)

2. **Preprocessing:**
    - first warp the image to the correct prespective
    - using hough transform to detect grid lines we first split the image into vertical lines and then into small cells.
   ![Vertical Lines](/grid/Docs/split.png)
   ![Cells](/grid/Docs/result.png)


3. **Detection Phase:**
   - Trains an SVM model to detect numbers, codes, lines, rectangles, and other drawings with an accuracy of 96%.
   - Utilizes pytesseract OCR for hand-drawn numbers detection(you have the option to use pytesseract or out SVM model).



## Dynamic Configuration

The Grade Sheet Module offers dynamic configuration options. Since it automatically detects lines, you have the flexibility to configure it according to your needs. By specifying what each vertical line corresponds to, you can achieve infinite possibilities in interpreting and grading your sheets.
