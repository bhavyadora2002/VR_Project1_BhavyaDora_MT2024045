# a. Binary Classification Using Handcrafted Features and ML Classifiers (4 Marks)
## i. Extract handcrafted features from the dataset.
## ii. Train and evaluate at least two machine learning classifiers (e.g., SVM, Neural
network) to classify faces as "with mask" or "without mask."
## iii. Report and compare the accuracy of the classifiers.
## **Dataset:** 
My dataset contains 4095 images.Out of which, 2165 images consists of faces with masks and 1930 images consists of faces without masks.
* Structure: Dataset->with_mask,without_mask(Dataset folder consists of 2 sub folders i.e, with_mask, without_mask)
* Source: https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset
## **Methodology:**
1. **Accessing Data:** 
* Iterated through each image and assigned labels to them based on their category. 0 for images with mask and 1 for images without mask.
2. **Feature Extraction:** Used Canny Edge Detection, Sobel Edge Detection and Contour Edge Detection for Preprocessing.
* Canny Edge Detection: Converted images to grayscale, resized them, and applied Canny edge detection with thresholding(50,150). Flattened the extracted edge features and stored them for training.
* Sobel Edge Detection: Computed Sobel gradients in both x and y directions, combined their magnitudes to get edge strength. Flattened the Sobel feature map and stored it for training.
* Contours Detection: Used Canny edges to find contours and drew them on a blank image. Flattened the contour image and stored it as features for training.
* Appended labels for the extracted features of each category. Now my data is ready to perform classification.
3. **Data Splitting:**
* As the data is not balanced(2165 images in with_mask and 1930 images in without_mask), We used Stratified Split to maintain class balance.
4. **Model Training**:
* Performed classification task on features obtained from Canny,Sobel and Contours using Random Forest and Support Vector Machine Classifiers.
  Calculated Test Accuracy to compare results.









* **Training**
  
