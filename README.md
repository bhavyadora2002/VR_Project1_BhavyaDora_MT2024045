# a. Binary Classification Using Handcrafted Features and ML Classifiers (4 Marks)
i. Extract handcrafted features from the dataset.
ii. Train and evaluate at least two machine learning classifiers (e.g., SVM, Neural
network) to classify faces as "with mask" or "without mask."
iii. Report and compare the accuracy of the classifiers.
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
  ![image](https://github.com/user-attachments/assets/a56929d6-40ae-4056-a4ba-a4752595911e)
3. **Data Splitting:**
* As the data is not balanced(2165 images in with_mask and 1930 images in without_mask), We used Stratified Split to maintain class balance.
4. **Model Training**:
* Performed classification task on features obtained from Canny,Sobel and Contours using Random Forest and Support Vector Machine Classifiers.
  Calculated Test Accuracy to compare results.
## **Results:**
* Test Accuracy of RF on Canny: 0.74
* Test Accuracy of SVC on Canny: 0.76
* Test Accuracy of RF on Sobel: 0.85
* Test Accuracy of SVC on Sobel: 0.71
* Test Accuracy of RF on Contours: 0.75
* Test Accuracy of SVC on Contours: 0.76
* Random Forest on Sobel gives good accuracy compared to others
## **Observations and Analysis:**
* We did analysis for RF on Sobel as it is giving high accuracy.
* To check where my classifier is working good and bad, We displayed 5 images for True Positives, True Negatives, False Positives and False Negatives.
* We found that classifier failed for images with text, images with other objects,rotated images and images with no clear boundaries etc.
![image](https://github.com/user-attachments/assets/0724d595-a618-4d2d-99fd-b3900f355609)
![image](https://github.com/user-attachments/assets/25102b75-1d0a-4442-9877-2b1c8cd5ff25)
## How to run code:
* Upload the dataset in your google drive.
* Path to upload dataset: /content/drive/MyDrive/Colab Notebooks
* Open VR_Task_a_b.ipynb in your colab and run it.
# b. Binary Classification Using CNN
i. Design and train a Convolutional Neural Network (CNN) to perform binary
classification on the same dataset.
ii. Try a few hyper-parameter variations (e.g., learning rate, batch size, optimizer,
activation function in the classification layer) and report the results.
iii. Compare the CNN's performance with the ML classifiers.
## **Dataset:** 
My dataset contains 4095 images.Out of which, 2165 images consists of faces with masks and 1930 images consists of faces without masks.
* Structure: Dataset->with_mask,without_mask(Dataset folder consists of 2 sub folders i.e, with_mask, without_mask)
* Source: https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset
## **Methodology:**
1. **Accessing Data:** 
Loaded images from the dataset, converted them to RGB format, resized them to 64x64, and stored them in a list. Converted the image data and labels into NumPy arrays for model training.
2. **Data Splitting:**
* As the data is not balanced(2165 images in with_mask and 1930 images in without_mask), We used Stratified Split to maintain class balance.
* Splitted the data to 64% tarin data,16% Validation data and 20% test data.
3. **Model Training:**
* Used Convolution Neural Network to train the model. Tried with different Hyper-Paramneters to find the best Hyper-parameters for the model.
## **Hyperparameters and Experiements:**
1. **Optimizer**:
* Performed model training using Adam,SGD,RMSProp and AdaGrad and calculated accuracy to compare them.
* Adam Test Accuracy: 0.8950, SGD Test Accuracy: 0.8962, RMSprop Test Accuracy: 0.8462, Adagrad Test Accuracy: 0.8303.
* Adam and SGD shows better results compared to other classifiers.SGD is slightly better than Adam while comparing validation accuracy and test accuracy.
2. **Activation Function in Classification Layer:**
* Performed model training using sigmoid,softmax,reLu,tanh and calculated accuracy to compare them.
* sigmoid Test Accuracy: 0.9048, softmax Test Accuracy: 0.4713, tanh Test Accuracy: 0.5287, relu Test Accuracy: 0.4725
* By observing results, Sigmoid works better than other activations.
3. **Learning Rate:**
* Performed model training for different learning rates(0.01,0.005,0.001) and calculated accuracy to compare them.
* 0.001 Test Accuracy: 0.9011, 0.005 Test Accuracy: 0.9121, 0.01 Test Accuracy: 0.9109
* The results are almost similar.We chose 0.01 for faster convergence
4. **Batch Size:**
* Performed model training for different batch sizes(16,32,64) and calculated accuracy to compare them.
* batchsize 16 Test Accuracy: 0.9109, batchsize 32 Test Accuracy: 0.9353, batchsize 64 Test Accuracy: 0.7216
* Training with batch size 32 gives better accuracy.
## **Model training:**
* Performed model training for CNN using SGD as optimizer, sigmoid as Activation Function, learning rate 0.01, and batchsize 32.
## Observations and Analysis:
* Test Accuracy of RF on Canny: 0.74
* Test Accuracy of SVC on Canny: 0.76
* Test Accuracy of RF on Sobel: 0.85
* Test Accuracy of SVC on Sobel: 0.71
* Test Accuracy of RF on Contours: 0.75
* Test Accuracy of SVC on Contours: 0.76
* Test Accuracy of CNN: 0.9487
1.  CNN (0.9487 accuracy) is significantly better than RF (0.85) and SVM (0.76)
2.  CNN is better because
*   It will automatically learns features, while ML classifiers rely on manually extracted features.
*   ML classifiers rely on preprocessing, while CNN learns from raw pixel data
## How to run code:
* Upload the dataset in your google drive.
* Path to upload dataset: /content/drive/MyDrive/Colab Notebooks
* Open VR_Task_a_b.ipynb in your colab and run it.





  






































# c. Region Segmentation Using Traditional Techniques (3 Marks)
i. Implement a region-based segmentation method (e.g., thresholding, edge detection) to segment the mask regions for faces identified as "with mask."

## Introduction
* The aim of this task is to identify the mask using region based segmentation

## Dataset
* A Masked Face Segmentation Dataset with ground truth face masks can be accessed here: https://github.com/sadjadrz/MFSD
    * This was the folder structure of the dataset.
      ![image](https://github.com/user-attachments/assets/6c7383b8-abbc-451b-a34e-1d5064b9b4ad)

        * img
            * Contains the images of people with mask and without mask.
        * dataset.csv
            * There were multiple entries of a image with labelled bounding boxes i.e:- if the mask was present then labelled as True or else False.
        * face_crop 
            * Using the dataset.csv file taken out the bounding boxes of only with mask label and stored the face_crop file.
            * This folder containe 9383 images which are faces with masks extracted from the images in the folder img.
        * Ground truth:- face_crop_segmentation
            * This folder contains the binary images of the masks which act as a ground truth so that we compare our results with these images using the Intersection over Union(IOU) metrics.

## Steps
* Converted the images to gray scale as it works best for single channel.
* Applied the guassian blur with a 5,5 kernel to remove noise and smoothen the image.
* Identify the images by applying the Canny edge detection with threshold values of 50 to 150.
* Adaptive otsu's threshold:- Used this instead of fixed threshold because we don't know 
    1)  cv2.THRESH_BINARY → Converts pixels to either 0 (black) or 255 (white) based on a threshold.
    2)  cv2.THRESH_OTSU → Automatically determines the optimal threshold value by minimizing the variance between foreground and background pixel intensities.
* Morphological operation that removes small holes (black regions) inside white objects in a binary image.
    1) Dilation → Expands the white regions.
    2) Erosion → Shrinks the expanded regions back.
* Finding contours to identify the shape of the image.
* An empty mask is initialized and the detected contours are overlaid on this by coloring the identified mask with blue colour.

## Results
* IoU Formula:
    * Intersection: Common area between predicted and ground truth masks.
    * Union: Total area covered by both masks.
* Displaying the best 5 results so check the results
* Mean iou:- 0.3226
![image](https://github.com/user-attachments/assets/db5c15aa-ab3d-478a-82bd-ad1165e798d5)

## Observations and Challenges
* As the dataset is very large and computing and storing the results occupies large amount of memory so we are using moving average and displaying only top 5 results.
* Also when we used fixed threshold we were not able to perform very well because most of the time the mask was regioned as a part of face so to avoid that we used adaptive threshold where based on the surrounding region thresholding was done.
* Some images had low IoU values, suggesting under-segmentation (missing parts of the object) or over-segmentation (extra regions included).

## How to run
* Load the dataset.
* Change the path in the code according to the loaded dataset.
* Run the file.


  
