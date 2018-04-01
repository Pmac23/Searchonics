# USAGE
# python search.py --index index.csv --query queries/103100.png --result-path dataset

# import the necessary packages
from pyimagesearch.colordescriptor import ColorDescriptor
from pyimagesearch.searcher import Searcher
import argparse
import cv2
from scipy.stats import wasserstein_distance
from scipy.ndimage import imread
import numpy as np
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import os, os.path
from PIL import Image


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True,
    help = "Path to where the computed index will be stored")
ap.add_argument("-q", "--query", required = True,
    help = "Path to the query image")
ap.add_argument("-r", "--result-path", required = True,
    help = "Path to the result path")
args = vars(ap.parse_args())


# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(query, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("query", query)


# load the query image and describe it
query = cv2.imread(args["query"])
clone = query

# call the method to crop
cv2.namedWindow("query")
cv2.setMouseCallback("query", click_and_crop)



# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("query", query)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)


    
# compare the similarity between two images


#mean squared error
def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = abs(np.sum((imageA.astype("float") - imageB.astype("float")) ** 2))
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err




#convert opencv image to PIL
roiI = Image.fromarray(roi)

#resize the original - region of interest
new_width  = 300
new_height = 300
roiI = roiI.resize((new_width, new_height), Image.ANTIALIAS)

#convert it back to original
original = np.array(roiI)
    

# convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

#compare images

#image path and valid extensions
imageDir = "C:/Users/Prajakta Pardeshi/images/im/dataset/" #specify your path here
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] #specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]

#create a list all files in directory and
#append files with a vaild extention to image_path_list
for file in os.listdir(imageDir):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDir, file))
    
    #loop through image_path_list to open each image
    # Remember, as the MSE increases the images are less similar
    # as opposed to the SSIM where smaller values indicate less similarity
for imagePath in image_path_list:
    #read the image
    image = cv2.imread(imagePath)
    
    if image is not None:
    
        #format the image and convert to PIL
        contrast1 = Image.fromarray(image)
    
        # resize the file
        contrast1 = contrast1.resize((new_width, new_height), Image.ANTIALIAS)
        
        #convert back to opencv format
        contrast = np.array(contrast1)
        
        #convert to grayscale
        contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    
       # compare_images(original, contrast, "Original vs. Contrast")
        m = mse(original, contrast)
        
        s = ssim(original, contrast)
        print(s)
        
        # Remember, as the MSE increases the images are less similar
        # as opposed to the SSIM where smaller values indicate less similarity
        if m < 350 or s > 0.4:
       # if s > 0.3:
            cv2.imshow("Result", image)
            cv2.waitKey(0)

    
#cv2.imshow("Query", query)
