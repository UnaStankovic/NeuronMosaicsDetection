import numpy as np
import cv2 as cv

import pandas as pd
import os

ROW_LEN = 130

#Format of the features in csv file: for each image and each Keypoint there is one row in csv file
#containing columns: image_name, KeyPoint id and vector of length 128 representing descriptor of the KeyPoint 
#last column contains class name true/false

#Parameters: input_dirname is path to the input directory
#			 output_filename is the name of the output csv file
#			 classname is the name of the class	

#Function extracts features and writes them in the csv file in explained format

def extract_features_to_csv(input_dirname, output_filename, classname='true'):
    images_list = os.listdir(input_dirname)
	#initialize data as an empty array
    data = np.empty((0, ROW_LEN))
	#for each image in the directory
    for image_name in images_list:
		#read the image
        image = cv.imread(os.path.join(input_dirname, image_name))
		#convert image to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		#create SIFT object
        sift = cv.xfeatures2d.SIFT_create()
		#detect and compute descriptors for every keypoint in the image
        kp, des = sift.detectAndCompute(gray, None)
        #images not containing keyPoints will not appear in the output csv file
        if not kp:
            continue
        kp_num = len(kp)   
		#append rows for each keypoint to the resulting array
        for i in range(0, kp_num):
            new_row = np.hstack(([image_name, kp[i]], des[i]))
            data = np.vstack((data, new_row))
    indexes = [i for i in range(0,128)]        
    data_frm = pd.DataFrame(data, columns=['image_name', 'keypoint'] + indexes)
    data_frm['class'] = classname
	#write data to csv file
    data_frm.to_csv(output_filename)