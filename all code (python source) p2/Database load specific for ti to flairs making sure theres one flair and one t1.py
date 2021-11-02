import os
import random
import pydicom as dicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from skimage.measure import marching_cubes_lewiner as marching_cubes
import stl
from stl import mesh
import shutil
import ntpath

import tensorflow as tf
from tensorflow import keras
import skimage.transform
import nibabel as nib
import h5py
import scipy

import time
start_time = time.time()


print("imported modules")

##### main program

input_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/raw ct files/TCGA-GBM DATA"

output_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/raw ct files/TCGA GBM DATA t1 to flair"

removed_folders = 0


total_removing = 0

print("recognizing_data...\n")

for set_ in os.listdir(input_path):
    dir_path = input_path + "/" + set_
    has_t1 = False
    has_flair = False

    if len(os.listdir(dir_path)) == 2:
        total_removing += 1


for set_ in os.listdir(input_path):
    dir_path = input_path + "/" + set_
    if len(os.listdir(dir_path)) == 2:
        shutil.move(dir_path, os.path.join(output_path, set_))
        removed_folders += 1
        print("moving:", dir_path)
        print(str(round(round(removed_folders/total_removing, 4) * 100, 2)) + "% done.")





                    
print("moved " + str(removed_folders) + " folders.")

                    
            



