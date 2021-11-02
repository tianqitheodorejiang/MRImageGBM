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

clean_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/raw ct files/TCGA-GBM DATA"
needs = ["flair", "t1"]

removed_folders = 0


total_removing = 0

print("recognizing_data...\n")

for set_ in os.listdir(clean_path):
    dir_path = clean_path + "/" + set_
    has_t1 = False
    has_flair = False
    for folder in os.listdir(dir_path):
        if "t1" in folder.lower() and "flair" not in folder.lower():
            has_t1 = True
        if "ax" in folder.lower() and "flair" in folder.lower():
            has_flair = True

    if not (has_flair and has_t1):
        total_removing += 1


for set_ in os.listdir(clean_path):
    dir_path = input_path + "/" + set_
    has_t1 = False
    has_flair = False
    for folder in os.listdir(dir_path):
        if "t1" in folder.lower() and "flair" not in folder.lower():
            has_t1 = True
        if "ax" in folder.lower() and "flair" in folder.lower():
            has_flair = True

    if not (has_flair and has_t1):
        shutil.rmtree(dir_path)
        removed_folders += 1
        print("removing:", dir_path)
        print(str(round(round(removed_folders/total_removing, 4) * 100, 2)) + "% done.")





                    
print("removed " + str(removed_folders) + " folders.")

                    
            



