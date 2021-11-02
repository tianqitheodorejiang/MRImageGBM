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

for path, dirs, files in os.walk(clean_path, topdown=False):
    for dir_ in dirs:
        dir_path = path + "/" + dir_
        satisfied_needs = []
        good_folders = []
        for need in needs:
            satisfied_needs.append(0)
        for folder in os.listdir(dir_path):
            for n, need in enumerate(needs):
                if need in folder.lower():
                    satisfied_needs[n] = 1
                    good_folders.append(os.path.join(dir_path, folder))
        if min(satisfied_needs) != 1:
            is_set = False
            for item in os.listdir(dir_path):
                if os.path.isdir(os.path.join(dir_path, item)):
                    is_set = True
            if is_set:
                total_removing += 1
            
 


for path, dirs, files in os.walk(clean_path, topdown=False):
    for dir_ in dirs:
        dir_path = path + "/" + dir_
        satisfied_needs = []
        good_folders = []
        for need in needs:
            satisfied_needs.append(0)
        for folder in os.listdir(dir_path):
            for n, need in enumerate(needs):
                if need in folder.lower():
                    satisfied_needs[n] = 1
                    good_folders.append(os.path.join(dir_path, folder))
        if min(satisfied_needs) != 1:
            is_set = False
            for item in os.listdir(dir_path):
                if os.path.isdir(os.path.join(dir_path, item)):
                    is_set = True
            if is_set:
                shutil.rmtree(dir_path)
                removed_folders += 1
                print("removing:", dir_path)
                print(str(round(round(removed_folders/total_removing, 4) * 100, 2)) + "% done.")
            
            
                    
print("removed " + str(removed_folders) + " folders.")

                    
            



