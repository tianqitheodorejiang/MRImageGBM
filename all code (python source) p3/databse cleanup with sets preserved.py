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

##### main progra

input_path = "C:/Users/JiangQin/Documents/data/raw ct files/TCGA-GBM"
output_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/raw ct files/TCGA with only flair t1 and t2"
needs = ["flair", "t1","t2"]
deletes = []
sets_moved = 0
folders_moved = 0

folders_moving = 0

print("recognizing sets...\n")
for path, dirs, files in os.walk(input_path, topdown=False):
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
        if min(satisfied_needs) == 1:
            moving = False
            set_path = os.path.join(output_path, dir_)
            for folder in good_folders:
                new_folder_path = os.path.join(set_path, os.path.basename(folder))
                if not os.path.exists(new_folder_path):
                    folders_moving += 1
 


for path, dirs, files in os.walk(input_path, topdown=False):
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
        if min(satisfied_needs) == 1:
            moving = False
            set_path = os.path.join(output_path, dir_)
            if not os.path.exists(set_path):
                sets_moved += 1
                print("\ncopying set:" ,set_path, "\n")
                os.makedirs(set_path)
                moving = True
            for folder in good_folders:
                new_folder_path = os.path.join(set_path, os.path.basename(folder))
                if not os.path.exists(new_folder_path):
                    folders_moved += 1
                    print("copying folder:", new_folder_path)
                    os.makedirs(new_folder_path)
                    for file in os.listdir(folder):
                        shutil.copy(os.path.join(folder, file), new_folder_path)
                    if folders_moved % 5 == 0 and folders_moved != 0:
                        print ("\n" + str(round(round(folders_moved/folders_moving, 4) * 100, 2)) + "% done.")
                
            
                    
print("sets copied:", sets_moved)
print("folders_copied:", folders_moved)
                    
            



