import time
start_time = time.time()

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

import tensorflow as tf
from tensorflow import keras
import skimage.transform
import nibabel as nib
import h5py
import scipy

precise_model_path = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/Tumor Seg Model 17/Model 41"
area_model_path1 = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models//Model 6"
area_model_path2 = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/Tumor Seg Model 16/Model 41"
area_model_path3 = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/Tumor Seg Model 17/Model 41"
brain_seg_model = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/Model 34"
brain_seg_model_get_max = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/Brain Seg 4/Model 2"
brain_seg_model_patchup = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/True area/Model 9"
output_image_path = "/home/jiangl/Documents/python/ct to tumor identifier project/image ct  visualizations/Machine Learning 2 models test"



def binarize(array, min_):
    #array = array/np.max(array)
    binary = array.copy()
    binary[array < min_] = 0
    binary[array >= min_] = 1

    return binary


def circle_highlighted(reference, binary, color):
    circled = np.squeeze(reference.copy()[0], axis = 3)
    binary = np.squeeze(binary.copy()[0], axis = 3)
    binary[binary > 0] = 1
    
    
    for n, image in enumerate(binary):
        contours, _ = cv2.findContours(image.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(circled[n], contours, -1,color, 1)

    

    return np.stack([np.stack([circled], axis = 3)])

def write_images(array, test_folder_path):
    array = array[0]/np.max(array)
    for n,image in enumerate(array):
        ##finds the index of the corresponding file name in the original input path from the resize factor after resampling
        file_name = str(str(n) +'.png')

        cv2.imwrite(os.path.join(test_folder_path, file_name), np.squeeze(image*255, axis = 2))


def get_file_path(path, index, img_id):
    pathes = []
    for path, dirs, files in os.walk(path, topdown=False):
        for file in files:
            if img_id in file:
                pathes.append(os.path.join(path, file))
                
    return pathes[index]

def get_folder_path(path, index, img_id):
    pathes = []
    for path, dirs, files in os.walk(path, topdown=False):
        for dir_ in dirs:
            if img_id in dir_.lower():
                pathes.append(os.path.join(path, dir_))
    
    return pathes[index]
    
def combine_zeros(array1,array2):
    masked = np.squeeze(array1.copy()[0], axis = 3)
    
    binary = np.squeeze(array2.copy()[0], axis = 3)
    binary[:] = 255
    binary[np.squeeze(array2.copy()[0], axis = 3) < 0.1] = 0
    binary[np.squeeze(array2.copy()[0], axis = 3) < 0.1] = 0

    masked[binary == 0] = 0
    
    return np.stack([np.stack([masked], axis = 3)])



def touching_island(reference, array):
    array = np.squeeze(array.copy()[0], axis = 3)
    reference = np.squeeze(reference.copy()[0], axis = 3)
    
    masked = array.copy()
    masked[:] = 0
    
    touching_structure_3d =[[[0,0,0],
                             [0,1,0],
                             [0,0,0]],

                            [[0,1,0],
                             [1,1,1],
                             [0,1,0]],

                            [[0,0,0],
                             [0,1,0],
                             [0,0,0]]]


   ##uses a label to find the largest object in the 3d array and only keeps that (if you are trying to highlight something like bones, that have multiple parts, this method may not be suitable)
    markers, num_features = scipy.ndimage.measurements.label(array, touching_structure_3d)
    reference_idx = np.unique(markers[reference == 1])
    for idx in reference_idx:
        masked[markers == idx] = 1

    masked[array == 0] = 0
    
    return np.stack([np.stack([masked], axis = 3)])
 


def combine_white_binary(array1, array2):
    masked = array1.copy()
    
    binary = array1.copy()
    binary[:] = 0
    binary[array1 > 0] = 255
    binary[array2 > 0] = 255

    masked[binary == 255] = 255

    return binary


class highlight_ct:

    def __init__(self, input_path):  
        self.input_path = input_path
        self.file_names = os.listdir(input_path)

    def load_scan(self):
        ##loads and sorts the data in as a dcm type array
        raw_data = [dicom.read_file(self.input_path + '/' + s) for s in os.listdir(self.input_path)]
        raw_data.sort(key = lambda x: int(x.InstanceNumber))

        ##sets the slice thickness 
        try:
            slice_thickness = np.abs(raw_data[0].ImagePositionPatient[2] - raw_data[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(raw_data[0].SliceLocation - raw_data[1].SliceLocation)

        for s in raw_data:
            s.SliceThickness = slice_thickness



        self.raw_data = raw_data ##update the output
        
    def generate_pixel_data(self):
        ## creates a 3d array of pixel data from the raw_data
        unprocessed_pixel_data = np.stack([s.pixel_array for s in self.raw_data])
        #unprocessed_pixel_data = (np.maximum(unprocessed_pixel_data,0) / unprocessed_pixel_data.max()) * 255.0



        self.original_pixel_array = unprocessed_pixel_data ##update the output
        return self.original_pixel_array
        
        
    def resample_array(self):
        ##resamples the array using the slice thickness obtained earlier
        new_spacing=[1,1,1]
        spacing = map(float, ([self.raw_data[0].SliceThickness, self.raw_data[0].PixelSpacing[0], self.raw_data[0].PixelSpacing[1]]))
        spacing = np.array(list(spacing))

        resize_factor = spacing / new_spacing
        new_real_shape = self.original_pixel_array.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / self.original_pixel_array.shape




        self.resize_factor = real_resize_factor ##creates a value called resize factor that can be used for image outputting later on
        
        self.resampled_array = scipy.ndimage.interpolation.zoom(self.original_pixel_array, real_resize_factor) ##update the output

        return self.resampled_array


def locate_bounds(array):
    left = np.squeeze(array.copy()[0], axis = 3).shape[2]
    right = 0 
    low = np.squeeze(array.copy()[0], axis = 3).shape[1]
    high = 0
    shallow = np.squeeze(array.copy()[0], axis = 3).shape[0]
    deep = 0

    array_3d = np.squeeze(array.copy()[0], axis = 3)
    for z,Slice in enumerate(array_3d):
        for y,line in enumerate(Slice):
            for x,pixel in enumerate(line):
                if pixel > 0.05:
                    if z > deep:
                        deep = z
                    if z < shallow:
                        shallow = z
                    if y > high:
                        high = y
                    if y < low:
                        low = y
                    if x > right:
                        right = x
                    if x < left:
                        left = x
                    
    #print([left,right,low,high,shallow,deep])
    
    return [left,right,low,high,shallow,deep]


def dilate_up(array, original, size):
    binary = np.squeeze(array.copy()[0], axis = 3)
    masked = np.squeeze(original.copy()[0], axis = 3)

    binary[:] = 0
    binary[np.squeeze(array.copy()[0], axis = 3) > 0] = 255




    ##creates a kernel which is a 3 by 3 square of ones as the main kernel for all denoising
    kernel = scipy.ndimage.generate_binary_structure(3, 1)

    ##erodes away the white areas of the 3d array to seperate the loose parts
    blew_up = scipy.ndimage.binary_dilation(binary.astype('uint8'), kernel, iterations=size)

    masked[blew_up == 0] = 0

    return np.stack([np.stack([masked], axis = 3)])



image_size = 128
index = 12
file = get_folder_path("/home/jiangl/Documents/python/ct to tumor identifier project/raw ct files/Brain-Tumor-Progression", index, "flair")

print("\n\nSet:", index, "\n\n")


print("loading scans...")
brain = highlight_ct(file)
brain.load_scan()
pixel = brain.generate_pixel_data()
image_data = brain.resample_array()
print("loaded scans.")

print("\nrescaling data...")
image_data = image_data/np.max(image_data)

z_zoom = image_size/image_data.shape[0]
y_zoom = image_size/image_data.shape[1]
x_zoom = image_size/image_data.shape[2]

image_data1 = skimage.transform.rescale(image_data, (z_zoom, y_zoom, x_zoom))

original_array1 = image_data1
original_array1[:] = 0

image_data = np.stack([np.stack([image_data], axis = 3)])

bounds = locate_bounds(image_data)

[left,right,low,high,shallow,deep] = bounds


x_size = abs(left-right)
y_size = abs(low-high)
z_size = abs(shallow-deep)

max_size = np.max([x_size, y_size, z_size])
rescale_factor = (image_size*0.8)/max_size


image_data = skimage.transform.rescale(np.squeeze(image_data.copy()[0], axis = 3), (rescale_factor, rescale_factor, rescale_factor))

for z,Slice in enumerate(image_data):
    for y,line in enumerate(Slice):
        for x,pixel in enumerate(line):
            original_array1[z][y][x] = pixel
original_array = np.stack([np.stack([original_array1], axis = 3)])
blank = original_array.copy()

blank[:] = 1

print("rescaled data.")


print("\nsegmenting brain...")
brain_seg_area = keras.models.load_model(brain_seg_model)

brain_prediction_area = brain_seg_area.predict(original_array)

binary_brain_area = binarize(brain_prediction_area, 0.2)

brain_seg_area = original_array.copy()

brain_seg_area[binary_brain_area == 0] = 0


brain_seg_precise = keras.models.load_model(brain_seg_model)

brain_prediction_precise = brain_seg_precise.predict(original_array)

binary_brain_precise = binarize(brain_prediction_precise, 0.2)


brain_seg_pinpoint = keras.models.load_model(brain_seg_model_get_max)

brain_prediction_pinpoint = brain_seg_pinpoint.predict(original_array)

binary_brain_pinpoint = binarize(brain_prediction_pinpoint, 1)

brain_seg_pinpoint = original_array.copy()

brain_seg_pinpoint[binary_brain_pinpoint == 0] = 0

print("brain segmented.")



final_brain_area = brain_seg_area/np.max(brain_seg_area)




print("\nsegmenting tumor...")

precise_model = keras.models.load_model(precise_model_path)

precise_tumor_prediction = precise_model.predict(final_brain_area)

precise_tumor_binary = binarize(precise_tumor_prediction, 1)



area_model1 = keras.models.load_model(area_model_path1)

area_prediction1 = area_model1.predict(final_brain_area)

area_binary1 = binarize(area_prediction1, 0.3)



area_model2 = keras.models.load_model(area_model_path2)

area_prediction2 = area_model2.predict(final_brain_area)

area_binary2 = binarize(area_prediction2, 0.95)


area_model3 = keras.models.load_model(area_model_path3)

area_prediction3 = area_model3.predict(final_brain_area)

area_binary3 = binarize(area_prediction3, 0.95)


print("tumor segmented.")


tumor_area_binary = combine_white_binary(area_binary1, area_binary2)
tumor_area_binary = combine_white_binary(tumor_area_binary, area_binary3)

tumor_area_binary = combine_zeros(tumor_area_binary , binary_brain_precise)


tumor_precise_binary = combine_zeros(precise_tumor_binary, binary_brain_precise)




correct_tumor_binary = touching_island(tumor_precise_binary, tumor_area_binary)


circled = circle_highlighted(original_array, tumor_area_binary, 1)


write_images(circled, output_image_path)


print ('\nFinished in', int((time.time() - start_time)/60), 'minutes and ', int((time.time() - start_time) % 60), 'seconds.')




