import os
import random
import pydicom as dicom
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import skimage.transform
import nibabel as nib
import h5py
import scipy

import time
start_time = time.time()

nii_path = "/home/jiangl/Documents/python/ct to tumor identifier project/raw ct files/MICCAI_BraTS_2019_Data_Training"
nii_path2 = "/home/jiangl/Documents/python/ct to tumor identifier project/raw ct files/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_18_1/BraTS19_2013_18_1_seg.nii.gz"
dicom_path = "/media/jiangl/50EC5AFF0AA889DF/CPTAC-GBM/C3L-00016/11-15-1999-MR BRAIN WOW CONTRAST-47088/6-Ax Flair irFSE H-80553"
precise_model_path1 = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/True area/Model 9"
precise_model_path2 = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/True area/Model 5"
area_model_path1 = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/True area/Model 9"
#brain_seg_model = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/True area/Model 6"
brain_seg_model = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/Model 32"
output_image_path = "/home/jiangl/Documents/python/ct to tumor identifier project/image ct  visualizations/Machine Learning 2 models test"
output_image_path_Seg = "/home/jiangl/Documents/python/ct to tumor identifier project/image ct  visualizations/brain 1"

output_image_path_Seg2 = "/home/jiangl/Documents/python/ct to tumor identifier project/image ct  visualizations/brain 1 (2)"


def load_array(path, image_size = 128):
    if os.path.exists(path):
        if os.path.isfile(path):
            if path.endswith("nii.gz"):
                image = nib.load(path)
                image_data = image.get_data().T
 
                ##normalizing the data to range from 0 to 1
                image_data = image_data/np.max(image_data)
                
                
                ##calculating the zoom factors and reshaping
                z_zoom = image_size/image_data.shape[0]
                y_zoom = image_size/image_data.shape[1]
                x_zoom = image_size/image_data.shape[2]

                image_data = skimage.transform.rescale(image_data, (z_zoom, y_zoom, x_zoom))

                return np.stack([np.stack([image_data], axis = 3)])
            
            elif path.endswith(".dcm"):
                print("This file format is not yet supported")


            
        elif os.path.isdir(path):
            if os.listdir(path)[0].endswith(".dcm"):
                                
                raw_data = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
                raw_data.sort(key = lambda x: int(x.InstanceNumber))

                ##sets the slice thickness 
                try:
                    slice_thickness = np.abs(raw_data[0].ImagePositionPatient[2] - raw_data[1].ImagePositionPatient[2])
                except:
                    slice_thickness = np.abs(raw_data[0].SliceLocation - raw_data[1].SliceLocation)

                for s in raw_data:
                    s.SliceThickness = slice_thickness

                unprocessed_pixel_data = np.stack([s.pixel_array for s in raw_data])
                #unprocessed_pixel_data = (np.maximum(unprocessed_pixel_data,0) / unprocessed_pixel_data.max()) * 255.0
 
                        
                new_spacing=[1,1,1]
                spacing = map(float, ([raw_data[0].SliceThickness, raw_data[0].PixelSpacing[0], raw_data[0].PixelSpacing[1]]))
                spacing = np.array(list(spacing))

                resize_factor = spacing / new_spacing
                new_real_shape = unprocessed_pixel_data.shape * resize_factor
                new_shape = np.round(new_real_shape)
                real_resize_factor = new_shape / unprocessed_pixel_data.shape

                image_data = scipy.ndimage.interpolation.zoom(unprocessed_pixel_data, real_resize_factor)
                
                if np.max(image_data <= 255):
                    image_data = image_data/255
                #elif np.max(image_data <= 1460):
                #    image_data = image_data/1460

                
                ##calculating the zoom factors and reshaping
                z_zoom = image_size/image_data.shape[0]
                y_zoom = image_size/image_data.shape[1]
                x_zoom = image_size/image_data.shape[2]

                image_data = skimage.transform.rescale(image_data, (z_zoom, y_zoom, x_zoom))

                return np.stack([np.stack([image_data], axis = 3)])


            
    else:
        print("Path does not exist")

def write_images(array, test_folder_path):
    array = array[0]/np.max(array)
    for n,image in enumerate(array):
        ##finds the index of the corresponding file name in the original input path from the resize factor after resampling
        file_name = str(str(n) +'.png')

        cv2.imwrite(os.path.join(test_folder_path, file_name), np.squeeze(image*255, axis = 2))



def binarize(array, min_):
    array = array/np.max(array)
    binary = array.copy()
    binary[array < min_] = 0
    binary[array >= min_] = 1

    return binary


    
    
def dilate_up(array, size):
    binary = np.squeeze(array.copy()[0], axis = 3)


    ##creates a kernel which is a 3 by 3 square of ones as the main kernel for all denoising
    kernel = scipy.ndimage.generate_binary_structure(3, 1)

    ##erodes away the white areas of the 3d array to seperate the loose parts
    blew_up = scipy.ndimage.binary_dilation(binary.astype('uint8'), kernel, iterations=size)

    return np.stack([np.stack([blew_up], axis = 3)])



def circle_highlighted(reference, binary, color):
    circled = np.squeeze(reference.copy()[0], axis = 3)
    binary = np.squeeze(binary.copy()[0], axis = 3)
    binary[binary > 0] = 1
    
    
    for n, image in enumerate(binary):
        contours, _ = cv2.findContours(image.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(circled[n], contours, -1,color, 1)

    

    return np.stack([np.stack([circled], axis = 3)])

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
    


def biggest_island(input_array):
    masked = np.squeeze(input_array.copy()[0], axis = 3)
    
    touching_structure_3d =[[[0,0,0],
                             [0,255,0],
                             [0,0,0]],

                            [[0,255,0],
                             [255,255,255],
                             [0,255,0]],

                            [[0,0,0],
                             [0,255,0],
                             [0,0,0]]]

    binary = np.squeeze(input_array.copy()[0], axis = 3)
    binary[:] = 0
    binary[np.squeeze(input_array.copy()[0], axis = 3) > 0] = 255

    ##uses a label to find the largest object in the 3d array and only keeps that (if you are trying to highlight something like bones, that have multiple parts, this method may not be suitable)
    markers, num_features = scipy.ndimage.measurements.label(binary,touching_structure_3d)
    binc = np.bincount(markers.ravel())
    binc[0] = 0
    noise_idx = np.where(binc != np.max(binc))
    mask = np.isin(markers, noise_idx)
    binary[mask] = 0

    masked[binary == 0] = 0
    
    return np.stack([np.stack([masked], axis = 3)])


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
 
def blur(array, blur_precision):

    return scipy.ndimage.gaussian_filter(array, blur_precision)

def adaptive_threshold(array, course, precise, blur_precision = 0):
    thresholded_array = np.squeeze(array.copy()[0], axis = 3)
    thresholded_array = thresholded_array*255/np.max(thresholded_array)
    blurred = scipy.ndimage.gaussian_filter(thresholded_array, blur_precision)
    
    adap = []
    for image in blurred:
        thresh = cv2.adaptiveThreshold(image.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, course, 1)
        thresh2 = cv2.adaptiveThreshold(image.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, precise, 1)

        thresh3 = thresh.copy()
        thresh3[:] = 255

        thresh3[thresh2 == 0] = 0
        thresh3[thresh == 0] = 0

        adap.append(thresh3)

        
        
    adap = np.stack(adap)
            
        
    thresholded_array[adap == 0] = 0




    return np.stack([np.stack([thresholded_array], axis = 3)])


def kill_small_islands(array, denoise_iterations):
    binary = np.squeeze(array.copy()[0], axis = 3)
    masked = np.squeeze(array.copy()[0], axis = 3)

    binary[:] = 0
    binary[np.squeeze(array[0], axis = 3) > 0] = 255

    touching_structure_3d =[[[0,0,0],
                             [0,255,0],
                             [0,0,0]],

                            [[0,255,0],
                             [255,255,255],
                             [0,255,0]],

                            [[0,0,0],
                             [0,255,0],
                             [0,0,0]]]


    touching_structure_2d = [[0,255,0],
                             [255,255,255],
                             [0,255,0]]



    ##creates a kernel which is a 3 by 3 square of ones as the main kernel for all denoising
    kernel = scipy.ndimage.generate_binary_structure(3, 1)

    ##erodes away the white areas of the 3d array to seperate the loose parts
    if denoise_iterations > 0:
        eroded_3d = scipy.ndimage.binary_erosion(binary.astype('uint8'), kernel, iterations=denoise_iterations)
        eroded_3d = eroded_3d.astype('uint8') * 255

    else:
        eroded_3d = binary


    ##uses a label to find the largest object in the 3d array and only keeps that (if you are trying to highlight something like bones, that have multiple parts, this method may not be suitable)
    markers, num_features = scipy.ndimage.measurements.label(eroded_3d,touching_structure_3d)
    binc = np.bincount(markers.ravel())
    binc[0] = 0
    noise_idx = np.where(binc != np.max(binc))
    mask = np.isin(markers, noise_idx)
    eroded_3d[mask] = 0

    ##dilates the entire thing back up to get the basic shape before
    if denoise_iterations > 0:
        dilate_3d = scipy.ndimage.binary_dilation(eroded_3d.astype('uint8'), kernel, iterations=denoise_iterations)
        dilate_3d = dilate_3d.astype('uint8') * 255

    else:
        dilate_3d = eroded_3d

    masked[dilate_3d == 0] = 0
    
    return np.stack([np.stack([masked], axis = 3)])


def kill_smaller_islands(array, denoise_iterations, thresh = 50):
    binary = np.squeeze(array.copy()[0], axis = 3)
    masked = np.squeeze(array.copy()[0], axis = 3)

    binary[:] = 0
    binary[np.squeeze(array[0], axis = 3) > 0] = 255

    touching_structure_3d =[[[0,0,0],
                             [0,255,0],
                             [0,0,0]],

                            [[0,255,0],
                             [255,255,255],
                             [0,255,0]],

                            [[0,0,0],
                             [0,255,0],
                             [0,0,0]]]


    touching_structure_2d = [[0,255,0],
                             [255,255,255],
                             [0,255,0]]



    ##creates a kernel which is a 3 by 3 square of ones as the main kernel for all denoising
    kernel = scipy.ndimage.generate_binary_structure(3, 1)

    ##erodes away the white areas of the 3d array to seperate the loose parts
    if denoise_iterations > 0:
        eroded_3d = scipy.ndimage.binary_erosion(binary.astype('uint8'), kernel, iterations=denoise_iterations)
        eroded_3d = eroded_3d.astype('uint8') * 255

    else:
        eroded_3d = binary


    ##uses a label to find the largest object in the 3d array and only keeps that (if you are trying to highlight something like bones, that have multiple parts, this method may not be suitable)
    markers, num_features = scipy.ndimage.measurements.label(eroded_3d,touching_structure_3d)
    binc = np.bincount(markers.ravel())
    binc[0] = 0
    noise_idx = np.where(binc <= thresh)
    mask = np.isin(markers, noise_idx)
    eroded_3d[mask] = 0

    ##dilates the entire thing back up to get the basic shape before
    if denoise_iterations > 0:
        dilate_3d = scipy.ndimage.binary_dilation(eroded_3d.astype('uint8'), kernel, iterations=denoise_iterations)
        dilate_3d = dilate_3d.astype('uint8') * 255

    else:
        dilate_3d = eroded_3d

    masked[dilate_3d == 0] = 0
    
    return np.stack([np.stack([masked], axis = 3)])

def combine_white_binary(array1, array2):
    masked = array1.copy()
    
    binary = array1.copy()
    binary[:] = 0
    binary[array1 > 0] = 255
    binary[array2 > 0] = 255

    masked[binary == 255] = 255

    return binary


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


def fill_holes(array, sense):
    binary = np.squeeze(array.copy()[0], axis = 3)

    binary_original = np.squeeze(array.copy()[0], axis = 3)

    binary_original[:] = 0
    binary_original[np.squeeze(array.copy()[0], axis = 3) > 0.1] = 255
    
    binary[:] = 0
    binary[np.squeeze(array.copy()[0], axis = 3) < 0.1] = 255




    touching_structure_2d = [[0,255,0],
                             [255,255,255],
                             [0,255,0]]


    denoised = []
    for n,image in enumerate(binary):
        markers, num_features = scipy.ndimage.measurements.label(image,touching_structure_2d)
        omit = markers[0][0]
        flat = markers.ravel()
        binc = np.bincount(flat)
        binc_not = np.bincount(flat[flat == omit])
        noise_idx2 = np.where(binc > sense)
        noise_idx1 = np.where(binc == np.max(binc_not))
        
        mask1 = np.isin(markers, noise_idx1)
        mask2 = np.isin(markers, noise_idx2)
        
        image[mask1] = 0
        image[mask2] = 0
        denoised.append(image)

    denoised = np.stack(denoised)

    binary_original[denoised == 255] = 1

        
    return np.stack([np.stack([binary_original], axis = 3)])


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

    def circle_highlighted(self, array, color):
        circled = self.resampled_array.copy()
        binary = array.copy()
        
        binary[:] = 0
        binary[array > 0] = 255

        
        cont = []
        for n, image in enumerate(binary):
            contours, _ = cv2.findContours(image.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            #cont.append(contours)
            cv2.drawContours(circled[n], contours, -1,color, 1)

        circled[binary == 0] = 0

        return circled
 

    def write_test_images(self, array_3d, test_folder_path):
        array_3d = array_3d/np.max(array_3d)
        print(np.max(array_3d))
        for n,image in enumerate(array_3d):
            ##finds the index of the corresponding file name in the original input path from the resize factor after resampling
            file_name = str(str(n) +'.png')

            ##writes the resulting image as a png in the test_folder_path

            cv2.imwrite(os.path.join(test_folder_path, file_name), image*255)

            
    def erode_down(self, array, size):
        binary = np.squeeze(array.copy()[0], axis = 3)
        masked = np.squeeze(array.copy()[0], axis = 3)

        binary[:] = 0
        binary[np.squeeze(array.copy()[0], axis = 3) > 0] = 255




        ##creates a kernel which is a 3 by 3 square of ones as the main kernel for all denoising
        kernel = scipy.ndimage.generate_binary_structure(3, 1)

        ##erodes away the white areas of the 3d array to seperate the loose parts
        blew_up = scipy.ndimage.binary_erosion(binary.astype('uint8'), kernel, iterations=size)

        masked[blew_up == 0] = 0

        return np.stack([np.stack([masked], axis = 3)])


    def invert(self, array):
        masked = self.resampled_array.copy()

        binary = array.copy()
        binary[:] = 1
        binary[array > 0] = 0

        masked[binary == 0] = 0

        return masked

    def find_median_grayscale(self, array):
        
        zero_pixels = float(np.count_nonzero(array==0))


        single_dimensional = array.flatten().tolist()

        
        single_dimensional.extend(np.full((1, int(zero_pixels)), 1000).flatten().tolist())



        return np.median(single_dimensional)

    def find_mode_grayscale(self, array):
        array = array.copy()
        array[array < 0.1] = 2
        array = array*100

        array = array.flatten().tolist()

        array = [round(i) for i in array]


        array_len = len(array)
        zero_pixels = float(np.count_nonzero(array==200))
      
        array.sort()

        non_zero_end = array.index(200)

        del array[non_zero_end:]

        return scipy.stats.mode(array, axis=None)



        
image_size = 128
index = 5
file = get_folder_path("/home/jiangl/Documents/python/ct to tumor identifier project/raw ct files/Brain-Tumor-Progression", index, "flair")

brain = highlight_ct(file)
print('initialized')
brain.load_scan()
print('loaded scans')
pixel = brain.generate_pixel_data()
print('generated pixel array')
image_data = brain.resample_array()

image_data = image_data/np.max(image_data)

print("""

Set: """ + str(index) + """


""")

##calculating the zoom factors and reshaping
z_zoom = image_size/image_data.shape[0]
y_zoom = image_size/image_data.shape[1]
x_zoom = image_size/image_data.shape[2]

image_data = skimage.transform.rescale(image_data, (z_zoom, y_zoom, x_zoom))

original_array = np.stack([np.stack([image_data], axis = 3)])


blurred = blur(original_array, 1)


original_array = original_array/(np.max(blurred)*1.2)

white_bumped_up = original_array.copy()


adapted = adaptive_threshold(original_array, 15, 45, 0)



precise_model = keras.models.load_model(precise_model_path1)

area_prediction = precise_model.predict(original_array)

area_tumor_binary = binarize(area_prediction, 0.03)

com = combine_zeros(area_tumor_binary, adapted)


erode = brain.erode_down(com, 2)




brain_seg1 = keras.models.load_model(brain_seg_model)

brain_mask = brain_seg1.predict(original_array)

area_binary_brain = binarize(brain_mask, 0.3)



white_bumped_up[area_tumor_binary == 0] *= 0.8

white_bumped_up[area_binary_brain == 0] = 0

write_images(white_bumped_up, output_image_path_Seg)





brain_seg_array = original_array.copy()


brain_seg1 = keras.models.load_model(area_model_path1)

brain_mask = brain_seg1.predict(white_bumped_up)

area_binary_brain = binarize(brain_mask, 0.4)


correct = touching_island(area_binary_brain, erode)

brain_seg_array[correct == 0] = 0

binary = binarize(brain_seg_array, 0.7)

correct = touching_island(binary, erode)


correct = dilate_up(correct, original_array, 5)

circled = circle_highlighted(original_array, binary, 1)


write_images(circled, output_image_path)

print ('Finished in', int((time.time() - start_time)/60), 'minutes and ', int((time.time() - start_time) % 60), 'seconds.')




