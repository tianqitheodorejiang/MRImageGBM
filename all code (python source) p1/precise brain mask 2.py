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

import time
start_time = time.time()

nii_path = "/home/jiangl/Documents/python/ct to tumor identifier project/raw ct files/MICCAI_BraTS_2019_Data_Training"
nii_path2 = "/home/jiangl/Documents/python/ct to tumor identifier project/raw ct files/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_18_1/BraTS19_2013_18_1_seg.nii.gz"
dicom_path = "/media/jiangl/50EC5AFF0AA889DF/CPTAC-GBM/C3L-00016/11-15-1999-MR BRAIN WOW CONTRAST-47088/6-Ax Flair irFSE H-80553"
area_model_path1 = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/Model 6"
area_model_path2 = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/Tumor Seg Model 17/Model 41"
area_model_path3 = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/Tumor Seg Model 16/Model 41"
precise_model_path = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/Tumor Seg Model 13/Model 41"
precise_model_path2 = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/Tumor Seg Model 16/Model 51"
area_model_precise = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/Model 43"
brain_seg_model_area = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/Model 34"
brain_seg_model_precise1 = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/Brain Seg 4/Model 1"
brain_seg_model_precise2 = "/home/jiangl/Documents/python/ct to tumor identifier project/code files/Saved Models/Brain Seg 4/Model 2"
output_image_path = "/home/jiangl/Documents/python/ct to tumor identifier project/image ct  visualizations/Machine Learning 2 models test"
output_image_path_Seg = "/home/jiangl/Documents/python/ct to tumor identifier project/image ct  visualizations/brain 1"

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
    #array = array/np.max(array)
    binary = array.copy()
    binary[array < min_] = 0
    binary[array >= min_] = 1

    return binary


def binarize_max(array, min_):
    array = array/np.max(array)
    binary = array.copy()
    binary[array < min_] = 0
    binary[array >= min_] = 1

    return binary

def binarize_blurred(array, min_, blur_prec):
    array = array/np.max(array)
    array = blur(array, blur_prec)
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
    print(reference_idx)
    for idx in reference_idx:
        masked[markers == idx] = 1

    masked[array == 0] = 0
    
    return np.stack([np.stack([masked], axis = 3)])
 


def adaptive_threshold(array, course, precise, blur_precision = 0):
    thresholded_array = np.squeeze(array.copy()[0], axis = 3)
    thresholded_array = thresholded_array*255/np.max(thresholded_array)
    blurred = scipy.ndimage.gaussian_filter(thresholded_array, blur_precision)
    
    adap = []
    for image in blurred:
        thresh = cv2.adaptiveThreshold(image.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, course, 2)
        thresh2 = cv2.adaptiveThreshold(image.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, precise, 2)

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

def blur(array, blur_precision):

    return np.stack([np.stack([scipy.ndimage.gaussian_filter(np.squeeze(array.copy()[0], axis = 3), blur_precision)], axis = 3)])   



def branch(array, brancher, iterations):

    reference_blank = array.copy()
    reference_blank[:] = 1
    
    for i in range(0, iterations):
        dilate = dilate_up(array, reference_blank, 1)
        array = combine_zeros(dilate, brancher)
        array = biggest_island(array)

    return array
        

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

            
    def generate_stl(self, array_3d, stl_file_path, name, stl_resolution):
        print('Generating mesh...')
        ##transposes the image to be the correct shape because np arrays are technically flipped
        transposed = np.squeeze(array_3d.copy()[0], axis = 3)#.transpose(2,1,0)

        ##uses the marching cubes algorithm to make a list of vertices, faces, normals, and values
        verts, faces, norm, val = marching_cubes(transposed, 0.01, step_size = stl_resolution, allow_degenerate=True)

        mesh = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
        print('Vertices obatined:', len(verts))
        print('')
        for i, f in enumerate(faces):
            for j in range(3):
                mesh.vectors[i][j] = verts[f[j],:]
        path = stl_file_path + '/' + name
        mesh.save(path)

            
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

def find_median_grayscale(array):
    
    zero_pixels = float(np.count_nonzero(array==0))


    single_dimensional = array.flatten().tolist()

    
    single_dimensional.extend(np.full((1, int(zero_pixels)), 1000).flatten().tolist())



    return np.median(single_dimensional)


def find_median(array):
    
    single_dimensional = array.flatten().tolist()

    return np.median(single_dimensional)

def find_mode_grayscale(array):
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

def translate(array, empty, translation):
    original_array = np.squeeze(array.copy()[0], axis = 3)

    array_translated = np.squeeze(empty.copy()[0], axis = 3)
    array_translated[:] = 0
    for z,Slice in enumerate(original_array):
        for y,line in enumerate(Slice):
            for x,pixel in enumerate(line):
                if pixel > 0.05:
                    array_translated[z+translation[0]][y+translation[1]][x+translation[2]] = pixel

    return np.stack([np.stack([array_translated], axis = 3)])
    

def cast_bounding(array, wanted_bounds, current_bounds):
    array = np.squeeze(array.copy()[0], axis = 3)


    [left,right,low,high,shallow,deep] = current_bounds

    
    x_size = abs(left-right)
    y_size = abs(low-high)
    z_size = abs(shallow-deep)

    [left_wanted,right_wanted,low_wanted,high_wanted,shallow_wanted,deep_wanted] = wanted_bounds


    x_wanted = abs(left_wanted-right_wanted)
    y_wanted = abs(low_wanted-high_wanted)
    z_wanted = abs(shallow_wanted-deep_wanted)

    z_zoom = z_wanted/z_size
    y_zoom = y_wanted/y_size
    x_zoom = x_wanted/x_size

    rescaled_array = skimage.transform.rescale(array, (z_zoom, y_zoom, x_zoom))

    [left_rescaled,right_rescaled,low_rescaled,high_rescaled,shallow_rescaled,deep_rescaled] = locate_bounds(np.stack([np.stack([rescaled_array], axis = 3)]))


    translate_x = right_wanted-right_rescaled
    translate_y = high_wanted-high_rescaled
    translate_z = deep_wanted-deep_rescaled
    translated = translate(np.stack([np.stack([rescaled_array], axis = 3)]), np.stack([np.stack([array], axis = 3)]), (translate_z,translate_y,translate_x))


    translated = np.squeeze(translated.copy()[0], axis = 3)

    return np.stack([np.stack([translated], axis = 3)])

def locate_bounds_2d(array):
    left = array.copy().shape[1]
    right = 0 
    low = array.copy().shape[0]
    high = 0

    array_2d = array.copy()
    for y,line in enumerate(array_2d):
        for x,pixel in enumerate(line):
            if pixel > 0.05:
                if y > high:
                    high = y
                if y < low:
                    low = y
                if x > right:
                    right = x
                if x < left:
                    left = x
                    
    return [left,right,low,high]



def translate_2d(array, empty, translation):
    original_array = array.copy()

    array_translated = empty.copy()
    array_translated[:] = 0
    for y,line in enumerate(original_array):
        for x,pixel in enumerate(line):
            if pixel > 0.1:
                array_translated[y+translation[0]][x+translation[1]] = pixel

    return array_translated



def bounds_scale_2d(array, wanted_bounds, current_bounds):
    array = array.copy()


    [left,right,low,high] = current_bounds

    
    x_size = abs(left-right)
    y_size = abs(low-high)


    [left_wanted,right_wanted,low_wanted,high_wanted] = wanted_bounds


    x_wanted = abs(left_wanted-right_wanted)
    y_wanted = abs(low_wanted-high_wanted)



    y_zoom = y_wanted/y_size
    x_zoom = x_wanted/x_size

    average_zoom = (y_zoom + x_zoom)/2

    rescaled_array = skimage.transform.rescale(array, (average_zoom, average_zoom))

    [left_rescaled,right_rescaled,low_rescaled,high_rescaled] = locate_bounds_2d(rescaled_array)


    translate_x = right_wanted-right_rescaled
    translate_y = high_wanted-high_rescaled
    
    translated = translate_2d(rescaled_array, array, (translate_y,translate_x))

    return translated

def scale_casting_with_rotation(array2d, reference, rotation):
    back_rotated = imutils.rotate(array2d.copy(),rotation)
    back_rotated_reference = imutils.rotate(reference.copy(),rotation)



    bounding_casted = bounds_scale_2d(back_rotated,locate_bounds_2d(back_rotated_reference), locate_bounds_2d(back_rotated))

    rotation_original = imutils.rotate(bounding_casted,-rotation)

    return rotation_original
    


    
def correct_rotation_z_axis(array, reference_array):

    rotated_array = np.squeeze(array.copy()[0], axis = 3)
    
    binary_array = np.squeeze(reference_array.copy()[0], axis = 3)
    binary_array[:] = 0
    binary_array[np.squeeze(array.copy()[0], axis = 3) > 0.1] = 1

    binary_reference = np.squeeze(reference_array.copy()[0], axis = 3)
    binary_reference[:] = 0
    binary_reference[np.squeeze(reference_array.copy()[0], axis = 3) > 0.1] = 1

    flat_z_axis_average_array = binary_array[0].copy()

    flat_z_axis_reference = binary_reference[0].copy()
    

    for image in binary_array:
        flat_z_axis_average_array[image > 0] += 1

    for image in binary_reference:
        flat_z_axis_reference[image > 0] += 1

    wanted_flat = flat_z_axis_reference
    
    wanted_flat_bounds = locate_bounds(np.stack([np.stack([np.stack([wanted_flat,wanted_flat])], axis = 3)]))

    

    non_zeros = []

    for i in range(0,360):
        print(i)
        eliminated = wanted_flat.copy()
        rotate_flat = imutils.rotate_bound(flat_z_axis_average_array.copy(),i)
        
        y_zoom = image_size/rotate_flat.shape[0]
        x_zoom = image_size/rotate_flat.shape[1]

        rotate_flat = skimage.transform.rescale(rotate_flat, (y_zoom, x_zoom))

 
        casted = scale_casting_with_rotation(rotate_flat, wanted_flat, i)
        
        eliminated = eliminated-casted
        rangE = np.max(eliminated)-np.min(eliminated)
        non_zeros.append(rangE)

    rotation_angle_z_axis = non_zeros.index(np.min(non_zeros))
    print(rotation_angle_z_axis)

    for n,image in enumerate(rotated_array):
        a2d_rotated_image = imutils.rotate_bound(image, rotation_angle_z_axis)
        
        y_zoom = image_size/a2d_rotated_image.shape[0]
        x_zoom = image_size/a2d_rotated_image.shape[1]

        a2d_rotated_image = skimage.transform.rescale(a2d_rotated_image, (y_zoom, x_zoom))

        rotated_array[n] = a2d_rotated_image
        


    rotated_bounds = locate_bounds(np.stack([np.stack([rotated_array], axis = 3)]))
    wanted_bounds = locate_bounds(reference_array)
    
    casted = cast_bounding(np.stack([np.stack([rotated_array], axis = 3)]), wanted_bounds, rotated_bounds)
    
    rescaled_bounds = locate_bounds(casted)
    
   

    [left_rotated,right_rotated,low_rotated,high_rotated,shallow_rotated,deep_rotated] = rotated_bounds
    [left_wanted,right_wanted,low_wanted,high_wanted,shallow_wanted,deep_wanted] = wanted_bounds
    [left_rescaled,right_rescaled,low_rescaled,high_rescaled,shallow_rescaled,deep_rescaled] = rescaled_bounds

    print("rotated bounds: " + str(rotated_bounds))
    print("wanted bounds: " + str(wanted_bounds))

    
    print("casted bounds: " + str(rescaled_bounds))


    translate_x = right_wanted-right_rescaled
    translate_y = high_wanted-high_rescaled
    translate_z = deep_wanted-deep_rescaled
 

    translated = translate(casted,reference_array, (translate_z,translate_y,translate_x))
    print("translated bounds: " + str(locate_bounds(translated)))
    return np.stack([np.stack([rotated_array], axis = 3)])

      
        
  

image_size = 128
index = 2
file = get_folder_path("/home/jiangl/Documents/python/ct to tumor identifier project/raw ct files/Brain-Tumor-Progression", index, "flair")

print("\n\nSet:", index, "\n\n")

brain = highlight_ct(file)
print('initialized')
brain.load_scan()
print('loaded scans')
pixel = brain.generate_pixel_data()
print('generated pixel array')
image_data = brain.resample_array()

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

print(np.max([x_size, y_size, z_size]))


rescale_factor = (image_size*0.8)/max_size


image_data = skimage.transform.rescale(np.squeeze(image_data.copy()[0], axis = 3), (rescale_factor, rescale_factor, rescale_factor))


for z,Slice in enumerate(image_data):
    for y,line in enumerate(Slice):
        for x,pixel in enumerate(line):
            original_array1[z][y][x] = pixel

original_array = np.stack([np.stack([original_array1], axis = 3)])

write_images(original_array, output_image_path_Seg)

blank = original_array.copy()

blank[:] = 1

adapted = binarize_max(original_array, 0.1)




blurred = blur(original_array, 3)

binary = binarize_blurred(original_array, 0.1, 2)


white = binarize(original_array, 0.25)

brain.resampled_array = original_array


brain_seg_array_precise = original_array.copy()
brain_seg_array_area = original_array.copy()

tumor_threshed_array = original_array.copy()
tumor_seg_array = original_array.copy()




print("median: ", find_median_grayscale(original_array))
print("max: ", np.max(original_array))


brain_seg = keras.models.load_model(brain_seg_model_area)

brain_mask1 = brain_seg.predict(original_array)

binary_brain_area = binarize(brain_mask1, 0.3)

brain_seg_area_final = original_array.copy()

brain_seg_area_final[binary_brain_area == 0] = 0



median = find_median_grayscale(brain_seg_area_final)




brain_seg_precise1 = keras.models.load_model(brain_seg_model_precise1)

brain_seg_precise2 = keras.models.load_model(brain_seg_model_precise2)

print("divide value:", median/0.05)

new_array = original_array/(median/0.05)

brain_mask2_precise1 = brain_seg_precise1.predict(new_array)

brain_mask2_precise2 = brain_seg_precise2.predict(new_array)



binary_brain_area2 = binarize(brain_mask2_precise1, 0.1)

binary_brain_area1 = binarize(brain_mask2_precise2, 0.8)


brain_seged = original_array.copy()


brain_seged[binary_brain_area2 == 0] = 0



precise_model = keras.models.load_model(precise_model_path)



precise_tumor_prediction = precise_model.predict(brain_seged)

precise_tumor_binary = binarize(precise_tumor_prediction, 0.2)



#binary_brain_area2 = combine_white_binary(precise_tumor_binary, binary_brain_area1)

brain_seg = keras.models.load_model(brain_seg_model_area)

new_array = original_array/(median/0.2)

brain_mask1 = brain_seg.predict(new_array)

binary_brain_area1 = binarize(brain_mask1, 0.3)

binary_brain_area1 = combine_zeros(binary_brain_area1, binary_brain_area2)


threshed = binarize(original_array, 0.05)

binary_brain_area1 = combine_zeros(binary_brain_area1, threshed)


#binary_brain_area1 = branch(binary_brain_area1, binary_brain_area, 2)

binary_brain_area1 = kill_small_islands(binary_brain_area1, 2)




circled = circle_highlighted(original_array, binary_brain_area2, 0.8)




write_images(circled, output_image_path)
write_images(brain_mask2, output_image_path_Seg)

brain.generate_stl(array_3d, stl_file_path, name, stl_resolution)


print ('Finished in', int((time.time() - start_time)/60), 'minutes and ', int((time.time() - start_time) % 60), 'seconds.')




