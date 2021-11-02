from mri_modules.utils import *
import os
import numpy as np
import cv2
import shutil
from skimage.measure import marching_cubes_lewiner as marching_cubes
import stl
from stl import mesh
import tensorflow as tf
from tensorflow.keras.models import load_model
import skimage.transform
import nibabel as nib
import h5py
import scipy
import pydicom as dicom
import tensorflow.keras as keras

def binarize(array, min_):
    binary = array.copy()
    binary[array < min_] = 0
    binary[array >= min_] = 1

    return binary

    
    
def dilate_up(array, size, stacked = True):
    if stacked:
        binary = np.squeeze(array.copy()[0], axis = 3)
    else:
        binary = array.copy()
    binary[binary > 0] = 1

    kernel = scipy.ndimage.generate_binary_structure(3, 1)
    blew_up = scipy.ndimage.binary_dilation(binary.astype('uint8'), kernel, iterations=size)
    
    if stacked:
        return np.stack([np.stack([blew_up], axis = 3)])
    else:
        return blew_up

def translate_3d(array, translation):
    original_array = array.copy()

    array_translated = array.copy()
    array_translated[:] = 0
    for z,Slice in enumerate(original_array):
        for y,line in enumerate(Slice):
            for x,pixel in enumerate(line):
                try:
                    array_translated[z+translation[0]][y+translation[1]][x+translation[2]] = pixel
                except:
                    pass

    return array_translated


def touching_island(reference, array, stacked = True):
    if stacked:
        array = np.squeeze(array.copy()[0], axis = 3)
        array[array > 0] = 1
        reference = np.squeeze(reference.copy()[0], axis = 3)
        reference[reference > 0] = 1
    else:
        array[array > 0] = 1
        reference[reference > 0] = 1
    
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

    markers, num_features = scipy.ndimage.measurements.label(array, touching_structure_3d)
    reference_idx = np.unique(markers[reference == 1])
    for idx in reference_idx:
        masked[markers == idx] = 1

    masked[array == 0] = 0

    if stacked:
        return np.stack([np.stack([masked], axis = 3)])
    else:
        return masked
 


def biggest_island(input_array, stacked = True):
    if stacked:
        masked = np.squeeze(input_array.copy()[0], axis = 3)
        binary = np.squeeze(input_array.copy()[0], axis = 3)
        binary[:] = 0
        binary[np.squeeze(input_array[0], axis = 3) > 0] = 1

    else:
        masked = input_array.copy()
        binary = input_array.copy()
        binary[:] = 0
        binary[input_array > 0] = 1
   
    touching_structure_3d =[[[0,0,0],
                             [0,1,0],
                             [0,0,0]],

                            [[0,1,0],
                             [1,1,1],
                             [0,1,0]],

                            [[0,0,0],
                             [0,1,0],
                             [0,0,0]]]


    markers,_ = scipy.ndimage.measurements.label(binary,touching_structure_3d)
    markers[binary == 0] = 0
    counts = np.bincount(markers.ravel())
    counts[0] = 0
    noise_idx = np.where(counts != np.max(counts))
    noise = np.isin(markers, noise_idx)
    binary[noise] = 0

    masked[binary == 0] = 0
    if stacked:
        return np.stack([np.stack([masked], axis = 3)])
    else:
        return masked


def combine_zeros(arrays):
    combined = arrays[0].copy()
    for array in arrays:
        combined[array < 0.1] = 0
    return combined



def adaptive_threshold(array, course, precise, blur_precision = 0, stacked = True):
    if stacked:
        thresholded_array = np.squeeze(array.copy()[0], axis = 3)
        thresholded_array = thresholded_array*255
        thresholded_array[thresholded_array > 255] = 255
    else:
        thresholded_array = array.copy()
        thresholded_array = thresholded_array*255
        thresholded_array[thresholded_array > 255] = 255

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

    if stacked:
        return np.stack([np.stack([thresholded_array/255], axis = 3)])
    else:
        return thresholded_array/255



       
def generate_stl(array_3d, stl_file_path, stl_resolution):
    array = array_3d.copy()
    verts, faces, norm, val = marching_cubes(array, 0.01, step_size = stl_resolution, allow_degenerate=True)
    mesh = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh.vectors[i][j] = verts[f[j],:]
    
    if not stl_file_path.endswith(".stl"):
        stl_file_path += ".stl"
    if not os.path.exists(os.path.dirname(stl_file_path)):
        os.makedirs(os.path.dirname(stl_file_path))
    mesh.save(stl_file_path)


def find_median_grayscale(array):    
    zero_pixels = float(np.count_nonzero(array==0))
    single_dimensional = array.flatten().tolist()
    single_dimensional.extend(np.full((1, int(zero_pixels)), 1000).flatten().tolist())

    return np.median(single_dimensional)

def get_folder_path(path, index, img_id):
    pathes = []
    for path, dirs, files in os.walk(path, topdown=False):
        for dir_ in dirs:
            if img_id in dir_.lower():
                pathes.append(os.path.join(path, dir_))
    
    return pathes[index]
    
class highlight_ct:

    def __init__(self, input_path):  
        self.input_path = input_path
        self.file_names = os.listdir(input_path)

    def load_scan(self):
        ##loads and sorts the data in as a dcm type array
        raw_data = [dicom.read_file(self.input_path + '/' + s) for s in os.listdir(self.input_path) if s.endswith(".dcm")]
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
        transposed = np.squeeze(array_3d.copy()[0], axis = 3)

        ##uses the marching cubes algorithm to make a list of vertices, faces, normals, and values
        verts, faces, norm, val = marching_cubes(transposed, 0.01, step_size = stl_resolution, allow_degenerate=True)

        mesh = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
        print('Vertices obatined:', len(verts))
        print('')
        for i, f in enumerate(faces):
            for j in range(3):
                mesh.vectors[i][j] = verts[f[j],:]
        path = stl_file_path + '/' + name
        if not path.endswith(".stl"):
            path += ".stl"
        if not os.path.exists(stl_file_path):
            os.makedirs(stl_file_path)
        mesh.save(path)

    def threshold_scans(self, input_array, lower_thresh, upper_thresh, blur_precision):
        input_array = np.squeeze(input_array.copy()[0], axis = 3)
        ##updates the object with the chosen lower and upper threshold
        self.lower_thresh = lower_thresh
        self.upper_thresh = upper_thresh
        
        ##blurs the scan to do very simple denoising 
        blurred_scans = scipy.ndimage.gaussian_filter(input_array, blur_precision)

        masked_array = np.squeeze(self.resampled_array.copy()[0], axis = 3)

        ##creates a mask that is the same shape as the original array and sets it to 255
        mask = masked_array.copy()
        mask[:] = 255

        ##sets the areas of the mask where the blurred image is not within the threshold to 0
        mask[blurred_scans > upper_thresh] = 0
        mask[blurred_scans < lower_thresh] = 0

        ##sets the masked off areas in the masked image output to 0
        masked_array[mask == 0] = 0

        ##finds the contours and draws them in the image with circled areas
        self.thresholded_array = masked_array ##update the output
        self.blurred_array = blurred_scans
        return np.stack([np.stack([self.thresholded_array], axis = 3)])

            
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



def locate_bounds(array, stacked = True):
    if stacked:
        left = np.squeeze(array.copy()[0], axis = 3).shape[2]
        right = 0 
        low = np.squeeze(array.copy()[0], axis = 3).shape[1]
        high = 0
        shallow = np.squeeze(array.copy()[0], axis = 3).shape[0]
        deep = 0
        array_3d = np.squeeze(array.copy()[0], axis = 3)
    else:
        left = array.copy().shape[2]
        right = 0 
        low = array.copy().shape[1]
        high = 0
        shallow = array.copy().shape[0]
        deep = 0
        array_3d = array.copy()
    

    
    for z,Slice in enumerate(array_3d):
        for y,line in enumerate(Slice):
            for x,pixel in enumerate(line):
                if pixel > 0:
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
                        
    return [left,right,low,high,shallow,deep]

def pad(array):
    padded = []
    for image in array:
        padded.append(image)
    padded.append(np.zeros((array.shape[1],array.shape[2])))
    padded.append(np.zeros((array.shape[1],array.shape[2])))
    final = translate_3d(np.stack(padded), [1,1,1])
    return final
    

def write_images(array, test_folder_path):
    for n,image in enumerate(array):
        file_name = str(str(n) +'.png')
        cv2.imwrite(os.path.join(test_folder_path, file_name), image*255)



def circle_highlighted(reference, binary, color):
    circled = reference.copy()
    binary = binary.copy()
    binary[binary > 0] = 1
    
    
    for n, image in enumerate(binary):
        contours, _ = cv2.findContours(image.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(circled[n], contours, -1,color, 1)

    return circled

def down_block(x, filters, dropout, kernel_size=(3, 3, 3), padding="same", strides=1):
    print(x.shape)
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu", input_shape = x.shape[1:], kernel_initializer='he_normal')(x)
    c = keras.layers.Dropout(dropout)(c)
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu", input_shape = c.shape[1:], kernel_initializer='he_normal')(c)
    p = keras.layers.MaxPool3D(pool_size = (2, 2, 2))(c)
    return c, p

def up_block(x, skip, filters, dropout,kernel_size=(3, 3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling3D((2, 2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu", input_shape = concat.shape[1:], kernel_initializer='he_normal')(concat)
    c = keras.layers.Dropout(dropout)(c)
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu", input_shape = c.shape[1:], kernel_initializer='he_normal')(c)
    return c

def bottleneck(x, filters, dropout, kernel_size=(3, 3, 3), padding="same", strides=1):
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu", input_shape = x.shape[1:], kernel_initializer='he_normal')(x)
    c = keras.layers.Dropout(dropout) (c)
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu", input_shape = c.shape[1:], kernel_initializer='he_normal')(c)
    return c

def ConvNetTumor(x,y,z):
    inputs = keras.layers.Input((x,y,z, 1))
    
    p0 = inputs
    c1, p1 = down_block(p0, 16, 0.1) #128 -> 64
    print(p1.shape)
    c2, p2 = down_block(p1, 32, 0.1) #64 -> 32
    c3, p3 = down_block(p2, 64, 0.2) #32 -> 16
    c4, p4 = down_block(p3, 128, 0.3) #16->8
    
    bn = bottleneck(p4, 256, 0.4)
    print(bn.shape)
    
    u1 = up_block(bn, c4, 128, 0.3) #8 -> 16
    u2 = up_block(u1, c3, 64, 0.2) #16 -> 32
    u3 = up_block(u2, c2, 32, 0.1) #32 -> 64
    u4 = up_block(u3, c1, 16, 0.1) #64 -> 128
    
    outputs = tf.keras.layers.Conv3D(1, (1, 1, 1),padding='same', activation="sigmoid")(u4)
    #outputs = keras.layers.Conv3D(1, (1, 1, 1), padding="same", activation="relu")(u4)
    print("out")
    print(outputs.shape)
    model = keras.models.Model(inputs, outputs)
    return model

image_size = 128

brain_seg_model_top = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_brain_top.h5"
brain_seg_model_front = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_brain_front.h5"
brain_seg_model_side = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_brain_side.h5"
    
tumor_seg_model1 = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_tumor.h5"
tumor_seg_model2 = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/Model 34.h5"


output_image_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/image ct  visualizations/Machine Learning 2 models test/tumor seg comparison"

for n in range(0,40):
    index = n
    file = get_folder_path("C:/Users/JiangQin/Documents/python/ct to tumor identifier project/raw ct files/Brain-Tumor-Progression", index, "flair")

    print("\n\nSet:", index, "\n\n")

    brain = highlight_ct(file)
    print('initialized')
    brain.load_scan()
    print('loaded scans')
    pixel = brain.generate_pixel_data()
    print('generated pixel array')
    image_data = brain.resample_array()

    image_data = image_data/np.max(image_data)


    blank_unscaled_array = image_data.copy()
    blank_unscaled_array[:] = 0

    z_zoom = image_size/image_data.shape[0]
    y_zoom = image_size/image_data.shape[1]
    x_zoom = image_size/image_data.shape[2]


    rescaled_blank = skimage.transform.rescale(blank_unscaled_array, (z_zoom, y_zoom, x_zoom))

    update_label("Tranforming data...")
    image_data = np.stack([np.stack([image_data], axis = 3)])
    bounds_finder = image_data.copy()
    bounds_finder = adaptive_threshold(bounds_finder, 101, 45, 1)
    bounds_finder = biggest_island(bounds_finder)
    image_data = biggest_island(image_data)

    bounds = locate_bounds(bounds_finder)
    [left,right,low,high,shallow,deep] = bounds
    x_size = abs(left-right)
    y_size = abs(low-high)
    z_size = abs(shallow-deep)

    max_size = np.max([x_size, y_size, z_size])

    rescale_factor = (image_size*0.8)/max_size
    backscale_factor = 1/rescale_factor

    image_data = skimage.transform.rescale(np.squeeze(image_data.copy()[0], axis = 3), (rescale_factor, rescale_factor, rescale_factor))

    bounds_finder = image_data.copy()
    bounds_finder = adaptive_threshold(bounds_finder, 101, 45, 1, stacked = False)
    bounds_finder = biggest_island(bounds_finder, stacked = False)
    image_data = biggest_island(image_data, stacked = False)

    bounds = locate_bounds(np.stack([np.stack([bounds_finder], axis = 3)]))
    [left,right,low,high,shallow,deep] = bounds


    image_data = translate_3d(image_data, [-shallow,-low,-left])
    original_unscaled_array = skimage.transform.rescale(image_data, (backscale_factor, backscale_factor, backscale_factor))

    rescaled_array = rescaled_blank.copy()
    for z,Slice in enumerate(image_data):
        for y,line in enumerate(Slice):
            for x,pixel in enumerate(line):
                try:
                    rescaled_array[z][y][x] = pixel
                except:
                    pass

    original_array_top = np.stack([np.stack([rescaled_array], axis = 3)])
    original_array_front = np.stack([np.stack([np.rot90(rescaled_array, axes = (1,0))], axis = 3)])
    original_array_side = np.stack([np.stack([np.rot90(rescaled_array.T, axes = (1,2))], axis = 3)])

    update_label("Performing brain segmentation for median calculation...")

    segmentations = []
    brain_seg_top = load_model(brain_seg_model_top)
    brain_mask_top = brain_seg_top.predict(original_array_top)
    binary_brain_top = binarize(brain_mask_top, 0.5)
    binary_brain_top_top_ized = np.squeeze(binary_brain_top.copy()[0], axis = 3)
    segmentations.append(binary_brain_top_top_ized)


    brain_seg_front = load_model(brain_seg_model_front)
    brain_mask_front = brain_seg_front.predict(original_array_front)
    binary_brain_front = binarize(brain_mask_front, 0.5)
    binary_brain_front_top_ized = np.rot90(np.squeeze(binary_brain_front.copy()[0], axis = 3), axes = (0,1))
    segmentations.append(binary_brain_front_top_ized)


    brain_seg_side = load_model(brain_seg_model_side)
    brain_mask_side = brain_seg_side.predict(original_array_side)
    binary_brain_side = binarize(brain_mask_side, 0.5)
    binary_brain_side_top_ized = np.rot90(np.squeeze(binary_brain_side.copy()[0], axis = 3), axes = (2,1)).T
    segmentations.append(binary_brain_side_top_ized)


    binary_brain_wo_median_combined = combine_zeros(segmentations)


    median = find_median_grayscale(np.squeeze(original_array_top[0], axis = 3)[binary_brain_wo_median_combined > 0])


    update_label("Performing brain segmentation with median...")

    segmentations = []

    brain_seg_top = load_model(brain_seg_model_top)
    new_array_top = original_array_top/(median/0.2)
    brain_mask_top = brain_seg_top.predict(new_array_top)
    binary_brain_top = binarize(brain_mask_top, 0.7)
    binary_brain_top_top_ized = np.squeeze(binary_brain_top.copy()[0], axis = 3)
    segmentations.append(binary_brain_top_top_ized)


    brain_seg_front = load_model(brain_seg_model_front)
    new_array_front = original_array_front/(median/0.2)
    brain_mask_front = brain_seg_front.predict(new_array_front)
    binary_brain_front = binarize(brain_mask_front, 0.5)
    binary_brain_front_top_ized = np.rot90(np.squeeze(binary_brain_front.copy()[0], axis = 3), axes = (0,1))
    segmentations.append(binary_brain_front_top_ized)

    brain_seg_side = load_model(brain_seg_model_side)
    new_array_side = original_array_side/(median/0.2)
    brain_mask_side = brain_seg_side.predict(new_array_side)
    binary_brain_side = binarize(brain_mask_side, 0.5)
    binary_brain_side_top_ized = np.rot90(np.squeeze(binary_brain_side.copy()[0], axis = 3), axes = (2,1)).T
    segmentations.append(binary_brain_side_top_ized)


    binary_brain_final_combined = combine_zeros(segmentations)


    update_label("Performing tumor segmentation...")

    only_brain = original_array_top.copy()
    only_brain[np.stack([np.stack([binary_brain_final_combined], axis = 3)]) == 0] = 0

    tumor_seg_top = load_model(tumor_seg_model1)
    new_array = only_brain/(median/0.2)
    tumor_mask = tumor_seg_top.predict(new_array)
    binary_tumor1 = np.squeeze(binarize(tumor_mask, 0.9)[0], axis = 3)

    tumor_seg_top = ConvNetTumor(128,128,128)
    tumor_seg_top.load_weights(tumor_seg_model2)
    new_array = only_brain/(median/0.2)
    tumor_mask = tumor_seg_top.predict(new_array)
    binary_tumor2 = np.squeeze(binarize(tumor_mask, 0.5)[0], axis = 3)

    update_label("Upsampling segmentations...")

    brain_mask_upscaled = skimage.transform.rescale(binary_brain_final_combined, (backscale_factor, backscale_factor, backscale_factor))
    brain_mask_rescaled = blank_unscaled_array.copy()
    for z,Slice in enumerate(brain_mask_upscaled):
        for y,line in enumerate(Slice):
            for x,pixel in enumerate(line):
                try:
                    brain_mask_rescaled[z][y][x] = pixel
                except:
                    pass

    tumor_mask_upscaled = skimage.transform.rescale(binary_tumor1, (backscale_factor, backscale_factor, backscale_factor))
    tumor_mask_rescaled1 = blank_unscaled_array.copy()
    for z,Slice in enumerate(tumor_mask_upscaled):
        for y,line in enumerate(Slice):
            for x,pixel in enumerate(line):
                try:
                    tumor_mask_rescaled1[z][y][x] = pixel
                except:
                    pass


    tumor_mask_upscaled = skimage.transform.rescale(binary_tumor2, (backscale_factor, backscale_factor, backscale_factor))
    tumor_mask_rescaled2 = blank_unscaled_array.copy()
    for z,Slice in enumerate(tumor_mask_upscaled):
        for y,line in enumerate(Slice):
            for x,pixel in enumerate(line):
                try:
                    tumor_mask_rescaled2[z][y][x] = pixel
                except:
                    pass


    circled1 = circle_highlighted(original_unscaled_array/(median/0.35), tumor_mask_rescaled1, 1)
    circled2 = circle_highlighted(original_unscaled_array/(median/0.35), tumor_mask_rescaled2, 1)
    if not os.path.exists((output_image_path + "/" + str(n))):
        os.makedirs((output_image_path + "/" + str(n)))
    else:
        shutil.rmtree((output_image_path + "/" + str(n)))
        os.makedirs((output_image_path + "/" + str(n)))
        
    write_images(circled1, (output_image_path + "/" + str(n) + " old"))
    write_images(circled2, (output_image_path + "/" + str(n) + " new"))
 
