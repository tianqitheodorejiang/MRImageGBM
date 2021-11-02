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
from mri_modules.load_in_arrays import *
import time


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



image_size = 128

brain_seg_model_top = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_brain_top.h5"
brain_seg_model_front = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_brain_front.h5"
brain_seg_model_side = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_brain_side.h5"
brain_seg_model_edges = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_brain_edges.h5"

tumor_seg_model = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/Model 34.h5"

input_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/raw ct files/Brain-Tumor-Progression"
output_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/raw ct files/Brain-Tumor-Progression/loaded arrays"
sets = []

for set_ in os.listdir(input_path):
    print(set_)
    set_path = input_path + "/" + set_
    scans = []
    scan_dates = []
    for scan in os.listdir(set_path):
        scan_path = set_path + '/' + scan
        if os.path.isdir(scan_path):
            for mri in os.listdir(scan_path):
                if "flair" in mri.lower() and os.path.isdir(scan_path + "/" + mri):
                    scans.append(scan_path + "/" + mri)
                    scan_month = int(scan[0:2])
                    scan_day = int(scan[3:5])
                    scan_year = int(scan[6:10])
                    scan_dates.append(int((365*scan_year)+(30.5*scan_month)+(scan_day)))
                    print(scan_month,scan_day,scan_year)
                    print(scan)
                    break
    if len(scans) > 0:
        for i,scan in enumerate(scans):
            newer_date = np.max(scan_dates[i])
            newer_mr_path = scans[scan_dates.index(newer_date)]
            del scans[scan_dates.index(newer_date)]
            print("should be 1:",len(scans))
            orig_mr_path = scans[0]
            sets.append([newer_mr_path, orig_mr_path])
        
print(sets)
print(len(sets))

images = []
masks = []

start = 0

start_time = time.time()

for Set in range(start, len(sets)):
    try:
        print("\n\nSet " + str(Set) + "\n\n")
        new_mr_path = sets[Set][0]
        image_data, valid = load_dicom_folder(new_mr_path)
        
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

        print("tranformed arrays")
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

        print("segmented brain")
        binary_brain_wo_median_combined = combine_zeros(segmentations)


        median = find_median_grayscale(np.squeeze(original_array_top[0], axis = 3)[binary_brain_wo_median_combined > 0])


        segmentations = []

        brain_seg_top = load_model(brain_seg_model_top)
        new_array_top = original_array_top/(median/0.2)
        brain_mask_top = brain_seg_top.predict(new_array_top)
        binary_brain_top = binarize(brain_mask_top, 0.7)
        binary_brain_top_top_ized = np.squeeze(binary_brain_top.copy()[0], axis = 3)
        segmentations.append(binary_brain_top_top_ized)

        binary_brain_final_combined = combine_zeros(segmentations)
          
        brain_seg_edges = load_model(brain_seg_model_edges)
        new_array_edges = original_array_top/(median/0.2)
        brain_mask_edges = brain_seg_edges.predict(new_array_edges)
        binary_brain_edges_top_ized = np.argmax(brain_mask_edges.copy()[0], axis = 3)
        segmentations.append(binary_brain_edges_top_ized)

        print("refined brain segmentation")

        binary_brain_final_edges = combine_zeros(segmentations)



        only_brain = original_array_top.copy()
        only_brain[np.stack([np.stack([binary_brain_final_combined], axis = 3)]) == 0] = 0

        tumor_seg_top = load_model(tumor_seg_model, compile = False)
        new_array = only_brain/(median/0.2)
        tumor_mask = tumor_seg_top.predict(new_array)
        binary_tumor = np.squeeze(binarize(tumor_mask, 0.5)[0], axis = 3)


        image = np.squeeze(only_brain[0], axis = 3)
        image = np.stack([image,image],axis = -1)
        image[:,:,:,1] = 0
        image[:,:,:,1][binary_tumor == 1] = 1

        mask = binary_tumor.copy()
        print("mask shape:", mask.shape)

        save_array(image, output_path + "/" + str(Set) + "/mask.h5")

        print("saved mask\n")

        #------------------------------------------------------------------------------------------
        old_mr_path = sets[Set][1]
        image_data, valid = load_dicom_folder(old_mr_path)
        
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

        print("transformed array")
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

        print("segmented brain")
        binary_brain_wo_median_combined = combine_zeros(segmentations)


        median = find_median_grayscale(np.squeeze(original_array_top[0], axis = 3)[binary_brain_wo_median_combined > 0])


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

        
        brain_seg_edges = load_model(brain_seg_model_edges)
        new_array_edges = original_array_top/(median/0.2)
        brain_mask_edges = brain_seg_edges.predict(new_array_edges)
        binary_brain_edges_top_ized = np.argmax(brain_mask_edges.copy()[0], axis = 3)
        segmentations.append(binary_brain_edges_top_ized)
          

        binary_brain_final_edges = combine_zeros(segmentations)
        print("refined brain segmentation")


        only_brain = original_array_top.copy()
        only_brain[np.stack([np.stack([binary_brain_final_combined], axis = 3)]) == 0] = 0

        tumor_seg_top = load_model(tumor_seg_model, compile = False)
        new_array = only_brain/(median/0.2)
        tumor_mask = tumor_seg_top.predict(new_array)
        binary_tumor = np.squeeze(binarize(tumor_mask, 0.5)[0], axis = 3)

        image = np.squeeze(only_brain[0], axis = 3)
        image = np.stack([image,image],axis = -1)
        image[:,:,:,1] = 0
        image[:,:,:,1][binary_tumor == 1] = 1

        print("image shape:", image.shape)
        
        save_array(image, output_path + "/" + str(Set) + "/image.h5")

        print("saved image")
        print ('Finished one set in', int((time.time() - start_time)/60), 'minutes and ', int((time.time() - start_time) % 60), 'seconds.')
    except Exception as e:
        print(e)
        print("\n\nFAILED\n\n")
