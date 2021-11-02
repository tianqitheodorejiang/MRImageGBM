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
import random
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import nilearn
import SimpleITK as sitk
import imregpoc
from math import pi

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
    if not os.path.exists(test_folder_path):
        os.makedirs(test_folder_path)
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

def preprocess_data(path):
    image_data, valid = load_dicom_folder(path, updating_labels = False)
    print(image_data.shape)
    print(valid)
    print(path)
    if "t2" in path.lower():
        image_data = np.rot90(image_data, axes = (2,1)).T
    
    image_data = image_data/np.max(image_data)



    blank_unscaled_array = image_data.copy()
    blank_unscaled_array[:] = 0

    z_zoom = image_size/image_data.shape[0]
    y_zoom = image_size/image_data.shape[1]
    x_zoom = image_size/image_data.shape[2]
    print(z_zoom, y_zoom, x_zoom)

    rescaled_blank = skimage.transform.rescale(blank_unscaled_array, (z_zoom, y_zoom, x_zoom))

    image_data = np.stack([np.stack([image_data], axis = 3)])
    bounds_finder = image_data.copy()
    bounds_finder = adaptive_threshold(bounds_finder, 101, 45, 1)
    bounds_finder = biggest_island(bounds_finder)

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

    return rescaled_array

def ConvNetsemantic(x,y,z):
    inputs = keras.layers.Input((x,y,z, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, 8, 0.1) #128 -> 64
    print(p1.shape)
    c2, p2 = down_block(p1, 16, 0.1) #64 -> 32
    c3, p3 = down_block(p2, 32, 0.2) #32 -> 16
    c4, p4 = down_block(p3, 64, 0.3) #16->8
    c5, p5 = down_block(p4, 128, 0.3) #16->8
    c6, p6 = down_block(p5, 256, 0.3) #16->8
    c7, p7 = down_block(p6, 512, 0.3) #16->8
    
    bn = bottleneck(p7, 1024, 0.4)
    print(bn.shape)
    
    u1 = up_block(bn, c7, 512, 0.3) #8 -> 16
    u2 = up_block(u1, c6, 256, 0.2) #16 -> 32
    u3 = up_block(u2, c5, 128, 0.1) #32 -> 64
    u4 = up_block(u3, c4, 64, 0.1) #64 -> 128
    u5 = up_block(u4, c3, 32, 0.1) #64 -> 128
    u6 = up_block(u5, c2, 16, 0.1) #64 -> 128
    u7 = up_block(u6, c1, 8, 0.1) #64 -> 128
    
    outputs = tf.keras.layers.Conv3D(4, (1, 1, 1),padding='same', activation="softmax")(u7)
    #outputs = keras.layers.Conv3D(1, (1, 1, 1), padding="same", activation="relu")(u4)
    print("out")
    print(outputs.shape)
    model = keras.models.Model(inputs, outputs)
    return model

def down_block(x, filters, dropout, kernel_size=(3, 3, 3), padding="same", strides=1):
    print(x.shape)
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="elu", input_shape = x.shape[1:], kernel_initializer='he_normal')(x)
    c = keras.layers.Dropout(dropout)(c)
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="elu", input_shape = c.shape[1:], kernel_initializer='he_normal')(c)
    p = keras.layers.MaxPool3D(pool_size = (2, 2, 2))(c)
    return c, p

def up_block(x, skip, filters, dropout,kernel_size=(3, 3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling3D((2, 2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="elu", input_shape = concat.shape[1:], kernel_initializer='he_normal')(concat)
    c = keras.layers.Dropout(dropout)(c)
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="elu", input_shape = c.shape[1:], kernel_initializer='he_normal')(c)
    return c

def bottleneck(x, filters, dropout, kernel_size=(3, 3, 3), padding="same", strides=1):
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="elu", input_shape = x.shape[1:], kernel_initializer='he_normal')(x)
    c = keras.layers.Dropout(dropout) (c)
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="elu", input_shape = c.shape[1:], kernel_initializer='he_normal')(c)
    return c

def command_iteration(method) :
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                   method.GetMetricValue(),
                                   method.GetOptimizerPosition()))

def register(fixed, moving):
    R = sitk.ImageRegistrationMethod()

    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins = 50)

    sample_per_axis=12
    if fixed.GetDimension() == 2:
        tx = sitk.Euler2DTransform()
        # Set the number of samples (radius) in each dimension, with a
        # default step size of 1.0
        R.SetOptimizerAsExhaustive([sample_per_axis//2,0,0])
        # Utilize the scale to set the step size for each dimension
        R.SetOptimizerScales([2.0*pi/sample_per_axis, 1.0,1.0])
    elif fixed.GetDimension() == 3:
        tx = sitk.Euler3DTransform()
        R.SetOptimizerAsExhaustive([sample_per_axis//2,sample_per_axis//2,sample_per_axis//4,0,0,0])
        R.SetOptimizerScales([2.0*pi/sample_per_axis,2.0*pi/sample_per_axis,2.0*pi/sample_per_axis,1.0,1.0,1.0])

    # Initialize the transform with a translation and the center of
    # rotation from the moments of intensity.
    tx = sitk.CenteredTransformInitializer(fixed, moving, tx)

    R.SetInitialTransform(tx)

    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )

    outTx = R.Execute(fixed, moving)
    print(outTx)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed);
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(1)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)

    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)

    return sitk.GetArrayFromImage(cimg)[:,:,:,0]

output_image_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/image ct  visualizations/Machine Learning 2 models test"

image_size = 128

brain_seg_model_top = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_brain_top.h5"
brain_seg_model_front = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_brain_front.h5"
brain_seg_model_side = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_brain_side.h5"
brain_seg_model_edges = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_brain_edges.h5"

tumor_seg_model = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_tumor.h5"

input_path = "C:/Users/JiangQin/Documents/data/raw ct files/QIN GBM Treatment Response"
output_path = "C:/Users/JiangQin/Documents/data/raw ct files/QIN GBM Treatment Response/loaded arrays 2"
sets = []

for set_ in os.listdir(input_path):
    print(set_)
    set_path = input_path + "/" + set_
    scans = []
    scan_dates = []
    for scan in os.listdir(set_path):
        flair = None
        t1 = None
        t2 = None
        
        scan_path = set_path + '/' + scan
        if os.path.isdir(scan_path):
            for mri in os.listdir(scan_path):
                if "t2" in mri.lower() and "space" in mri.lower() and os.path.isdir(scan_path + "/" + mri):
                    t2 = mri
                if "t1" in mri.lower() and "axial" in mri.lower() and "post" in mri.lower() and os.path.isdir(scan_path + "/" + mri):
                    t1 = mri
                if "flair" in mri.lower() and os.path.isdir(scan_path + "/" + mri):
                    flair = mri
            if flair is not None and t1 is not None and t2 is not None:
                scans.append([scan_path + "/" + flair,scan_path + "/" + t1,scan_path + "/" + t2])
                scan_month = int(scan[0:2])
                scan_day = int(scan[3:5])
                scan_year = int(scan[6:10])
                scan_dates.append(int((365*scan_year)+(30.5*scan_month)+(scan_day)))
                print(scan_month,scan_day,scan_year)
                print(scan)
    print(scans)
    if len(scans) == 2:
        for i,scan in enumerate(scans):
            newer_date = np.max(scan_dates)
            newer_mr_path = scans[scan_dates.index(newer_date)]
            del scans[scan_dates.index(newer_date)]
            print("should be 1:",len(scans))
            orig_mr_path = scans[0]
            sets.append([newer_mr_path, orig_mr_path])



print("sets:",sets)
print(len(sets))

semantic_model_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/code files/Saved models/Tumor seg semantic with t1ce t2 flair/Model 16.h5"

start = 0

start_time = time.time()

backwards_indexes = []

for i in range(0, int(len(sets)/2)):
    backwards = random.randint(0, len(sets))
    while backwards in backwards_indexes:
        backwards = random.randint(0, len(sets))
    backwards_indexes.append(backwards)

print(backwards_indexes)

for Set in range(start, len(sets)):
    print("\n\nSet " + str(Set) + "\n\n")
    new_mr_path_flair = sets[Set][0][0]
    flair = preprocess_data(new_mr_path_flair)
    print(flair.shape)
    flair_image = sitk.GetImageFromArray(flair)
    print(sitk.GetArrayFromImage(flair_image).shape)
    print("loaded flair")


    new_mr_path_t1 = sets[Set][0][1]
    t1 = preprocess_data(new_mr_path_t1)
    t1_image = sitk.GetImageFromArray(t1)    
    t1 = register(t1_image, flair_image)
    t1[flair == 0] = 0
    print(t1.shape)
    print("loaded t1")

    new_mr_path_t2 = sets[Set][0][2]
    t2 = preprocess_data(new_mr_path_t2)
    t2_image = sitk.GetImageFromArray(t2)    
    t2 = register(flair_image, t2_image)
    t2[flair == 0] = 0
    print(t2.shape)
    print("loaded t2")

    write_images((t2/np.max(t2)), output_image_path)
    cont = input()


    
    print("tranformed arrays")
    segmentations = []

    brain_seg_top = load_model(brain_seg_model_top)
    brain_mask_top = brain_seg_top.predict(np.stack([np.stack([flair], axis = -1)]))
    binary_brain_top = binarize(brain_mask_top, 0.5)
    binary_brain_top_top_ized = np.squeeze(binary_brain_top.copy()[0], axis = -1)
    segmentations.append(binary_brain_top_top_ized)

    print("segmented brain")
    binary_brain_wo_median_combined = combine_zeros(segmentations)


    median = find_median_grayscale(flair[binary_brain_wo_median_combined > 0])


    segmentations = []

    brain_seg_top = load_model(brain_seg_model_top)
    new_array_top = np.stack([np.stack([flair], axis = -1)])/(median/0.2)
    brain_mask_top = brain_seg_top.predict(new_array_top)
    binary_brain_top = binarize(brain_mask_top, 0.7)
    binary_brain_top_top_ized = np.squeeze(binary_brain_top.copy()[0], axis = -1)
    segmentations.append(binary_brain_top_top_ized)

    binary_brain_final_combined = combine_zeros(segmentations)


    only_brain = np.stack([flair,t1,t2],axis = -1)
    only_brain[np.stack([binary_brain_final_combined,binary_brain_final_combined,binary_brain_final_combined],axis = -1) == 0] = 0

    
    tumor_seg_channeled = ConvNetsemantic(128,128,128)

    tumor_seg_channeled.load_weights(semantic_model_path)

    tumor_mask = tumor_seg_channeled.predict(np.stack([only_brain]))

    tumor_colored = np.argmax(tumor_mask[0], axis = -1)

    image = np.stack([only_brain[:,:,:,0],only_brain[:,:,:,1],only_brain[:,:,:,2],only_brain[:,:,:,0],only_brain[:,:,:,0],only_brain[:,:,:,0]],axis = -1)
    image[:,:,:,3:6] = 0
    image[:,:,:,3][tumor_colored == 1] = 1
    image[:,:,:,4][tumor_colored == 2] = 1
    image[:,:,:,5][tumor_colored == 3] = 1

    new_image = image.copy()

    write_images(image[:,:,:,0], output_image_path+"/treatment eval testing/"+str(Set)+"/image1")
    write_images(image[:,:,:,1], output_image_path+"/treatment eval testing/"+str(Set)+"/image2")
    write_images(image[:,:,:,2], output_image_path+"/treatment eval testing/"+str(Set)+"/image3")
    write_images(image[:,:,:,3], output_image_path+"/treatment eval testing/"+str(Set)+"/image4")
    write_images(image[:,:,:,4], output_image_path+"/treatment eval testing/"+str(Set)+"/image5")
    write_images(image[:,:,:,5], output_image_path+"/treatment eval testing/"+str(Set)+"/image6")

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


    binary_brain_final_combined = combine_zeros(segmentations)


    only_brain = original_array_top.copy()
    only_brain[np.stack([np.stack([binary_brain_final_combined], axis = 3)]) == 0] = 0

    #write_images(np.squeeze(only_brain[0], axis = 3), "C:/ProgramData/MRImage3D/metadata/temp_processed_storage/images/"+str(Set)+str(i)+"/only brain old")


    #tumor_seg_top = ConvNetTumor(128,128,128)
    #tumor_seg_top.load_weights(tumor_seg_model)
    tumor_seg_top = load_model(tumor_seg_model)
    new_array = only_brain/(median/0.3)
    tumor_mask = tumor_seg_top.predict(new_array)
    binary_tumor = np.squeeze(binarize(tumor_mask, 0.5)[0], axis = 3)

    #write_images(binary_tumor, "C:/ProgramData/MRImage3D/metadata/temp_processed_storage/images/"+str(Set)+str(i)+"/tumor mask old")

    #write_images(original_unscaled_array, "C:/ProgramData/MRImage3D/metadata/temp_processed_storage/images/"+str(Set)+str(i)+"/original array 2")

    image = np.squeeze(only_brain[0], axis = 3)
    image = np.stack([image,image],axis = -1)
    image[:,:,:,1] = 0
    image[:,:,:,1][binary_tumor == 1] = 1
    old_image = image.copy()


    if Set in backwards_indexes:
        complete_image = np.stack([new_image[:,:,:,0],old_image[:,:,:,0]],axis = -1)
        save_array(complete_image, output_path + "/" + str(Set) + "/image.h5")
        save_array([1], output_path + "/" + str(Set) + "/mask.h5")
        #open("C:/ProgramData/MRImage3D/metadata/temp_processed_storage/images/"+str(Set)+str(i)+"/mask.txt","w+").write(str(1))
        
    else:
        complete_image = np.stack([old_image[:,:,:,0],new_image[:,:,:,0]],axis = -1)
        save_array(complete_image, output_path + "/" + str(Set) + "/image.h5")
        save_array([0], output_path + "/" + str(Set) + "/mask.h5")
        #open("C:/ProgramData/MRImage3D/metadata/temp_processed_storage/images/"+str(Set)+str(i)+"/mask.txt","w+").write(str(0))
               
    print("\n\nComplete image shape:",complete_image.shape,"\n\n")

    print("image shape:", image.shape)
    
    
    

    print("saved image")
    print ('Finished one set in', int((time.time() - start_time)/60), 'minutes and ', int((time.time() - start_time) % 60), 'seconds.')
