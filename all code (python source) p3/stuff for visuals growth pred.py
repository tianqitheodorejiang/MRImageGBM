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
import sys
from skimage.morphology import convex_hull_image
import csv
import linecache

start_time = time.time()


def printe():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print ('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

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

def preprocess_data(path,rescale_factor = None,translation = None):
    print(path)
    image_data, valid = load_dicom_folder(path, updating_labels = False)
    print(image_data.shape)
    print(valid)
    print(np.max(image_data))
    print("oofoofoofoofofofofofof")
    
    '''if "t2" in path.lower():
        image_data = np.rot90(image_data, axes = (2,1)).T'''
    
    image_data = image_data/np.max(image_data)



    blank_unscaled_array = image_data.copy()
    blank_unscaled_array[:] = 0

    z_zoom = image_size/image_data.shape[0]
    y_zoom = image_size/image_data.shape[1]
    x_zoom = image_size/image_data.shape[2]
    print(z_zoom, y_zoom, x_zoom)

    rescaled_blank = skimage.transform.rescale(blank_unscaled_array, (z_zoom, y_zoom, x_zoom))

    image_data = np.stack([np.stack([image_data], axis = 3)])

    if rescale_factor is None:
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
    if translation is None:
        bounds_finder = image_data.copy()
        bounds_finder = adaptive_threshold(bounds_finder, 101, 45, 1, stacked = False)
        bounds_finder = biggest_island(bounds_finder, stacked = False)

        bounds = locate_bounds(np.stack([np.stack([bounds_finder], axis = 3)]))
    else:
        bounds=translation
    print("\n\nbounds:",bounds,"\n\n")
    [left,right,low,high,shallow,deep] = bounds


    image_data = translate_3d(image_data, [-shallow,-low,-left])

    rescaled_array = rescaled_blank.copy()
    for z,Slice in enumerate(image_data):
        for y,line in enumerate(Slice):
            for x,pixel in enumerate(line):
                try:
                    rescaled_array[z][y][x] = pixel
                except:
                    pass

    return rescaled_array, rescale_factor,bounds

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


def register(fixed_image, moving_image, orig, transform = None):
    if transform is None:
        resamples = []
        metrics = []
        transforms = []
        for i in range (1,10):
            ImageSamplingPercentage = 1
            initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.ScaleVersor3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=200)
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(float(ImageSamplingPercentage)/100)
            registration_method.SetInterpolator(sitk.sitkLinear)
            registration_method.SetOptimizerAsGradientDescent(learningRate=0.001, numberOfIterations=10**5, convergenceMinimumValue=1e-6, convergenceWindowSize=100) #Once
            registration_method.SetOptimizerScalesFromPhysicalShift() 
            registration_method.SetInitialTransform(initial_transform)
            #registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
            #registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
            #registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

            transform = registration_method.Execute(fixed_image, moving_image)
            #print(transform)
            print("number:",i)
            print(registration_method.GetMetricValue())
            metrics.append(registration_method.GetMetricValue())
            resamples.append(sitk.Resample(orig, fixed_image, transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID()))
            transforms.append(transform)
        print(np.min(metrics))
        return sitk.GetArrayFromImage(resamples[metrics.index(np.min(metrics))]),transforms[metrics.index(np.min(metrics))]
    else:
        return sitk.GetArrayFromImage(sitk.Resample(orig, fixed_image, transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())),transform



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

def convex_border(array, thickness):
    contour_only = array.copy()
    binary = array.copy()

    contour_only[:] = 0
    
    binary[:] = 0
    binary[array > 0] = 255

    
    cont = []
    for n, image in enumerate(binary):
        contours, _ = cv2.findContours(image.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            hull = cv2.convexHull(contour)
            cv2.drawContours(contour_only[n], [hull], -1, 200, thickness)
        

    return contour_only


def convex_hull(array):
    contour_only = array.copy()
    binary = array.copy()

    hull = []
    
    binary[:] = 0
    binary[array > 0.05] = 255

    
    cont = []
    for n, image in enumerate(binary):
        convex = np.array(convex_hull_image(image.astype('uint8')),dtype="float64")
        hull.append(convex)
            
    return np.stack(hull)

def fill_holes_binary(array, sense):
    binary = array.copy()

    binary_original = array.copy()

    binary_original[:] = 0
    binary_original[array > 0] = 1
    
    binary[:] = 0
    binary[array == 0] = 1




    touching_structure_2d = [[0,1,0],
                             [1,1,1],
                             [0,1,0]]


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

    binary_original[denoised == 1] = 1

        
    return binary_original

def convex_shape(input_array):
    #binary = adaptive_threshold(input_array, 101, 45, 1, stacked = False)
    #binary[input_array < 0.1] = 0
    binary = np.array(input_array > 0.1,dtype = "float64")
    binary = biggest_island(binary, stacked = False)
    binary = convex_hull(binary)
    binary = biggest_island(binary, stacked = False)

    return binary

def trim(flair,t1,t2):
    flair_cut = flair.copy()
    t1_cut = t1.copy()
    t1_cut[:] = 0
    t2_cut = t2.copy()
    t2_cut[:] = 0
    for n,image in enumerate(flair):
        if np.max(flair) > 0 and np.max(t1) > 0 and np.max(t2) > 0:
            flair_cut[n] = flair[n]
            t1_cut[n] = t1[n]
            t2_cut[n] = t2[n]

    return flair_cut, t1_cut, t2_cut


def normalize(scans):
    outputs = []
    flair = scans[0]
    flair = flair/np.max(flair)
    blank_unscaled_array = flair.copy()

    blank_unscaled_array[:] = 0




    z_zoom = image_size/flair.shape[0]
    y_zoom = image_size/flair.shape[1]
    x_zoom = image_size/flair.shape[2]

    image_data1 = skimage.transform.rescale(flair, (z_zoom, y_zoom, x_zoom))

    original_array1 = image_data1.copy()
    original_array1[:] = 0






    image_data = np.stack([np.stack([flair], axis = 3)])

    original_unscaled_array = image_data.copy()


    bounds = locate_bounds(image_data)



    [left,right,low,high,shallow,deep] = bounds



    x_size = abs(left-right)
    y_size = abs(low-high)
    z_size = abs(shallow-deep)

    max_size = np.max([x_size, y_size, z_size])


    image_data = translate_3d(np.squeeze(image_data.copy()[0], axis = 3), [-shallow,-low,-left])


    rescale_factor = (image_size*0.8)/max_size

    print("rescale factor:", rescale_factor)

    backscale_factor = 1/rescale_factor


    image_data = skimage.transform.rescale(image_data, (rescale_factor, rescale_factor, rescale_factor))

    
    original_scaled_down = image_data.copy()


    for z,Slice in enumerate(image_data):
        for y,line in enumerate(Slice):
            for x,pixel in enumerate(line):
                try:
                    original_array1[z][y][x] = pixel
                except:
                    pass
    flair = original_array1.copy()
    outputs.append(flair)

    
    ###########################       T1CE   ####################################
    for t1 in scans[1:]:
        t1 = t1/np.max(t1)

        image_data = translate_3d(t1, [-shallow,-low,-left])

        image_data = skimage.transform.rescale(image_data, (rescale_factor, rescale_factor, rescale_factor))

        original_array1[:] = 0
        for z,Slice in enumerate(image_data):
            for y,line in enumerate(Slice):
                for x,pixel in enumerate(line):
                    try:
                        original_array1[z][y][x] = pixel
                    except:
                        pass
        t1ce = original_array1.copy()
        outputs.append(t1ce)


    return outputs

def preprocess_data_set(pathes):
    arrays = []
    images, valid = load_dicom_set_folders(pathes, updating_labels = False, direction = False)
    image_data = images[0]
    print(image_data.shape)
    print(valid)
    print(np.max(image_data))
    print("oofoofoofoofofofofofof")
        
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

    print("\n\nbounds:",bounds,"\n\n")
    [left,right,low,high,shallow,deep] = bounds


    image_data = translate_3d(image_data, [-shallow,-low,-left])

    rescaled_array = rescaled_blank.copy()
    for z,Slice in enumerate(image_data):
        for y,line in enumerate(Slice):
            for x,pixel in enumerate(line):
                try:
                    rescaled_array[z][y][x] = pixel
                except:
                    pass
    arrays.append(rescaled_array)

    for i in range(1,len(images)):
        image_data = images[i]
        image_data = image_data/np.max(image_data)



        blank_unscaled_array = image_data.copy()
        blank_unscaled_array[:] = 0

        z_zoom = image_size/image_data.shape[0]
        y_zoom = image_size/image_data.shape[1]
        x_zoom = image_size/image_data.shape[2]
        print(z_zoom, y_zoom, x_zoom)

        image_data = np.stack([np.stack([image_data], axis = 3)])

        rescaled_blank = skimage.transform.rescale(blank_unscaled_array, (z_zoom, y_zoom, x_zoom))
        image_data = skimage.transform.rescale(np.squeeze(image_data.copy()[0], axis = 3), (rescale_factor, rescale_factor, rescale_factor))
        image_data = translate_3d(image_data, [-shallow,-low,-left])

        rescaled_array = rescaled_blank.copy()
        for z,Slice in enumerate(image_data):
            for y,line in enumerate(Slice):
                for x,pixel in enumerate(line):
                    try:
                        rescaled_array[z][y][x] = pixel
                    except:
                        pass
        arrays.append(rescaled_array)

                          

    return arrays


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

def ConvNetbinary(x,y,z):
    inputs = keras.layers.Input((x,y,z, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, 4, 0.1) #128 -> 64
    print(p1.shape)
    c2, p2 = down_block(p1, 8, 0.1) #64 -> 32
    c3, p3 = down_block(p2, 16, 0.2) #32 -> 16
    c4, p4 = down_block(p3, 32, 0.3) #16->8
    c5, p5 = down_block(p4, 64, 0.3) #16->8
    c6, p6 = down_block(p5, 128, 0.3) #16->8
    c7, p7 = down_block(p6, 256, 0.3) #16->8
    
    bn = bottleneck(p7, 512, 0.4)
    print(bn.shape)
    
    u1 = up_block(bn, c7, 256, 0.3) #8 -> 16
    u2 = up_block(u1, c6, 128, 0.2) #16 -> 32
    u3 = up_block(u2, c5, 64, 0.1) #32 -> 64
    u4 = up_block(u3, c4, 32, 0.1) #64 -> 128
    u5 = up_block(u4, c3, 16, 0.1) #64 -> 128
    u6 = up_block(u5, c2, 8, 0.1) #64 -> 128
    u7 = up_block(u6, c1, 4, 0.1) #64 -> 128
    
    outputs = tf.keras.layers.Conv3D(1, (1, 1, 1),padding='same', activation="sigmoid")(u7)
    #outputs = keras.layers.Conv3D(1, (1, 1, 1), padding="same", activation="relu")(u4)
    print("out")
    print(outputs.shape)
    model = keras.models.Model(inputs, outputs)
    return model
def down_block_e(x, filters, dropout, kernel_size=(3, 3, 3), padding="same", strides=1):
    print(x.shape)
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="elu", input_shape = x.shape[1:], kernel_initializer='he_normal')(x)
    c = keras.layers.Dropout(dropout)(c)
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="elu", input_shape = c.shape[1:], kernel_initializer='he_normal')(c)
    p = keras.layers.MaxPool3D(pool_size = (2, 2, 2))(c)
    return c, p

def up_block_e(x, skip, filters, dropout,kernel_size=(3, 3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling3D((2, 2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="elu", input_shape = concat.shape[1:], kernel_initializer='he_normal')(concat)
    c = keras.layers.Dropout(dropout)(c)
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="elu", input_shape = c.shape[1:], kernel_initializer='he_normal')(c)
    return c

def bottleneck_e(x, filters, dropout, kernel_size=(3, 3, 3), padding="same", strides=1):
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="elu", input_shape = x.shape[1:], kernel_initializer='he_normal')(x)
    c = keras.layers.Dropout(dropout) (c)
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="elu", input_shape = c.shape[1:], kernel_initializer='he_normal')(c)
    return c


def ConvNetSemantic64(x,y,z):
    inputs = keras.layers.Input((x,y,z, 3))
    
    p0 = inputs
    c1, p1 = down_block_e(p0, 16, 0.1) #128 -> 64
    print(p1.shape)
    c2, p2 = down_block_e(p1, 32, 0.1) #64 -> 32
    c3, p3 = down_block_e(p2, 64, 0.2) #32 -> 16
    c4, p4 = down_block_e(p3, 128, 0.3) #16->8
    c5, p5 = down_block_e(p4, 256, 0.3) #16->8
    c6, p6 = down_block_e(p5, 512, 0.3) #16->8
    
    bn = bottleneck_e(p6, 1024, 0.4)
    print(bn.shape)
    
    u1 = up_block_e(bn, c6, 512, 0.3) #8 -> 16
    u2 = up_block_e(u1, c5, 256, 0.2) #16 -> 32
    u3 = up_block_e(u2, c4, 128, 0.1) #32 -> 64
    u4 = up_block_e(u3, c3, 64, 0.1) #64 -> 128
    u5 = up_block_e(u4, c2, 32, 0.1) #64 -> 128
    u6 = up_block_e(u5, c1, 16, 0.1) #64 -> 128
    
    outputs = tf.keras.layers.Conv3D(4, (1, 1, 1),padding='same', activation="softmax")(u6)
    #outputs = keras.layers.Conv3D(1, (1, 1, 1), padding="same", activation="relu")(u4)
    print("out")
    print(outputs.shape)
    model = keras.models.Model(inputs, outputs)
    return model

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

def ConvNetRough(x,y,z):
    inputs = keras.layers.Input((x,y,z, 1))
    
    p0 = inputs
    c1, p1 = down_block(p0, 32, 0.1) #128 -> 64
    print(p1.shape)
    c2, p2 = down_block(p1, 64, 0.1) #64 -> 32
    c3, p3 = down_block(p2, 128, 0.2) #32 -> 16
    
    bn = bottleneck(p3, 256, 0.4)
    print(bn.shape)
    
    u1 = up_block(bn, c3, 128, 0.3) #16 -> 32
    u2 = up_block(u1, c2, 64, 0.2) #16 -> 64
    u3 = up_block(u2, c1, 32, 0.1) #32 -> 128
    
    outputs = tf.keras.layers.Conv3D(1, (1, 1, 1),padding='same', activation="sigmoid")(u3)

    print("out")
    print(outputs.shape)
    model = keras.models.Model(inputs, outputs)
    return model

output_image_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/image ct  visualizations/Machine Learning 2 models test"

image_size = 128

brain_seg_model_top = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_brain_top.h5"
brain_seg_model_top2 = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_brain_top.h5"
brain_seg_model_front = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_brain_front.h5"
brain_seg_model_side = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_brain_side.h5"
brain_seg_model_edges = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_brain_edges.h5"

tumor_seg_model = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_tumor.h5"

input_path = "C:/Users/JiangQin/Documents/data/raw ct files/Brain-Tumor-Progression"
output_path = "C:/Users/JiangQin/Documents/data/raw ct files/Brain-Tumor-Progression/loaded arrays for growth prediction"

binary_model_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/code files/Saved models/Tumor seg binary with t1ce t2 flair/Model 16.h5"
binary_model_path2 = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_tumor.h5"
binary_model_path3 = "C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/Model 34.h5"

semantic_model_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/code files/Saved models/Tumor seg semantic 64x with t1ce t2 flair/Model 81.h5"
brain_seg_t1_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/code files/Saved models/Brain Seg OASIS 36 top view/Model 51 (2).h5"
start = 9
sets = []
from datetime import datetime

for set_ in os.listdir(input_path):
    set_path = input_path + "/" + set_
    scans = []
    scan_dates = []

    for scan in os.listdir(set_path):
        flair = None
        t1 = None
        t2 = None
        adc = None
        
        scan_path = set_path + '/' + scan
        if os.path.isdir(scan_path):
            for mri in os.listdir(scan_path):
                if "t2" in mri.lower() and os.path.isdir(scan_path + "/" + mri):
                    if t2!=None:
                        bru+=1
                    t2 = mri
                if "t1post" in mri.lower() and os.path.isdir(scan_path + "/" + mri):
                    if t1!=None:
                        bru+=1
                    t1 = mri
                if "flair" in mri.lower() and os.path.isdir(scan_path + "/" + mri):
                    if flair!=None:
                        bru+=1
                    flair = mri
                if "adc" in mri.lower() and os.path.isdir(scan_path + "/" + mri):
                    if adc!=None:
                        bru+=1
                    adc = mri
            if flair is not None and t1 is not None and t2 is not None:
                month,day,yr = scan.split("-")[:3]
                month,day,yr = int(month),int(day),int(yr)
                print(month,day,yr)
                scans.append([scan_path + "/" + flair,scan_path + "/" + t1,scan_path + "/" + t2,scan_path + "/" + adc])
                scan_dates.append(datetime(yr,month,day).timestamp())
        
    if len(scans)>0:
        scans = [x for x,_ in sorted(zip(scans,scan_dates),key=lambda x:x[1])]
        sets.append(scans)



for Set in range(start, len(sets)):
    pathes = []
    
    print("\n\nSet " + str(Set) + "\n\n")
    for i in range(0,10):
        print("\nGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG\n")
    print(sets[Set])
    print("blub:",pathes)
    flair,t1,t2,adc = preprocess_data_set(sets[Set][0])
    write_images(t2, output_image_path+"/tumor growth pred/"+str(Set)+"/image3")
    write_images(adc, output_image_path+"/tumor growth pred/"+str(Set)+"/image4")
    flair_binary = np.array(flair >0.1,dtype = "float64")
    write_images(flair_binary, output_image_path)
    flair_binary_image_og = sitk.GetImageFromArray(flair_binary)
    

    write_images(np.stack([flair,t1,t2],axis=-1), output_image_path)
    write_images(np.stack([flair,t1,t2],axis=-1), output_image_path+"/tumor growth pred/"+str(Set)+"/image_full")

    ##flair brain seg
    print("tranformed arrays",np.max(flair),np.max(t1),np.max(t2))
    segmentations = []

    brain_seg_top = load_model(brain_seg_model_top)
    brain_mask_top = brain_seg_top.predict(np.stack([np.stack([flair], axis = -1)]))
    binary_brain_top = binarize(brain_mask_top, 0.5)
    binary_brain_top_top_ized = np.squeeze(binary_brain_top.copy()[0], axis = -1)
    segmentations.append(binary_brain_top_top_ized)

    print("segmented brain")
    binary_brain_wo_median_combined = combine_zeros(segmentations)


    median_flair = find_median_grayscale(flair[binary_brain_wo_median_combined > 0])


    segmentations = []

    brain_seg_top = load_model(brain_seg_model_top)
    new_array_top = np.stack([np.stack([flair], axis = -1)])/(median_flair/0.2)
    brain_mask_top = brain_seg_top.predict(new_array_top)
    binary_brain_top = binarize(brain_mask_top, 0.7)
    binary_brain_top_top_ized = np.squeeze(binary_brain_top.copy()[0], axis = -1)
    segmentations.append(binary_brain_top_top_ized)

    binary_brain_final_combined2 = combine_zeros(segmentations)

    ##t1 brain seg

    segmentations = []

    model_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/code files/Saved models/Brain Seg OASIS 36 top view/Model 51 (2).h5"
    
    brain_seg = ConvNetRough(128,128,128)
    brain_seg.load_weights(model_path)
    brain_mask_top = brain_seg.predict(np.stack([np.stack([t1], axis = -1)]))
    binary_brain_top = binarize(brain_mask_top, 0.5)
    binary_brain_top_top_ized = np.squeeze(binary_brain_top.copy()[0], axis = -1)
    segmentations.append(binary_brain_top_top_ized)

    print("segmented brain")
    print(np.stack([np.stack([t1], axis = -1)]).shape)
    binary_brain_wo_median_combined = combine_zeros(segmentations)
    only_brain_t1 = t1.copy()
    only_brain_t1[binary_brain_wo_median_combined == 0] = 0



    median_t1 = find_median_grayscale(only_brain_t1)


    segmentations = []
    model_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/code files/Saved models/Brain Seg OASIS 36 top view/Model 51 (2).h5"
    
    brain_seg = ConvNetRough(128,128,128)
    brain_seg.load_weights(model_path)
    brain_mask_top = brain_seg.predict(np.stack([np.stack([t1/(median_t1/0.3)], axis = -1)]))
    binary_brain_top = binarize(brain_mask_top, 0.5)
    binary_brain_top_top_ized = np.squeeze(binary_brain_top.copy()[0], axis = -1)
    segmentations.append(binary_brain_top_top_ized)

    


    binary_brain_final_combined1 = combine_zeros(segmentations)

    #write_images(binary_brain_final_combined1, output_image_path+"/tumor growth pred/"+str(Set)+"/imageblub")
    #input("HUDADAWUBUDAWUP")

    ##t2 brain seg


    segmentations = []

    brain_seg_top = load_model(brain_seg_model_top)
    brain_mask_top = brain_seg_top.predict(np.stack([np.stack([t2], axis = -1)]))
    binary_brain_top = binarize(brain_mask_top, 0.5)
    binary_brain_top_top_ized = np.squeeze(binary_brain_top.copy()[0], axis = -1)
    segmentations.append(binary_brain_top_top_ized)

    print("segmented brain")
    binary_brain_wo_median_combined = combine_zeros(segmentations)


    median = find_median_grayscale(t2[binary_brain_wo_median_combined > 0])


    segmentations = []

    brain_seg_top = load_model(brain_seg_model_top)
    new_array_top = np.stack([np.stack([t2], axis = -1)])/(median/0.2)
    brain_mask_top = brain_seg_top.predict(new_array_top)
    binary_brain_top = binarize(brain_mask_top, 0.7)
    binary_brain_top_top_ized = np.squeeze(binary_brain_top.copy()[0], axis = -1)
    segmentations.append(binary_brain_top_top_ized)

    binary_brain_final_combined = binary_brain_final_combined2.copy()#combine_zeros(segmentations)#binary_brain_final_combined2.copy()
    binary_brain_final_combined[binary_brain_final_combined2 > 0] = 1
    #binary_brain_final_combined[binary_brain_final_combined1 < 1] = 0

    binary_brain_final_combined = biggest_island(binary_brain_final_combined,stacked=False)

    write_images(binary_brain_final_combined, output_image_path+"/tumor growth pred/"+str(Set)+"/imageblub")


    #-------------

    flair[binary_brain_final_combined==0] = 0
    t1[binary_brain_final_combined==0] = 0
    t2[binary_brain_final_combined==0] = 0
    adc[binary_brain_final_combined==0] = 0

    #stuff = normalize([flair,t1,t2,adc])

    #[flair,t1,t2,adc] = stuff

    t1 /=(find_median_grayscale(t1)/0.35)
    t2 /=(find_median_grayscale(t2)/0.35)
    flair /=(find_median_grayscale(flair)/0.35)
    write_images(np.stack([flair,t1,t2],axis=-1), output_image_path+"/tumor growth pred/"+str(Set)+"/image_brain")

    flairc = skimage.transform.rescale(flair, (2.0,2.0,2.0))
    t1c = skimage.transform.rescale(t1, (2.0,2.0,2.0))
    t2c = skimage.transform.rescale(t2, (2.0,2.0,2.0))
    adcc = skimage.transform.rescale(adc, (2.0,2.0,2.0))
    
    write_images(flairc, output_image_path+"/tumor growth pred/"+str(Set)+"/image_brainf")
    write_images(t1c, output_image_path+"/tumor growth pred/"+str(Set)+"/image_braint")
    write_images(t2c, output_image_path+"/tumor growth pred/"+str(Set)+"/image_braintt")
    write_images(adcc, output_image_path+"/tumor growth pred/"+str(Set)+"/image_brainadc")
  
    
    only_brain = np.stack([flair,t1,t2],axis = -1)

    only_brain = skimage.transform.rescale(only_brain, (0.5,0.5,0.5,1))
    adc = skimage.transform.rescale(adc, (0.5,0.5,0.5))
 
    write_images(only_brain, output_image_path+"/tumor growth pred/"+str(Set)+"/imagebraib")
    
    tumor_seg_binary = load_model(binary_model_path2)

    tumor_mask = tumor_seg_binary.predict(np.stack([np.stack([flair/(median_flair/0.3)],axis=-1)]))

    tumor_binary = np.squeeze(tumor_mask[0] > 0.5, axis = -1)


    tumor_seg_channeled = ConvNetSemantic64(64,64,64)

    tumor_seg_channeled.load_weights(semantic_model_path)

    tumor_mask = tumor_seg_channeled.predict(np.stack([only_brain]))
    print(tumor_mask.shape)
    print(np.max(tumor_mask))

    tumor_colored = np.argmax(tumor_mask[0], axis = -1)
    print(np.max(tumor_colored))
    print(tumor_colored.shape)
    tumor_colored = skimage.transform.rescale(tumor_colored/3, (2.0,2.0,2.0))
    print(np.max(tumor_colored))
    tumor_colored = np.round(tumor_colored*3)
    write_images(tumor_colored/3, output_image_path+"/tumor growth pred/"+str(Set)+"/image_seg")
    tumor_colored[tumor_binary == 0] = 0
    write_images(tumor_colored/3, output_image_path+"/tumor growth pred/"+str(Set)+"/image_seg_bin")
    tumor_colored = skimage.transform.rescale(tumor_colored/3, (0.5,0.5,0.5))
    print(np.max(tumor_colored))
    tumor_colored = np.round(tumor_colored*3)
    print(tumor_colored.shape)


    tumor_mask = tumor_mask[0]
    tumor_mask = skimage.transform.rescale(tumor_mask, (2.0,2.0,2.0,1))

    tumor_mask[:,:,:,1:][np.stack([tumor_binary,tumor_binary,tumor_binary],axis=-1)==0] = 0
    tumor_mask = skimage.transform.rescale(tumor_mask, (2.0,2.0,2.0,1))
    write_images(tumor_mask[:,:,:,1:], output_image_path+"/tumor growth pred/"+str(Set)+"b/tumorold")


    #cont = input("YEEEEEEEEEEEEEEEEETTTTTTTTTTTTTT")
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------

    flair,t1,t2,adc = preprocess_data_set(sets[Set][1])
    
    flair_binary = np.array(flair >0.1,dtype = "float64")
    flair_binary_image = sitk.GetImageFromArray(flair_binary) 
    flair_image = sitk.GetImageFromArray(flair)
    flair, tran = register(flair_binary_image_og, flair_binary_image, flair_image)


    t1_binary = np.array(t1 >0.1,dtype = "float64")
    t1_binary_image = sitk.GetImageFromArray(t1_binary) 
    t1_image = sitk.GetImageFromArray(t1)
    t1,_ = register(flair_binary_image_og, t1_binary_image, t1_image,tran)

    t2_binary = np.array(t2 >0.1,dtype = "float64")
    t2_binary_image = sitk.GetImageFromArray(t2_binary) 
    t2_image = sitk.GetImageFromArray(t2)
    t2,_ = register(flair_binary_image_og, t2_binary_image, t2_image,tran)
    
    adc_binary = np.array(adc >0.1,dtype = "float64")
    adc_binary_image = sitk.GetImageFromArray(adc_binary) 
    adc_image = sitk.GetImageFromArray(adc)
    adc,_ = register(flair_binary_image_og, adc_binary_image, adc_image,tran)
    

    write_images(np.stack([flair,t1,t2],axis=-1), output_image_path)
    write_images(np.stack([flair,t1,t2],axis=-1), output_image_path+"/tumor growth pred/"+str(Set)+"b/image_full")

    ##flair brain seg
    print("tranformed arrays",np.max(flair),np.max(t1),np.max(t2))
    segmentations = []

    brain_seg_top = load_model(brain_seg_model_top)
    brain_mask_top = brain_seg_top.predict(np.stack([np.stack([flair], axis = -1)]))
    binary_brain_top = binarize(brain_mask_top, 0.5)
    binary_brain_top_top_ized = np.squeeze(binary_brain_top.copy()[0], axis = -1)
    segmentations.append(binary_brain_top_top_ized)

    print("segmented brain")
    binary_brain_wo_median_combined = combine_zeros(segmentations)


    median_flair = find_median_grayscale(flair[binary_brain_wo_median_combined > 0])


    segmentations = []

    brain_seg_top = load_model(brain_seg_model_top)
    new_array_top = np.stack([np.stack([flair], axis = -1)])/(median_flair/0.2)
    brain_mask_top = brain_seg_top.predict(new_array_top)
    binary_brain_top = binarize(brain_mask_top, 0.7)
    binary_brain_top_top_ized = np.squeeze(binary_brain_top.copy()[0], axis = -1)
    segmentations.append(binary_brain_top_top_ized)

    binary_brain_final_combined2 = combine_zeros(segmentations)

    ##t1 brain seg

    segmentations = []

    model_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/code files/Saved models/Brain Seg OASIS 36 top view/Model 51 (2).h5"
    
    brain_seg = ConvNetRough(128,128,128)
    brain_seg.load_weights(model_path)
    brain_mask_top = brain_seg.predict(np.stack([np.stack([t1], axis = -1)]))
    binary_brain_top = binarize(brain_mask_top, 0.5)
    binary_brain_top_top_ized = np.squeeze(binary_brain_top.copy()[0], axis = -1)
    segmentations.append(binary_brain_top_top_ized)

    print("segmented brain")
    print(np.stack([np.stack([t1], axis = -1)]).shape)
    binary_brain_wo_median_combined = combine_zeros(segmentations)
    only_brain_t1 = t1.copy()
    only_brain_t1[binary_brain_wo_median_combined == 0] = 0



    median_t1 = find_median_grayscale(only_brain_t1)


    segmentations = []
    model_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/code files/Saved models/Brain Seg OASIS 36 top view/Model 51 (2).h5"
    
    brain_seg = ConvNetRough(128,128,128)
    brain_seg.load_weights(model_path)
    brain_mask_top = brain_seg.predict(np.stack([np.stack([t1/(median_t1/0.3)], axis = -1)]))
    binary_brain_top = binarize(brain_mask_top, 0.5)
    binary_brain_top_top_ized = np.squeeze(binary_brain_top.copy()[0], axis = -1)
    segmentations.append(binary_brain_top_top_ized)

    


    binary_brain_final_combined1 = combine_zeros(segmentations)

    #write_images(binary_brain_final_combined1, output_image_path+"/tumor growth pred/"+str(Set)+"/imageblub")
    #input("HUDADAWUBUDAWUP")

    ##t2 brain seg


    segmentations = []

    brain_seg_top = load_model(brain_seg_model_top)
    brain_mask_top = brain_seg_top.predict(np.stack([np.stack([t2], axis = -1)]))
    binary_brain_top = binarize(brain_mask_top, 0.5)
    binary_brain_top_top_ized = np.squeeze(binary_brain_top.copy()[0], axis = -1)
    segmentations.append(binary_brain_top_top_ized)

    print("segmented brain")
    binary_brain_wo_median_combined = combine_zeros(segmentations)


    median = find_median_grayscale(t2[binary_brain_wo_median_combined > 0])


    segmentations = []

    brain_seg_top = load_model(brain_seg_model_top)
    new_array_top = np.stack([np.stack([t2], axis = -1)])/(median/0.2)
    brain_mask_top = brain_seg_top.predict(new_array_top)
    binary_brain_top = binarize(brain_mask_top, 0.7)
    binary_brain_top_top_ized = np.squeeze(binary_brain_top.copy()[0], axis = -1)
    segmentations.append(binary_brain_top_top_ized)

    binary_brain_final_combined = binary_brain_final_combined2.copy()#combine_zeros(segmentations)#
    binary_brain_final_combined[binary_brain_final_combined2 > 0] = 1
    #binary_brain_final_combined[binary_brain_final_combined1 < 1] = 0

    binary_brain_final_combined = biggest_island(binary_brain_final_combined,stacked=False)

    write_images(binary_brain_final_combined, output_image_path+"/tumor growth pred/"+str(Set)+"b/imageblub")


    #-------------

    flair[binary_brain_final_combined==0] = 0
    t1[binary_brain_final_combined==0] = 0
    t2[binary_brain_final_combined==0] = 0
    adc[binary_brain_final_combined==0] = 0

    #stuff = normalize([flair,t1,t2,adc])

    #[flair,t1,t2,adc] = stuff

    t1 /=(find_median_grayscale(t1)/0.4)
    t2 /=(find_median_grayscale(t2)/0.4)
    flair /=(find_median_grayscale(flair)/0.4)
    write_images(np.stack([flair,t1,t2],axis=-1), output_image_path+"/tumor growth pred/"+str(Set)+"b/image_brain")


    
    only_brain = np.stack([flair,t1,t2],axis = -1)

    only_brain = skimage.transform.rescale(only_brain, (0.5,0.5,0.5,1))
    adc = skimage.transform.rescale(adc, (0.5,0.5,0.5))
 
    write_images(only_brain, output_image_path+"/tumor growth pred/"+str(Set)+"b/imagebraib")
    
    tumor_seg_binary = load_model(binary_model_path2)

    tumor_mask = tumor_seg_binary.predict(np.stack([np.stack([flair/(median_flair/0.3)],axis=-1)]))

    tumor_binary = np.squeeze(tumor_mask[0] > 0.5, axis = -1)


    tumor_seg_channeled = ConvNetSemantic64(64,64,64)

    tumor_seg_channeled.load_weights(semantic_model_path)

    tumor_mask = tumor_seg_channeled.predict(np.stack([only_brain]))
    print(tumor_mask.shape)
    print(np.max(tumor_mask))

    tumor_colored = np.argmax(tumor_mask[0], axis = -1)
    print(np.max(tumor_colored))
    print(tumor_colored.shape)
    tumor_colored = skimage.transform.rescale(tumor_colored/3, (2.0,2.0,2.0))
    print(np.max(tumor_colored))
    tumor_colored = np.round(tumor_colored*3)
    write_images(tumor_colored/3, output_image_path+"/tumor growth pred/"+str(Set)+"b/image_seg")
    tumor_colored[tumor_binary == 0] = 0
    write_images(tumor_colored/3, output_image_path+"/tumor growth pred/"+str(Set)+"b/image_seg_bin")
    tumor_colored = skimage.transform.rescale(tumor_colored/3, (0.5,0.5,0.5))
    print(np.max(tumor_colored))
    tumor_colored = np.round(tumor_colored*3)
    print(tumor_colored.shape)

    tumor_mask = tumor_mask[0]
    tumor_mask = skimage.transform.rescale(tumor_mask, (2.0,2.0,2.0,1))

    tumor_mask[:,:,:,1:][np.stack([tumor_binary,tumor_binary,tumor_binary],axis=-1)==0] = 0
    tumor_mask = skimage.transform.rescale(tumor_mask, (2.0,2.0,2.0,1))
    write_images(tumor_mask[:,:,:,1:], output_image_path+"/tumor growth pred/"+str(Set)+"b/tumornew")


    print("saved mask\n")
    write_images(image[:,:,:,0], output_image_path+"/tumor growth pred/"+str(Set)+"b/image1")
    write_images(image[:,:,:,1], output_image_path+"/tumor growth pred/"+str(Set)+"b/image2")
    write_images(image[:,:,:,2], output_image_path+"/tumor growth pred/"+str(Set)+"b/image3")
    
    write_images(image[:,:,:,0], output_image_path+"/tumor growth pred/"+str(Set)+"b/image1")
    write_images(image[:,:,:,1], output_image_path+"/tumor growth pred/"+str(Set)+"b/image2")
    write_images(image[:,:,:,2], output_image_path+"/tumor growth pred/"+str(Set)+"b/image3")
    

    new_image = image.copy()

    print("saved mask\n")


    #-----------------------------------------------------------------------------------------------------------------------------------------


    complete_image = np.stack([old_image[:,:,:,0],old_image[:,:,:,1],old_image[:,:,:,2],old_image[:,:,:,3],old_image[:,:,:,4],old_image[:,:,:,5],old_image[:,:,:,6]],axis = -1)
    complete_image2 = np.stack([new_image[:,:,:,3],new_image[:,:,:,4],new_image[:,:,:,5]],axis = -1)
    
    save_array(complete_image, output_path + "/" + str(Set) + "/image.h5")
    save_array([complete_image2], output_path + "/" + str(Set) + "/mask.h5")
    print("\n\nComplete image shape:",complete_image.shape,"\n\n")
    print("max and min:",np.max(complete_image),np.min(complete_image))

    print("image shape:", image.shape)
    
    
    

    print("saved image")
    print ('Finished one set in', int((time.time() - start_time)/60), 'minutes and ', int((time.time() - start_time) % 60), 'seconds.')

    
