import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import time
import pandas
import h5py
import cv2
import random
import scipy.ndimage
import skimage.transform
import shutil

output_image_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/image ct  visualizations/Machine Learning 2 models test"

array_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/raw ct files/flair tumor seg arrays"


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
                if pixel > 0.01:
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


def translate_3d(array, translation):
    original_array = array.copy()

    array_translated = array.copy()
    array_translated[:] = 0
    for z,Slice in enumerate(original_array):
        for y,line in enumerate(Slice):
            for x,pixel in enumerate(line):
                if pixel > 0.1:
                    array_translated[z+translation[0]][y+translation[1]][x+translation[2]] = pixel

    return array_translated

def write_images(array, test_folder_path):
    #array = array/np.max(array)
    for n,image in enumerate(array):
        ##finds the index of the corresponding file name in the original input path from the resize factor after resampling
        file_name = str(str(n) +'.png')

        cv2.imwrite(os.path.join(test_folder_path, file_name), image*255)



def blur(array, blur_precision):
    return scipy.ndimage.gaussian_filter(array, blur_precision)

def find_median_grayscale(array):
    
    zero_pixels = float(np.count_nonzero(array==0))


    single_dimensional = array.flatten().tolist()

    
    single_dimensional.extend(np.full((1, int(zero_pixels)), 1000).flatten().tolist()

                              )
    return np.median(single_dimensional)


def clip_neck_area(array, mask, amt):
    #print(np.max(array))
    cut_off = array.copy()
    new_mask = mask.copy()
    found_non_empty = False
    for n,image in enumerate(array):
        if np.max(mask[n]) > 0 and found_non_empty == False:
            first_non_empty = n
            found_non_empty = True            

    cut_point = first_non_empty + amt
    for n, image in enumerate(cut_off):
        if n <= cut_point:
            image[:] = 0
    for n, image in enumerate(new_mask):
        if n <= cut_point:
            image[:] = 0
    return cut_off, new_mask


def mask_off(foreground,background,mask,invert = False):
    if invert:
        new_mask = mask.copy()
        new_mask[:] = 0
        new_mask[mask == 0] = 1
        mask = new_mask.copy()
    masked_off = foreground.copy()
    masked_off[mask == 0] = 0
    
    background_new = background.copy()
    background_new[mask > 0] = 0
    
    masked_off += background_new


        

    return masked_off


def change_contrast(array, factor):
    new_array = array.copy()

    new_array *= factor

    new_array += (0.5-(factor/2))

    

    return new_array

def circle_highlighted(reference, binary, color):
    circled = reference.copy()
    binary[binary > 0] = 1
    
    
    for n, image in enumerate(binary):
        contours, _ = cv2.findContours(image.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(circled[n], contours, -1,color, 1)

    

    return circled


def dilate_up(array, size):
    binary = array.copy()


    ##creates a kernel which is a 3 by 3 square of ones as the main kernel for all denoising
    kernel = scipy.ndimage.generate_binary_structure(3, 1)

    ##erodes away the white areas of the 3d array to seperate the loose parts
    blew_up = scipy.ndimage.binary_dilation(binary.astype('uint8'), kernel, iterations=size)

    return blew_up


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



def convex_hull(array, thickness):
    contour_only = array.copy()
    binary = array.copy()

    contour_only[:] = 0
    
    binary[:] = 0
    binary[array > 0.05] = 1

    
    cont = []
    for n, image in enumerate(binary):
        contours, _ = cv2.findContours(image.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_only[n], contours, -1, 1, thickness)


    return contour_only
  
def kill_small_islands(array, denoise_iterations):
    binary = array.copy()
    masked = array.copy()

    binary[:] = 0
    binary[array > 0] = 1

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
    omit = markers[0][0][0]
    flat = markers.ravel()
    binc = np.bincount(flat)
    binc_not = np.bincount(flat[flat == omit])
    binc[binc == binc_not] = 0
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
    
    return masked


def save_array(array, path):
    if os.path.exists(path):
        os.remove(path)

    hdf5_store = h5py.File(path, "a")
    hdf5_store.create_dataset("all_data", data = array, compression="gzip")



def seperate_sets(images, masks, set_size):
    image_sets = []
    mask_sets = []
    loop = 0
    current_set = -1
    num_sets = len(images)
    
    while True:
        if loop % set_size == 0:
            image_sets.append([])
            mask_sets.append([])
            current_set += 1

        image_sets[current_set].append(images[loop])
        mask_sets[current_set].append(masks[loop])
        loop += 1

        if loop >= num_sets:
            break
        
    return image_sets, mask_sets


def save_data_set(set_, save_path, save_name):
    if os.path.exists(os.path.join(save_path, save_name)+".h5"):
        os.remove(os.path.join(save_path, save_name)+".h5")

    hdf5_store = h5py.File(os.path.join(save_path, save_name)+".h5", "a")
    hdf5_store.create_dataset("all_data", data = set_, compression="gzip")

def split_train_val_test(set_):
    total = len(set_)
    train_end_val_beginning = round(0.7 * total)
    val_end_test_beginning = round(0.85 * total)


    train_images = set_[:train_end_val_beginning]
    val_images = set_[train_end_val_beginning:val_end_test_beginning]
    test_images = set_[val_end_test_beginning:]

    return train_images, val_images, test_images



def get_files(path, img_id = "flair", mask_id = "seg", save_folder_path = None,image_size = 128, start_index = 0):
    images = []
    masks = []
    start_time = time.time()
    non3 = 0

    num_files = 0
    for path, dirs, files in os.walk(path, topdown=False):
        for dirr in dirs:
            usable = 0
            set_path = os.path.join(path, dirr)
            for file in os.listdir(set_path):
                if mask_id in file:
                    usable+=1
                    mask_path = os.path.join(set_path, file)
                if "flair" in file:
                    usable+=1
                    flair_path = os.path.join(set_path, file)
                if "t1ce" in file:
                    usable+=1
                    t1ce_path = os.path.join(set_path, file)
                if "t2" in file:
                    usable+=1
                    t2_path = os.path.join(set_path, file)
            if usable == 4:
                images.append([flair_path,t1ce_path,t2_path])
                masks.append(mask_path)
                num_files+=2

    print("Total files to be loaded: " + str(num_files) + "\n")
    finished_files = start_index*2

    for i in range(start_index, num_files):
        print("\n\n" + str(i) + "\n\n")
        image = nib.load(images[i][0])
        flair = image.get_data().T.astype("float64")
        flair = flair/(find_median_grayscale(flair)/0.2)


        finished_files += 1
        print("finished " + str(finished_files) + " files out of " + str(num_files) + " files. " +
              str(round((finished_files*100/num_files), 2)) + "% done.") 
        
        mask = nib.load(masks[i])
        mask = mask.get_data().T.astype("float64")

        if len(np.unique(mask))>4:
            non3 +=1
        print("non3s:",non3)
        
        print("mask unique:", np.unique(mask))

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

        z_zoom = image_size/mask.shape[0]
        y_zoom = image_size/mask.shape[1]
        x_zoom = image_size/mask.shape[2]

        image_data1 = skimage.transform.rescale(mask, (z_zoom, y_zoom, x_zoom))

        original_array1 = image_data1.copy()
        original_array1[:] = 0

        mask = translate_3d(mask, [-shallow,-low,-left])

        image_data = skimage.transform.rescale(mask.copy(), (rescale_factor, rescale_factor, rescale_factor))


        original_scaled_down = image_data.copy()


        for z,Slice in enumerate(image_data):
            for y,line in enumerate(Slice):
                for x,pixel in enumerate(line):
                    try:
                        original_array1[z][y][x] = pixel
                    except:
                        pass
        mask = original_array1.copy()
        
        ###########################       T1CE   ####################################

        
        image = nib.load(images[i][1])
        t1ce = image.get_data().T.astype("float64")
        t1ce = t1ce/(find_median_grayscale(t1ce)/0.2)

        image_data = translate_3d(t1ce, [-shallow,-low,-left])

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

        ###########################       T2   ####################################
        image = nib.load(images[i][2])
        t2 = image.get_data().T.astype("float64")
        t2 = t2/(find_median_grayscale(t2)/0.2)

        image_data = translate_3d(t2, [-shallow,-low,-left])

        image_data = skimage.transform.rescale(image_data, (rescale_factor, rescale_factor, rescale_factor))

        original_array1[:] = 0
        for z,Slice in enumerate(image_data):
            for y,line in enumerate(Slice):
                for x,pixel in enumerate(line):
                    try:
                        original_array1[z][y][x] = pixel
                    except:
                        pass
        t2 = original_array1.copy()


        #t1 = np.rot90(t1, axes = (1,0))
        

        print(np.max(mask))
        channeled_mask = mask.copy()
        channeled_mask = np.round(channeled_mask)
        mask = channeled_mask.copy()

        print(np.max(mask/4))

        write_images(mask/4, output_image_path)
        print("unique mask:",np.unique(mask/4))


        array_output_path = os.path.join(array_path, str(i))

        if not os.path.exists(array_output_path):
            os.makedirs(array_output_path)
        else:
            shutil.rmtree(array_output_path)
            os.makedirs(array_output_path)
         
        input_array = np.stack([flair,t1ce,t2],axis=-1)

        write_images(input_array, output_image_path)
        print("input_shape: ",input_array.shape)
        save_array(input_array, array_output_path + "/original array.h5")
        save_array(mask, array_output_path + "/rough brain seg.h5")
        
           

        #circled = circle_highlighted(flair, mask, 0.8)

        print()
        #write_images(mask/4, output_image_path)

        
        

        finished_files += 1

        print("finished " + str(finished_files) + " files out of " + str(num_files) + " files. " +
              str(round((finished_files*100/num_files), 2)) + "% done.")

        print ('Finished in', int((time.time() - start_time)/60), 'minutes and ', int((time.time() - start_time) % 60), 'seconds.')





get_files("D:/Users/JiangQin/Documents/data/MICCAI_BraTS_2019_Data_Training", img_id = "flair",
          save_folder_path = "C:/Users/JiangQin/Documents/data/raw ct files/MICCAI_BraTS_2019_Data_Training/Images and Masks For Tumor Segmentation")

