import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nb
import time
import pandas
import h5py
import skimage.transform
import pydicom as dicom
import scipy
import cv2
import random
from PIL import Image, ImageEnhance
from skimage.measure import marching_cubes_lewiner as marching_cubes
import stl
from stl import mesh
import shutil


output_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/image ct  visualizations/Machine Learning 2 models test/brain border"

def write_images(array, test_folder_path):
    if not os.path.exists(test_folder_path):
        os.makedirs(test_folder_path)

    for n,image in enumerate(array):
        file_name = str(str(n) +'.png')
        cv2.imwrite(os.path.join(test_folder_path, file_name), image*255)


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

    def circle_highlighted(self, array, mask, color):
        circled = self.resampled_array.copy()
        binary = array.copy()
        
        binary[:] = 0
        binary[array > 0] = 255

        
        cont = []
        for n, image in enumerate(binary):
            contours, _ = cv2.findContours(image.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            #cont.append(contours)
            cv2.drawContours(circled[n], contours, -1,color, 1)

        

        return circled
 

    def write_test_images(self, array_3d, test_folder_path):
        array_3d = array_3d/np.max(array_3d)
        print(np.max(array_3d))
        for n,image in enumerate(array_3d):
            ##finds the index of the corresponding file name in the original input path from the resize factor after resampling
            file_name = str(str(n) +'.png')

            ##writes the resulting image as a png in the test_folder_path

            cv2.imwrite(os.path.join(test_folder_path, file_name), image*255)


def dilate_up_binary(array, size):
    binary = array.copy()

    binary[:] = 0
    binary[array > 0] = 255




    ##creates a kernel which is a 3 by 3 square of ones as the main kernel for all denoising
    kernel = scipy.ndimage.generate_binary_structure(3, 1)

    ##erodes away the white areas of the 3d array to seperate the loose parts
    blew_up = scipy.ndimage.binary_dilation(binary.astype('uint8'), kernel, iterations=size)


    return blew_up > 0


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

def flip(array):
    flipped = array.copy()
    for n,image in enumerate(array):
        flipped[-n] = image
    return flipped
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
    
def biggest_island(input_array):
    masked = input_array.copy()
    
    touching_structure_3d =[[[0,0,0],
                             [0,1,0],
                             [0,0,0]],

                            [[0,1,0],
                             [1,1,1],
                             [0,1,0]],

                            [[0,0,0],
                             [0,1,0],
                             [0,0,0]]]

    binary = input_array.copy()
    binary[:] = 0
    binary[input_array.copy() > 0] = 1

    ##uses a label to find the largest object in the 3d array and only keeps that (if you are trying to highlight something like bones, that have multiple parts, this method may not be suitable)
    markers, num_features = scipy.ndimage.measurements.label(binary,touching_structure_3d)
    binc = np.bincount(markers.ravel())
    binc[0] = 0
    noise_idx = np.where(binc != np.max(binc))
    mask = np.isin(markers, noise_idx)
    binary[mask] = 0

    masked[binary == 0] = 0
    
    return masked

def convex_border(array, thickness):
    contour_only = array.copy()
    binary = array.copy()

    contour_only[:] = 0
    
    binary[:] = 0
    binary[array > 0] = 255

    
    cont = []
    for n, image in enumerate(binary):
        contours, _ = cv2.findContours(image.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            biggest_contour = max(contours, key = cv2.contourArea)
            #contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
            hull = cv2.convexHull(biggest_contour)
            cv2.drawContours(contour_only[n], [hull], -1, 200, thickness)
        else:
            cv2.drawContours(contour_only[n], contours, -1, 255, thickness)
        

    return contour_only

    
def cut_neck(array, mask):
    wo_neck = array.copy()
    for n, Slice in enumerate(mask):
        wo_neck[n] = 0
        if not np.max(Slice) == 0:
            break
    return wo_neck, mask
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




def blur(array, blur_precision):
    return scipy.ndimage.gaussian_filter(array, blur_precision)

def mask_off(foreground,background,mask):
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

def mask_off(foreground,background,mask):
    masked_off = foreground.copy()
    masked_off[mask == 0] = 0
    
    background_new = background.copy()
    background_new[mask > 0] = 0
    
    masked_off += background_new


        

    return masked_off
    
def dilate_up(array, size):
    binary = array.copy()


    ##creates a kernel which is a 3 by 3 square of ones as the main kernel for all denoising
    kernel = scipy.ndimage.generate_binary_structure(3, 1)

    ##erodes away the white areas of the 3d array to seperate the loose parts
    blew_up = scipy.ndimage.binary_dilation(binary.astype('uint8'), kernel, iterations=size)

    return blew_up


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

def trim_from_top(array, mask, amt):
    cut_off = array.copy()
    new_mask = mask.copy()
    first_non_empty = False
    ends_on_black = False
    for n, image in enumerate(array):
        if np.max(image) > 0:
            first_non_empty = True
        if first_non_empty == True and np.max(image) == 0:
            last = n
            ends_on_black = True
            break
    if not ends_on_black:
        last = array.shape[0]-1

    print(np.max(array[last]))

    cut_point = last - amt
    
    for n, image in enumerate(cut_off):
        if n >= cut_point:
            image[:] = 0
    for n, image in enumerate(new_mask):
        if n >= cut_point:
            image[:] = 0

    return cut_off, new_mask
   
        
            
        
        
def generate_stl(array_3d, stl_file_path, stl_resolution):
    array = array_3d.copy()
    verts, faces, norm, val = marching_cubes(array, 0.5, step_size = stl_resolution, allow_degenerate=True)
    mesh = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh.vectors[i][j] = verts[f[j],:]
    
    if not stl_file_path.endswith(".stl"):
        stl_file_path += ".stl"
    if not os.path.exists(os.path.dirname(stl_file_path)):
        os.makedirs(os.path.dirname(stl_file_path))
    mesh.save(stl_file_path)

            
          

def get_files(path, img_id = "T1", mask_id = "aseg", save_folder_path = None, chunk_size = 64, image_size = 128):
    images = []
    masks = []
    start_time = time.time()


    set_size = int(chunk_size/2)

    num_files = 0
    for path, dirs, files in os.walk(path, topdown=False):
        for file in files:
            if img_id in file or mask_id in file:
                num_files += 1

    num_sets = int(num_files/2)

    print("Total files to be loaded: " + str(num_files) + "\n")
    
    finished_files = 0
    #for n, set in enumerate()
    for set_num,SET in enumerate(os.listdir(path)):
        if "OAS" in SET:

            set_path = os.path.join(path, SET)




            img = nb.load(os.path.join(set_path, "T1.mgz"))


            img_array = img.get_fdata()

            img_array = img_array/np.max(img_array)

            img_array = np.rot90(img_array, axes = (2,0))
            img_array = np.rot90(img_array, axes = (1,0))
            img_array = flip(img_array)

            t1 = img_array/np.max(img_array)





            mask = nb.load(os.path.join(set_path, "aparc+aseg.mgz"))




            mask_array = mask.get_fdata()

            mask_array = mask_array/np.max(mask_array)

            mask_array = np.rot90(mask_array, axes = (2,0))
            mask_array = np.rot90(mask_array, axes = (1,0))



            mask = flip(mask_array)

            mask[mask > 0] = 1
            first = False
            for n, image in enumerate(mask):
                if np.max(image) > 0 and first ==  False:
                    first = True
                if np.max(image) == 0 and first == True:
                    last  = n

            for n, image in enumerate(mask):
                if n >= last:
                    image[:] = 0

            t1, mask = cut_neck(t1, mask)


            decontrasted_array = t1.copy()


            #decontrasted_array = blur(decontrasted_array, 1)

            only_brain = decontrasted_array.copy()

            only_brain[mask == 0] = 0

            decontrast_brain = random.randint(3,6)
            decontrast_skull = random.randint(7,9)


            decontrasted_brain = change_contrast(only_brain, (decontrast_brain/10))
            decontrasted_skull = change_contrast(decontrasted_array, (decontrast_skull/10))




            big_mask = dilate_up(mask, 3)

            t11 = mask_off(decontrasted_brain, decontrasted_skull, big_mask)

            decontrasted_array = blur(t11, 0.5)

            only_brain = decontrasted_array.copy()

            only_brain[mask == 0] = 0


            median_brain = find_median_grayscale(only_brain)

            print("median brain:", median_brain)



            non_brain = decontrasted_array.copy()

            non_brain[mask > 0] = 0

            median_skull = find_median_grayscale(non_brain)




            black = convex_hull(t1, 1)
            black = fill_holes_binary(black, 100000000000000000)
            black = kill_small_islands(black, 1)
            black = dilate_up(black, 3)
            black = fill_holes_binary(black, 100000000000000000)

            t11[black == 0] = 0

            brain_median = random.randint(15, 40)

            blur_amt = random.randint(2, 10)


            t11[big_mask > 0] *= ((brain_median/100)/median_brain)
            #t11[big_mask == 0] *= (0.05/median_skull)

            t11 = blur(t11, (blur_amt/10))

            top_clip = random.randint(0,25)

            bottom_clip = random.randint(0,5)


            t11, mask = trim_from_top(t11, mask, top_clip)

            t11, mask = clip_neck_area(t11, mask, bottom_clip)
            

            finished_files += 1
                
            print("finished " + str(finished_files) + " files out of " + str(num_files) + " files. " +
                  str(round((finished_files*100/num_files), 2)) + "% done.") 



            blank_unscaled_array = t11.copy()

            blank_unscaled_array[:] = 0




            z_zoom = image_size/t1.shape[0]
            y_zoom = image_size/t1.shape[1]
            x_zoom = image_size/t1.shape[2]

            image_data1 = skimage.transform.rescale(t11, (z_zoom, y_zoom, x_zoom))

            original_array1 = image_data1.copy()
            original_array1[:] = 0






            image_data = np.stack([np.stack([t11], axis = 3)])

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
                        if pixel > 0.01:
                            original_array1[z][y][x] = pixel
            t1 = original_array1.copy()


            #t1 = np.rot90(t1, axes = (1,0))
            

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
                        if pixel > 0.01:
                            original_array1[z][y][x] = pixel
            mask = original_array1.copy()

            #mask = np.rot90(mask, axes = (1,0))

            mask[mask >0.1] = 1

            mask[mask < 1] = 0

            

                

            images.append(t1)

        

            #mask[mask > 0.5] = 1

            folder_output = output_path + "/" + str(set_num)
            

            #mask[mask < 1] = 0

            
            print(mask.shape,t1.shape)
            write_images(t1, folder_output+"/image")

            #mask = blur(mask, 1)

            

            #generate_stl(mask.T, folder_output+"/skin_skull.stl", 1)

            

            
            
            dilated_mask = dilate_up_binary(mask, 5)

            border = dilated_mask.copy()
            border[mask > 0] = 0
            mask[border > 0] = 2

            only_brain = t1.copy()
            only_brain[mask < 1] = 0
            only_brain[mask == 2] = 0

            write_images(mask/2, folder_output+"/mask")

            print(np.max(mask),np.min(mask),np.max(t1),np.min(t1))
            
            #write_images(mask, folder_output)
            
            masks.append(mask)
            
            #circled = circle_highlighted(t1, np.argmax(mask_channeled, axis = 3), 0.8)


            #for image in mask:
            #    print(np.max(image))

            #write_images(mask_channeled[:,:,:,1], output_image_path)

            finished_files += 1
                
            print("finished " + str(finished_files) + " files out of " + str(num_files) + " files. " +
                  str(round((finished_files*100/num_files), 2)) + "% done.\n\n\n") 

    print("Finished. Loaded " + str(num_files) + " files in ", int((time.time() - start_time)/60), "minutes and ",
          int(((time.time() - start_time) % 60)+1), "seconds.")


   

        
    print("\nSplitting data into training and testing...")

    print("Splitting data into chunks of 64 datapoints...")



    image_sets, mask_sets = seperate_sets(images, masks, set_size)

    start_time = time.time()


    print("\nSaving loaded data in: " + save_folder_path + "...")
    
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    for n, set_ in enumerate(image_sets):
        train_images, val_images, test_images = split_train_val_test(set_)
        
        save_data_set(train_images, save_folder_path, "Train Images "+str(n))
        save_data_set(val_images, save_folder_path, "Val Images "+str(n))
        save_data_set(test_images, save_folder_path, "Test Images "+str(n))

    print("Finished saving images. Proceeding to save masks...")

    for n, set_ in enumerate(mask_sets):
        train_masks, val_masks, test_masks = split_train_val_test(set_)
        
        save_data_set(train_masks, save_folder_path, "Train Masks "+str(n))
        save_data_set(val_masks, save_folder_path, "Val Masks "+str(n))
        save_data_set(test_masks, save_folder_path, "Test Masks "+str(n))

    print("Finished saving masks.")
    print("\nAll data finished saving in", int((time.time() - start_time)/60), "minutes and ",
        int(((time.time() - start_time) % 60)+1), "seconds.")


tiny = 2
mini = 1
main = 0

data = main

if data ==  tiny:
    get_files("/media/jiangl/50EC5AFF0AA889DF/Tiny Database", img_id = "flair",
              save_folder_path = "/media/jiangl/50EC5AFF0AA889DF/Tiny Database/Images and Masks For Tumor Segmentation")

if data ==  mini:
    get_files("/media/jiangl/50EC5AFF0AA889DF/Mini Database", img_id = "flair",
              save_folder_path = "/media/jiangl/50EC5AFF0AA889DF/Mini Database/Images and Masks For Tumor Segmentation")

if data ==  main:
    get_files("C:/Users/JiangQin/Documents/python/ct to tumor identifier project/raw ct files/Brain Seg Data Oasis 2/Actual Data",
              save_folder_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/raw ct files/Brain Seg Data Oasis top view/Actual Data/Images and Masks For Tumor Segmentation")








######brain seg oasis 






