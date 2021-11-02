import nibabel as nb
import numpy as np
import cv2
import os
from skimage.measure import marching_cubes_lewiner as marching_cubes
import stl
from stl import mesh
import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nb
import time
start_time = time.time()

import pandas
import h5py
import skimage.transform
import pydicom as dicom
import scipy


t2_path = "/home/jiangl/Downloads/OAS30001_MR_d0129 (3)/anat4/NIFTI/sub-OAS30001_ses-d0129_T2w.nii.gz"
orig_t1_path = "/home/jiangl/Downloads/OAS30001_MR_d0129 (3)/OAS30001_Freesurfer53_d0129/DATA/OAS30001_MR_d0129/mri/T1.mgz"
ref_t1_path =  "/home/jiangl/Downloads/TheoJiang-20200430_233308/OAS30001_MR_d0129/anat3/NIFTI/sub-OAS30001_ses-d0129_run-02_T1w.nii.gz"
mask_path = "/home/jiangl/Downloads/OAS30001_MR_d0129 (3)/OAS30001_Freesurfer53_d0129/DATA/OAS30001_MR_d0129/mri/aparc+aseg.mgz"


output_image_path = "/home/jiangl/Documents/python/ct to tumor identifier project/image ct  visualizations/Machine Learning 2 models test"
output_image_path_Seg = "/home/jiangl/Documents/python/ct to tumor identifier project/image ct  visualizations/brain 1"

def write_images(array, test_folder_path):
    array = array/np.max(array)
    for n,image in enumerate(array):
        ##finds the index of the corresponding file name in the original input path from the resize factor after resampling
        file_name = str(str(n) +'.png')

        cv2.imwrite(os.path.join(test_folder_path, file_name), image*255)
def locate_bounds(array):
    left = array.shape[2]
    right = 0 
    low = array.shape[1]
    high = 0
    shallow = array.shape[0]
    deep = 0

    array_3d = array.copy()
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



def translate_2d(array, empty, translation):
    original_array = array.copy()

    array_translated = empty.copy()
    array_translated[:] = 0
    for y,line in enumerate(original_array):
        for x,pixel in enumerate(line):
            if pixel > 0.1:
                array_translated[y+translation[0]][x+translation[1]] = pixel

    return array_translated


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
    


def flip(array):
    flipped = array.copy()
    for n,image in enumerate(array):
        flipped[-n] = image
    return flipped



def circle_highlighted(reference, binary, color):
    circled = reference.copy()
    binary[binary > 0] = 1
    
    
    for n, image in enumerate(binary):
        contours, _ = cv2.findContours(image.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(circled[n], contours, -1,color, 1)

    

    return circled


def combine_zeros(array1,array2):
    masked = array1.copy()
    
    binary = array2.copy()
    binary[:] = 255
    binary[array2 < 0.1] = 0
    binary[array2 < 0.1] = 0

    masked[binary == 0] = 0
    
    return masked

def generate_stl(array_3d, stl_file_path, name, stl_resolution):
    print('Generating mesh...')
    ##transposes the image to be the correct shape because np arrays are technically flipped
    transposed = array_3d.copy()

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


def biggest_island(input_array):
    masked = input_array.copy()
    
    touching_structure_3d =[[[0,0,0],
                             [0,255,0],
                             [0,0,0]],

                            [[0,255,0],
                             [255,255,255],
                             [0,255,0]],

                            [[0,0,0],
                             [0,255,0],
                             [0,0,0]]]

    binary = input_array.copy()
    binary[:] = 0
    binary[input_array.copy() > 0] = 255

    ##uses a label to find the largest object in the 3d array and only keeps that (if you are trying to highlight something like bones, that have multiple parts, this method may not be suitable)
    markers, num_features = scipy.ndimage.measurements.label(binary,touching_structure_3d)
    binc = np.bincount(markers.ravel())
    binc[0] = 0
    noise_idx = np.where(binc != np.max(binc))
    mask = np.isin(markers, noise_idx)
    binary[mask] = 0

    masked[binary == 0] = 0
    
    return masked


def binarize(array, min_):
    #array = array/np.max(array)
    binary = array.copy()
    binary[array < min_] = 0
    binary[array >= min_] = 1

    return binary

#T1

image_size = 128



mask = nb.load(orig_t1_path)




mask_array = mask.get_fdata()

print("t1 shape: ", mask_array.shape)


mask_array = mask_array/np.max(mask_array)

mask_array = np.rot90(mask_array, axes = (2,0))
mask_array = np.rot90(mask_array, axes = (1,0))



t1 = flip(mask_array)

t1 = binarize(t1,0.05)

t1 = biggest_island(t1)


bounds = locate_bounds(t1)



[left,right,low,high,shallow,deep] = bounds

t1_bounds = bounds


x_size = abs(left-right)
y_size = abs(low-high)
z_size = abs(shallow-deep)


print("t1 sizes: ", [x_size, y_size, z_size])




######################################################################################################################
#T2


img = nb.load(ref_t1_path)


img_array = img.get_fdata()





img_array = img_array/np.max(img_array)
img_array = np.rot90(img_array, axes = (2,0))
img_array = np.rot90(img_array, axes = (1,2))
img_array = np.rot90(img_array, axes = (1,2))
#img_array = flip(img_array)

orig_t2 = img_array/np.max(img_array)

t2 = binarize(orig_t2,0.05)

t2 = biggest_island(t2)


t2_rescaled = t1.copy()
t2_rescaled[:] = 0

for z,Slice in enumerate(t2):
    for y,line in enumerate(Slice):
        for x,pixel in enumerate(line):
            if pixel > 0:
                t2_rescaled[z][y][x] = pixel

print("t2 shape: ", t2_rescaled.shape)

bounds = locate_bounds(t2_rescaled)



[left,right,low,high,shallow,deep] = bounds



t2_bounds = bounds


x_size = abs(left-right)
y_size = abs(low-high)
z_size = abs(shallow-deep)

print("t2 sizes: ", [x_size, y_size, z_size])








################################################################################3











t2_x_center = (t2_bounds[0]+t2_bounds[1])/2
t2_y_center = (t2_bounds[2]+t2_bounds[3])/2
t2_z_center = (t2_bounds[4]+t2_bounds[5])/2


t1_x_center = (t1_bounds[0]+t1_bounds[1])/2
t1_y_center = (t1_bounds[2]+t1_bounds[3])/2
t1_z_center = (t1_bounds[4]+t1_bounds[5])/2

print(t2_x_center,t2_y_center,t2_z_center)
print(t1_x_center,t1_y_center,t1_z_center)

x_translate = int(t1_x_center-t2_x_center)
y_translate = int(t1_y_center-t2_y_center)
z_translate = int(t1_z_center-t2_z_center)

print([z_translate,y_translate,x_translate])

translated_t2 = translate_3d(t2_rescaled, [0,y_translate,0])








########################################################################################################3
#Mask

mask = nb.load(mask_path)




mask_array = mask.get_fdata()

print("t1 shape: ", mask_array.shape)


mask_array = mask_array/np.max(mask_array)

mask_array = np.rot90(mask_array, axes = (2,0))
mask_array = np.rot90(mask_array, axes = (1,0))



mask = flip(mask_array)





circled = circle_highlighted(translated_t2, mask, 0.8)

write_images(t2, output_image_path)
write_images(t1, output_image_path_Seg)

#generate_stl(mask.T, "/home/jiangl/Documents/python/ct to tumor identifier project/3d stl ct visualizations", "brain new.stl", 1)
print ('Finished in', int((time.time() - start_time)/60), 'minutes and ', int((time.time() - start_time) % 60), 'seconds.')
