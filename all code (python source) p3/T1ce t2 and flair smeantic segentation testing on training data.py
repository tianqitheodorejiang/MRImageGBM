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

output_image_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/image ct  visualizations/Machine Learning 2 models test"

def write_images(array, test_folder_path):
    array = array/np.max(array)
    if not os.path.exists(test_folder_path):
        os.makedirs(test_folder_path)
    for n,image in enumerate(array):
        ##finds the index of the corresponding file name in the original input path from the resize factor after resampling
        file_name = str(str(n) +'.png')

        cv2.imwrite(os.path.join(test_folder_path, file_name), image*255)



def circle_highlighted(reference, binary, color):
    circled = reference.copy()
    binary[binary > 0] = 1
    
    
    for n, image in enumerate(binary):
        contours, _ = cv2.findContours(image.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(circled[n], contours, -1,color, 1)

    

    return circled
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

        

        return circled
 

    def write_test_images(self, array_3d, test_folder_path):
        array_3d = array_3d/np.max(array_3d)
        print(np.max(array_3d))
        for n,image in enumerate(array_3d):
            ##finds the index of the corresponding file name in the original input path from the resize factor after resampling
            file_name = str(str(n) +'.png')

            ##writes the resulting image as a png in the test_folder_path

            cv2.imwrite(os.path.join(test_folder_path, file_name), image*255)




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


def save_array(array, path):
    if os.path.exists(path):
        os.remove(path)

    hdf5_store = h5py.File(path, "a")
    hdf5_store.create_dataset("all_data", data = array, compression="gzip")



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

def dilate_up_binary(array, size):
    binary = array.copy()

    binary[:] = 0
    binary[array > 0] = 255




    ##creates a kernel which is a 3 by 3 square of ones as the main kernel for all denoising
    kernel = scipy.ndimage.generate_binary_structure(3, 1)

    ##erodes away the white areas of the 3d array to seperate the loose parts
    blew_up = scipy.ndimage.binary_dilation(binary.astype('uint8'), kernel, iterations=size)


    return blew_up > 0

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

def binarize(array, min_):
    #array = array/np.max(array)
    binary = array.copy()
    binary[array < min_] = 0
    binary[array >= min_] = 1

    return binary

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


semantic_model_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/code files/Saved models/Tumor seg semantic 64x with t1ce t2 flair/Model 81.h5"
binary_model_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/code files/Saved models/Tumor seg binary with t1ce t2 flair/Model 16.h5"

def iou(y_true, y_pred): 
    # Compute weights: "the contribution of each label is corrected by the inverse of its volume"
    Ncl = y_pred.shape[-1]
    print("NCL:",Ncl)
    score = 0
    smooth = 0.000001
    for l in range(0,Ncl):
        if np.max(y_true[:,:,:,l])>0:
            print(np.sum(y_true[:,:,:,l]+y_pred[:,:,:,l]))
            ist = y_true[:,:,:,l]+y_pred[:,:,:,l]
            ist[ist>1] = 1
            score+=np.sum(y_true[:,:,:,l]*y_pred[:,:,:,l])/(np.sum(ist)+smooth)
        else:
            print("\n\n\nBRUUUUU")
            Ncl-=1
    score /=Ncl
    
    return score

def dice(y_true, y_pred): 
    # Compute weights: "the contribution of each label is corrected by the inverse of its volume"
    Ncl = y_pred.shape[-1]
    print("NCL:",Ncl)
    score = 0
    smooth = 0.000001
    for l in range(0,Ncl):
        if np.max(y_true[:,:,:,l])>0:
            print(np.sum(y_true[:,:,:,l]+y_pred[:,:,:,l]))
            score+=2*np.sum(y_true[:,:,:,l]*y_pred[:,:,:,l])/(np.sum(y_true[:,:,:,l]+y_pred[:,:,:,l])+smooth)
        else:
            print("\n\n\nBRUUUUU")
            Ncl-=1
    score /=Ncl
    
    return score

binary_model_path2="C:/Users/JiangQin/Documents/c++/build-MRImage3D-Desktop_Qt_5_15_0_MSVC2015_64bit-Debug/models/flair_tumor.h5"

def find_median_grayscale(array):    
    zero_pixels = float(np.count_nonzero(array==0))
    single_dimensional = array.flatten().tolist()
    single_dimensional.extend(np.full((1, int(zero_pixels)), 1000).flatten().tolist())

    return np.median(single_dimensional)

def get_files(path, img_id = "original array", mask_id = "rough brain", save_folder_path = None, chunk_size = 64, image_size =64, bad_indexes = []):
    dices = 0
    tot = 0
    images = []
    masks = []
    start_time = time.time()


    set_size = int(chunk_size/2)

    num_files = 0
    for set_ in os.listdir(path):
        try:
            if int(set_) not in bad_indexes:
                num_files += 2
        except:
            pass

    num_sets = int(num_files/2)

    print("Total files to be loaded: " + str(num_files) + "\n")
    
    finished_files = 0
    #for n, set in enumerate()
    sets = []
    for set_ in os.listdir(path):
        sets.append(set_)
    print(sets)

    new_indexes = []
    for i in range(0,len(sets)):
        index = random.randint(0,len(sets)-1)
        while index in new_indexes:
            index = random.randint(0,len(sets)-1)
        new_indexes.append(index)

    print(new_indexes)

    new_sets = []
    for index in new_indexes:
        print(index)
        new_sets.append(sets[index])

    sets = new_sets

    print(sets)
    for j in range(0,int(num_files/chunk_size)+1):
        images = []
        masks = []
        for i in range(set_size*j,set_size*(j+1)):
            try:
                set_path = os.path.join(path, sets[i])
                image_path = os.path.join(set_path, "original array.h5")
                h5f = h5py.File(image_path,'r')
                image = h5f['all_data'][:]
                h5f.close()

                flair = image[:,:,:,0]
                t1 = image[:,:,:,1]
                t2 = image[:,:,:,2]

                t1 /=(find_median_grayscale(t1)/0.35)
                t2 /=(find_median_grayscale(t2)/0.35)
                flair /=(find_median_grayscale(flair)/0.35)

                
                only_brain = np.stack([flair,t1,t2],axis = -1)

                median_flair = find_median_grayscale(flair)

                tumor_seg_binary = load_model(binary_model_path2)

                tumor_mask = tumor_seg_binary.predict(np.stack([np.stack([flair/(median_flair/0.3)],axis=-1)]))

                tumor_binary = np.squeeze(tumor_mask[0] > 0.5, axis = -1)

                only_brain = skimage.transform.rescale(only_brain, (0.5,0.5,0.5,1))


                tumor_seg_channeled = ConvNetSemantic64(64,64,64)

                tumor_seg_channeled.load_weights(semantic_model_path)

                tumor_mask = tumor_seg_channeled.predict(np.stack([only_brain]))

                tumor_mask = skimage.transform.rescale(tumor_mask[0], (2.0,2.0,2.0,1))
                
                tumor_mask[:,:,:,1:][np.stack([tumor_binary,tumor_binary,tumor_binary],axis=-1)==0] = 0
                

                
     

                finished_files += 1
                print("finished " + str(finished_files) + " files out of " + str(num_files) + " files. " +
                      str(round((finished_files*100/num_files), 2)) + "% done.") 
                

                masks_path = os.path.join(set_path, "rough brain seg.h5")
                h5f = h5py.File(masks_path,'r')
                mask = h5f['all_data'][:]

                mask[mask==4] = 3
                actual_mask = np.stack([mask,mask,mask,mask],axis=-1)
                actual_mask[:] = 0
                actual_mask[:,:,:,0][mask==0] = 1
                actual_mask[:,:,:,1][mask==1] = 1
                actual_mask[:,:,:,2][mask==2] = 1
                actual_mask[:,:,:,3][mask==3] = 1
                dices+=iou(actual_mask,tumor_mask)
                tot+=1

                print("IOU:",dices/tot)


                

                
                print("mask unique:",np.unique(mask))
                print("0:",np.count_nonzero(mask==0),"1:",np.count_nonzero(mask==1),"2:",np.count_nonzero(mask==2),"3:",np.count_nonzero(mask==3),"4:",np.count_nonzero(mask==4))

                finished_files += 1
            
                print("finished " + str(finished_files) + " files out of " + str(num_files) + " files. " +
                      str(round((finished_files*100/num_files), 2)) + "% done.")
            except:
                pass

        print("Finished. Loaded " + str(num_files) + " files in ", int((time.time() - start_time)/60), "minutes and ",
              int(((time.time() - start_time) % 60)+1), "seconds.")


   

            
        print("\nSplitting data into training and testing...")

        print("Splitting data into chunks of 64 datapoints...")




        print("\nSaving loaded data in: " + save_folder_path + "...")
        
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        train_images, val_images, test_images = split_train_val_test(images)
        
        save_data_set(train_images, save_folder_path, "Train Images "+str(j))
        save_data_set(val_images, save_folder_path, "Val Images "+str(j))
        save_data_set(test_images, save_folder_path, "Test Images "+str(j))

        print("Finished saving images. Proceeding to save masks...")

        train_masks, val_masks, test_masks = split_train_val_test(masks)
        
        save_data_set(train_masks, save_folder_path, "Train Masks "+str(j))
        save_data_set(val_masks, save_folder_path, "Val Masks "+str(j))
        save_data_set(test_masks, save_folder_path, "Test Masks "+str(j))

        print("Finished saving masks.")
        print("\nAll data finished saving in", int((time.time() - start_time)/60), "minutes and ",
            int(((time.time() - start_time) % 60)+1), "seconds.")


tiny = 2
mini = 1
main = 0

data = main

good = [0, 2, 6, 8, 9, 10, 11, 13, 14, 16, 18, 20, 21, 22, 23, 24, 25, 27, 28, 29, 31, 32, 33, 36, 37, 38, 40, 42, 45, 48, 50, 51, 52, 55, 56, 58, 59, 60, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 76, 77, 78]
bad = [3, 17, 57, 62, 69]
ok = [1, 4, 5, 7, 12, 15, 19, 30, 34, 35, 39, 41, 43, 44, 46, 47, 49, 53, 54, 61, 75]

print(len(good))
if data ==  tiny:
    get_files("/media/jiangl/50EC5AFF0AA889DF/Tiny Database", img_id = "flair",
              save_folder_path = "/media/jiangl/50EC5AFF0AA889DF/Tiny Database/Images and Masks For Tumor Segmentation")

if data ==  mini:
    get_files("/media/jiangl/50EC5AFF0AA889DF/Mini Database", img_id = "flair",
              save_folder_path = "/media/jiangl/50EC5AFF0AA889DF/Mini Database/Images and Masks For Tumor Segmentation")

if data ==  main:
    get_files("C:/Users/JiangQin/Documents/python/ct to tumor identifier project/raw ct files/flair tumor seg arrays",
              save_folder_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/raw ct files/flair tumor seg arrays/Images and Masks For Tumor Segmentation binary")
















