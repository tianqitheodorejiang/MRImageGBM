import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import time
import pandas
import h5py
import skimage.transform
import pydicom as dicom
import scipy
import cv2
import random


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

output_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/image ct  visualizations/Machine Learning 2 models test/tumor border"

def write_images(array, test_folder_path):
    if not os.path.exists(test_folder_path):
        os.makedirs(test_folder_path)
    for n,image in enumerate(array):
        file_name = str(str(n) +'.png')
        cv2.imwrite(os.path.join(test_folder_path, file_name), image*255)


  
def get_files(path, img_id = "original array", mask_id = "rough brain", save_folder_path = None, chunk_size = 64, image_size = 128, bad_indexes = []):
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
    for n,set_ in enumerate(os.listdir(path)):
        try:
            if int(set_) not in bad_indexes:
                for i in range(0,4):
                    set_path = os.path.join(path, set_)
                    image_path = os.path.join(set_path, "image.h5")
                    h5f = h5py.File(image_path,'r')
                    image = h5f['all_data'][:]
                    h5f.close()

                    image[:,:,:,0] *= (random.randint(6,11)/10)

                    print(image.shape)

                    images.append(image)

                    finished_files += 1
                    print("finished " + str(finished_files) + " files out of " + str(num_files) + " files. " +
                          str(round((finished_files*100/num_files), 2)) + "% done.") 
                    

                    masks_path = os.path.join(set_path, "mask.h5")
                    h5f = h5py.File(masks_path,'r')
                    mask = h5f['all_data'][:]
                    h5f.close()

                    mask[mask > 0] = 1
                    dilated_mask = dilate_up_binary(mask, 5)

                    border = dilated_mask.copy()
                    border[mask > 0] = 0
                    mask[border > 0] = 2

                    folder_output = output_path + "/" + str(n)
                    
                    write_images(mask/2, folder_output)

                    print(mask.shape)
                   
                    masks.append(mask)

                    finished_files += 1
                
                    print("finished " + str(finished_files) + " files out of " + str(num_files) + " files. " +
                          str(round((finished_files*100/num_files), 2)) + "% done.")
        except Exception as e:
            print(e)
            pass

    print("Finished. Loaded " + str(num_files) + " files in ", int((time.time() - start_time)/60), "minutes and ",
          int(((time.time() - start_time) % 60)+1), "seconds.")


    new_indexes = []
    for i in range(0,len(images)):
        index = random.randint(0,len(images)-1)
        while index in new_indexes:
            index = random.randint(0,len(images)-1)
        new_indexes.append(index)

    print(new_indexes)
    print(len(images))

    new_images = []
    new_masks = []
    for index in new_indexes:
        print(index)
        new_images.append(images[index])
        new_masks.append(masks[index])

        
    print("\nSplitting data into training and testing...")

    print("Splitting data into chunks of 64 datapoints...")



    image_sets, mask_sets = seperate_sets(new_images, new_masks, set_size)   

        

    start_time = time.time()


    print("\nSaving loaded data in: " + save_folder_path + "...")
    
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

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
    get_files("C:/Users/JiangQin/Documents/python/ct to tumor identifier project/raw ct files/Brain-Tumor-Progression/loaded arrays for growth prediction",
              save_folder_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/raw ct files/Brain-Tumor-Progression/Images and Masks For Tumor Segmentation")
















