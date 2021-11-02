import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nb
import time
import pandas
import h5py
import skimage.transform
import matplotlib.pyplot as plt
import cv2



output_image_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/image ct  visualizations/Machine Learning 2 models test"

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


def circle_highlighted(reference, binary, color):
    circled = reference.copy()
    binary[binary > 0] = 1
    
    
    for n, image in enumerate(binary):
        contours, _ = cv2.findContours(image.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(circled[n], contours, -1,color, 1)

    

    return circled

def write_images(array, test_folder_path):
    array = array/np.max(array)
    for n,image in enumerate(array):
        ##finds the index of the corresponding file name in the original input path from the resize factor after resampling
        file_name = str(str(n) +'.png')

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


def get_files(path, mask_id, img_id, save_folder_path = None, chunk_size = 64, image_size = 128):
    images = []
    masks = []
    start_time = time.time()


    set_size = int(chunk_size/2)

    num_files = 0
    for path, dirs, files in os.walk(path, topdown=False):
        for file in files:
            if mask_id in file:
                set_path = os.path.join(path, dirr)
                usable = 0
                for file in os.listdir(set_path):
                    #print(file)
                    if mask_id in file:
                        mask_name = file
                        usable += 1
                    if img_id in file:
                        id_name = file
                        usable += 1
                if usable == 2:
                    num_files += 1

    num_sets = int(num_files/2)

    print("Total files to be loaded: " + str(num_files) + "\n")
    
    finished_files = 0
    #for n, set in enumerate()
    for path, dirs, files in os.walk(path, topdown=False):
        for dirr in dirs:
            set_path = os.path.join(path, dirr)
            usable = 0
            for file in os.listdir(set_path):
                #print(file)
                if mask_id in file:
                    mask_name = file
                    usable += 1
                if img_id in file:
                    id_name = file
                    usable += 1
            if usable == 2:
                img = nb.load(os.path.join(set_path, id_name))


                img_array = img.get_fdata()

                img_array = img_array/np.max(img_array)

                img_array = np.rot90(img_array, axes = (2,0))
                t1 = img_array/np.max(img_array)





                mask = nb.load(os.path.join(set_path, mask_name))




                mask_array = mask.get_fdata()

                mask_array = mask_array/np.max(mask_array)

                mask_array = np.rot90(mask_array, axes = (2,0))




                circled = circle_highlighted(t1, mask_array, 1)

                write_images(circled, output_image_path)

                finished_files += 1

                
                print("finished " + str(finished_files) + " files out of " + str(num_files) + " files. " +
                      str(round((finished_files*100/num_files), 2)) + "% done.") 

    print("Finished. Loaded " + str(num_files) + " files in ", int((time.time() - start_time)/60), "minutes and ",
          int(((time.time() - start_time) % 60)+1), "seconds.")


   

        
    print("\nSplitting data into training and testing...")

    total = len(images)
    train_end_val_beginning = round(0.7 * total)
    val_end_test_beginning = round(0.85 * total)


    train_images = images[:train_end_val_beginning]
    val_images = images[train_end_val_beginning:val_end_test_beginning]
    test_images = images[val_end_test_beginning:]

    train_masks = masks[:train_end_val_beginning]
    val_masks = masks[train_end_val_beginning:val_end_test_beginning]
    test_masks = masks[val_end_test_beginning:]

    print("Splitting data into chunks of 64 datapoints...")



    image_sets, mask_sets = seperate_sets(images, masks, set_size)

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

if data ==  tiny:
    get_files("/media/jiangl/50EC5AFF0AA889DF/Tiny Database", img_id = "flair",
              save_folder_path = "/media/jiangl/50EC5AFF0AA889DF/Tiny Database/Images and Masks For Tumor Segmentation")

if data ==  mini:
    get_files("/media/jiangl/50EC5AFF0AA889DF/Mini Database", img_id = "flair",
              save_folder_path = "/media/jiangl/50EC5AFF0AA889DF/Mini Database/Images and Masks For Tumor Segmentation")

if data ==  main:
    get_files("C:/Users/JiangQin/Documents/python/ct to tumor identifier project/raw ct files/MICCAI_BraTS_2019_Data_Training", img_id = "flair", mask_id = "seg",
              save_folder_path = "C:/Users/JiangQin/Documents/python/ct to tumor identifier project/raw ct files/MICCAI_BraTS_2019_Data_Training/Images and Masks For Tumor Segmentation")

