import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import time
import pandas
import h5py


def get_files(path, img_id = "t2", mask_id = "seg", save_folder_path = None):
    images = []
    masks = []
    start_time = time.time()

    num_files = 0
    for path, dirs, files in os.walk(path, topdown=False):
        for file in files:
            if img_id in file or mask_id in file:
                num_files += 1

    print("Total files to be loaded: " + str(num_files) + "\n")
    finished_files = 0

    for path, dirs, files in os.walk(path, topdown=False):
        for file in files:
            if img_id in file:
                image = nib.load(os.path.join(path, file))
                image_data = image.get_data().T
                image_data = image_data/np.max(image_data)
                images.append(image_data)

                finished_files += 1
                print("finished " + str(finished_files) + " files out of " + str(num_files) + " files. " +
                      str(round((finished_files*100/num_files), 2)) + "% done.") 
                
            elif mask_id in file:
                mask = nib.load(os.path.join(path, file))
                mask_data = mask.get_data().T
                binary = mask_data.copy()
                binary[:] = 0
                binary[mask_data < 4] = 1
                binary[mask_data == 0] = 0

                
                masks.append(binary)

                finished_files += 1
            
                print("finished " + str(finished_files) + " files out of " + str(num_files) + " files. " +
                      str(round((finished_files*100/num_files), 2)) + "% done.") 

    print("Finished. Loaded " + str(num_files) + " files in ", int((time.time() - start_time)/60), "minutes and ",
          int(((time.time() - start_time) % 60)+1), "seconds.")

    print("Splitting data into training and testing...")

    total = len(images)
    sliced = round(0.7 * total)

    train_images = images[:sliced]
    test_images = images[sliced:]

    train_masks = masks[:sliced]
    test_masks = masks[sliced:]

    start_time = time.time()

    if save_folder_path:
        print("\nSaving loaded data in: " + save_folder_path + "...")
        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)

        if os.path.exists(os.path.join(save_folder_path, "Train Images.h5")):
            os.remove(os.path.join(save_folder_path, "Train Images.h5"))

        hdf5_store = h5py.File(os.path.join(save_folder_path, "Train Images.h5"), "a")
        hdf5_store.create_dataset("all_data", data = train_images, compression="gzip")

        if os.path.exists(os.path.join(save_folder_path, "Test Images.h5")):
            os.remove(os.path.join(save_folder_path, "Test Images.h5"))

        hdf5_store = h5py.File(os.path.join(save_folder_path, "Test Images.h5"), "a")
        hdf5_store.create_dataset("all_data", data = test_images, compression="gzip")

        print("Finished saving images. Proceeding to save masks...")


        if os.path.exists(os.path.join(save_folder_path, "Train Masks.h5")):
            os.remove(os.path.join(save_folder_path, "Train Masks.h5"))
            
        hdf5_store = h5py.File(os.path.join(save_folder_path, "Train Masks.h5"), "a")
        hdf5_store.create_dataset("all_data", data = train_masks, compression="gzip")

        if os.path.exists(os.path.join(save_folder_path, "Test Masks.h5")):
            os.remove(os.path.join(save_folder_path, "Test Masks.h5"))
            
        hdf5_store = h5py.File(os.path.join(save_folder_path, "Test Masks.h5"), "a")
        hdf5_store.create_dataset("all_data", data = test_masks, compression="gzip")

        print("Finished saving masks.")
        print("\nAll data finished saving in", int((time.time() - start_time)/60), "minutes and ",
              int(((time.time() - start_time) % 60)+1), "seconds.")

    return images, masks
    





get_files("/media/jiangl/50EC5AFF0AA889DF/MICCAI_BraTS_2019_Data_Training", img_id = "flair",
          save_folder_path = "/media/jiangl/50EC5AFF0AA889DF/MICCAI_BraTS_2019_Data_Training/Images and Masks For Tumor Segmentation")

