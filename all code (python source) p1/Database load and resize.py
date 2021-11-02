import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import time
import pandas
import h5py
import skimage.transform


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


def save_data_set(sets, save_path, save_name):
    if os.path.exists(os.path.join(save_path, save_name)+".h5"):
        os.remove(os.path.join(save_path, save_name)+".h5")

    hdf5_store = h5py.File(os.path.join(save_path, save_name)+".h5", "a")
    hdf5_store.create_dataset("all_data", data = sets, compression="gzip")


def get_files(path, img_id = "t2", mask_id = "seg", save_folder_path = None, chunk_size = 64, image_size = 128):
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
    for path, dirs, files in os.walk(path, topdown=False):
        for file in files:
            if img_id in file:
                image = nib.load(os.path.join(path, file))
                image_data = image.get_data().T
                image_data = image_data/255
     
                z_zoom = image_size/image_data.shape[0]
                y_zoom = image_size/image_data.shape[1]
                x_zoom = image_size/image_data.shape[2]

                image_data = skimage.transform.rescale(image_data, (z_zoom, y_zoom, x_zoom))

                   
                images.append(image_data)

                finished_files += 1
                print("finished " + str(finished_files) + " files out of " + str(num_files) + " files. " +
                      str(round((finished_files*100/num_files), 2)) + "% done.") 
                
            elif mask_id in file:
                mask = nib.load(os.path.join(path, file))
                mask_data = mask.get_data().T
                #mask_data = mask_data/255

                               
                binary = mask_data.copy()
                binary[:] = 0
                binary[mask_data  < 2] = 255
                binary[mask_data == 0] = 0

                binary = binary/255



                z_zoom = image_size/mask_data.shape[0]
                y_zoom = image_size/mask_data.shape[1]
                x_zoom = image_size/mask_data.shape[2]

                mask_data = skimage.transform.rescale(binary, (z_zoom, y_zoom, x_zoom))


                binary = mask_data.copy()
                binary[:] = 0
                binary[mask_data > 0] = 1
           
                masks.append(binary)

                finished_files += 1
            
                print("finished " + str(finished_files) + " files out of " + str(num_files) + " files. " +
                      str(round((finished_files*100/num_files), 2)) + "% done.") 

    print("Finished. Loaded " + str(num_files) + " files in ", int((time.time() - start_time)/60), "minutes and ",
          int(((time.time() - start_time) % 60)+1), "seconds.")



    print("Splitting data into chunks of 64 datapoints...")

   

        
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


    start_time = time.time()

    if save_folder_path:
        
        print("\nSaving loaded data in: " + save_folder_path + "...")
        
        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)

        save_data_set(train_images, save_folder_path, "Train Images")
        save_data_set(val_images, save_folder_path, "Val Images")
        save_data_set(test_images, save_folder_path, "Test Images")

        print("Finished saving images. Proceeding to save masks...")
            
        save_data_set(train_masks, save_folder_path, "Train Masks")
        save_data_set(val_masks, save_folder_path, "Val Masks")
        save_data_set(test_masks, save_folder_path, "Test Masks")


        print("Finished saving masks.")
        print("\nAll data finished saving in", int((time.time() - start_time)/60), "minutes and ",
              int(((time.time() - start_time) % 60)+1), "seconds.")

    return images, masks



mini = 1
main = 0

data = mini

if data ==  mini:
    get_files("/media/jiangl/50EC5AFF0AA889DF/Mini Database", img_id = "flair",
              save_folder_path = "/media/jiangl/50EC5AFF0AA889DF/Mini Database/Images and Masks For Tumor Segmentation")

if data ==  main:
    get_files("/media/jiangl/50EC5AFF0AA889DF/MICCAI_BraTS_2019_Data_Training", img_id = "flair",
              save_folder_path = "/media/jiangl/50EC5AFF0AA889DF/MICCAI_BraTS_2019_Data_Training/Images and Masks For Tumor Segmentation")

