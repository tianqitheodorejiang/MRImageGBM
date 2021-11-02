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


    
def get_files(path, img_id = "flair", mask_id = "seg", save_folder_path = None, chunk_size = 64, image_size = 128):
    images = []
    masks = []
    start_time = time.time()


    set_size = int(chunk_size/2)

    num_files = 0
    for path, dirs, files in os.walk(path, topdown=False):
        for dir_ in dirs:
            if img_id in dir_.lower() or mask_id in dir_.lower():
                num_files += 1

    num_sets = int(num_files/2)

    print("Total files to be loaded: " + str(num_files) + "\n")
    
    finished_files = 0
    #for n, set in enumerate()
    for path, dirs, files in os.walk(path, topdown=False):
        for dir_ in dirs:
            if img_id in dir_.lower():
                brain = highlight_ct(os.path.join(path, dir_))
                brain.load_scan()
                pixel = brain.generate_pixel_data()
                image_data = brain.resample_array()

                image_data = image_data/np.max(image_data)


                ##calculating the zoom factors and reshaping
                z_zoom = image_size/image_data.shape[0]
                y_zoom = image_size/image_data.shape[1]
                x_zoom = image_size/image_data.shape[2]

                image_data = skimage.transform.rescale(image_data, (z_zoom, y_zoom, x_zoom))

                   
                images.append(image_data)

                finished_files += 1
                print("finished " + str(finished_files) + " files out of " + str(num_files) + " files. " +
                      str(round((finished_files*100/num_files), 2)) + "% done.") 
                
            elif mask_id in dir_.lower():
                brain = highlight_ct(os.path.join(path, dir_))
                brain.load_scan()
                pixel = brain.generate_pixel_data()
                mask_data = brain.resample_array()

                               
                binary = mask_data.copy()
                binary[:] = 0
                binary[mask_data  > 0] = 255

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


   

        
    print("\nSplitting data into training and testing...")

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
    get_files("/home/jiangl/Documents/python/ct to tumor identifier project/raw ct files/QIN-BRAIN-DSC-MRI", img_id = "anatomic", mask_id = "maskbrain",
              save_folder_path = "/home/jiangl/Documents/python/ct to tumor identifier project/raw ct files/QIN-BRAIN-DSC-MRI/Images and Masks For Tumor Segmentation")















