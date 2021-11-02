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
                if pixel > 0.05:
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
    for SET in os.listdir(path):
        if "OAS" in SET:
            set_path = os.path.join(path, SET)


            img = nb.load(os.path.join(set_path, "T1.mgz"))

            img_array = img.get_fdata()

            img_array = img_array/np.max(img_array)

            img_array = np.rot90(img_array, axes = (2,0))
            img_array = np.rot90(img_array, axes = (1,0))
            image_data = flip(img_array)

            image_data = image_data/np.max(image_data)

            blank_unscaled_array = image_data.copy()

            blank_unscaled_array[:] = 0




            z_zoom = image_size/image_data.shape[0]
            y_zoom = image_size/image_data.shape[1]
            x_zoom = image_size/image_data.shape[2]

            image_data1 = skimage.transform.rescale(image_data, (z_zoom, y_zoom, x_zoom))

            original_array1 = image_data1
            original_array1[:] = 0






            image_data = np.stack([np.stack([image_data], axis = 3)])

            original_unscaled_array = image_data.copy()


            bounds = locate_bounds(image_data)



            [left,right,low,high,shallow,deep] = bounds


            x_size = abs(left-right)
            y_size = abs(low-high)
            z_size = abs(shallow-deep)

            max_size = np.max([x_size, y_size, z_size])

            
            rescale_factor = (image_size*0.8)/max_size

            print("\nrescale factor:", rescale_factor)

            backscale_factor = 1/rescale_factor


            image_data = skimage.transform.rescale(np.squeeze(image_data.copy()[0], axis = 3), (rescale_factor, rescale_factor, rescale_factor))

            original_scaled_down = image_data.copy()


            for z,Slice in enumerate(image_data):
                for y,line in enumerate(Slice):
                    for x,pixel in enumerate(line):
                        original_array1[z][y][x] = pixel
            original_image_array = original_array1.copy()

            
            

            finished_files += 1
                
            print("finished " + str(finished_files) + " files out of " + str(num_files) + " files. " +
                  str(round((finished_files*100/num_files), 2)) + "% done.") 


            mask = nb.load(os.path.join(set_path, "aparc+aseg.mgz"))


            mask_array = mask.get_fdata()

            mask_array[mask_array>0] = 1

            mask_array = np.rot90(mask_array, axes = (2,0))
            mask_array = np.rot90(mask_array, axes = (1,0))



            image_data = flip(mask_array)

            
            
            z_zoom = image_size/image_data.shape[0]
            y_zoom = image_size/image_data.shape[1]
            x_zoom = image_size/image_data.shape[2]

            image_data1 = skimage.transform.rescale(image_data, (z_zoom, y_zoom, x_zoom))

            original_array1 = image_data1
            original_array1[:] = 0
            
            image_data = skimage.transform.rescale(image_data.copy(), (rescale_factor, rescale_factor, rescale_factor))

            original_scaled_down = image_data.copy()


            for z,Slice in enumerate(image_data):
                for y,line in enumerate(Slice):
                    for x,pixel in enumerate(line):
                        original_array1[z][y][x] = pixel
            mask = original_array1.copy()



            t1, mask = cut_neck(original_image_array, mask)
            
            images.append(t1)

            
            masks.append(mask)
            


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
    get_files("/home/jiangl/Documents/python/ct to tumor identifier project/raw ct files/Brain Seg Data Oasis 2/Actual Data",
              save_folder_path = "/home/jiangl/Documents/python/ct to tumor identifier project/raw ct files/Brain Seg Data Oasis/Actual Data/Images and Masks For Tumor Segmentation")















