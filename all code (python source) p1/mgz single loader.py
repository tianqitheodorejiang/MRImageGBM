import nibabel as nb
import numpy as np
import cv2
import os
from skimage.measure import marching_cubes_lewiner as marching_cubes
import stl

opath = "/home/jiangl/Downloads/TheoJiang-20200419_143524/OAS30003_MR_d3731/OAS30003_Freesurfer53_d3731/DATA/OAS30003_MR_d3731/mri/T1.mgz"
spath = "/home/jiangl/Downloads/TheoJiang-20200419_143524/OAS30003_MR_d3731/OAS30003_Freesurfer53_d3731/DATA/OAS30003_MR_d3731/mri/brain.mgz"
output_image_path = "/home/jiangl/Documents/python/ct to tumor identifier project/image ct  visualizations/Machine Learning 2 models test"

def write_images(array, test_folder_path):
    array = array/np.max(array)
    for n,image in enumerate(array):
        ##finds the index of the corresponding file name in the original input path from the resize factor after resampling
        file_name = str(str(n) +'.png')

        cv2.imwrite(os.path.join(test_folder_path, file_name), image*255)

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


def generate_stl(array_3d, stl_file_path, name, stl_resolution):
    print('Generating mesh...')
    verts, faces, norm, val = marching_cubes(array_3d, 0.01, step_size = stl_resolution, allow_degenerate=True)

    mesh = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
    print('Vertices obatined:', len(verts))
    print('')
    for i, f in enumerate(faces):
        for j in range(3):
            mesh.vectors[i][j] = verts[f[j],:]
    path = stl_file_path + '/' + name
    mesh.save(path)


def get_files(path, index, mask = "aseg", orig = "T1"):
    images = []
    masks = []
    
    for path, dirs, files in os.walk(path, topdown=False):
        for file in files:
            if mask in file and file.endswith("mgz"):
                masks.append(os.path.join(path, file))
            elif orig in file and file.endswith("mgz"):
                images.append(os.path.join(path, file))
                

    img = nb.load(images[index])

    array = img.get_fdata()

    array = array/np.max(array)

    array = np.rot90(array, axes = (2,0))
    array = np.rot90(array, axes = (1,0))
    original_array = flip(array)


    img = nb.load(masks[index])


    array = img.get_fdata()

    array[array>0] = 1

    array = np.rot90(array, axes = (2,0))
    array = np.rot90(array, axes = (1,0))



    mask = flip(array)
    return original_array, mask

array, mask = get_files("/home/jiangl/Downloads/TheoJiang-20200419_143524/OAS30003_MR_d3731", 0)

circled = circle_highlighted(array, mask, 1)

write_images(circled, output_image_path)
generate_stl(mask, "/home/jiangl/Documents/python/ct to tumor identifier project/3d stl ct visualizations", 'brain1.stl', 1)

print(array.shape)
