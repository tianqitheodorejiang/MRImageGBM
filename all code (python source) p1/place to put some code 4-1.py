write_images(binary_brain_area2, output_image_path_Seg)

mask_upscaled = skimage.transform.rescale(np.squeeze(binary_brain_area1.copy()[0], axis = 3), (backscale_factor, backscale_factor, backscale_factor))

mask_unscaled = blank_unscaled_array.copy()

for z,Slice in enumerate(mask_upscaled):
    for y,line in enumerate(Slice):
        for x,pixel in enumerate(line):
            try:
                mask_unscaled[z][y][x] = pixel
            except:
                pass



brain_mask_precise_rescaled = np.stack([np.stack([mask_unscaled], axis = 3)])

mask_upscaled = skimage.transform.rescale(np.squeeze(binary_brain_area2.copy()[0], axis = 3), (backscale_factor, backscale_factor, backscale_factor))

mask_unscaled = blank_unscaled_array.copy()

for z,Slice in enumerate(mask_upscaled):
    for y,line in enumerate(Slice):
        for x,pixel in enumerate(line):
            try:
                mask_unscaled[z][y][x] = pixel
            except:
                pass



brain_mask_area_rescaled = np.stack([np.stack([mask_unscaled], axis = 3)])



tumorless_array = original_unscaled_array.copy()

tumorless_array[brain_mask_area_rescaled == 0] *= 0.5


tumorless_array = np.squeeze(tumorless_array.copy()[0], axis = 3)

import nibabel as nb
from deepbrain import Extractor

# Load a nifti as 3d numpy image [H, W, D]

ext = Extractor()

# `prob` will be a 3d numpy image containing probability 
# of being brain tissue for each of the voxels in `img`
prob = ext.run(np.squeeze(original_unscaled_array.copy()[0], axis = 3)) 

# mask can be obtained as:
mask = prob > 0.5

mask = np.stack([np.stack([mask], axis = 3)])

final_mask = combine_white_binary(mask, brain_mask_precise_rescaled)

print(mask.shape)



