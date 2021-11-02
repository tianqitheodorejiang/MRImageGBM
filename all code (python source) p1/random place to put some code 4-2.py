
    blurred = blur(original_array, 3)

    binary = binarize_blurred(original_array, 0.1, 2)


    white = binarize(original_array, 0.25)

    brain.resampled_array = original_array


    brain_seg_array_precise = original_array.copy()
    brain_seg_array_area = original_array.copy()

    tumor_threshed_array = original_array.copy()
    tumor_seg_array = original_array.copy()




    print("median: ", find_median_grayscale(original_array))
    print("max: ", np.max(original_array))


    brain_seg = keras.models.load_model(brain_seg_model_area)

    brain_mask1 = brain_seg.predict(original_array)

    binary_brain_area = binarize(brain_mask1, 0.3)

    brain_seg_area_final = original_array.copy()

    brain_seg_area_final[binary_brain_area == 0] = 0



    median = find_median_grayscale(brain_seg_area_final)




    brain_seg = keras.models.load_model(brain_seg_model_precise)

    print("divide value:", median/0.05)

    new_array = original_array/(median/0.05)

    brain_mask2 = brain_seg.predict(new_array)

    binary_brain_area2 = binarize(brain_mask2, 0.8)

    binary_brain_area = binarize(brain_mask2, 0.8)


    brain_seged = original_array.copy()


    brain_seged[binary_brain_area2 == 0] *=2



    brain_seg = keras.models.load_model(brain_seg_model_area)

    new_array = original_array/(median/0.2)

    brain_mask1 = brain_seg.predict(new_array)

    binary_brain_area1 = binarize(brain_mask1, 0.2)

    binary_brain_area1 = combine_zeros(binary_brain_area1, binary_brain_area2)


    threshed = binarize(original_array, 0.05)

    binary_brain_area1 = combine_zeros(binary_brain_area1, threshed)


    #binary_brain_area1 = branch(binary_brain_area1, binary_brain_area, 2)

    binary_brain_area1 = kill_small_islands(binary_brain_area1, 2)


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
    prob = ext.run(tumorless_array) 

    # mask can be obtained as:
    mask = prob > 0.1

    mask = np.stack([np.stack([mask], axis = 3)])

    final_mask = combine_white_binary(mask, brain_mask_precise_rescaled)

    print(mask.shape)

    circled = circle_highlighted(original_unscaled_array, final_mask, 0.8)

    print(circled.shape)

    path = output_image_path + "/" + str(index)

    if not os.path.exists(path):
        os.mkdir(path)

    print(path)
