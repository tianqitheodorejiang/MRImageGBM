import SimpleITK as sitk
import numpy as np
import os

outpath = "C:/Users/JiangQin/Documents/data/raw ct files/IvyGAP/W1/10-25-1996-MR BRAIN WITHOUT AND WITH CONTRAST D-1-34997/10.000000-COR T2 blub"

# Load moving image
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames("C:/Users/JiangQin/Documents/data/raw ct files/IvyGAP/W1/10-25-1996-MR BRAIN WITHOUT AND WITH CONTRAST D-1-34997/10.000000-COR T2 FSE-XL FS POST-57816")
reader.SetFileNames(dicom_names)
moving_image = reader.Execute()

# Load fixed image
dicom_names = reader.GetGDCMSeriesFileNames("C:/Users/JiangQin/Documents/data/raw ct files/IvyGAP/W1/10-25-1996-MR BRAIN WITHOUT AND WITH CONTRAST D-1-34997/6.000000-AX T2 FLAIR 3MM-67253")
reader.SetFileNames(dicom_names)
fixed_image = reader.Execute()

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed_image)
resampler.SetOutputSpacing(fixed_image.GetSpacing())
resampler.SetOutputOrigin(fixed_image.GetOrigin())
resampler.SetOutputDirection(fixed_image.GetDirection())
resampler.SetInterpolator(sitk.sitkLinear)
out = resampler.Execute(moving_image)

writer = sitk.ImageFileWriter()
for i in range(out.GetDepth()):
    image_slice = out[:,:,i]

    # Slice specific tags.
    for j,key in enumerate(out.GetMetaDataKeys()):
        image_slice.SetMetaData(key, out.GetMetaData(key))

    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    writer.SetFileName(os.path.join(outpath,str(i)+'.dcm'))
    writer.Execute(image_slice)


print(resampler,sitk.GetArrayFromImage(fixed_image).shape,sitk.GetArrayFromImage(moving_image).shape)
