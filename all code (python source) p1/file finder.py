import os

path1 = "/home/jiangl/Documents/python/ct to tumor identifier project/raw ct files/TCGA-GBM"
path2 = "/home/jiangl/Documents/python/ct to tumor identifier project/raw ct files/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations"

dates = []
pathes = []

for folder1 in os.listdir(path1):
    for file in os.listdir(os.path.join(path1, folder1)):
        day = file[3:5]
        month = file[0:2]
        year = file[6:10]
        date = str(month) + "-" + str(day) + "-" + str(year)

        dates.append(date)
        pathes.append(os.path.join(os.path.join(path1, folder1), file))

for folder1 in os.listdir(path2):
    for file in os.listdir(os.path.join(path2, folder1)):
        if "flair" in file.lower():
            day = file[21:23]
            month = file[18:20]
            year = file[13:17]
            date = str(month) + "-" + str(day) + "-" + str(year)
            if date in dates:
                print("")
                print(os.path.join(os.path.join(path2, folder1), file))
                print(pathes[dates.index(date)])
                
        
        
