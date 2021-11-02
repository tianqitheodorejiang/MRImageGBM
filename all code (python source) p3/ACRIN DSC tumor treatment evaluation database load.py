import os
import numpy as np
import pydicom as dicom

input_path = "C:/Users/JiangQin/Documents/data/raw ct files/ACRIN-DSC-MR-Brain"
path = "C:/Users/JiangQin/Documents/data/raw ct files/ACRIN-DSC-MR-Brain/Clinical data/ACRIN-DSC-MR-Brain TCIA Anonymized"
path2 = "C:/Users/JiangQin/Documents/data/raw ct files/ACRIN-DSC-MR-Brain/Clinical data/ACRIN-DSC-MR-Brain-HB TCIA Anonymized"
alphabet = ["A","B","C","D"]

progressions = []
dates = []

for i in range(0,14):
    if i >= 10:
        i=alphabet[i%10]
    data = []
    blub = open(os.path.join(path,str("M"+str(i))+".csv")).read()
    lines = blub.split("\n")
    del lines[0]
    del lines[-1]
    for n,line in enumerate(lines):
        chars = line.split(",")
        data.append([])
        for char in chars:
            data[n].append(char)
    data = np.stack(data)
    sets = data[:,0]
    for n,num in enumerate(sets):
        num = int(num)
        if len(dates) < num:
            for i in range(0,num-len(dates)+1):
                dates.append([])
                progressions.append([])
        if data[:,43][n] != "":
            dates[num].append(int(data[:,8][n]))
            progressions[num].append(int(data[:,43][n]))
        
for i in range(0,14):
    if i >= 10:
        i=alphabet[i%10]
    data = []
    blub = open(os.path.join(path2,str("M"+str(i))+".csv")).read()
    print(blub)
    lines = blub.split("\n")
    del lines[0]
    del lines[-1]
    for n,line in enumerate(lines):
        chars = line.split(",")
        data.append([])
        for char in chars:
            data[n].append(char)
    data = np.stack(data)
    sets = data[:,0]
    for n,num in enumerate(sets):
        num = int(num)
        if len(dates) < num:
            for i in range(0,num-len(dates)+1):
                dates.append([])
                progressions.append([])
        if data[:,43][n] != "":
            dates[num].append(int(data[:,8][n]))
            progressions[num].append(int(data[:,43][n]))
        
        
print(dates)
print(progressions)



bru = 0
valid_indexes = []
scans = []
sets = []

for set_ in os.listdir(input_path):
    set_path = input_path + "/" + set_
    scans = []
    scan_dates = []
    try:
        set_num = int(set_[-3:])
        #print(set_num)
        progression_datas = []
        if set_num<len(dates):
            for scan in os.listdir(set_path):
                flair = None
                t1 = None
                t2 = None
                
                scan_path = set_path + '/' + scan
                if os.path.isdir(scan_path):
                    for mri in os.listdir(scan_path):
                        if "t2" in mri.lower() and "cor" not in mri.lower() and "sag" not in mri.lower()  and "trace" not in mri.lower() and os.path.isdir(scan_path + "/" + mri):
                            if t2!=None:
                                bru+=1
                            t2 = mri
                        if "t1" in mri.lower() and "cor" not in mri.lower() and "sag" not in mri.lower()  and "post" in mri.lower() and os.path.isdir(scan_path + "/" + mri):
                            if t1!=None:
                                bru+=1
                            t1 = mri
                        if "flair" in mri.lower() and "cor" not in mri.lower() and "sag" not in mri.lower()  and "t1" not in mri.lower() and os.path.isdir(scan_path + "/" + mri):
                            if flair!=None:
                                bru+=1
                            flair = mri
                    if flair is not None and t1 is not None and t2 is not None:
                        
                        date = dicom.read_file(scan_path + "/" + flair+"/"+os.listdir(scan_path + "/" + flair)[0]).ClinicalTrialTimePointID
                        #print(int(date),dates[set_num])
                        if int(date) in dates[set_num]:
                            scan_dates.append(date)
                            scans.append([scan_path + "/" + flair,scan_path + "/" + t1,scan_path + "/" + t2])
                            progression_datas.append(progressions[set_num][dates[set_num].index(int(date))])
            if len(scans) > 0:
                sets.append([])
                for s,scan in enumerate(scans):
                    sets[-1].append([scan,progression_datas[s],int(scan_dates[s])])
            
    except Exception as e:
        print(e)
        pass

print(len(scans))
print(bru)

print(sets)

for n,set_ in enumerate(sets):
    set_ = sorted(set_,key = lambda x:x[2])
    print(set_)
    
progression_datas = []
set_datas = []
dates = []

for i in range(0,10):
    data = []
    blub = open(os.path.join(path,str("M"+str(i))+".csv")).read()
    lines = blub.split("\n")
    del lines[0]
    del lines[-1]
    for n,line in enumerate(lines):
        chars = line.split(",")
        data.append([])
        for char in chars:
            data[n].append(char)
    data = np.stack(data)
    progression_datas.append(data[:,43])
    dates.append(data[:,8])
    set_datas.append(data[:,0])

'''sets_num = np.max(np.array(set_datas[0]).astype(np.int))
print(sets_num)

sets = [[] for i in range(sets_num+1)]

for n,data in enumerate(progression_datas):
    for i in range(sets_num+1):
        if str(i) in set_datas[n]:
            if data[np.where(set_datas[n]==str(i))][0] != "":
                progress = int(data[np.where(set_datas[n]==str(i))][0])
                sets[i].append([progress,dates[np.where(set_datas[n]==str(i))]])
            else:
                sets[i].append(0)
        else:
            sets[i].append(0)'''



progression_data = np.stack(sets)
print(progression_data.shape)

binary = progression_data.copy()
binary[binary>0] = 1
print(np.sum(binary))

'''for n,thing in enumerate(sets):
    print(n,thing)'''

