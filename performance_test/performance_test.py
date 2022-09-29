import os
path=os. getcwd()
print(path)

# #<<<<<<<<<<<<<<<<<<<RPP<<<<<<<<<<<<<<<<<<<<<<<
os.system("git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp.git")
os.chdir('./rpp/')
os.system('mkdir build')
os.chdir('./build/')
os.system('cmake ..')
os.system('sudo make -j21 install')
os.chdir(path)

# #<<<<<<<<<<<<<<<<<<<<<image mivisionx<<<<<<<<<<<<<<<<<<

os.system("mkdir MivisionxTOT")
os.chdir('./MivisionxTOT/')
os.system("pwd")
os.system("git clone https://github.com/shobana-mcw/MIVisionX.git")
os.chdir('./MIVisionX/')
os.system("git checkout ak/tensor_performance_benchmarking")
os.system('mkdir build')
os.chdir('./build/')
os.system('cmake ..')
os.system('sudo make -j21 install')
os.chdir("./../utilities/rocAL/rocAL_performance_tests/")
os.system('mkdir build')
os.chdir('./build/')
os.system('cmake ..')
os.system('sudo make -j21 install')
os.chdir("./../")
os.system("./performance_testing.sh '/media/sample_test/coco/val2017_10_images/' 300 300 48 0")

os.chdir(path)

#>>>>>>>>tensor branch <<<<<<<<<<<<<<<<<<<<<<
os.system("mkdir MivisionxTensor")
os.chdir('./MivisionxTensor/')
os.system("pwd")
os.system("git clone https://github.com/fiona-gladwin/MIVisionX.git")
os.chdir('./MIVisionX/')
os.system("git checkout ak/rocal_tensor_v1")
os.system('mkdir build')
os.chdir('./build/')
os.system('cmake ..')
os.system('sudo make -j21 install')
os.chdir("./../utilities/rocAL/rocAL_performance_tests/")
os.system('mkdir build')
os.chdir('./build/')
os.system('cmake ..')
os.system('sudo make -j21 install')
os.chdir("./../")
os.system("./performance_tensor.sh '/media/sample_test/coco/val2017_10_images/' 300 300 48 0")

os.chdir(path)


import csv

header = ['test case', 'Load     time', 'Decode   time','Process  time', 'Transfer time', 'Total time']
new_list, new_list1, new_list2, new_list3, new_list4, new_list5, tot_list = [], [], [], [], [], [], []
aug_list = ["rocalResize", "rocalCropResize", "rocalRotate", "rocalBrightness", "rocalGamma", "rocalContrast", "rocalFlip", "rocalBlur", "rocalBlend", "rocalWarpAffine", "rocalFishEye", "rocalVignette", "rocalVignette", "rocalSnPNoise", "rocalSnow", "rocalRain", "rocalColorTemp", "rocalFog", "rocalLensCorrection", "rocalPixelate", "rocalExposure", "rocalHue", "rocalSaturation", "rocalCopy", "rocalColorTwist", "rocalCropMirrorNormalize", "rocalCrop", "rocalResizeCropMirror", "No-Op"]
idx =0
aug_list1 = ["rocalResize", "rocalBrightness", "rocalGamma", "rocalContrast", "rocalFlip",  "rocalBlend",  "rocalSnPNoise",  "rocalExposure",  "rocalColorTwist", "rocalCropMirrorNormalize"]
# input file name with extension
op_paths = [path+"/MivisionxTOT/MIVisionX/utilities/rocAL/rocAL_performance_tests//output_folder/",path+"/MivisionxTensor/MIVisionX/utilities/rocAL/rocAL_performance_tests//output_folder/"]
print(op_paths[0])
for j in range(2):
    for file in aug_list:
        file_name = op_paths[j]+file +".txt"
        if(os.path.exists(file_name)):
            # opening and reading the file
            file_read = open(file_name, "r")
            lines = file_read.readlines()
            
            for line in lines:
                
                if header[0] in line:
                    words= line.split()
                    a=int(words[-1])
                    new_list.append(aug_list[a])

                if header[1] in line:
                    words= line.split()
                    new_list1.append( words[-1])
                
                if header[2] in line:
                    words= line.split()
                    new_list2.append( words[-1])


                if header[3] in line:
                    words= line.split()
                    new_list3.append( words[-1])

                if header[4] in line:
                    words= line.split()
                    new_list4.append( words[-1])

                if header[5] in line:
                    words= line.split()
                    words[-1]=int(words[-1])/1000000
                    new_list5.append( words[-1])

            # closing file after reading

            file_read.close()
            if len(new_list)==0:
                print( "\ not found in \"" +file_name+ "\"!")
            else:
                lineLen = len(new_list)
                if new_list[-1] in aug_list1:
                    tot_list.append([new_list[-1],new_list1[-1],new_list2[-1],new_list3[-1],new_list4[-1],new_list5[-1]])
                print(tot_list)
        with open('tensor_performance.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write the data
            for i in range(len(tot_list)):
                writer.writerow(tot_list[i])


##>>>>>>>>>>>>>>>>>>>>>>>>graph plotting <<<<<<<<<<<<<<<<<<<<<<<

import matplotlib.pyplot as plt 
import csv
import numpy as np
Names=[]
Values=[]
# Names
Values1 =[]
k=0
with open(path+'/tensor_performance.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    header = next(lines)
    if header != None:
        for row in lines:
            
            if(k<10):
                Names.append(row[0])
                Values.append(float(row[-1]))
            else:
                Values1.append(float(row[-1]))
            k=k+1
print(Values)
print()
print(Values1)
     
# plt.scatter(Names, Values, color = 'g',s = 100)
X_axis = np.arange(len(Names))

plt.bar(X_axis-0.2,Values, 0.4, label = 'Image time ')
plt.bar(X_axis+0.2,Values1, 0.4, label = 'Tensor time ')
plt.legend()

# print("hello", values)
plt.xticks(X_axis, Names,rotation=25)

plt.xlabel('Augmentation')
plt.ylabel('Time in ms')
plt.title('Timing', fontsize = 20)
plt.savefig(path+"/img1.png",bbox_inches="tight",pad_inches=0.3)