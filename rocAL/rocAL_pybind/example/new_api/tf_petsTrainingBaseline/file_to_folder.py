import os
import shutil
from tqdm import tqdm


os.mkdir("images_subdir")
path = "./images/"
temp="rm -rf ./images/american_bull*"
os.system (temp)
dir_list = os.listdir(path)
# print(dir_list)
for file_name in tqdm(dir_list):
    file =file_name.split("_")
    file.pop()
    separator = '_'
    dir_name=separator.join(file)
    path_each_subdir ="./images_subdir/"+dir_name
    if(not os.path.exists(path_each_subdir)):
        os.mkdir(path_each_subdir)
    cmd="cp ./images/"+dir_name+"* ./images_subdir/"+dir_name+"/"
    # print("hello", cmd)
    os.system(cmd)
