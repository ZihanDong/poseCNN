# Import the converted model's class
import numpy as np
import cv2
from tqdm import tqdm
import os, os.path


# Set this path to your dataset directory
directory = 'D:/Dataset/Cleaned/'
new_dir = 'D:/Dataset/down_sampled/'
# category number you intend to process, from start to end 
start_idx = 82
end_idx = 128
# Specify the illumination level
illumination = ['_max.png', '_3_r5000.png', '_5_r5000.png']

def centeredCrop(img, output_side_length):
    height, width, depth = img.shape
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = output_side_length * height / width
    else:
        new_width = output_side_length * width / height
    height_offset = int((new_height - output_side_length) / 2)
    width_offset = int((new_width - output_side_length) / 2)
    cropped_img = img[height_offset:height_offset + output_side_length, width_offset:width_offset + output_side_length]
    return cropped_img

def sub_sample():
    for i in range(start_idx, end_idx+1):
        cwd = directory + 'scan' + str(i) + '/'
        new = new_dir + 'scan' + str(i) + '/'
        number_of_files = len([name for name in os.listdir(cwd) if os.path.isfile(cwd+name)])
        number_of_poses = 49 if number_of_files < 64*8 else 64
        # Create Dir
        if not os.path.exists(new):
            os.makedirs(new)
        for n in tqdm(range(1, number_of_poses+1)):
            idx_num = '0'+str(n) if n<10 else str(n)
            for illu_level in range(3):
                X = cv2.imread(cwd +'clean_0' + idx_num + illumination[illu_level])
                X = cv2.resize(X, (320, 240))
                X = centeredCrop(X, 224)
                cv2.imwrite(new +'clean_0' + idx_num + illumination[illu_level], X)
        print('Category ' + str(i) + ' finished.')

if __name__ == '__main__':
    sub_sample()
    
    
    