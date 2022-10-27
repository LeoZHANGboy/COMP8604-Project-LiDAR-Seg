import os
from distutils.dir_util import copy_tree

calib_path = 'D:/dataset/data_odometry_calib'

label_path = 'D:/dataset/data_odometry_labels'

pc_path = 'D:/dataset/data_odometry_velodyne'

further_path = '/dataset/sequences/'

calib_path += further_path
label_path += further_path
pc_path += further_path

seq_list = os.listdir(calib_path)

for seq in seq_list:
    copy_tree(os.path.join(calib_path, seq), os.path.join(pc_path, seq))
    copy_tree(os.path.join(label_path, seq), os.path.join(pc_path, seq))
    # print(os.path.join(calib_path, seq), os.path.join(pc_path, seq))
    # print(os.path.join(label_path, seq), os.path.join(pc_path, seq))