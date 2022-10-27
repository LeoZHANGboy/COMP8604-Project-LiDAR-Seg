import matplotlib.pyplot as plt
import random
import os
import numpy as np
import matplotlib.patches as mpatches
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings(action='ignore')
import sys

sys.stderr = open(os.devnull, "w")  # silence stderr
sys.stderr = sys.__stderr__  # unsilence stderr
import matplotlib.colors as mcolors
import matplotlib as mpl
# import matplotlib.pyplot as plt
from matplotlib import cm
parser = argparse.ArgumentParser()
parser.add_argument('--pred_type', default='pred', help='use ground truth or predictions [choose from: gt and pred]')
FLAGS = parser.parse_args()

label_to_names = {0: 'unlabeled',
                  1: 'car',
                  2: 'bicycle',
                  3: 'motorcycle',
                  4: 'truck',
                  5: 'other-vehicle',
                  6: 'person',
                  7: 'bicyclist',
                  8: 'motorcyclist',
                  9: 'road',
                  10: 'parking',
                  11: 'sidewalk',
                  12: 'other-ground',
                  13: 'building',
                  14: 'fence',
                  15: 'vegetation',
                  16: 'trunk',
                  17: 'terrain',
                  18: 'pole',
                  19: 'traffic-sign'}
color_map = [[0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
             [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [0, 130, 180],
             [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230],
             [119, 11, 32]]
seq = '08'

data_path = './demo_data/pcs/sequences'
if FLAGS.pred_type == 'pred':
    label_path = './demo_data/outputs/prediction/08/predictions'
else:
    label_path = data_path + '/' + seq + '/labels/'

# idx = 160

random.seed(0)
print(len(mcolors.TABLEAU_COLORS))
#
pc_path = data_path + '/' + seq + '/velodyne/'

label_files = os.listdir(label_path)
pc_files = os.listdir(pc_path)
label_files = sorted(label_files)
pc_files = sorted(pc_files)

# rand_idx = np.random.randint(0, len(labels), len(labels))
# labels = labels[rand_idx]
# pc = pc[rand_idx]


def pc_plot_3d(pc, labels, idx):
    all_class = np.arange(0, 20)
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(projection='3d')
    color_patch = []
    for cls in all_class:

        point_cur_class = pc[labels == cls]
        # if len(point_cur_class) == 0:
        #     continue
        cur_sequence_containing_x_vals = point_cur_class[:, 0]
        cur_sequence_containing_y_vals = point_cur_class[:, 1]
        cur_sequence_containing_z_vals = point_cur_class[:, 2]
        #
        # random.shuffle(sequence_containing_x_vals)
        # random.shuffle(sequence_containing_y_vals)
        # random.shuffle(sequence_containing_z_vals)
        # cur_color = np.repeat(np.array(color_map[cls]), len(point_cur_class), axis=0)

        ax.scatter(cur_sequence_containing_x_vals, cur_sequence_containing_y_vals, cur_sequence_containing_z_vals,
                   color=np.array(color_map[cls]) / 255., s=0.2,
                   )
        color_patch.append(mpatches.Patch(color=np.array(color_map[cls]) / 255., label=label_to_names[cls]))
    if FLAGS.pred_type == 'pred':
        outpath = f'./demo_data/outputs/pred_plot/{idx}.png'
        title = 'Pred Segmentation'
    else:
        outpath = f'./demo_data/outputs/GT_plot/{idx}.png'
        title = 'GT Segmentation'
    ax.set(zlim=(5, -5), xlim=(-15, 15), ylim=(-15, 15), title=title)

    ax.legend(handles=color_patch, bbox_to_anchor=(1, 0), loc="lower right",
              bbox_transform=fig.transFigure, ncol=10)
    ax.view_init(50, 30)
    ax.set_axis_off()
    plt.tight_layout()

    plt.savefig(outpath, dpi=600)
    plt.clf()
    plt.close()
    # plt.show()


for idx in tqdm(range(len(label_files))):
    select_label_file = os.path.join(label_path, label_files[idx])
    select_pc_file = os.path.join(pc_path, pc_files[idx])

    cur_labels = np.load(select_label_file).reshape(-1)
    cur_pc = np.load(select_pc_file)
    pc_plot_3d(cur_pc, cur_labels, idx)
    # break

# pcs = np.load('/project/RandLA-Net/demo_data/pcs/sequences/08/velodyne/000000.npy')
# labels = np.load('/project/RandLA-Net/demo_data/pcs/sequences/08/velodyne/000000.npy')
# print(a)