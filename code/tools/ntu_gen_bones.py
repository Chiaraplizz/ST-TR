import os
import numpy as np
from numpy.lib.format import open_memmap

from tqdm import tqdm

'''
Function adapted from: https://github.com/kenziyuliu/Unofficial-DGNN-PyTorch 
'''

paris = {
    'xview': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
        (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
        (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
        (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
        (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
    ),
    'xsub': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
        (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
        (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
        (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
        (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
    ),

    'kinetics': (
        (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1),
        (6, 5), (7, 6), (8, 2), (9, 8), (10, 9), (11, 5),
        (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)
    )
}

sets = {'val'}
datasets = {'xsub'}


def gen_bone_data():
    """Generate bone data from joint data for NTU skeleton dataset"""
    for dataset in datasets:
        for set in sets:
            print(dataset, set)
            data = np.load(
                './Output_skeletons_without_missing_skeletons/{}/{}_data_joint_filtered.npy'.format(
                    dataset, set))

            data1 = np.load(
                './Output_skeletons_without_missing_skeletons/{}/{}_data_joint_bones.npy'.format(
                    dataset, set))

           # N, C, T, V, M = data.shape
            # fp_sp = open_memmap(
            #      '/multiverse/datasets/plizzari/new_data_processed/{}/{}_data_joint_filtered_bone_new.npy'.format(
            #          dataset, set),
            #      dtype='float32',
            #      mode='w+',
            #      shape=(N, 3, T, V, M))

            # Copy the joints data to bone placeholder tensor
           # fp_sp[:, :C, :, :, :] = data
            for v1, v2 in tqdm(paris[dataset]):
                # Reduce class index for NTU datasets
                if dataset != 'kinetics':
                    v1 = v1 - 1
                    v2 = v2 - 1


                # Assign bones to be joint1 - joint2, the pairs are pre-determined and hardcoded
                # There also happens to be 25 bones

                print(data1[:, 3:6, :, v1, :] == (data[:, :, :, v1, :] - data[:, :, :, v2, :]))


if __name__ == '__main__':
    gen_bone_data()
