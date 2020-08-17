import numpy as np
from tools.rotation import angle_between, rotation_matrix
from tqdm import tqdm


def pre_normalization(data, zaxis=[0, 1], xaxis=[8, 4]):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # to (N, M, T, V, C)

    print('pad the null frames with the previous frames')
    for i_s, skeleton in enumerate(tqdm(s)):  # Dimension N
        if skeleton.sum() == 0:
            print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton): # Dimension M (# person)
            # `person` has shape (T, V, C)
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                # `index` of frames that have non-zero nodes
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                # Shift non-zero nodes to beginning of frames
                person *= 0
                person[:len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                # Each frame has shape (V, C)
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        # Repeat all the frames up to now (`i_f`) till the max seq len
                        rest = len(person) - i_f
                        reps = int(np.ceil(rest / i_f))
                        pad = np.concatenate([person[:i_f] for _ in range(reps)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break

    print('sub the center joint #1 (spine joint in ntu and neck joint in kinetics)')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        # Use the first skeleton's body center (`1:2` along the nodes dimension)
        main_body_center = skeleton[0][:, 1:2, :].copy()    # Shape (T, 1, C)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            # For all `person`, compute the `mask` which is the non-zero channel dimension
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            # Subtract the first skeleton's centre joint
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    print('parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        # Shapes: (C,)
        joint_bottom = skeleton[0, 0, zaxis[0]]
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

    print('parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)

    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data


if __name__ == '__main__':
    data = np.load('../data/ntu/xview/val_data.npy')
    pre_normalization(data)
    np.save('../data/ntu/xview/data_val_pre.npy', data)