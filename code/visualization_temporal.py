import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import cv2
import copy
from copy import deepcopy
import PIL
from PIL import Image
import matplotlib.animation as animation

import io

vegetables = ["base of the spine", "middle of the spine", "neck", "head", "left shoulder", "left elbow", "left wrist",
              "left hand", "right shoulder", "right elbow", "right wrist", "right hand", "left hip", "left knee",
              "left ankle", "left foot", "right hip", "right knee", "right ankle", "right foot", "spine",
              "tip of the left hand", "left thumb", "tip of the right hand", "right thumb"]
farmers = ["base of the spine", "middle of the spine", "neck", "head", "left shoulder", "left elbow", "left wrist",
           "left hand", "right shoulder", "right elbow", "right wrist", "right hand", "left hip", "left knee",
           "left ankle", "left foot", "right hip", "right knee", "right ankle", "right foot", "spine",
           "tip of the left hand", "left thumb", "tip of the right hand", "right thumb"]

self_link = [(i, i) for i in range(25)]
neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                  (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                  (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                  (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                  (22, 23), (23, 8), (24, 25), (25, 12)]
neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
edge = self_link + neighbor_link

action = "Clapping_rel"

two_body = True


def visualize(weights, weights_rel, label, name, layer, file):
    B, T, Nh, V, V = weights.size()
    # Opens the Video file

    if (False):
        print(file)
        video, data = pose_estimation(file[0:20])

        features = []
        for t in range(0, T):
            if not two_body:
                w = weights[0, t, 0, :, :]
            else:
                w = weights[:, t, 0, :, :]
            print(w.shape)
            feature = torch.sum(w, dim=1)
            feature = feature.cpu().detach().numpy()
            print(feature.shape)
            min = np.min(feature)
            max = np.max(feature)
            feature = (feature - min) / (max - min)
            features.append(feature)
        print(np.shape(features))
        images = stgcn_visualize(data,
                                 edge,
                                 feature=features, video=video)
        # for (i, image) in enumerate(images):
        # image = image.astype(np.uint8)
        #       image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # cv2.imwrite('/multiverse/storage/plizzari/code/eccv/frames/Prova' + str(i).zfill(3) + '.jpg', image)
        # os.system(
        #   "ffmpeg -start_number 0 -r 1 -i /multiverse/storage/plizzari/code/eccv/frames/Prova%03d.jpg -vcodec mpeg4 -y Prova_T4.mp4")
        cv2.destroyAllWindows()
        print("DONE!")

    #     cap = cv2.VideoCapture('/multiverse/datasets/plizzari/nturgb+d_rgb/S007C002P025R001A017_rgb.avi')
    #     i = 0
    #     while (cap.isOpened()):
    #         ret, frame = cap.read()
    #         if ret == False:
    #             break
    #         cv2.imwrite('/multiverse/storage/plizzari/code/eccv/frames/ShoeOff' + str(i).zfill(3) + '.jpg', frame)
    #         i += 1
    # # -filter:v setpts=0.25*PTS
    #     os.system(
    #         "ffmpeg -start_number 0 -r 1 -i /multiverse/storage/plizzari/code/eccv/frames/ShoeOff%03d.jpg -vcodec mpeg4 -y ShoeOff_T4.mp4")
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     print("DONE!")
    # # Writer = animation.writers['ffmpeg']
    # # writer = Writer(fps=15, bitrate=1800)
    # ims = []

    # for i in range(Nh):

    # if hm:
    #     heatmap = weights[0, 0, :, :].squeeze()
    #     heatmap = np.maximum(heatmap, 0)
    #     plt.imsave('/multiverse/storage/plizzari/code/original_code/heatmaps/hm_'+name+'.png',
    #                heatmap.squeeze())
    ## opening videocapture
    # cap = cv2.VideoCapture(0)

    ## some videowriter props
    # sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # fps = 20
    # fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    # vout = cv2.VideoWriter()
    # vout.open('output.mp4', fourcc, fps, sz, True)
    if True:
        weights = weights.cpu().detach()
        weights_rel=weights_rel.cpu().detach()
        for t in range(0, T):

            # heatmap = torch.mean(weights[[0], i, :, :], dim=0).squeeze()

            heatmap0 = weights[0, t, 0, :, :].squeeze()
            heatmap1 = weights_rel[0, t, 0, :, :].squeeze()

            # relu on top of the heatmap
            # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
            heatmap0 = np.maximum(heatmap0, 0)
            heatmap1 = np.maximum(heatmap1, 0)

            # normalize the heatmap
            # heatmap /= torch.max(heatmap)

            # draw the heatmap
            # rint(heatmap0.size())

            fig, axes = plt.subplots(ncols=2, figsize=(20, 12))
            fig.suptitle("Clapping", fontsize=16)

            # fig, ax = plt.subplots()

            ax, ax1 = axes


            im = ax.imshow(heatmap0)
            # We want to show all ticks...
            ax.set_xticks(np.arange(len(farmers)))
            ax.set_yticks(np.arange(len(vegetables)))
            # ... and label them with the respective list entries
            ax.set_xticklabels(farmers, fontsize=10)
            ax.set_yticklabels(vegetables, fontsize=10)
            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor", fontsize=10)

            # Loop over data dimensions and create text annotations.
            for i in range(len(vegetables)):
                for j in range(len(farmers)):
                    text = ax.text(j, i, '',
                                   ha="center", va="center", color="w", fontsize=5)

            im1 = ax1.imshow(heatmap1)
            ax1.set_xticks(np.arange(len(farmers)))
            ax1.set_yticks(np.arange(len(vegetables)))
            ax1.set_xticklabels(farmers, fontsize=10)
            ax1.set_yticklabels(vegetables, fontsize=10)
            plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor", fontsize=10)
            for i in range(len(vegetables)):
                for j in range(len(farmers)):
                    text = ax1.text(j, i, '',
                                    ha="center", va="center", color="w", fontsize=5)

            # ax.set_title("Joint correlation")
            # plt.title("Handshaking")

            fig.tight_layout()

            # ims.append([fig])

            # plt.show()
            # cv2.imwrite('/multiverse/storage/plizzari/code/original_code/heatmaps/LastLayer_add_' + name + 'cv2.png', heatmap.squeeze())

            print(fig)
            # _, frame = cap.read()
            # vout.write(pil_img)
            print(t)

            # im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
            #                                    blit=True)
            # print(im_ani)
            # im_ani.save('im.mp4', writer=writer)
            # print("saved")
            fig.savefig(
                '/multiverse/storage/plizzari/code/eccv/frames/Layer' + str(layer) + action +'_'+name+'_' + str(t).zfill(
                    3) + '.png')

            plt.close(fig)
        os.system(
            "ffmpeg -start_number 0 -r 1 -i /multiverse/storage/plizzari/code/eccv/frames/Layer" + str(
                layer) + action + '_'+name+"_%03d.png -vcodec mpeg4 -y " + action + "_" + str(layer) +"_"+name+ "movie_rel.mp4")

        # im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
        #  blit=True)

        # print(im_ani)
        # im_ani.save('im.mp4', writer=writer)
        print("Saved")
    # vout.release()
    # cap.release()
    # plt.imsave('/multiverse/storage/plizzari/code/original_code/heatmaps/Prova'+str(layer)+'_ep120_head'+str(i)+'_' + name + '.png', fig)

    # plt.matshow(heatmap.squeeze())
    # plt.show(block=True)


#  heatmap = torch.mean(weights[[0], :, :, :], dim=1).squeeze()
#  heatmap = np.maximum(heatmap, 0)
#
#  # normalize the heatmap
# # heatmap /= torch.max(heatmap)
#
#  # draw the heatmap
#  print(heatmap.size())
#  plt.imsave('/multiverse/storage/plizzari/code/original_code/heatmaps/first_mean_head_'+name+'.png', heatmap.squeeze())

def pose_estimation(filename):
    video_capture = cv2.VideoCapture(
        '/multiverse/datasets/plizzari/nturgb+d_rgb/' + filename + '_rgb.avi')
    video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # pose estimation
    frame_index = 0
    video = list()

    while (True):

        # get image
        ret, orig_image = video_capture.read()
        if orig_image is None:
            break
        source_H, source_W, _ = orig_image.shape
        # orig_image = cv2.resize(
        #     orig_image, (256 * source_W // source_H, 256))
        H, W, _ = orig_image.shape
        video.append(orig_image)

        frame_index += 1

    # pose estimation

    data = read_xyz(
        '/multiverse/datasets/plizzari/Skeletons/nturgb+d_skeletons/' + filename[0:20] + '.skeleton')

    return video, data


def stgcn_visualize(pose,
                    edge,
                    feature,
                    video,
                    height=1080,
                    fps=None):
    _, T, V, M = pose.shape
    T = len(video)
    images = []

    pos_track = [None] * M
    for t in range(T):
        print(t)

        frame = video[t]
        # image resize
        H, W, c = frame.shape
        # frame = cv2.resize(frame, (height * W // H // 2, height//2))
        H, W, c = frame.shape
        scale_factor = 2 * height / 1080

        # draw skeleton
        skeleton = frame * 0
        text = frame * 0
        for m in range(M):

            # score = pose[2, t, :, m].max()
            # if score < 0.3:
            #     continue

            for i, j in edge:
                xi = pose[0, t, i, m]
                yi = pose[1, t, i, m]
                xj = pose[0, t, j, m]
                yj = pose[1, t, j, m]
                if xi + yi == 0 or xj + yj == 0:
                    continue
                else:
                    xi = int((xi))
                    yi = int((yi))
                    xj = int((xj))
                    yj = int((yj))
                    # xi = int((xi + 0.5) * W)
                    # yi = int((yi + 0.5) * H)
                    # xj = int((xj + 0.5) * W)
                    # yj = int((yj + 0.5) * H)
                cv2.line(skeleton, (xi, yi), (xj, yj), (255, 255, 255),
                         int(np.ceil(2 * scale_factor)))

            x_nose = int((pose[0, t, 0, m]))
            y_nose = int((pose[1, t, 0, m]))
            x_neck = int((pose[0, t, 1, m]))
            y_neck = int((pose[1, t, 1, m]))

            half_head = int(((x_neck - x_nose) ** 2 + (y_neck - y_nose) ** 2) ** 0.5)
            pos = (x_nose + half_head, y_nose - half_head)
            if pos_track[m] is None:
                pos_track[m] = pos
            else:
                new_x = int(pos_track[m][0] + (pos[0] - pos_track[m][0]) * 0.2)
                new_y = int(pos_track[m][1] + (pos[1] - pos_track[m][1]) * 0.2)
                pos_track[m] = (new_x, new_y)
            # cv2.putText(text, body_label, pos_track[m],
            #       cv2.FONT_HERSHEY_TRIPLEX, 0.5 * scale_factor,
            #       (255, 255, 255))

        # generate mask
        mask = frame * 0
        feature = np.abs(feature)
        feature = feature / feature.mean()
        for m in range(M):
            # score = pose[2, t, :, m].max()
            # if score < 0.3:
            #     continue

            if not two_body:
                f = feature[t // 4, :] ** 5
            else:
                f = feature[t // 4, m, :] ** 5

            if f.mean() != 0:
                f = f / f.mean()
            for v in range(V):
                x = pose[0, t, v, m]
                y = pose[1, t, v, m]
                if x + y == 0:
                    continue
                else:
                    x = int((x))
                    y = int((y))
                cv2.circle(mask, (x, y), 0, (255, 255, 255),
                           int(np.ceil(f[v] ** 0.5 * 8 * scale_factor)))
        blurred_mask = cv2.blur(mask, (12, 12))

        skeleton_result = blurred_mask.astype(float) * 0.75
        skeleton_result += skeleton.astype(float) * 0.25
        skeleton_result += text.astype(float)
        skeleton_result[skeleton_result > 255] = 255
        skeleton_result.astype(np.uint8)

        rgb_result = blurred_mask.astype(float) * 0.75
        rgb_result += frame.astype(float) * 0.5
        rgb_result += skeleton.astype(float) * 0.25
        rgb_result[rgb_result > 255] = 255
        rgb_result.astype(np.uint8)

        #
        # text_1 = cv2.imread(
        #     './resource/demo_asset/original_video.png', cv2.IMREAD_UNCHANGED)
        # text_2 = cv2.imread(
        #     './resource/demo_asset/pose_estimation.png', cv2.IMREAD_UNCHANGED)
        # text_3 = cv2.imread(
        #     './resource/demo_asset/attention+prediction.png', cv2.IMREAD_UNCHANGED)

        # img0 = np.concatenate((frame, skeleton), axis=1)
        # img1 = np.concatenate((skeleton_result, rgb_result), axis=1)
        # img = np.concatenate((img0, img1), axis=0)
        img = rgb_result
        # img = img[200:1080 - 200, 600:1920 - 500]

        # if fps is not None:
        #   put_text(img, 'fps:{:.2f}'.format(fps), (0.9, 0.5))
        # img = cv2.resize(img, (height * W // H // 2, height // 2))
        cv2.imwrite('/multiverse/storage/plizzari/code/eccv/frames/' + action + str(t).zfill(3) + '.jpg', img)

        images.append(img)
    os.system(
        "ffmpeg -start_number 0 -r 1 -i /multiverse/storage/plizzari/code/eccv/frames/" + action + "%03d.jpg -vcodec mpeg4 -y " + action + ".mp4")


def put_text(img, text, position, scale_factor=1):
    t_w, t_h = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_TRIPLEX, scale_factor, thickness=1)[0]
    H, W, _ = img.shape
    position = (int(W * position[1] - t_w * 0.5),
                int(H * position[0] - t_h * 0.5))
    params = (position, cv2.FONT_HERSHEY_TRIPLEX, scale_factor,
              (255, 255, 255))
    cv2.putText(img, text, *params)


def blend(background, foreground, dx=20, dy=10, fy=0.7):
    foreground = cv2.resize(foreground, (0, 0), fx=fy, fy=fy)
    h, w = foreground.shape[:2]
    b, g, r, a = cv2.split(foreground)
    mask = np.dstack((a, a, a))
    rgb = np.dstack((b, g, r))

    canvas = background[-h - dy:-dy, dx:w + dx]
    imask = mask > 0
    canvas[imask] = rgb[imask]


def read_skeleton(file):
    with open(os.path.join("", file), 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((3, 300, num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['colorX'], v['colorY'], 0]
                else:
                    pass
    return data


def read_xyz_true_coord(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((3, 300, num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data


if __name__ == '__main__':
    torch.manual_seed(13696642)
    weights = torch.randint(2, 17777, (2, 2, 3, 3), dtype=torch.float32)
    print(weights)
    visualize(weights, 'prova')
