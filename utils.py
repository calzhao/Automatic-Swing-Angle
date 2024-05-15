import cv2
import numpy as np
import os
from math import atan2, degrees
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

import urllib.request

def angle_estimation(mask):
    kernel = np.ones((2, 2), np.uint8)
    cleaned_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    y_coords, x_coords = np.where(mask)
    points = np.column_stack((x_coords, y_coords))

    if len(points) == 0:
        return -1, 0

    pca = PCA(n_components=2)
    pca.fit(points)
    center = pca.mean_
    first_component = pca.components_[0]

    angle = atan2(first_component[1], first_component[0])
    y_coords, x_coords = np.where(mask)
    m = np.tan(angle)
    b = center[1] - m * center[0]  # y = mx + b => b = y - mx
    predicted_y = m * x_coords + b
    r_squared = r2_score(y_coords, predicted_y)

    orientation_angle = degrees(angle)
    orientation_angle = (orientation_angle + 360) % 180

    est_angle = orientation_angle
    if est_angle > 90:
        est_angle = 180 - est_angle

    return est_angle, r_squared

def frame_detection(scores, bboxes, vid_shape, thresh, kf_range=(190, 220)):
    lb, ub = kf_range
    peaks = find_peaks(scores, l_bound=lb, r_bound=ub - 1)
    candidates = context_proposal(scores, peaks, l_bound=lb, r_bound=ub - 1)
    cen_dist, hor_len, ver_len = get_bbox_info(bboxes, vid_shape)
    selected_indices = np.asarray([idx for idx in candidates if cen_dist[idx - lb] < thresh])

    if len(selected_indices) == 0:
        return -1, center_dist(0, 0)

    min_idx = np.argmin(cen_dist[selected_indices - lb])
    detected_frame = selected_indices[min_idx]

    return detected_frame, int(np.min(cen_dist[selected_indices - lb]))

def frame_detection_v2(model, scores, bboxes, threshold=0.8, kf_range=(190, 250), win_len=3, reso=(1280, 720)):
    lb, ub = kf_range
    peaks = find_peaks(scores, l_bound=lb, r_bound=ub - 1)
    proposals = context_proposal(scores, peaks, l_bound=lb, r_bound=ub - 1)
    bboxes_feature = normalize_box(bboxes, C_X=reso[0] / 2, C_Y=reso[1] / 2)
    padded_bboxes = uni_pad(bboxes_feature, win_len, pad_dir='top_bottom')
    vid_bbox_candidates = []
    for p in proposals:
        left = p - lb
        right = p - lb + win_len * 2 + 1
        contextualized_bbox = padded_bboxes[left: right]
        vid_bbox_candidates.append(contextualized_bbox)
    if len(vid_bbox_candidates) == 0:
        return -1, 0
    vid_bbox_candidates = np.stack(vid_bbox_candidates)

    test_bboxes = vid_bbox_candidates.reshape(vid_bbox_candidates.shape[0], -1)
    test_bboxes = torch.tensor(test_bboxes, dtype=torch.float32)

    model.eval()
    outputs = model(test_bboxes)
    predicted = torch.sigmoid(outputs)
    
    pred_idx = torch.argmax(predicted)
    if len(vid_bbox_candidates) == 1:
        return proposals[0], predicted.detach().numpy()

    return proposals[pred_idx], predicted[pred_idx].detach().numpy()

def loadVid(path, max_frame):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    cap = cv2.VideoCapture(path)

    frames = []
    frame_counter = 0

    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while (cap.isOpened() and frame_counter < max_frame):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            frame_counter += 1
        else:
            break
        
    cap.release()
    frames = np.stack(frames)
    return frames

def process_predictions(predictions, vid_shape):
    fields = predictions['instances'].get_fields()
    pred_masks = fields['pred_masks'].to('cpu').numpy()
    pred_scores = fields['scores'].to('cpu').numpy()
    pred_boxes = fields['pred_boxes'].tensor.to('cpu').numpy()
    # pred_classes = fields['pred_classes'].to('cpu').numpy()

    if len(pred_scores) > 1:
        idx = np.argmax(pred_scores)
        pred_masks = pred_masks[idx]
        pred_boxes = pred_boxes[idx]
        pred_scores = pred_scores[idx]
    elif len(pred_scores) == 0:
        pred_scores = np.array([0])
        pred_boxes = np.zeros((4,))
        # pred_masks = np.zeros((720, 1280))
        pred_masks = np.zeros(vid_shape)

    return pred_scores.squeeze(), pred_boxes.squeeze(), pred_masks.squeeze()

def saveVid(predictor, vidName, vidSrcRoot, detResRoot, args, proc_bar = None):
    pred_info = get_prediction(predictor, vidName, vidSrcRoot, (args.left_win,args.right_win), proc_bar=proc_bar)
    boxes, scores, masks = pred_info['boxes'], pred_info['scores'], pred_info['masks']
    save_path = os.path.join(detResRoot, '{}.npz'.format(vidName))
    np.savez(save_path, boxes=boxes, scores=scores, masks=masks)
    
def get_prediction(predictor, vidName, vidSrcRoot, window, proc_bar = None, vid_link = None):
    vid_masks = []
    vid_boxes = []
    vid_scores = []
    if vid_link != None:
        try:
            urllib.request.urlretrieve(vid_link, 'tmp/temp_vid.mp4') 
        except:
            print("Error downloading video.")
            return -1
        video = loadVid('tmp/temp_vid.mp4', max_frame=window[1])
    else:
        video = loadVid(os.path.join(vidSrcRoot, vidName+'.mp4'), max_frame=window[1])
    video = video[window[0]:window[1], :, :]

    for i in range(len(video)):
        if proc_bar:
            proc_bar.set_description('[{:03d}/{:03d}] in {}'.format(i+1, len(video), vidName))
        input_frame = video[i, :, :, :]

        predictions, visualized_output = predictor.run_on_image(input_frame)
        H, W, _ = input_frame.shape
        pred_scores, pred_boxes, pred_masks = process_predictions(predictions, (H, W))
        # print(pred_boxes.shape,pred_masks.shape)

        vid_scores.append(pred_scores)
        vid_boxes.append(pred_boxes)
        vid_masks.append(pred_masks.astype(np.uint8))
        # print(i, pred_scores, pred_scores.shape)

    vid_scores = np.stack(vid_scores)
    vid_boxes = np.stack(vid_boxes)
    vid_masks = np.stack(vid_masks)
    # print(vid_masks.shape)

    return {'boxes':vid_boxes, 'scores':vid_scores, 'masks':vid_masks}
    
def center_dist(xc, yc, C_X=640, C_Y=360):
    return np.sqrt((xc - C_X) ** 2 + (yc - C_Y) ** 2)

def get_bbox_info(bboxes):
    xlength = bboxes[:, 2] - bboxes[:, 0]
    ylength = bboxes[:, 3] - bboxes[:, 1]
    ratio = xlength / ylength
    area = xlength * ylength
    return xlength, ylength, ratio, area

def uni_pad(array, pad_len, pad_dir='left', pad_value=0):
    if pad_dir == 'left':
        pad_width = ((0, 0), (pad_len, 0))
    elif pad_dir == 'right':
        pad_width = ((0, 0), (0, pad_len))
    elif pad_dir == 'top':
        pad_width = ((pad_len, 0), (0, 0))
    elif pad_dir == 'bottom':
        pad_width = ((0, pad_len), (0, 0))
    elif pad_dir == 'top_bottom':
        pad_width = ((pad_len, pad_len), (0, 0))
    else:
        raise NotImplementedError('pad_dir can only be chosen within [left, right, top, bottom]')

    return np.pad(array, pad_width=pad_width, mode='constant', constant_values=pad_value)

def normalize_box(boxes, C_X=640, C_Y=360):
    boxes[:, 0] -= C_X
    boxes[:, 0] /= C_X
    boxes[:, 1] -= C_Y
    boxes[:, 1] /= C_Y
    boxes[:, 2] -= C_X
    boxes[:, 2] /= C_X
    boxes[:, 3] -= C_Y
    boxes[:, 3] /= C_Y
    # print("C_X, C_Y: ",C_X, C_Y, np.max(boxes, axis=0), np.min(boxes, axis=0))
    return boxes

def process_box(bboxes, C_X=640, C_Y=360):
    shape = list(bboxes.shape)
    # shape[-1]+=1
    box_feat = np.zeros(shape)
    xlength = bboxes[:, 2] - bboxes[:, 0]
    ylength = bboxes[:, 3] - bboxes[:, 1]
    ratio = xlength / (ylength + 1)
    # area = xlength * ylength
    
    box_feat[:, 0] = xlength
    box_feat[:, 1] = ylength
    box_feat[:, 2] = (bboxes[:, 0] + bboxes[:, 2]) / 2 # x_center
    box_feat[:, 3] = (bboxes[:, 1] + bboxes[:, 3]) / 2 # y_center
    # box_feat[:, 4] = ratio

    min_val = np.min(box_feat, axis=0)  # Min value for each feature
    max_val = np.max(box_feat, axis=0)  # Max value for each feature
    normalized_feat = (box_feat - min_val) / (max_val - min_val)

    mean = np.mean(box_feat, axis=0)
    std_dev = np.std(box_feat, axis=0)
    standarlized_feat = (box_feat - mean) / std_dev
    
    final_feat = np.zeros(shape)
    final_feat[:, 0:2] = normalized_feat[:, 0:2]
    final_feat[:, 2:] = standarlized_feat[:, 2:]
    # final_feat[:, 4] = normalized_feat[:, 4]
    
    # print(final_feat.shape, np.max(final_feat, axis=0), np.min(final_feat, axis=0))
    return final_feat
    
def get_bbox_info(bboxes, vid_shape):
    x_c = (bboxes[:, 0] + bboxes[:, 2]) / 2
    y_c = (bboxes[:, 1] + bboxes[:, 3]) / 2
    euc_dist = center_dist(x_c, y_c, C_X=vid_shape[1]/2, C_Y=vid_shape[0]/2)
    
    x_length = bboxes[:, 2] - bboxes[:, 0]
    y_length = bboxes[:, 3] - bboxes[:, 1]
    return euc_dist, x_length, y_length


def find_peaks(arr, l_bound=190, r_bound=249):
    peaks = []
    n = len(arr)

    # Check if the array has fewer than 3 elements; cannot have a peak
    if n < 3:
        return peaks

    # Check the first element
    if arr[0] >= arr[1]:
        peaks.append(l_bound)

    # Check for peaks in the middle of the array
    for i in range(1, n - 1):
        if arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
            peaks.append(i + l_bound)

    # Check the last element
    if arr[-1] >= arr[-2]:
        peaks.append(l_bound+n-1)

    return np.array(peaks)


def context_proposal(conf_seq, peaks, context_window=3, l_bound=190, r_bound=250):
    candidates = []
    # print(len(conf_seq), l_bound, r_bound, peaks)
    for p in peaks:
        for idx in range(max(p - context_window, l_bound, 0), \
            min(p + context_window + 1, r_bound, len(conf_seq)+l_bound)):
            if conf_seq[idx - l_bound] > 0.4 and idx not in candidates:
                candidates.append(idx)
    return np.asarray(candidates)

def randAug(normalized_bboxes, shift_size=0.2, scale_range=(0.9, 1.1)):
    num_frames = normalized_bboxes.shape[0]
    shift_array = np.random.uniform(-shift_size, shift_size, (2))

    # Apply shifts to the bounding boxes
    # shifted_bboxes = normalized_bboxes
    shifted_bboxes = np.copy(normalized_bboxes)
    shifted_bboxes[:, 0] += shift_array[0]
    shifted_bboxes[:, 1] += shift_array[1]
    shifted_bboxes[:, 2] += shift_array[0]
    shifted_bboxes[:, 3] += shift_array[1]

    scale_factor = np.random.uniform(*scale_range)
    # print(scale_factor)

    box_widths = shifted_bboxes[:, 2] - shifted_bboxes[:, 0]
    box_heights = shifted_bboxes[:, 3] - shifted_bboxes[:, 1]

    centers_x = shifted_bboxes[:, 0] + box_widths / 2
    centers_y = shifted_bboxes[:, 1] + box_heights / 2

    new_widths = box_widths * scale_factor
    new_heights = box_heights * scale_factor

    shifted_bboxes[:, 0] = centers_x - new_widths / 2
    shifted_bboxes[:, 1] = centers_y - new_heights / 2
    shifted_bboxes[:, 2] = centers_x + new_widths / 2
    shifted_bboxes[:, 3] = centers_y + new_heights / 2

    shifted_bboxes = np.clip(shifted_bboxes, -1, 1)


    return shifted_bboxes

def draw_plot(df, exp_name):
    total = len(df)
    if 'Angle2d' not in df.columns:
        df['Angle2d'] = df['PiratesManualBatAngle']

    df[df['pred_conf_angle']>0]['pred_conf_angle'].hist()
    # df['angle_confidence'].hist()

    def mean_error_by_confidence(df, error_col, kf_conf_threshold, conf_col):
        if kf_conf_threshold is not None:
            df_filtered = df[df['kf_conf'] >= kf_conf_threshold]
        else:
            df_filtered = df

        df_sorted = df_filtered.sort_values(by=conf_col, ascending=False)
        confidences = np.arange(0.1, 1, 0.1)

        mean_errors = []
        for conf in confidences:
            mean_error = df_sorted[df_sorted[conf_col] >= conf][error_col].mean()
            mean_errors.append(mean_error)

        recall = len(df_sorted[df_sorted[conf_col] >= 0.8])
        return confidences, mean_errors,recall

    plt.figure(figsize=(14, 10))
    df['error_Angle2d'] = abs(df['Angle2d'] - df['gt_est_angle'])
    confidences, mean_errors_Angle2d, recall = mean_error_by_confidence(df, 'error_Angle2d', None, 'gt_conf_angle')
    plt.plot(confidences, mean_errors_Angle2d, marker='o', label=f'GT keyframe recall {recall/total:.2f}')
    df['error_Angle2d'] = abs(df['Angle2d'] - df['pred_est_angle'])

    # Thresholds to consider for filtering
    thresholds = np.arange(0.1, 1, 0.1)
    for threshold in thresholds:
        confidences, mean_errors_Angle2d, recall = mean_error_by_confidence(df, 'error_Angle2d', threshold, 'pred_conf_angle')
        plt.plot(confidences, mean_errors_Angle2d, marker='o', label=f'Box Confidence Threshold {threshold:.2f}, recall {recall/total:.2f}')

    plt.xlabel('Angle Confidence')
    plt.ylabel('Mean Error of 2D Bat Angle')
    plt.title('Mean 2D Bat Angle Error vs. Angle Confidence')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./{exp_name}.png")
