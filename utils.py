import cv2
import numpy as np
import os
from math import atan2, degrees
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

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

def frame_detection(scores, bboxes, thresh, kf_range=(190, 220)):
    lb, ub = kf_range
    peaks = find_peaks(scores, l_bound=lb, r_bound=ub - 1)
    candidates = context_proposal(scores, peaks, l_bound=lb, r_bound=ub - 1)
    cen_dist, hor_len, ver_len = get_bbox_info(bboxes)
    selected_indices = np.asarray([idx for idx in candidates if cen_dist[idx - lb] < thresh])

    if len(selected_indices) == 0:
        return -1, center_dist(0, 0)

    min_idx = np.argmin(cen_dist[selected_indices - lb])
    detected_frame = selected_indices[min_idx]

    return detected_frame, int(np.min(cen_dist[selected_indices - lb]))

def loadVid(path):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    cap = cv2.VideoCapture(path)
    frames = []

    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
        
    cap.release()
    frames = np.stack(frames)
    return frames

def process_predictions(predictions):
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
        pred_masks = np.zeros((720, 1280))

    return pred_scores.squeeze(), pred_boxes.squeeze(), pred_masks.squeeze()

def saveVid(predictor, vidName, vidSrcRoot, detResRoot, args, proc_bar = None):
    vid_scores, vid_boxes, vid_masks = get_prediction(predictor, vidName, vidSrcRoot, (args.left_win,args.right_win), proc_bar=proc_bar)
    save_path = os.path.join(detResRoot, '{}.npz'.format(vidName))
    np.savez(save_path, boxes=vid_boxes, scores=vid_scores, masks=vid_masks)
    
def get_prediction(predictor, vidName, vidSrcRoot, window, proc_bar = None):
    vid_masks = []
    vid_boxes = []
    vid_scores = []
    video = loadVid(os.path.join(vidSrcRoot, vidName+'.mp4'))
    video = video[window[0]:window[1], :, :]

    for i in range(len(video)):
        if proc_bar:
            proc_bar.set_description('[{:03d}/{:03d}] in {}'.format(i+1, len(video), vidName))
        input_frame = video[i, :, :, :]

        predictions, visualized_output = predictor.run_on_image(input_frame)
        pred_scores, pred_boxes, pred_masks = process_predictions(predictions)

        vid_scores.append(pred_scores)
        vid_boxes.append(pred_boxes)
        vid_masks.append(pred_masks.astype(np.uint8))
        # print(i, pred_scores, pred_scores.shape)

    vid_scores = np.stack(vid_scores)
    vid_boxes = np.stack(vid_boxes)
    vid_masks = np.stack(vid_masks)

    return {'boxes':vid_boxes, 'scores':vid_scores, 'masks':vid_masks}
    
def center_dist(xc, yc, C_X=640, C_Y=360):
    return np.sqrt((xc - C_X) ** 2 + (yc - C_Y) ** 2)

def get_bbox_info(bboxes):
    x_c = (bboxes[:, 0] + bboxes[:, 2]) / 2
    y_c = (bboxes[:, 1] + bboxes[:, 3]) / 2
    euc_dist = center_dist(x_c, y_c)
    
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
        peaks.append(r_bound)

    return np.array(peaks)


def context_proposal(conf_seq, peaks, context_window=1, l_bound=190, r_bound=250):
    candidates = []
    for p in peaks:
        for idx in range(max(p - context_window, l_bound), min(p + context_window + 1, r_bound)):
            if conf_seq[idx - l_bound] > 0.4 and idx not in candidates:
                candidates.append(idx)
    return np.asarray(candidates)