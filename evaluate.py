import argparse
import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd

from detectron2.config import get_cfg
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.predictor import VisualizationDemo

from utils import saveVid, get_prediction, angle_estimation, frame_detection

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg

def kf_angle_pred_with_load(det_res_root, vid_file, kf_range, thresh=float('inf')):
    f_path = os.path.join(det_res_root, vid_file+'.npz')
    vid_info = np.load(f_path)
    return kf_angle_pred(vid_info, kf_range, thresh)

def kf_angle_pred(vid_info, kf_range, thresh=float('inf')):
    boxes = vid_info['boxes']
    scores = vid_info['scores']
    masks = vid_info['masks']
    
    keyframe, kf_dist = frame_detection(scores, boxes, thresh, kf_range)
    if keyframe > 0:
        kf_mask = masks[keyframe-kf_range[0]]
        est_angle, conf_angle = angle_estimation(kf_mask)
    else:
        est_angle, conf_angle = -1, 0

    return keyframe, kf_dist, est_angle, conf_angle


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument(
        "--vocabulary",
        default="custom",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default='baseball_bat',
        help="",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'],
        nargs=argparse.REMAINDER,
    )
    
    # self-defined arguments
    parser.add_argument(
        "--left_win",
        help="The starting frame index (inclusive) of the keyframe",
        default=190,
        type=int
    )
    parser.add_argument(
        "--right_win",
        help="The ending frame index (not inclusive) of the keyframe",
        default=250,
        type=int
    )
    parser.add_argument(
        "--dist_thresh",
        help="threshold for filtering the keyframe candidates",
        default=float('inf'),
        type=float
    )
    parser.add_argument(
        "--vid_source_root",
        help="The path to the video root",
        default="./video_data/demo_videos/",
        type=str
    )
    parser.add_argument(
        "--det_result_root",
        help="The path to the video root",
        default= None,
        type=str
    )
    parser.add_argument(
        "--result_path",
        help="the path to save the keyframe detection results",
        default='./result_pred.csv',
        type=str
    )
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    predictor = VisualizationDemo(cfg, args)
    vid_source_root = args.vid_source_root
    det_result_root = args.det_result_root
    if det_result_root:
        os.makedirs(det_result_root, exist_ok=True)

    df = pd.DataFrame(columns=['PitchID', 'Keyframe_Pred', 'Keyframe_BBOX_Dist', 'Angle_Est', 'Angle_Conf'])
    name_list, keyframe_list, kf_dist_list, est_angle_list, conf_angle_list = [], [], [], [], []
    vid_list = sorted(os.listdir(vid_source_root))
    proc_bar = tqdm(range(len(vid_list)))
    for vid_idx in proc_bar:
        vid_name = vid_list[vid_idx].split('.')[0]
        if det_result_root:
            saveVid(predictor, vid_name, vid_source_root, det_result_root, args, proc_bar=proc_bar)
            kf_idx, kf_dist, est_angle, conf_angle = kf_angle_pred_with_load(det_result_root, vid_name,
                                                    kf_range = (args.left_win, args.right_win))
        else:
            pred_info = get_prediction(predictor, vid_name, vid_source_root, window=(args.left_win,args.right_win), proc_bar=proc_bar)
            kf_idx, kf_dist, est_angle, conf_angle = kf_angle_pred(pred_info,
                                                    kf_range = (args.left_win, args.right_win), thresh=args.dist_thresh)        
        
        name_list.append(vid_name)
        keyframe_list.append(kf_idx)
        kf_dist_list.append(kf_dist)
        est_angle_list.append(est_angle)
        conf_angle_list.append(conf_angle)
        proc_bar.set_description('vid [{}]'.format(vid_name))

    df['PitchID'] = name_list
    df['Keyframe_Pred'] = keyframe_list
    df['Keyframe_BBOX_Dist'] = kf_dist_list
    df['Angle_Est'] = est_angle_list
    df['Angle_Conf'] = conf_angle_list
    df.to_csv(args.result_path, index=False)