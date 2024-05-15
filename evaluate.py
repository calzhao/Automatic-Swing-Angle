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

from utils import saveVid, get_prediction, angle_estimation, frame_detection_v2, draw_plot
from model_utils import load_model

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

def kf_angle_pred_with_load(det_res_root, vid_file, model, gt_keyframe=None, kf_range=(190, 250), win_len=3):
    f_path = os.path.join(det_res_root, vid_file+'.npz')
    vid_info = np.load(f_path)
    return kf_angle_pred(vid_info, model, gt_keyframe, kf_range, win_len=win_len)

def kf_angle_pred(vid_info, model, gt_keyframe=None, kf_range = (190, 250), win_len=3):
    boxes = vid_info['boxes']
    scores = vid_info['scores']
    masks = vid_info['masks']

    _, H, W = masks.shape
    keyframe, kf_conf = frame_detection_v2(model, scores, boxes, kf_range=kf_range, 
                                           win_len=win_len)

    est_angle, est_conf_angle = -1, 0
    if keyframe > 0:
        kf_mask = masks[keyframe-kf_range[0]]
        est_angle, est_conf_angle = angle_estimation(kf_mask)
    # else:
    #     est_angle, est_conf_angle = -1, 0

    gt_angle, gt_conf_angle = -1, 0
    if gt_keyframe:
        score = scores[gt_keyframe-kf_range[0]]
        # print(gt_keyframe, score)
        if score == 0:
            gt_angle, gt_conf_angle = -1, 0
        else:
            gt_mask = masks[gt_keyframe-kf_range[0]]
            gt_angle, gt_conf_angle = angle_estimation(gt_mask)

    return keyframe, kf_conf, est_angle, est_conf_angle, gt_angle, gt_conf_angle


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
        "--win_size",
        help="the context window size",
        default=5,
        type=int
    )
    parser.add_argument(
        "--vid_source_root",
        help="The path to the video root",
        default="./demo_videos",
        type=str
    )
    parser.add_argument(
        "--det_result_root",
        help="The path to the video root",
        default=None,
        type=str
    )
    parser.add_argument(
        "--result_path",
        help="the path to save the keyframe detection results",
        default='./BatEstimation_v2.csv',
        type=str
    )
    parser.add_argument(
        "--vid_link_input",
        help="file containing video links",
        default=None,
        type=str
    )
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    predictor = VisualizationDemo(cfg, args)
    vid_source_root = args.vid_source_root
    det_result_root = args.det_result_root
    vid_link_input = args.vid_link_input

    exp_name = "batangle_v2"
    file_name = "/v2.pth"
    model_path = "./checkpoint/"+exp_name+file_name
    window_size = args.win_size
    model = load_model(model_path,win_len=window_size)
    model.eval()

    name_list, keyframe_list, est_angle_list, est_conf_angle_list, \
        keyframe_conf_list, gt_anle_list, gt_conf_angle_list = [], [], [], [], [], [], []
    if vid_link_input != None:
        vid_link_file = pd.read_csv(vid_link_input)
        pitch_list = list(vid_link_file['PitchId'].values)
        vid_list = list(vid_link_file['VideoLink'].values)
    else:
        vid_list = sorted(os.listdir(vid_source_root))
    proc_bar = tqdm(range(len(vid_list)))

    #with open(args.result_path, 'w') as file:
    #    file.write('PitchID,KeyFrame_Pred,Keyframe_Confidence,Angle_Est,Angle_Conf\n')

    for idx in proc_bar:
        if vid_link_input != None:
            vid_name = pitch_list[idx]
        else:
            vid_name = vid_list[idx].split('.')[0]
        kf_gt = None
        if det_result_root:
            saveVid(predictor, vid_name, vid_source_root, det_result_root, args, proc_bar=proc_bar)
            kf_idx, kf_conf, est_angle, est_conf_angle, gt_anle, gt_conf_angle \
                = kf_angle_pred_with_load(det_result_root, vid_name, model,
                gt_keyframe = kf_gt ,kf_range = (args.left_win, args.right_win), 
                win_len=window_size)
        else:
            pred_info = get_prediction(predictor, vid_name, vid_source_root, \
                    window=(args.left_win,args.right_win), proc_bar=proc_bar, vid_link=vid_list[idx])
            if pred_info == -1:
                continue
            kf_idx, kf_conf, est_angle, est_conf_angle, gt_anle, gt_conf_angle \
                    = kf_angle_pred(pred_info, model, gt_keyframe = kf_gt,
                                    kf_range = (args.left_win, args.right_win), 
                                    win_len=window_size)       

        str_to_write = f'{vid_name},{kf_idx},{kf_conf},{est_angle},{est_conf_angle}\n'
        with open(args.result_path, 'a') as file:
            file.write(str_to_write)
