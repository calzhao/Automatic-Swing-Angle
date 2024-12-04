import argparse
import os
import sys
from tqdm import tqdm
import pandas as pd
import torch

from detectron2.config import get_cfg
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.predictor import VisualizationDemo

from dino_keyframe import  DinoCNN, keyframe_detection, get_prediction, angle_estimation

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

def kf_angle_pred(vid_info):
    box = vid_info['boxes']
    score = vid_info['scores']
    mask = vid_info['masks']

    est_angle, est_conf_angle = -1, 0
    if score > 0:
        est_angle, est_conf_angle = angle_estimation(mask)

    return est_angle, est_conf_angle


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
        "--vid_source_root",
        help="The path to the video root",
        default="./demo_videos",
        type=str
    )
    parser.add_argument(
        "--result_path",
        help="the path to save the keyframe detection results",
        default='./BatEstimation_v2.csv',
        type=str
    )
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    
    predictor = VisualizationDemo(cfg, args)
    vid_source_root = args.vid_source_root
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(device).eval()
    s2f_model = DinoCNN(input_dim=1536).to(device).eval()
    s2f_model.load_state_dict(torch.load('./checkpoint/batangle_v3_dino.pth'))
    s2f_model.eval()

    df = pd.DataFrame(columns=['PitchID', 'Keyframe_Pred', 'Angle_Est', 'Angle_Confidence'])
    name_list, keyframe_list, est_angle_list, est_conf_angle_list = [], [], [], []
    vid_list = sorted(os.listdir(vid_source_root))
    proc_bar = tqdm(range(len(vid_list)))

    for idx in proc_bar:
        vid_name = vid_list[idx].split('.')[0]
        vid_path = os.path.join(vid_source_root, vid_list[idx])

        keyframe, kf_idx = keyframe_detection(vid_path, dino_model, s2f_model, device)
        pred_info = get_prediction(predictor, keyframe)
        est_angle, est_conf_angle = kf_angle_pred(pred_info)

        name_list.append(vid_name)
        keyframe_list.append(kf_idx)
        est_angle_list.append(est_angle)
        est_conf_angle_list.append(est_conf_angle)
        proc_bar.set_description('vid [{}]'.format(vid_name))

    df['PitchID'] = name_list
    df['Keyframe_Pred'] = keyframe_list
    df['Angle_Est'] = est_angle_list
    df['Angle_Confidence'] = est_conf_angle_list
    df.to_csv(args.result_path, index=False)

