import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from pre_processing import *
from yolact.yolact import Yolact
from yolact.eval import evaluate_image
from to_implement import *

cuda_en=0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device",device)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=False, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                        emulate_playback=False)
    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True
    
    if args.seed is not None:
        random.seed(args.seed)

def init_seg_model():
    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')
        
        if cuda_en:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        
        print('Loading model...', end='')
        net = Yolact()
        net.load_weights('./yolact/weights/yolact_base_54_800000.pth')
        net.eval()
        print(' Done.')
        return net    
    
def segmentor(image):
  net=init_seg_model()
  with torch.no_grad():
       if cuda_en:
           net = net.cuda()
       _,mask,predictions,_=evaluate_image(net,image,args)
  return mask,predictions

def module_sol(net):
    parse_args()
    cwd = os.getcwd()
    instance=0
    
    command=load_command(instance) #this function load the verbal instruction from the user.
    
    #current processing environment instance
    image_path=cwd+'/images/image/scene_environment.png'
    #load the image from the iphone. default one is not in correct position so i have to rotate it by 90 clockwise 
    image= cv2.imread(image_path)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = cv2.resize(image, (550,550), interpolation = cv2.INTER_AREA) # resize the image width= 1440 and height= 1920

    #generate depth image
    depth_image_path='./images/depth/depth'+str(instance)+'.txt'
    depth_image=depth_image_generation(depth_image_path)
    depth_image = cv2.resize(depth_image, (550,550), interpolation = cv2.INTER_AREA)

    # Translation Matrix genertation
    matrix_path='./images/trans_matrix/matrix'+str(instance)+'.txt'
    t_matrx = translate_matrix_genertion(matrix_path)

    #Intrinsic camera matrix generation
    in_matrix_path='./images/intric_matrix/matrix'+str(instance)+'.txt'
    intrinsic_matrx= intric_matrix_genertion(in_matrix_path)
    
    obj_class= class_type_resolver_fromtext(command)

    mask,predictions=segmentor(image)
    
    x1,y1,z1,V = pointing_direction_solver(mask,predictions,depth_image,t_matrx,intrinsic_matrx)
    line=[x1,y1,z1,V]
    
    target_point_cloud = advanced_target_resolver(line,obj_class,mask,predictions,depth_image,t_matrx,intrinsic_matrx)
    
    width_dimension = module_dimension(target_point_cloud)
    
    print('object width is ',width_dimension)
    return [[5,6,5,6],width_dimension]

if __name__ == "__main__":
    net=init_seg_model()
    module_sol(net)

