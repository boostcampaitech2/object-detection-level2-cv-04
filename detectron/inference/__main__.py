import os
import copy
from tqdm import tqdm
import pandas as pd
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_json', default="../dataset/test.json",required=False)
parser.add_argument('--image_root', default = '../dataset/',required=False)
parser.add_argument('--modelzoo_config', default='COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',required=False)
parser.add_argument('--model_dir', default='../output/detectron/10-14_faster_resx101_fpn_01/train',required=False)
parser.add_argument('--model_name', default="model_0000009",required=False)
parser.add_argument('--output_csv_name',default="det_submission", required=False)

args, other = parser.parse_known_args()


# Register Dataset
try:
    register_coco_instances('coco_trash_test', {}, args.test_json, args.image_root)
except AssertionError:
    pass


# config 불러오기
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(args.modelzoo_config))

# config 수정하기
cfg.DATASETS.TEST = ('coco_trash_test',)

cfg.DATALOADER.NUM_WOREKRS = 2

cfg.OUTPUT_DIR = args.model_dir

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.model_name+".pth")

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3

# model
predictor = DefaultPredictor(cfg)

# mapper - input data를 어떤 형식으로 return할지
def MyMapper(dataset_dict):
    
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    dataset_dict['image'] = image
    
    return dataset_dict

# test loader
test_loader = build_detection_test_loader(cfg, 'coco_trash_test', MyMapper)

# output 뽑은 후 sumbmission 양식에 맞게 후처리 
prediction_strings = []
file_names = []

class_num = 10

for data in tqdm(test_loader):
    
    prediction_string = ''
    
    data = data[0]
    
    outputs = predictor(data['image'])['instances']
    
    targets = outputs.pred_classes.cpu().tolist()
    boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
    scores = outputs.scores.cpu().tolist()
    
    for target, box, score in zip(targets,boxes,scores):
        prediction_string += (str(target) + ' ' + str(score) + ' ' + str(box[0]) + ' ' 
        + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')
        
    prediction_strings.append(prediction_string)
    file_names.append(data['file_name'].replace('./dataset/',''))

submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv(os.path.join(cfg.OUTPUT_DIR,"../../..", args.output_csv_name), index=None)
submission.head()