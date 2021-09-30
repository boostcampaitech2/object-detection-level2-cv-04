from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader
import os

class BaseTrainer(DefaultTrainer):

  outputEvalDir = ""
  mapper = None
  
  def __init__(self, cfg):
    super().__init__(cfg)

  @classmethod
  def build_train_loader(cls, cfg, sampler=None):
      return build_detection_train_loader(
      cfg, mapper = cls.mapper, sampler = sampler
      )
    
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
      os.makedirs(cls.outputEvalDir)
      output_folder = cls.outputEvalDir
        
    return COCOEvaluator(dataset_name, cfg, False, output_folder)