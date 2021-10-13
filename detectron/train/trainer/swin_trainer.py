from train.model.SwinT_detectron2.swint.config import add_swint_config
from train.trainer.base_trainer import BaseTrainer

import itertools
import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.solver.build import maybe_add_gradient_clipping, get_default_optimizer_params


class SwinTrainer(BaseTrainer):

	def __init__(self, cfg, cDict):
			super().__init__(self.setup(cDict))

	@classmethod
	def build_evaluator(cls, cfg, dataset_name, output_folder=None):
			"""
			Create evaluator(s) for a given dataset.
			This uses the special metadata "evaluator_type" associated with each builtin dataset.
			For your own dataset, you can simply create an evaluator manually in your
			script and do not have to worry about the hacky if-else logic here.
			"""
			if output_folder is None:
					output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
			evaluator_list = []
			evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
			if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
					evaluator_list.append(
							SemSegEvaluator(
									dataset_name,
									distributed=True,
									num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
									ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
									output_dir=output_folder,
							)
					)
			if evaluator_type in ["coco", "coco_panoptic_seg"]:
					evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
			if evaluator_type == "coco_panoptic_seg":
					evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
			if evaluator_type == "cityscapes_instance":
					assert (
							torch.cuda.device_count() >= comm.get_rank()
					), "CityscapesEvaluator currently do not work with multiple machines."
					return CityscapesInstanceEvaluator(dataset_name)
			if evaluator_type == "cityscapes_sem_seg":
					assert (
							torch.cuda.device_count() >= comm.get_rank()
					), "CityscapesEvaluator currently do not work with multiple machines."
					return CityscapesSemSegEvaluator(dataset_name)
			elif evaluator_type == "pascal_voc":
					return PascalVOCDetectionEvaluator(dataset_name)
			elif evaluator_type == "lvis":
					return LVISEvaluator(dataset_name, cfg, True, output_folder)
			if len(evaluator_list) == 0:
					raise NotImplementedError(
							"no Evaluator for the dataset {} with the type {}".format(
									dataset_name, evaluator_type
							)
					)
			elif len(evaluator_list) == 1:
					return evaluator_list[0]
			return DatasetEvaluators(evaluator_list)

	@classmethod
	def test_with_TTA(cls, cfg, model):
			logger = logging.getLogger("detectron2.trainer")
			# In the end of training, run an evaluation with TTA
			# Only support some R-CNN models.
			logger.info("Running inference with test-time augmentation ...")
			model = GeneralizedRCNNWithTTA(cfg, model)
			evaluators = [
					cls.build_evaluator(
							cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
					)
					for name in cfg.DATASETS.TEST
			]
			res = cls.test(cfg, model, evaluators)
			res = OrderedDict({k + "_TTA": v for k, v in res.items()})
			return res

	def setup(self, cDict):
		cfg = get_cfg()
		add_swint_config(cfg)
		cfg.merge_from_file("train/model/SwinT_detectron2/configs/SwinT/faster_rcnn_swint_T_FPN_3x.yaml")
		cfg.MODEL.WEIGHTS = "train/model/SwinT_detectron2/configs/SwinT/swin_tiny_patch4_window7_224_d2.pth"
		
		cfg.DATASETS.TRAIN = ((cDict.name.train_dataset,))
		cfg.DATASETS.TEST = ((cDict.name.test_dataset,))

		cfg.DATALOADER.NUM_WOREKRS = cDict.general.num_workers

		cfg.SOLVER.IMS_PER_BATCH = cDict.hyperparam.ims_per_batch
		cfg.SOLVER.BASE_LR = cDict.hyperparam.base_lr
		cfg.SOLVER.MAX_ITER = cDict.hyperparam.max_iter
		cfg.SOLVER.STEPS = cDict.hyperparam.steps
		cfg.SOLVER.GAMMA = cDict.hyperparam.gamma
		cfg.SOLVER.CHECKPOINT_PERIOD = cDict.hyperparam.checkpoint_period

		cfg.OUTPUT_DIR = cDict.path.output_dir

		cfg.SEED = cDict.hyperparam.seed

		cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = cDict.hyperparam.roi_batch
		cfg.MODEL.ROI_HEADS.NUM_CLASSES = cDict.general.roi_num_classes

		cfg.TEST.EVAL_PERIOD = cDict.hyperparam.test_eval_period

		cfg.SOLVER.OPTIMIZER = cDict.general.optimizer

		return cfg