import argparse
from detectron2 import model_zoo
from detectron2.config import get_cfg 
from config.load_config import makeDictByConfig, refineVal
from datetime import datetime,timedelta
import os

class ConfigArgParser:
	
	def __init__(self):
		self.customDict = self._addArgParser(makeDictByConfig())
		self.customDict = self._refineOutputPath()
		self.config = get_cfg()
		
		self._mergeConfig(self.customDict.name.modelzoo_config)
		self.modifyConfig(self.customDict)

	def _refineOutputPath(self):
		# subPath = "train" if isTrain else "eval"
		nowTime = datetime.now() + timedelta(hours=9)

		cnt = 0
		
		while True:
			addedName = f'{nowTime.strftime("%m-%d")}_{self.customDict.name.custom_model}_{cnt:02}'
			path = os.path.join(self.customDict.path.output_dir,addedName)
			cnt+=1
			
			if not os.path.exists(path) :
				self.customDict.path.output_dir =  os.path.join(path,"train")
				self.customDict.path.output_eval_dir = os.path.join(path,"eval")
				return self.customDict

	def _addArgParser(self, customDict):
		parser = argparse.ArgumentParser()

		parser.add_argument('--train_json', required=False)
		parser.add_argument('--test_json', required=False)
		parser.add_argument('--image_root', required=False)
		parser.add_argument('--modelzoo_config', required=False)
		parser.add_argument('--custom_model', required=False)
		parser.add_argument('--mapper', required=False)
		parser.add_argument('--trainer', required=False)
		parser.add_argument('--sampler', required=False)
		parser.add_argument('--num_workers', required=False)

		parser.add_argument('--optimizer', required=False)
		parser.add_argument('--seed', required=False)
		parser.add_argument('--base_lr', required=False)
		parser.add_argument('--ims_per_batch', required=False)
		parser.add_argument('--max_iter', required=False)
		parser.add_argument('--steps', required=False)
		parser.add_argument('--gamma', required=False)
		parser.add_argument('--checkpoint_period', required=False)
		parser.add_argument('--test_eval_period', required=False)
		parser.add_argument('--roi_batch', required=False)

		args, other = parser.parse_known_args()

		for key,val in vars(args).items():
			if not val == None:
				for mainkey in customDict.keys():
					if key in customDict[mainkey] :
						_, rVal = refineVal((key,val))
						customDict[mainkey][key] = rVal
		
		return customDict
	
	def getConfig(self):
		return self.config

	def getCustomDict(self):
		return self.customDict

	def _mergeConfig(self, configName):
		self.config.merge_from_file(model_zoo.get_config_file(configName))


	def modifyConfig(self, cDict):
		self.config.DATASETS.TRAIN = ((cDict.name.train_dataset,))
		self.config.DATASETS.TEST = ((cDict.name.test_dataset,))

		self.config.DATALOADER.NUM_WOREKRS = cDict.general.num_workers

		self.config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cDict.name.modelzoo_config)

		self.config.SOLVER.IMS_PER_BATCH = cDict.hyperparam.ims_per_batch
		self.config.SOLVER.BASE_LR = cDict.hyperparam.base_lr
		self.config.SOLVER.MAX_ITER = cDict.hyperparam.max_iter
		self.config.SOLVER.STEPS = cDict.hyperparam.steps
		self.config.SOLVER.GAMMA = cDict.hyperparam.gamma
		self.config.SOLVER.CHECKPOINT_PERIOD = cDict.hyperparam.checkpoint_period

		self.config.OUTPUT_DIR = cDict.path.output_dir

		self.config.SEED = cDict.hyperparam.seed

		self.config.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = cDict.hyperparam.roi_batch
		self.config.MODEL.ROI_HEADS.NUM_CLASSES = cDict.general.roi_num_classes
		self.config.MODEL.RETINANET.NUM_CLASSES = cDict.general.roi_num_classes

		self.config.SOLVER.OPTIMIZER = cDict.general.optimizer

		self.config.TEST.EVAL_PERIOD = cDict.hyperparam.test_eval_period


		self.config.MODEL.MASK_ON = False