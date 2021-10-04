import argparse
from detectron2 import model_zoo
from detectron2.config import get_cfg 

class ConfigArgParser:
	
	def __init__(self, customDict):
		self._addArgParser()
		self.config = get_cfg()

		self._addArgParser()
		
		self._mergeConfig(customDict.name.modelzoo_config)
		self._modifyConfig(customDict)


	# TODO
	# arg따라 config 바꿔주기
	def _addArgParser(self):
		# print(dict(self.cfg))
		pass
	
	def getConfig(self):
		return self.config

	def _mergeConfig(self, configName):
		self.config.merge_from_file(model_zoo.get_config_file(configName))


	def _modifyConfig(self, cDict):
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

		self.config.TEST.EVAL_PERIOD = cDict.hyperparam.test_eval_period
