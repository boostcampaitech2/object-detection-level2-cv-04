from train.src.config_arg_parser import ConfigArgParser
from train.src.set_seed import setSeed
from train.src.register_dataset import registerDataset
from train.src.module_caller import getModule
from config.save_config import saveConfig
from train.src.get_dataset_dict import getDatasetDicts

import os

def main():

	# Set Detectron Config by Arg
	parser = ConfigArgParser()
	customDict = parser.getCustomDict()
	cfg = parser.getConfig()
	
	# Set Seed
	setSeed(customDict.hyperparam.seed)
	
	# print(cfg)
	# exit(0)

	# Register Dataset
	registerDataset(
		datasetName = customDict.name.train_dataset,
		jsonPath = customDict.path.train_json,
		imageDirPath = customDict.path.image_root,
		numOfClasses = customDict.general.classes,
		istrain=True
	)

	registerDataset(
		datasetName = customDict.name.test_dataset,
		jsonPath = customDict.path.test_json,
		imageDirPath = customDict.path.image_root,
		numOfClasses = customDict.general.classes,
		istrain=False
	)

	# Set Mapper
	mapperModule = getModule("mapper",customDict.name.mapper)
	if "albu" in customDict.name.mapper:
		mapper = mapperModule(cfg)
	else:
		mapper = mapperModule()
	
	# Set Sampler
	samplerModule = getModule("sampler",customDict.name.sampler)

	if "repeatfactor" in customDict.name.sampler:
		samplerModule.repeat_factors = getDatasetDicts(cfg)
		samplerModule.threshold = 0.5
	
	sampler = samplerModule(size = 3902, seed=customDict.hyperparam.seed)

	# Set Trainer
	trainerModule = getModule("trainer",customDict.name.trainer)
	trainerModule.outputEvalDir = customDict.path.output_eval_dir
	trainerModule.mapper = mapper
	trainerModule.sampler = sampler

	# Start Train
	os.makedirs(customDict.path.output_dir)
	saveConfig(customDict.path.output_dir, customDict)

	if "swin" in customDict.name.trainer:
		trainer = trainerModule(cfg,customDict)
	else:
		trainer = trainerModule(cfg = cfg)
		

	trainer.resume_or_load(resume=customDict.option.resume)
	trainer.train()
	
if __name__ == "__main__":
	main()
