from train.src.config_arg_parser import ConfigArgParser
from train.src.set_seed import setSeed
from train.src.register_dataset import registerDataset
from config.load_config import makeDictByConfig

from importlib import import_module

import os

def main():

	# Call Initialization File 
	customDict = makeDictByConfig()

	# Set Seed
	setSeed(customDict.hyperparam.seed)
	
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

	# Set Detectron Config by Arg
	parser = ConfigArgParser(customDict)

	# Set Mapper
	# TODO: 문자열로 되어있는 부분 바꾸기
	# mapperModule = getattr(import_module("train.mapper.base_mapper"), "BaseMapper")
	mapperModule = getattr(import_module("train.mapper.custom_mapper"), "CustomMapper")
	mapper = mapperModule()
	
	# Set Trainer
	# TODO: 문자열로 되어있는 부분 바꾸기
	trainerModule = getattr(import_module("train.trainer.base_trainer"), "BaseTrainer")
	trainerModule.outputEvalDir = customDict.path.output_eval_dir
	trainerModule.mapper = mapper
	
	print(trainerModule.outputEvalDir)

	# Start Train
	# TODO: 세분화하기
	os.makedirs(customDict.path.output_dir)
	# os.makedirs(trainerModule.outputEvalDir)
	trainer = trainerModule(cfg = parser.getConfig())
	trainer.resume_or_load(resume=customDict.option.resume)
	trainer.train()
	
if __name__ == "__main__":
	main()