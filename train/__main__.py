from train.src.config_arg_parser import ConfigArgParser
from train.src.set_seed import setSeed
from train.src.register_dataset import registerDataset
from train.src.module_caller import getModule
from config.save_config import saveConfig

import os

def main():

	# Set Detectron Config by Arg
	parser = ConfigArgParser()
	customDict = parser.getCustomDict()

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


	# Set Mapper
	mapperModule = getModule("mapper",customDict.name.mapper)
	mapper = mapperModule()
	
	# Set Sampler
	samplerModule = getModule("sampler",customDict.name.sampler)
	sampler = samplerModule(size = 3902, seed=customDict.hyperparam.seed)

	# Set Trainer
	trainerModule = getModule("trainer",customDict.name.trainer)
	trainerModule.outputEvalDir = customDict.path.output_eval_dir
	trainerModule.mapper = mapper
	trainerModule.sampler = sampler

	# Start Train
	os.makedirs(customDict.path.output_dir)
	saveConfig(customDict.path.output_dir, customDict)
	trainer = trainerModule(cfg = parser.getConfig())
	trainer.resume_or_load(resume=customDict.option.resume)
	trainer.train()
	
if __name__ == "__main__":
	main()
