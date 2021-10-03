import configparser
import re
from easydict import EasyDict as edict
import os
from datetime import datetime, timedelta

def refineVal(x):
	key,val = x
	
	if val.isdigit():
		val = int(val)

	elif re.match(r'^-?\d+(?:\.\d+)$', val):
		val = float(val)

	elif re.match("^\(",val):
		val = tuple(map(int,(val[1:-1].split(","))))
	
	elif key in "classes":
		val = [x for x in val.split("  ")]
	
	elif key in "resume":
		val = True if val in "True" else False

	return key, val

def refineOutputPath(isTrain, outputDir, customName):
	subPath = "train" if isTrain else "eval"
	nowTime = datetime.now() + timedelta(hours=9)

	cnt = 0
	
	while True:
		addedName = f'{nowTime.strftime("%m-%d")}_{customName}_{cnt:02}'
		path = os.path.join(outputDir,addedName,subPath)
		cnt+=1
		
		if not os.path.exists(path) : 
			# os.makedirs(path)
			return path

def makeDictByConfig(isTrain=True):
	parser = configparser.ConfigParser()
	parser.read('config/config.ini', encoding='utf-8')
	
	cfgDict = edict()

	for sect in parser.sections():
		item = [refineVal(x) for x in parser.items(sect)]
		cfgDict[sect] = dict(item)
	
	cfgDict.path.output_dir = refineOutputPath(isTrain, cfgDict.path.output_dir, cfgDict.name.custom_model)
	cfgDict.path.output_eval_dir = refineOutputPath(False, cfgDict.path.output_eval_dir, cfgDict.name.custom_model)
	return cfgDict

