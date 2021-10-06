import os
import configparser

def saveConfig(path, configDict):
	config = configparser.ConfigParser()

	for mainKey in configDict.keys():
		config[mainKey] = {}
		for subKey, val in configDict[mainKey].items():
			
			if subKey in "classes":
				config[mainKey][subKey] = "  ".join(val)
			elif subKey in "steps":
				config[mainKey][subKey] = ",".join([str(x) for x in val])
			else:
				config[mainKey][subKey] = str(val)

	with open(os.path.join(path,'config.ini'), 'w', encoding='utf-8') as configfile:
		config.write(configfile)
