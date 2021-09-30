import configparser
import re

def makeDictByConfig():
		parser = configparser.ConfigParser()
		parser.read('config/config.ini', encoding='utf-8')
		
		cfgDict = {}
		
		def refineVal(x):
			key,val = x
			
			if val.isdigit():
				val = int(val)

			elif re.match(r'^-?\d+(?:\.\d+)$', val):
				val = float(val)

			elif re.match("^\(",val):
				val = tuple(map(int,(val[1:-1].split(","))))
		
			return key, val

		for sect in parser.sections():
			item = [refineVal(x) for x in parser.items(sect)]
			cfgDict[sect] = dict(item)
		
		return cfgDict