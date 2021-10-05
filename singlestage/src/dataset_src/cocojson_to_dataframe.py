import json
import pandas as pd

def _readJson(jsonPath):
  with open(jsonPath,"r") as f:
    return json.load(f)

def _refineBbox(bbox):
  x1,y1,width,heigth = bbox
  return [x1,y1,round(x1+width,1),round(y1+heigth,1)]

def _refineData(data):
	dataList = []
	for row in data['annotations']:
		imgId = row['image_id']
		imgRow = data['images'][imgId]
		categoryId = row["category_id"]
		categoryRow = data["categories"][categoryId]
		dataList.append({
			"image_id": imgId,
			"width": imgRow["width"],
			"height" : imgRow["height"],
			"file_name" : imgRow["file_name"],
			"category_id" : categoryId,
			"category_name" : categoryRow["name"],
			"area" : row["area"],
			"bbox" : _refineBbox(row["bbox"]),
			"iscrowd" : row["iscrowd"]
		})
	return dataList


def jsonToDataframe(jsonPath):
  rowData = _readJson(jsonPath)
  refineData = _refineData(rowData)

  return pd.DataFrame(refineData)  
