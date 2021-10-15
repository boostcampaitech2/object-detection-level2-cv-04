import numpy as np
import pandas as pd
import json
from pandas import json_normalize

# 기존의 json data (Labeled data) ######################################################################
labeled_json_data_path = "./split/train_0.json" # 경로수정 필요
with open(labeled_json_data_path) as f:
	labeled_data = json.load(f)

# width     height      file_name      license     flickr_url  coco_url    date_captured  id
df_images = json_normalize(labeled_data['images'])
# image_id  category_id     area    bbox    iscrowd     id
df_annotations = json_normalize(labeled_data['annotations'])

# 마지막 요소의 값들 가져오기
# [1024 1024 'train/4882.jpg' 0 None None '2020-12-23 16:20:30' 4882]
width, height, _, license, flickr_url, coco_url, date_captured, image_id_1 = df_images.tail(1).values[0]
# [4882 1 149633.22 list([145.4, 295.4, 420.2, 356.1]) 0 23143]
image_id_2, category_id, area, bbox, iscrowd, anno_id = df_annotations.tail(1).values[0]

# 예측을 통해 나온 데이터 (Unlabeled data) ######################################################################
submission_csv = './submission/submission_libracnn.csv' # 경로수정 필요
data = pd.read_csv(submission_csv, keep_default_na=False)
data = data.values.tolist()
# print(data.head(5)) # class, confidence, x1, y1, x2, y2 형태

unlabeled = dict() # json 변환을 위한 dictionary
unlabeled['images'] = []
unlabeled['annotations'] = []
confidence_threshold = None

for predict, image in data:
    if predict == None: # 예측하지 못한 데이터는 pass
        continue
    predict = predict.strip() # 띄어쓰기만 있는 경우가 있을 수 있음
    if predict == '': # 예측하지 못한 데이터는 pass
        continue

    count = 0 # annotation 개수 체크
    split_predict = predict.split(' ')
    anns_length = len(split_predict) // 6 # annotation 개수
    
    image_save = False # 이미지 저장 여부
    temp_image = dict()
    for i in range(anns_length):
        temp_annotation = dict()

        class_ = int(split_predict[i*6])
        confidence = float(split_predict[(i*6)+1])
        Left = float(split_predict[(i*6)+2])
        Top = float(split_predict[(i*6)+3])
        Right = float(split_predict[(i*6)+4])
        Bottom = float(split_predict[(i*6)+5])
        Width = Right - Left
        Height = Bottom - Top
        Area = round(Width * Height, 2)
        if confidence_threshold != None: # confidence Threshold 걸은 경우
            if confidence < confidence_threshold:
                continue
        
        # Image 추가
        if image_save == False: # 추가된 이미지인지 확인
            image_id_2 += 1
            temp_image['width'] = width # 마지막 데이터 그대로 이용
            temp_image['height'] = height # 마지막 데이터 그대로 이용
            temp_image['file_name'] = image
            temp_image['license'] = license # 마지막 데이터 그대로 이용
            temp_image['flickr_url'] = flickr_url # 마지막 데이터 그대로 이용
            temp_image['coco_url'] = coco_url # 마지막 데이터 그대로 이용
            temp_image['date_captured'] = date_captured # 마지막 데이터 그대로 이용
            temp_image['id'] = image_id_2
            image_save = True

        # Annotation 추가
        anno_id += 1
        count += 1
        temp_annotation['image_id'] = image_id_2
        temp_annotation['category_id'] = class_
        temp_annotation['area'] = Area
        temp_annotation['bbox'] = [round(Left, 1), round(Top, 1), round(Width, 1), round(Height, 1)]
        temp_annotation['iscrowd'] = iscrowd # 마지막 데이터 그대로 이용
        temp_annotation['id'] = anno_id

    if count > 0: # 주석이 그려진게 있다면
        unlabeled['images'].append(temp_image)
        unlabeled['annotations'].append(temp_annotation)

# Labeled Data + Unlabeled Data ################################################################################
labeled_data['images'] += unlabeled['images']
labeled_data['annotations'] += unlabeled['annotations']
with open("./train_new.json", "w") as new_file:
	json.dump(labeled_data, new_file)