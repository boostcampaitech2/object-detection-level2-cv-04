import numpy as np
import pandas as pd
import os
import json
from pandas import json_normalize

# annotation data 읽기
with open('/opt/ml/detection/object-detection-level2-cv-04/dataset/train.json', 'r') as f:
    json_data = json.load(f)
df_images = json_normalize(json_data['images'])
df_annotations = json_normalize(json_data['annotations'])

# 주석 제거 목록
delete_list = list()
with open('/opt/ml/detection/object-detection-level2-cv-04/dataset/eda/delete.txt', 'r') as d:
    for delete_data in d.readlines():
        if delete_data == '': # 예외처리
            continue
        delete_list.append(int(delete_data.strip()))
print("주석 제거 : ", len(delete_list))

# 주석 제거
print("삭제 전 주석 개수 : ", len(df_annotations))
df_annotations.drop(index=delete_list, axis=0, inplace=True)
print("삭제 후 주석 개수 : ", len(df_annotations))

# 이미지 제거 목록
total_img_index = set(range(0,len(df_images)))
check_img_index = set(df_annotations.image_id.unique())
delete_img_index = list(total_img_index - check_img_index)
# print(delete_img_index)
print("이미지 제거 : ", len(delete_img_index))

# 이미지 제거
print("삭제 전 이미지 개수 : ", len(df_images))
df_images.drop(index=delete_img_index, axis=0, inplace=True)
print("삭제 후 이미지 개수 : ", len(df_images))

# print("260번 수정 전 클래스 : ", df_annotations.loc[260, 'category_id']) # 수정 전 확인 1
# 주석 업데이트
with open('/opt/ml/detection/object-detection-level2-cv-04/dataset/eda/update.txt', 'r') as d:
    for update_data in d.readlines():
        if update_data == '': # 예외처리
            continue
        idx, class_ = map(int, update_data.strip().split('>'))
        # 주석 업데이트
        df_annotations.loc[idx, 'category_id'] = class_
# print("260번 수정 후 클래스 : ", df_annotations.loc[260, 'category_id']) # 수정 후 확인 0

# json 파일 생성
json_images = df_images.transpose().to_dict().values()
json_annotations = df_annotations.transpose().to_dict().values()

# id 순으로 정렬
json_images = sorted(json_images, key=lambda image_: image_['id'])
json_annotations = sorted(json_annotations, key=lambda annotation_: annotation_['id'])

json_origin_dict = dict()
#기존 id 제거 새 id를 index와 맞추기
for i in range(len(json_images)):
    json_origin_dict[json_images[i]['id']] = i # 바뀌기전 백업
    json_images[i]['id'] = i
for i in range(len(json_annotations)):
    json_annotations[i]['id'] = i
    temp = json_origin_dict[json_annotations[i]['image_id']]
    json_annotations[i]['image_id'] = temp

json_data['images'] = json_images
json_data['annotations'] = json_annotations
with open("./train_modify.json", "w") as new_file:
	json.dump(json_data, new_file, indent='\t')