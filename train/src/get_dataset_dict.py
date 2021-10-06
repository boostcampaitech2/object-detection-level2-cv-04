from detectron2.data.build import get_detection_dataset_dicts

def getDatasetDicts(cfg):
	return get_detection_dataset_dicts(
			cfg.DATASETS.TRAIN,
			filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
			min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
			if cfg.MODEL.KEYPOINT_ON
			else 0,
			proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
	)