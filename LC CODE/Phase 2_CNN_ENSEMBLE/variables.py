import os
import sys
import re

WORKER_POOL_SIZE = 4

TARGET_VOXEL_MM = 1.00
MEAN_PIXEL_VALUE_nod = 41
LUNA_SUBSET_START_INDEX = 0
SEGMENTER_IMG_SIZE = 320


NDSB3_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "/nsdb_extracted_images/"
LUNA16_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "/luna16_extracted_images/"
LUNA_16_TRAIN_DIR = BASE_DIR_SSD + "/luna16_extracted_images/"
NDSB3_nod_DETECTION_DIR = BASE_DIR_SSD + "/nsdb_nod_predictions/"
LUNA_nod_DETECTION_DIR = BASE_DIR_SSD + "/luna_nod_predictions/"

BASE_DIR_SSD = "/media/pikachu/Seagate Backup Plus Drive/LC nod Detection"
BASE_DIR = "/media/pikachu/Seagate Backup Plus Drive/LC nod Detection"
EXTRA_DATA_DIR = "/media/pikachu/Seagate Backup Plus Drive/LC nod Detection/resources/"
NDSB3_RAW_SRC_DIR = BASE_DIR + "/nsdb_raw/stages/"
LUNA16_RAW_SRC_DIR = BASE_DIR + "/luna_raw/"

