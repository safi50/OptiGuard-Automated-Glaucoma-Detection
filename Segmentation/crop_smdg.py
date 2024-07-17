import math
import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import io

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

import time


class Segmentation():
    def __init__(self, MODEL_PATH):
        self.cfg = get_cfg()
        self.cfg.MODEL.DEVICE = "cpu"
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.INPUT.MASK_FORMAT = "bitmask"

        self.cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
        self.cfg.INPUT.MAX_SIZE_TRAIN = 1024

        self.cfg.INPUT.MIN_SIZE_TEST = 1024
        self.cfg.INPUT.MAX_SIZE_TEST = 1024

        self.cfg.DATALOADER.NUM_WORKERS = 8
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.SOLVER.IMS_PER_BATCH = 1
        self.cfg.SOLVER.BASE_LR = 1e-4
        self.cfg.SOLVER.MAX_ITER = 10200
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

        self.cfg.MODEL.WEIGHTS = MODEL_PATH
        self.predictor = DefaultPredictor(self.cfg)

    def cropImage(self, image, cup_mask, disc_mask):
        cup_indices = np.argwhere(cup_mask)
        disc_indices = np.argwhere(disc_mask)
        cup_bbox = cv2.boundingRect(cup_indices) if len(cup_indices) > 0 else None
        disc_bbox = cv2.boundingRect(disc_indices) if len(disc_indices) > 0 else None
        cup_cropped = image[cup_bbox[1]:cup_bbox[1] + cup_bbox[3], cup_bbox[0]:cup_bbox[0] + cup_bbox[2]] if cup_bbox is not None else None
        disc_cropped = image[disc_bbox[1]:disc_bbox[1] + disc_bbox[3], disc_bbox[0]:disc_bbox[0] + disc_bbox[2]] if disc_bbox is not None else None
        return cup_cropped, disc_cropped

    def doInference(self,imagePath):
        img = cv2.imread(imagePath)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        outputs = self.predictor(img)
        pred_classes = outputs["instances"].pred_classes.to("cpu").numpy()
        pred_masks = outputs["instances"].pred_masks.to("cpu").numpy()
        scores = outputs["instances"].scores.to("cpu").numpy()
    
        optic_disc_mask_pred = np.zeros_like(img[:, :, 0])
        optic_cup_mask_pred = np.zeros_like(img[:, :, 0])

        if np.any(pred_classes == 0):
            optic_disc_idx = pred_classes == 0
            highest_disc_score_idx = np.argmax(scores[optic_disc_idx])
            optic_disc_mask_pred = pred_masks[optic_disc_idx][highest_disc_score_idx]
        
        if np.any(pred_classes == 1):
            optic_cup_idx = pred_classes == 1
            highest_cup_score_idx = np.argmax(scores[optic_cup_idx])
            optic_cup_mask_pred = pred_masks[optic_cup_idx][highest_cup_score_idx]

        return optic_disc_mask_pred, optic_cup_mask_pred
    
    def doSegmentation(self,imagePath,croppedPath,imageId):
        imagePath=os.path.join(imagePath,imageId)
        disc_mask, cup_mask=self.doInference(imagePath)

        cup_cropped, disc_cropped=self.cropImage(cv2.imread(imagePath),cup_mask,disc_mask)

        disc_cropped_path=os.path.join(croppedPath,imageId)


        if disc_cropped is None:
            print("Error: Unable to read the image.")
        elif disc_cropped.size == 0:
            print("Error: The image is empty.")
        else:
            cv2.imwrite(disc_cropped_path,disc_cropped)
            #print("Cropped Image saved at ",disc_cropped_path)


seg=Segmentation("output/model_final.pth")
inputPath="/Users/huzaifa/Downloads/smdg/full-fundus/full-fundus/"
outputPath="cropped"


filenames=[]
for imageId in os.listdir(inputPath):
    filenames.append(imageId)

filesDone=[]
for imageId in os.listdir(outputPath):
    filesDone.append(imageId)

for i in range(len(filenames)):
    if filenames[i] in filesDone:
        print(f"{i}:File Already Done")
    else:
        seg.doSegmentation(inputPath,outputPath,filenames[i])
        filesDone.append(filenames[i])
        print(f"{i}:File Done")

    
print("Files Done: ",len(filesDone))
print("Files in Output Folder: ",len(os.listdir(outputPath)))