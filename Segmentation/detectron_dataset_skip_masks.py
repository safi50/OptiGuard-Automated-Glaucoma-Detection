import os
import json
import csv
import random
import pickle
import cv2
import numpy as np
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import label
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import cv2

# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
from PIL import Image
from sklearn.model_selection import train_test_split

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn
import pandas as pd

from detectron2.structures import BoxMode
from pycocotools import mask as coco_mask
from detectron2.data import DatasetCatalog, MetadataCatalog 



class GlaucomaDataset3(Dataset):
    def __init__(self, root_dir, split='train', max_images=None):
        # self.output_size = output_size
        self.root_dir = root_dir
        self.split = split
        self.images = []
        self.segs = []
        self.max_images = max_images
        self.img_width, self.img_height = (1024,1024)

        self.csv_df = pd.read_csv("G1020.csv")

        self.loaded_image_names = []  # Add this line to initialize an empty list


        # Load data index
        for direct in self.root_dir:
            self.image_filenames = []
            for path in os.listdir(os.path.join(direct, "Images_Square")):
                if(not path.startswith('.')):
                    self.image_filenames.append(path)

            num_images = 0
            for k in range(len(self.image_filenames)):
                # Skip loading if max_images is specified and the limit has been reached
                if max_images is not None and num_images >= max_images:
                    break

                #print('Loading {} image {}/{}...'.format(split, k, len(self.image_filenames)), end='\r')
                img_name = os.path.join(direct, "Images_Square", self.image_filenames[k])
                img = np.array(Image.open(img_name).convert('RGB'))

                if split != 'test':
                    seg_name = os.path.join(direct, "Masks_Square", self.image_filenames[k][:-3] + "png")
                    mask = np.array(Image.open(seg_name, mode='r'))
                    od = (mask==1.).astype(np.float32)
                    if np.any(mask == 2.):
                        oc = (mask == 2.).astype(np.float32)
                    else:
                        oc = np.zeros((1024, 1024), dtype=np.float32)
                    
                    img = transforms.functional.to_tensor(img)
                    self.images.append(img)
                    od = torch.from_numpy(od[None,:,:]) if np.any(od) else torch.zeros((1,1024, 1024), dtype=torch.float32)
                    oc = torch.from_numpy(oc[None,:,:]) if np.any(oc) else torch.zeros((1,1024, 1024), dtype=torch.float32)
                    self.segs.append(torch.cat([od, oc], dim=0))
                    num_images += 1
                    self.loaded_image_names.append(self.image_filenames[k])
                    
            print('Succesfully loaded {} dataset.'.format(split) + ' '*50)

            with open('loaded_image_names.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['image_name'])  # Header
                for name in self.loaded_image_names:
                    writer.writerow([name])

    def __len__(self):
        return len(self.images)
   
    def __getitem__(self, idx):
        # load image
        img = self.images[idx]
        # load segmentation masks (for both optic disk and optic cup)
        seg = self.segs[idx]
        # For instance segmentation, each mask should be a binary mask of shape (H, W).
        # Therefore, we need to split the combined mask into two separate masks.
        od_mask, oc_mask = seg[0], seg[1]
        

        # Find bounding boxes around each mask. The bounding box is represented as
        # [xmin, ymin, width, height], which is the format expected by Mask R-CNN.
        # Find bounding boxes around each mask. The bounding box is represented as
        # [xmin, ymin, width, height], w hich is the format expected by Mask R-CNN.
        od_bbox = torch.tensor(self.mask_to_bbox(od_mask.numpy()))
        oc_bbox = torch.tensor(self.mask_to_bbox(oc_mask.numpy()))

        # print("od_bbox:", od_bbox, "Type:", type(od_bbox), "Shape:", od_bbox.shape)
        # print("oc_bbox:", oc_bbox, "Type:", type(oc_bbox), "Shape:", oc_bbox.shape)



        # od_bbox = np.array(od_bbox, dtype=np.float32)
        # oc_bbox = np.array(oc_bbox, dtype=np.float32)
        # boxes = torch.tensor([od_bbox, oc_bbox], dtype=torch.float32)
        od_bbox = np.array(od_bbox)
        oc_bbox = np.array(oc_bbox)
        
        boxes = torch.tensor(np.array([od_bbox, oc_bbox]))





        # Check that the bounding boxes are valid
        img_height, img_width = self.img_height, self.img_width


        for bbox in [od_bbox]:
                assert bbox[0] >= 0, "xmin should be non-negative"
                assert bbox[1] >= 0, "ymin should be non-negative"
                assert bbox[2] > 0, "width should be positive"
                assert bbox[3] > 0, "height should be positive"
                try:
                    bbox[0] = min(bbox[0], img_width - bbox[2])
                    bbox[1] = min(bbox[1], img_height - bbox[3])
                    assert bbox[0] + bbox[2] <= img_width, "xmin + width should be within the image width"
                    assert bbox[1] + bbox[3] <= img_height, "ymin + height should be within the image height"
                except AssertionError:
                    print(f"Error with bounding box {bbox} for image of size {img.shape}")
                    raise

        # The labels are a tensor of class IDs. In this case, you might want to use
        # 0 for optic disc and 1 for optic cup, as you did when creating the masks.
        labels = torch.tensor([0,1], dtype=torch.int64)

        # Now, we need to put the masks and bounding boxes into the right format.
        # The masks should be a tensor of shape (num_objs, H, W),
        # and the bounding boxes should be in a (num_objs, 4) tensor.
        masks = torch.stack([od_mask, oc_mask])
        boxes = torch.tensor([od_bbox, oc_bbox])


        self.current_direct = self.find_directory_of_image(self.image_filenames[idx])
        img_filename = os.path.join(self.current_direct, "Images_Square", self.image_filenames[idx])
        img_filename_no_ext = os.path.splitext(self.image_filenames[idx])[0]

        csv_row = self.csv_df.loc[self.csv_df['imageID'] == img_filename]        
        # Retrieve 'loc' and 'cup' values from the CSV row
        loc = None
        cup = None
        if not csv_row.empty:
            loc = csv_row['loc'].values[0]
            cup = csv_row['cup'].values[0]


        json_filename = os.path.splitext(self.image_filenames[idx])[0] + ".json"
        json_path = os.path.join(self.current_direct, "Images_Json", json_filename)

        # Check if the JSON file exists
        if not os.path.exists(json_path):
            print(f"JSON Path not found for {json_filename}")
            return None  # or some default value
            

        with open(json_path, 'r') as f:
            data = json.load(f)

        od_points = []
        oc_points = []
        for shape in data['shapes']:
            if shape['label'] == 'disc':
                od_points = shape['points']
            elif shape['label'] == 'cup':
                oc_points = shape['points']

        # Convert points to tensors
        od_points = torch.tensor(od_points, dtype=torch.float32)
        oc_points = torch.tensor(oc_points, dtype=torch.float32)



        # # Pack the bounding boxes and labels into a dictionary
        # target = {"boxes": boxes, "labels": labels, "masks": masks}

        # # Convert bounding box format
        # target['boxes'][:, 2] += target['boxes'][:, 0]  # xmax = xmin + width
        # target['boxes'][:, 3] += target['boxes'][:, 1]  # ymax = ymin + height

        #     # Convert your binary masks to COCO RLE format
        # od_rle = coco_mask.encode(np.asfortranarray(od_mask.numpy().astype("uint8")))
        # oc_rle = coco_mask.encode(np.asfortranarray(oc_mask.numpy().astype("uint8")))

        #         # Convert bounding boxes to Detectron2 format (x_min, y_min, x_max, y_max)
        # od_bbox = [od_bbox[0], od_bbox[1], od_bbox[0] + od_bbox[2], od_bbox[1] + od_bbox[3]]
        # oc_bbox = [oc_bbox[0], oc_bbox[1], oc_bbox[0] + oc_bbox[2], oc_bbox[1] + oc_bbox[3]]
        
        objs = []
        
        # Always add the Optic Disk (assuming it's always present)
        od_bbox = [od_bbox[0], od_bbox[1], od_bbox[0] + od_bbox[2], od_bbox[1] + od_bbox[3]]
        ## Convert your binary masks to COCO RLE format
        od_rle = coco_mask.encode(np.asfortranarray(od_mask.numpy().astype("uint8")))
        od_obj = {
            "bbox": od_bbox,
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 0,
            "segmentation": od_rle  # RLE in Detectron2 format
        }
        objs.append(od_obj)

        # Only add the Optic Cup if it's present
        if np.sum(oc_mask.numpy()) > 0:  # Change this condition based on how you determine the presence of the optic cup
            oc_bbox = [oc_bbox[0], oc_bbox[1], oc_bbox[0] + oc_bbox[2], oc_bbox[1] + oc_bbox[3]]
            oc_rle = coco_mask.encode(np.asfortranarray(oc_mask.numpy().astype("uint8")))
            oc_obj = {
                "bbox": oc_bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 1,
                "segmentation": oc_rle  # RLE in Detectron2 format
            }
            objs.append(oc_obj)
        
        # Record for Detectron2
        record = {}
        record["file_name"] = img_filename
        record["image_id"] = img_filename_no_ext
        record["height"] = self.img_height
        record["width"] = self.img_width
        record["annotations"] = objs

        return record

    def find_directory_of_image(self, image_filename):
        for direct in self.root_dir:
            full_path = os.path.join(direct, "Images_Square", image_filename)
            if os.path.exists(full_path):
                return direct
        return None

    @staticmethod
    def mask_to_bbox(mask):
        # Find the bounding box of a binary mask.
        # This method assumes that the input is a binary mask with 0s and 1s.
        if mask.sum() == 0:
            #print("Empty mask, returning empty bbox...")
            return [0, 0, 1, 1]

        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax - xmin, ymax - ymin]


