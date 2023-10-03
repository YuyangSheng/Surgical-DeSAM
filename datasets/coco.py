# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from glob import glob
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import tifffile
import cv2
import os, sys
import time, json
import copy
import torchvision.transforms.functional as F

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import datasets.transforms as T

from pathlib import Path
from collections import defaultdict
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

def sam_preprocess(img, sam_size=1024):
    # pad
    h, w = img.shape[-2:]
    padh = sam_size - h
    padw = sam_size - w
    img = F.pad(img, (0, padw, 0, padh))
    return img


def make_coco_transforms(image_set):
    sam_size = 1024

    normalize = T.Compose([
        T.ToTensor(),
        # T.Pad(sam_size),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [819]
    # scales = [1024]
    # sizes = [224, 224]
    sizes = [1024, 1024]

    if image_set == 'train':
        return T.Compose([
            # T.RandomHorizontalFlip(),
            # T.RandomResize(scales, max_size=1024),
            T.Resize(sizes),
            # T.RandomSelect(
            #     T.RandomResize(scales, max_size=1280),
            #     T.Compose([
            #         T.RandomResize([400, 500, 600]),
            #         T.RandomSizeCrop(384, 600),
            #         T.RandomResize(scales, max_size=1280),
            #     ])
            # ),
            normalize,
            # T.Pad(sam_size),
        ])

    if image_set == 'val':
        return T.Compose([
            T.Resize(sizes),
            # T.RandomResize(scales, max_size=1024),
            normalize,
            # T.Pad(sam_size),
        ])

    raise ValueError(f'unknown {image_set}')


class EndovisDataset(Dataset):
    def __init__(self, root, transforms, mode='train', dataset='endovis17'):
        self._transforms = transforms
        self.root = root
        assert mode in ['train', 'val'], "mode variable can only be 'train' or 'val'"
        self.mode = mode
        self.dataset_name = dataset

    def __getitem__(self, index):
        if sys.version_info[0] == 2:
            import xml.etree.cElementTree as ET
        else:
            import xml.etree.ElementTree as ET

        if self.dataset_name == 'endovis17':
            INSTRUMENT_CLASSES = ('Bipolar Forceps', 'Prograsp Forceps', 'Large Needle Driver', 'Vessel Sealer',
                'Grasping Retractor', 'Monopolar Curved Scissors', 'Others')
            img_type = '.jpg'
            # There are totally 10 folders and each folder contains 225 frames
            # Choosing the first 8 folders as training set
            if self.mode == 'train':
                self.folder_id = int(index / 225) + 1
                file_idx = index - 225 * int(index / 225)
                assert self.folder_id < 9, "Exceed training set length"      
            elif self.mode == 'val':
                img_type = '.png'
                if index < 300:
                    self.folder_id = 9
                    file_idx = index
                elif index < 585:
                    self.folder_id = 10
                    file_idx = index - 300
                else:
                    print('Out of index')

            train_data_length = 1800
            self.folder_name = f'instrument_dataset_{self.folder_id}'
            image_path_name = 'images'
            ann_path_name = 'xml'
            mask_path_name = 'instruments_masks'

            self.img_folder = os.path.join(self.root, self.folder_name, image_path_name)
            self.ann_folder = os.path.join(self.root, self.folder_name, ann_path_name)
            ann_name = sorted(os.listdir(self.ann_folder))[file_idx]
            mask_path = os.path.join(self.mask_folder, os.path.basename(ann_name[:-4]) + '.png')
            img_path = os.path.join(self.img_folder, os.path.basename(ann_name[:-4]) + img_type)
            xml_path = os.path.join(self.ann_folder, ann_name)


        elif self.dataset_name == 'endovis18':
            INSTRUMENT_CLASSES = ('bipolar_forceps', 'prograsp_forceps', 'large_needle_driver', 'monopolar_curved_scissors',
                'ultrasound_probe', 'suction', 'clip_applier', 'stapler')
            img_type = '.png'

            train_data_length = 1438
            image_path_name = f'{self.mode}_images'
            ann_path_name = f'{self.mode}_xml'
            mask_path_name = f'{self.mode}_masks' # 'train_masks' or 'val_masks'

            self.mask_folder = os.path.join(self.root, mask_path_name)
            self.image_folder = os.path.join(self.root, image_path_name)
            self.ann_folder = os.path.join(self.root, ann_path_name)

            ann_name = sorted(os.listdir(self.ann_folder))[index]
            mask_path = os.path.join(self.mask_folder, ann_name[:-4] + '.png')
            img_path = os.path.join(self.image_folder, ann_name[:-4] + img_type)
            xml_path = os.path.join(self.ann_folder, ann_name)
        
        id = index
        img_id = torch.tensor([int(id)])
        if self.mode == 'val':
            img_id = torch.tensor([int(train_data_length + id)])

        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        # masks
        mask = cv2.imread(mask_path)[..., 0]

        # annotations
        _xml = ET.parse(xml_path).getroot()
        category_to_id = dict(zip(INSTRUMENT_CLASSES, range(len(INSTRUMENT_CLASSES))))

        target = {}
        boxes = []
        classes = [] # category id
        iscrowds = []
        area = []
        anns = []
        multi_mask = []

        # For objects on an image
        for obj in _xml.iter('objects'):
            name = obj.find('name').text.strip()
            # skip 'kidney' object
            if name == 'kidney':
                continue
            bbox = obj.find('bndbox')

            # generate one bounding box
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):       
                cur_pt = int(bbox.find(pt).text)
                bndbox.append(cur_pt)

            area.append((bndbox[2]-bndbox[0])*(bndbox[3]-bndbox[1]))
            boxes.append(bndbox)

            label_id = category_to_id[name]

            single_mask = np.zeros_like(mask)
            single_mask[mask == (label_id + 1)] = 1
            multi_mask.append(single_mask)

            classes.append(label_id) # start from 0
            iscrowds.append(0)
            

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = torch.tensor(classes, dtype=torch.int64)
        area = torch.tensor(area, dtype=torch.float32)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        # In case, there is no-object in one image
        if not multi_mask:
            target_masks = torch.tensor(np.zeros((1, mask.shape[0], mask.shape[1])))
        else:
            target_masks = torch.tensor(np.array(multi_mask))[keep]
        

        target['masks'] = np.array(target_masks)
        target['boxes'] = boxes
        target['orig_boxes'] = boxes
        target['area'] = area[keep]
        target['labels'] = classes
        target['image_id'] = img_id
        target["iscrowd"] = torch.tensor(iscrowds)[keep]


        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
    
    def __len__(self):
        if self.mode == 'train':
            if self.dataset_name == 'endovis17':
                return 225 * 8
            elif self.dataset_name == 'endovis18':
                return 1220
        elif self.mode == 'val':
            if self.dataset_name == 'endovis17':
                return 300 + 285
            elif self.dataset_name == 'endovis18':
                return 548
            
def build(image_set, args):
    # root = Path(args.coco_path)
    root = Path(args.endovis_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    # PATHS = {
    #     "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
    #     "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    # }

    # img_folder, ann_file = PATHS[image_set]
    # dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    # dataset = EnvidosDataset(root, transforms=make_coco_transforms(image_set), mode=image_set, dataset=args.dataset_file)
    dataset = EndovisDataset(root, transforms=make_coco_transforms(image_set), mode=image_set, dataset=args.dataset_file)
    return dataset

# Make endovis dataset has cocoapi's properties
class EndovisCOCO(COCO):
    def __init__(self, mode, dataset=None, root=None):
        # if dataset is None and root is not None:
        #     dataset = EndovisDataset(root, transforms=make_coco_transforms(mode), mode=mode)

        anns, cats, imgs = {}, {}, {}
        self.dataset = dict()
        self.dataset['annotations'], self.dataset['categories'], self.dataset['images'] = [], [], []
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        
        if dataset is not None:
            print('Start creating COCO-format dataset...')
            for idx in range(len(dataset)):
                # print(idx)
                img, anno = dataset[idx]
                

                # denormalize bbox
                img_h, img_w = anno['orig_size']
                scale_fct = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
                anno['boxes'] *= scale_fct

                # # convert boxes format from cxcywh => xywh
                anno['boxes'][:, 0] = anno['boxes'][:, 0] - anno['boxes'][:, 2] * 0.5
                anno['boxes'][:, 1] = anno['boxes'][:, 1] - anno['boxes'][:, 3] * 0.5


                # image_id is not equal to id!!
                # this part 'id' is actually 'image_id'
                if mode == 'train':
                    id = idx
                elif mode == 'val':
                    id = 1438 + idx
                
                anns[id] = anno # image_id
                # self.dataset['annotations'].append(anno)
                
                # create image dictionary
                img_info = {}
                img_info['license'] = 0
                img_info['file_name'] = f'{id:04d}.jpg'
                img_info['coco_url'], img_info['date_captured'], img_info['flickr_url'] = '', '', ''
                img_info['height'], img_info['width'] = anno['orig_size'][0], anno['orig_size'][1]
                img_info['id'] = id
                imgs[id] = img_info
                self.dataset['images'].append(img_info)
            
        # change the endovis dataset to coco-format by creating new anns based on bbox
        id = 0
        annotations = {}
        for img_id in anns.keys():
            ann = anns[img_id]
            for i in range(ann['boxes'].shape[0]):
                id += 1
                new = {'id': id, 'bbox': (ann['boxes'][i]).tolist(), 'category_id': int(ann['labels'][i]), 'area': float(ann['area'][i]), 'iscrowd': 0, 'image_id': img_id}
                
                # update dataset value => 'annotations'
                self.dataset['annotations'].append(new)
                annotations[id] = new
                imgToAnns[img_id].append(new)
                catToImgs[new['category_id']].append(new['image_id'])
            

        # INSTRUMENT_CLASSES = ('Bipolar Forceps', 'Prograsp Forceps', 'Large Needle Driver', 'Vessel Sealer',
        #     'Grasping Retractor', 'Monopolar Curved Scissors', 'Others')
        INSTRUMENT_CLASSES = ('bipolar_forceps', 'prograsp_forceps', 'large_needle_driver', 'monopolar_curved_scissors',
                'ultrasound_probe', 'suction', 'clip_applier', 'stapler')
        
        for i in range(len(INSTRUMENT_CLASSES)):
            cat = {}
            cat['supercategory'] = 'instrument'
            cat['id'] = i+1
            cat['name'] = INSTRUMENT_CLASSES[i]
            cats[i+1] = cat
            self.dataset['categories'].append(cat)

        # create class members
        self.anns = annotations
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

        print('Finish prepared!')

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = EndovisCOCO(mode='val')
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        anns = resFile
        # print('anns:', anns)

        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        # print(annsImgIds, self.getImgIds())
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
               'Results do not correspond to current coco set'
   
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            bb = ann['bbox']
            x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
            if not 'segmentation' in ann:
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann['area'] = bb[2]*bb[3]
            ann['id'] = id+1 # ?
            ann['iscrowd'] = 0

        # print('annotations:', anns )
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res
