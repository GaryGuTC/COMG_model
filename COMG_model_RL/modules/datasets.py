import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
sys.path.append("..")
from preprocess_mask.label_mapper import label_mapper_image_name
from torchvision import transforms
import copy
join = os.path.join
c = copy.deepcopy

need_mask_table = {
    "bone":["ribs", "ribs super"],
    "pleural":[], # None
    "lung":["lung zones", "lung halves", "lung lobes"],
    "heart":["heart region"],
    "mediastinum":["mediastinum"],
}

tfms = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor()
    ])

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
    
class BaseDataset_MIMIC_CXR(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['disease_detected_bone'] = torch.tensor([tokenizer.token2idx[each] for each in self.examples[i]['disease_detected_bone']])
            self.examples[i]['disease_detected_lung'] = torch.tensor([tokenizer.token2idx[each] for each in self.examples[i]['disease_detected_lung']])
            self.examples[i]['disease_detected_heart'] = torch.tensor([tokenizer.token2idx[each] for each in self.examples[i]['disease_detected_heart']])
            self.examples[i]['disease_detected_mediastinum'] = torch.tensor([tokenizer.token2idx[each] for each in self.examples[i]['disease_detected_mediastinum']])
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)
    
class BaseDataset_IU_XRay(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['disease_detected'] = torch.tensor([tokenizer.token2idx[each] for each in self.examples[i]['disease_detected']])
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset_IU_XRay):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        mask_bone_1, mask_lung_1, mask_heart_1, mask_mediastinum_1 = [load_pickle("../COMG_model/data/IU_xray_segmentation/{}/0_mask/{}_concat.pkl".format(image_id, each)) for each in ["bone", "lung", "heart", "mediastinum"]]
        mask_bone_2, mask_lung_2, mask_heart_2, mask_mediastinum_2 = [load_pickle("../COMG_model/data/IU_xray_segmentation/{}/1_mask/{}_concat.pkl".format(image_id, each)) for each in ["bone", "lung", "heart", "mediastinum"]]
        if self.transform is not None:
            image_1, mask_bone_1, mask_lung_1, mask_heart_1, mask_mediastinum_1 = \
                        self.transform(image_1, mask_bone_1, mask_lung_1, mask_heart_1, mask_mediastinum_1)
            image_2, mask_bone_2, mask_lung_2, mask_heart_2, mask_mediastinum_2 = \
                        self.transform(image_2, mask_bone_2, mask_lung_2, mask_heart_2, mask_mediastinum_2)
        image = torch.stack((image_1, image_2), 0)
        image_mask_bone = torch.stack((mask_bone_1, mask_bone_2), 0)
        image_mask_lung = torch.stack((mask_lung_1, mask_lung_2), 0)
        image_mask_heart = torch.stack((mask_heart_1, mask_heart_2), 0)
        image_mask_mediastinum = torch.stack((mask_mediastinum_1, mask_mediastinum_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        disease_detected = example['disease_detected']
        disease_detected_len = len(example['disease_detected'])
        sample = (image_id, image, image_mask_bone, image_mask_lung, image_mask_heart, image_mask_mediastinum, report_ids, report_masks, seq_length, disease_detected, disease_detected_len)
        return sample


class MimiccxrSingleImageDataset(BaseDataset_MIMIC_CXR):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        img_path = os.path.join(self.image_dir, image_path[0])
        image = Image.open(img_path).convert('RGB')
        ################ mask ###########################
        img_path = join(img_path.replace("mimic_cxr","mimic_cxr_segmentation").replace(".jpg",""), image_id)
        total_mask = []
        for need_class,v in need_mask_table.items():
            if len(v) == 0: continue
            masks = []
            for each_class in v:
                for each_img in label_mapper_image_name[each_class]:
                    img = Image.open(join(img_path,"{}.jpg".format(each_img)))
                    masks.append(tfms(img))
                    # masks += tfms(img)
            masks = torch.concat(masks, dim=0)
            total_mask.append(c(masks))
        mask_bone, mask_lung, mask_heart, mask_mediastinum = total_mask
        ################ mask ###########################
        if self.transform is not None:
            # image = self.transform(image)
            image, mask_bone, mask_lung, mask_heart, mask_mediastinum = \
                        self.transform(image, mask_bone, mask_lung, mask_heart, mask_mediastinum)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        
        disease_detected_bone = example['disease_detected_bone']
        disease_detected_lung = example['disease_detected_lung']
        disease_detected_heart = example['disease_detected_heart']
        disease_detected_mediastinum = example['disease_detected_mediastinum']
        
        disease_detected_bone_len = len(example['disease_detected_bone'])
        disease_detected_lung_len = len(example['disease_detected_lung'])
        disease_detected_heart_len = len(example['disease_detected_heart'])
        disease_detected_mediastinum_len = len(example['disease_detected_mediastinum'])
        
        sample = (image_id, 
                  image, 
                  mask_bone, 
                  mask_lung, 
                  mask_heart, 
                  mask_mediastinum,
                  report_ids, 
                  report_masks, 
                  seq_length, 
                  disease_detected_bone,
                  disease_detected_lung, 
                  disease_detected_heart, 
                  disease_detected_mediastinum, 
                  disease_detected_bone_len, 
                  disease_detected_lung_len, 
                  disease_detected_heart_len, 
                  disease_detected_mediastinum_len)
        
        return sample
