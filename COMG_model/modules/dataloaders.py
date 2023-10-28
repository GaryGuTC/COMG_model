import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset
import torchvision.transforms.functional as TF
import random

def transform_train(image, mask_bone, mask_lung, mask_heart, mask_mediastinum):
    # ref: https://blog.csdn.net/WANGWUSHAN/article/details/105329374
    # Resize
    resize = transforms.Resize(size=(256, 256), antialias=True)
    image = resize(image)
    # mask = resize(mask)
    mask_bone = resize(mask_bone)
    mask_lung = resize(mask_lung)
    mask_heart = resize(mask_heart)
    mask_mediastinum = resize(mask_mediastinum)

    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(
        image, output_size=(224, 224))
    image = TF.crop(image, i, j, h, w)
    mask_bone = TF.crop(mask_bone, i, j, h, w)
    mask_lung = TF.crop(mask_lung, i, j, h, w)
    mask_heart = TF.crop(mask_heart, i, j, h, w)
    mask_mediastinum = TF.crop(mask_mediastinum, i, j, h, w)
    # mask = TF.crop(mask, i, j, h, w)

    # Random horizontal flipping
    horizontal_flipping_prob = random.random()
    if horizontal_flipping_prob > 0.5:
        image = TF.hflip(image)
        mask_bone = TF.hflip(mask_bone)
        mask_lung = TF.hflip(mask_lung)
        mask_heart = TF.hflip(mask_heart)
        mask_mediastinum = TF.hflip(mask_mediastinum)
        # mask = TF.hflip(mask)

    # Random vertical flipping
    vertical_flipping_prob = random.random()
    if vertical_flipping_prob > 0.5:
        image = TF.vflip(image)
        mask_bone = TF.vflip(mask_bone)
        mask_lung = TF.vflip(mask_lung)
        mask_heart = TF.vflip(mask_heart)
        mask_mediastinum = TF.vflip(mask_mediastinum)

    # Transform to tensor
    image = TF.to_tensor(image)
    
    image = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))(image)
    
    return image, mask_bone, mask_lung, mask_heart, mask_mediastinum

def transform_infer(image, mask_bone, mask_lung, mask_heart, mask_mediastinum):
    # ref: https://blog.csdn.net/WANGWUSHAN/article/details/105329374
    # Resize
    resize = transforms.Resize(size=(224, 224), antialias=True)
    image = resize(image)
    mask_bone = resize(mask_bone)
    mask_lung = resize(mask_lung)
    mask_heart = resize(mask_heart)
    mask_mediastinum = resize(mask_mediastinum)
    # mask = resize(mask)

    # Transform to tensor
    image = TF.to_tensor(image)
    
    image = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))(image)
    
    return image, mask_bone, mask_lung, mask_heart, mask_mediastinum

class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transform_train
        else:
            self.transform = transform_infer

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
            self.collate_fn = self.collate_fn_IU_Xray
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
            self.collate_fn = self.collate_fn_MIMIC_CXR

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn_IU_Xray(data):
        (image_id_batch, 
        image_batch, 
        image_mask_bone_batch, 
        image_mask_lung_batch, 
        image_mask_heart_batch, 
        image_mask_mediastinum_batch, 
        report_ids_batch, 
        report_masks_batch, 
        seq_lengths_batch,
        disease_detected,
        disease_detected_len) = zip(*data)
        
        image_batch = torch.stack(image_batch, 0)
        image_mask_bone_batch = torch.stack(image_mask_bone_batch, 0)
        image_mask_lung_batch = torch.stack(image_mask_lung_batch, 0)
        image_mask_heart_batch = torch.stack(image_mask_heart_batch, 0)
        image_mask_mediastinum_batch = torch.stack(image_mask_mediastinum_batch, 0)
        
        max_seq_length = max(seq_lengths_batch)

        disease_batch = torch.zeros((len(report_ids_batch), max(disease_detected_len)), dtype=int)
        target_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
        target_masks_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
        
        for i, disease_ids in enumerate(disease_detected):
            disease_batch[i, :len(disease_ids)] = disease_ids

        for i, report_ids in enumerate(report_ids_batch):
            target_batch[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(report_masks_batch):
            target_masks_batch[i, :len(report_masks)] = report_masks

        return image_id_batch, \
               image_batch, \
               image_mask_bone_batch, \
               image_mask_lung_batch, \
               image_mask_heart_batch, \
               image_mask_mediastinum_batch, \
               torch.LongTensor(target_batch), \
               torch.Tensor(target_masks_batch), \
               disease_batch
        
    @staticmethod
    def collate_fn_MIMIC_CXR(data):
        (image_id_batch, 
         image_batch, 
         image_mask_bone_batch, 
         image_mask_lung_batch, 
         image_mask_heart_batch, 
         image_mask_mediastinum_batch, 
         report_ids_batch, 
         report_masks_batch, 
         seq_lengths_batch,
         disease_detected_bone, 
         disease_detected_lung, 
         disease_detected_heart, 
         disease_detected_mediastinum, 
         disease_detected_bone_len,
         disease_detected_lung_len,
         disease_detected_heart_len, 
         disease_detected_mediastinum_len) = zip(*data)
        image_batch = torch.stack(image_batch, 0)
        image_mask_bone_batch = torch.stack(image_mask_bone_batch, 0)
        image_mask_lung_batch = torch.stack(image_mask_lung_batch, 0)
        image_mask_heart_batch = torch.stack(image_mask_heart_batch, 0)
        image_mask_mediastinum_batch = torch.stack(image_mask_mediastinum_batch, 0)
        
        max_seq_length = max(seq_lengths_batch)

        disease_batch_bone = torch.zeros((len(report_ids_batch), max(disease_detected_bone_len)), dtype=int)
        disease_batch_lung = torch.zeros((len(report_ids_batch), max(disease_detected_lung_len)), dtype=int)
        disease_batch_heart = torch.zeros((len(report_ids_batch), max(disease_detected_heart_len)), dtype=int)
        disease_batch_mediastinum = torch.zeros((len(report_ids_batch), max(disease_detected_mediastinum_len)), dtype=int)
        
        target_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
        target_masks_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
        
        for i, disease_ids in enumerate(disease_detected_bone):
            disease_batch_bone[i, :len(disease_ids)] = disease_ids
            
        for i, disease_ids in enumerate(disease_detected_lung):
            disease_batch_lung[i, :len(disease_ids)] = disease_ids
        
        for i, disease_ids in enumerate(disease_detected_heart):
            disease_batch_heart[i, :len(disease_ids)] = disease_ids
        
        for i, disease_ids in enumerate(disease_detected_mediastinum):
            disease_batch_mediastinum[i, :len(disease_ids)] = disease_ids

        for i, report_ids in enumerate(report_ids_batch):
            target_batch[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(report_masks_batch):
            target_masks_batch[i, :len(report_masks)] = report_masks

        return  image_id_batch, \
                image_batch, \
                image_mask_bone_batch, \
                image_mask_lung_batch, \
                image_mask_heart_batch, \
                image_mask_mediastinum_batch, \
                torch.LongTensor(target_batch), \
                torch.Tensor(target_masks_batch), \
                torch.Tensor(disease_batch_bone), \
                torch.Tensor(disease_batch_lung), \
                torch.Tensor(disease_batch_heart), \
                torch.Tensor(disease_batch_mediastinum)
