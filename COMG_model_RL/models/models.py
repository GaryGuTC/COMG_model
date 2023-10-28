import numpy as np
import torch
import torch.nn as nn

from modules.base_cmn import BaseCMN
from modules.visual_extractor import VisualExtractor
import copy
cp = copy.deepcopy

total_disease_list = ["fracture",
                "pneumonia", "nodule", "opacity" ,"consolidation", "edema", "atelectasis", "lesion", "infiltrate", "mass", "emphysema", "fibrosis", 
                "cardiomegaly", "cardiomediastinal",
                "hernia" ]

disease_caption = [
    "fracture <SEP> normal", 
    "pneumonia <SEP> nodule <SEP> opacity <SEP> consolidation <SEP> edema <SEP> atelectasis <SEP> lesion <SEP> infiltrate <SEP> mass <SEP> emphysema <SEP> fibrosis <SEP> normal",
    "cardiomegaly <SEP> enlarged cardiomediastinal <SEP> normal",
    "hernia <SEP> normal"
]

def generate_disease_prompt(embeding_model, tokenizer, batch_size, device):
    four_caption = []
    for each_caption in disease_caption:
        for each in each_caption.split(" "):
            tmp = []
            tmp.append(tokenizer.token2idx[each]) 
            prompt = torch.tensor(tmp)
            prompt = prompt.expand(batch_size, *prompt.shape).to(device)
            four_caption.append(embeding_model(prompt)) 
    return four_caption
    
class BaseCMNModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseCMNModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = BaseCMN(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, 
                        images, 
                        image_mask_bone,
                        image_mask_lung, 
                        image_mask_heart, 
                        image_mask_mediastinum, 
                        disease_detected = None, 
                        targets=None, 
                        mode='train', 
                        update_opts={}):
        ########## generate disease token ##########
        disease_prompts = generate_disease_prompt(self.encoder_decoder.model.tgt_embed, self.tokenizer, images.shape[0], images.device) 
        disease_token_target = self.encoder_decoder.model.tgt_embed(disease_detected) if disease_detected != None else None
        ########## generate disease token ##########
        disease_token_feature_0, mask_feature_0, att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0], image_mask_bone[:,0],image_mask_lung[:,0], image_mask_heart[:,0], image_mask_mediastinum[:,0], disease_prompts)
        disease_token_feature_1, mask_feature_1, att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1], image_mask_bone[:,1],image_mask_lung[:,1], image_mask_heart[:,1], image_mask_mediastinum[:,1], disease_prompts)
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        mask_feats = torch.cat((mask_feature_0, mask_feature_1), dim=1)
        disease_token_feats = torch.cat((disease_token_feature_0, disease_token_feature_1), dim=1)
        if mode == 'train':
            text_embeddings, output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output, mask_feats, text_embeddings, disease_token_target, disease_token_feats
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError

    def forward_mimic_cxr(self, 
                          images, 
                          image_mask_bone,
                          image_mask_lung, 
                          image_mask_heart,
                          image_mask_mediastinum, 
                          disease_detected_bone = None,
                          disease_detected_lung = None, 
                          disease_detected_heart = None,
                          disease_detected_mediastinum = None, 
                          targets=None, 
                          mode='train', 
                          update_opts={}):
        ########## generate disease token ##########
        disease_prompts = generate_disease_prompt(self.encoder_decoder.model.tgt_embed, self.tokenizer, images.shape[0], images.device) 
        disease_token_target_bone = self.encoder_decoder.model.tgt_embed(disease_detected_bone) if disease_detected_bone != None else None 
        disease_token_target_lung = self.encoder_decoder.model.tgt_embed(disease_detected_lung) if disease_detected_lung != None else None
        disease_token_target_heart = self.encoder_decoder.model.tgt_embed(disease_detected_heart) if disease_detected_heart != None else None
        disease_token_target_mediastinum = self.encoder_decoder.model.tgt_embed(disease_detected_mediastinum) if disease_detected_mediastinum != None else None
        disease_token_target = {
            "bone":disease_token_target_bone,
            "lung":disease_token_target_lung,
            "heart":disease_token_target_heart,
            "mediastinum":disease_token_target_mediastinum
        }
        ########## generate disease token ##########
        mask_feature, att_feats, fc_feats, saved_disease_token = self.visual_extractor(images, image_mask_bone, image_mask_lung, image_mask_heart, image_mask_mediastinum, disease_prompts)
        saved_disease_tokens = {
            "bone":saved_disease_token[0],
            "lung":saved_disease_token[1],
            "heart":saved_disease_token[2],
            "mediastinum":saved_disease_token[3]}
        if mode == 'train':
            text_embeddings, output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output, mask_feature, text_embeddings, disease_token_target, saved_disease_tokens
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError
