import torch
import torch.nn as nn
import torchvision.models as models
from .base_cmn import MultiThreadMemory,MultiHeadedAttention,LayerNorm
import copy
cp = copy.deepcopy
        
class Reconstructed_resnet101(nn.Module):
        def __init__(self, pretrain):
            super().__init__()
            if pretrain:
                model101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            else:
                model101 = models.resnet101()
            model101 = nn.Sequential(*list(model101.children())[:-2])
        
            self.part1_resnet101 = nn.Sequential(
                *cp(list(model101.children())[:7]) # bs,256,56,56 / bs,1024,14,14
            )
        
            self.transfer_channel = nn.Sequential(
                nn.Conv2d(1024,512,3,padding=1),
                nn.BatchNorm2d(512),
                nn.Conv2d(512,256,3,padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

            self.part2_resnet101 = nn.Sequential(
                *cp(list(model101.children())[7:])
            )
            del model101
        
        def forward(self, x):
            tmp_feature = self.part1_resnet101(x)
            return  self.transfer_channel(tmp_feature), self.part2_resnet101(tmp_feature) # [bs, 256, 14, 14] | [bs, 2048, 7, 7]

class Reconstructed_resnet18(nn.Module):
    def __init__(self, pretrain):
        super().__init__()
        if pretrain:
            model18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            model18 = models.resnet18()
        self.model18 = nn.Sequential(*list(model18.children())[:-2])
    
    def forward(self,x):
        return self.model18(x) # [bs,512,7,7]

class Transfer_model(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,3,kernel_size=3,padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True))
    
    def forward(self,x):
        return self.model(x) # [bs,3,224,224]
 
class attn_part(nn.Module):
    def __init__(self, self_attn, cross_attn, Ln1, Ln2, Ln3, dropout1, dropout2, ffd) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.drop_out1 = dropout1
        self.drop_out2 = dropout2
        self.layerNorm1 = Ln1
        self.layerNorm2 = Ln2
        self.layerNorm3 = Ln3
        self.feedforward = ffd
        
    def forward(self, q, k, v):
        score1 = q + self.drop_out1(self.layerNorm1(self.self_attn(q,q,q)))
        score2 = score1 + self.drop_out2(self.layerNorm2(self.cross_attn(score1,k,v)))
        score = score2 + self.layerNorm3(self.feedforward(score2))
        return score

class decoder_part(nn.Module):
    def __init__(self, layer_num, decoder_layer) -> None:
        super().__init__()
        self.model = [cp(decoder_layer).to("cuda:0") for _ in range(layer_num)]
        
    def forward(self,q,k,v):
        for l in self.model:
            q = l(q,k,v)
        return q
        
class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        # model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        # modules = list(model.children())[:-2]
        # self.model = nn.Sequential(*modules)
        self.model = Reconstructed_resnet101(self.pretrained)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        ###########################################################
        ############### Experiment 1 (extral part-mask_CMN)########
        ###########################################################
        self.cross_attn = MultiHeadedAttention(args.num_heads, args.d_model)
        self.self_attn = MultiHeadedAttention(args.num_heads, args.d_model)
        self.Lnorm1 = LayerNorm(512)
        self.Lnorm2 = LayerNorm(512)
        self.Lnorm3 = LayerNorm(512)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.pffd = nn.Linear(512,512)
        self.decoder_layer = attn_part(
            self.self_attn,
            self.cross_attn,
            self.Lnorm1,
            self.Lnorm2,
            self.Lnorm3,
            self.dropout1,
            self.dropout2,
            self.pffd
        )
        self.disease_attn = decoder_part(1, self.decoder_layer)
        self.decoder = decoder_part(2, self.decoder_layer)
    
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(args.d_vf),) if args.use_bn else ()) +
                (nn.Linear(args.d_vf, args.d_model),
                 nn.ReLU(),
                 nn.Dropout(args.drop_prob_lm)) +
                ((nn.BatchNorm1d(args.d_model),) if args.use_bn == 2 else ())))
        
        self.model18 = Reconstructed_resnet18(self.pretrained)
        self.model_bone = Transfer_model(70)
        self.model_lung = Transfer_model(15)
        self.model_heart = Transfer_model(6)
        self.model_mediastinum = Transfer_model(6)
        self.model_list = [
            self.model_bone,
            self.model_lung,
            self.model_heart,
            self.model_mediastinum
        ]
        self.model_disease_param = [
            nn.Parameter(torch.tensor([0.5]),requires_grad= True).to("cuda:0"),
            nn.Parameter(torch.tensor([0.5]),requires_grad= True).to("cuda:0"),
            nn.Parameter(torch.tensor([0.5]),requires_grad= True).to("cuda:0"),
            nn.Parameter(torch.tensor([0.5]),requires_grad= True).to("cuda:0")
        ]
        self.forward = self.forward_IU_Xray if args.dataset_name == 'iu_xray' else self.forward_MIMIC_CXR
        
    def forward_MIMIC_CXR(self, images, image_mask_bone,image_mask_lung, image_mask_heart, image_mask_mediastinum, disease_prompt):
        tmp_feature, patch_feats = self.model(images)  # 16,256,14,14 | 16,2048,7,7
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1)) # 16,2048
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1) # 16,49,2048
        saved_disease_token = []
        ###########################################################
        ###################### Experiment 1 (extral part) #########
        ###########################################################
        mask_feature = []
        tmp_feature = tmp_feature.reshape(image_mask_bone.shape[0], 1, image_mask_bone.shape[2], image_mask_bone.shape[3])
        disease_token_feature = []
        for _,(each,each_model,disease,disease_param) in enumerate(zip([image_mask_bone,image_mask_lung, image_mask_heart, image_mask_mediastinum],self.model_list, disease_prompt,self.model_disease_param)):
            output = self.model18(each_model(tmp_feature.expand(image_mask_bone.shape[0], each.shape[1], image_mask_bone.shape[2], image_mask_bone.shape[3]) * each)) # bs,512,7,7
            output = output.reshape(images.shape[0],output.shape[1],-1).permute(0,2,1) # 16,49,512
            mask_feature.append(output)
            # disease_output = output.detach()
            output =  self.disease_attn(output, disease, disease) # bs,49,512
            part_disease_token = torch.mean(output, dim=1, keepdim=True)
            saved_disease_token.append(part_disease_token)
            output =  disease_param* part_disease_token# bs,1,512
            disease_token_feature.append(output)
        mask_feature = torch.concat(mask_feature, dim=1) # bs,196,512
        disease_token_feature = torch.concat(disease_token_feature, dim=1) # bs,4,512
        mask_feature_1 = torch.concat([disease_token_feature, mask_feature], dim=1) # bs, 196+4, 512
        # saved_disease_token = {
        #     "bone":saved_disease_token[0],
        #     "lung":saved_disease_token[1],
        #     "heart":saved_disease_token[2],
        #     "mediastinum":saved_disease_token[3]
        # }
        ######## Embedding
        att_feats = self.att_embed(patch_feats) # 16,49,512
        att_feats = self.decoder(att_feats, mask_feature_1, mask_feature_1)  # 16,49,512

        return mask_feature, att_feats, avg_feats, saved_disease_token

    def forward_IU_Xray(self, images, image_mask_bone,image_mask_lung, image_mask_heart, image_mask_mediastinum, disease_prompt):
        tmp_feature, patch_feats = self.model(images)  # 16,256,14,14 | 16,2048,7,7
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1)) # 16,2048
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1) # 16,49,2048
        
        ###########################################################
        ###################### Experiment 1 (extral part) #########
        ###########################################################
        mask_feature = []
        tmp_feature = tmp_feature.reshape(image_mask_bone.shape[0], 1, image_mask_bone.shape[2], image_mask_bone.shape[3])
        disease_token_feature = []
        for _,(each,each_model,disease,disease_param) in enumerate(zip([image_mask_bone,image_mask_lung, image_mask_heart, image_mask_mediastinum],self.model_list, disease_prompt,self.model_disease_param)):
            output = self.model18(each_model(tmp_feature.expand(image_mask_bone.shape[0], each.shape[1], image_mask_bone.shape[2], image_mask_bone.shape[3]) * each)) # bs,512,7,7
            output = output.reshape(images.shape[0],output.shape[1],-1).permute(0,2,1) # 16,49,512
            mask_feature.append(output)
            # disease_output = output.detach()
            output =  self.disease_attn(output, disease, disease) # bs,49,512
            output =  disease_param* torch.mean(output, dim=1, keepdim=True)# bs,1,512
            disease_token_feature.append(output)
        mask_feature = torch.concat(mask_feature, dim=1) # bs,196,512
        disease_token_feature = torch.concat(disease_token_feature, dim=1) # bs,4,512
        mask_feature_1 = torch.concat([disease_token_feature, mask_feature], dim=1) # bs, 196+4, 512
        
        ######## Embedding
        att_feats = self.att_embed(patch_feats) # 16,49,512
        att_feats = self.decoder(att_feats, mask_feature_1, mask_feature_1)  # 16,49,512

        return disease_token_feature, mask_feature, att_feats, avg_feats