from label_mapper import label_mapper_image_name
import os
import pickle
from pathlib import Path
import argparse
import torch
from PIL import Image
from torchvision import transforms
from tqdm import trange
import json

from cxas import CXAS
import cxas.visualize as cxas_vis
join = os.path.join

class_check_tables = [
        "thoracic spine", 
        "all vertebrae",  
        "cervical spine", 
        "lumbar spine",   
        "clavicle set",   
        "scapula set",    
        "ribs",           
        "ribs super",     
        "diaphragm",      
        "mediastinum",    
        "abdomen",        
        "heart region",   
        "breast tissue",  
        "trachea",        
        "lung zones",     
        "lung halves",    
        "vessels",        
        "lung lobes"     
    ]

need_mask_table = {
    "bone":["ribs", "ribs super"],
    "pleural":[],
    "lung":["lung zones", "lung halves", "lung lobes"],
    "heart":["heart region"],
    "mediastinum":["mediastinum"],
}

class config:    
    tfms = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor()
    ])
    #### Check table
    class_check_tables = [
        "thoracic spine", 
        "all vertebrae",  
        "cervical spine",
        "lumbar spine",   
        "clavicle set", 
        "scapula set",  
        "ribs",         
        "ribs super",   
        "diaphragm",    
        "mediastinum",  
        "abdomen",      
        "heart region", 
        "breast tissue",  
        "trachea",        
        "lung zones",     
        "lung halves",    
        "vessels",        
        "lung lobes"  
    ]
    
def save_as_pickle(path,content):
    with open(path, 'wb') as f:
        pickle.dump(content, f, protocol=pickle.HIGHEST_PROTOCOL)

def create_cxas_model():
    return CXAS(model_name = 'UNet_ResNet50_default',
                gpus       = '0')

def get_caption(caption_path):
    with open(caption_path) as f:
        data = json.load(f)
    return data

def cxas_infer(cxas_model, path_image, out_path, args):
    if args.dataset_name == "iu_xray":
        _ = cxas_model.process_file(
                filename = path_image,
                do_store = True, 
                output_directory = out_path,
                storage_type = 'npy',
                )
        
    _ = cxas_model.process_file(
            filename = path_image,
            do_store = True, 
            output_directory = out_path,
            storage_type = 'jpg',
            )
    return

def segmentation_cxas_iu_xray(cxas_model, caption_data, args):
    for k in caption_data.keys():
        print("{} part has been started !!!!!!".format(k))
        for i in trange(len(caption_data[k])):
            if not os.path.exists(config.iu_xray_saved_path): os.mkdir(config.iu_xray_saved_path)
            saved_path = join(config.iu_xray_saved_path, caption_data[k][i]["id"])
            if not os.path.exists(saved_path): os.mkdir(saved_path)
            for i,each in enumerate(caption_data[k][i]["image_path"]):
                img = join(config.iu_xray_image, each)
                cxas_infer(cxas_model, img, saved_path, args)
        
def saved_mask_pickle_iu_xray():
    print("Pickel saving part has been started !!!!!!")
    img_file_list = [join(config.iu_xray_saved_path, each.stem) for each in list(Path(config.iu_xray_saved_path).glob("*"))]
    for i in trange(len(img_file_list)):
        each = img_file_list[i]
        for i in range(2):
            img_path = join(each, str(i))
            saved_path = join(each, "{}_mask".format(i))
            # delete numpy file => save space
            numpy_file = join(each, "{}.npy".format(i))
            if os.path.exists(numpy_file): 
                os.remove(numpy_file)
            if not os.path.exists(saved_path): 
                os.mkdir(saved_path)
            for i,(need_class,v) in enumerate(need_mask_table.items()):
                if len(v) == 0: continue
                masks = [] 
                for each_class in v:
                    for each_img in label_mapper_image_name[each_class]:
                        img = Image.open(join(img_path,"{}.jpg".format(each_img)))
                        masks.append(config.tfms(img))
                masks = torch.concat(masks, dim=0)
                save_as_pickle(join(saved_path, "{}_concat.pkl".format(need_class)), masks)

def segmentation_cxas_mimic_cxr(cxas_model, mimic_cxr_caption, args):
    for key in mimic_cxr_caption.keys():
        print("{} part has been started !!!!!!".format(key))
        for i in trange(len(mimic_cxr_caption[key])):
            item = mimic_cxr_caption[key][i]
            item_id = item["id"]
            assert len(item["image_path"]) == 1, "{} has more than 1 images".format(item_id)
            item_image_path = item["image_path"][0]
            img_path = join(config.mimic_cxr_image, item_image_path)
            saved_path = img_path.replace("mimic_cxr","mimic_cxr_segmentation").replace(".jpg","")
            if os.path.exists(saved_path): continue
            if not os.path.exists(saved_path): os.makedirs(saved_path)
            #################### Segmentation ####################
            cxas_infer(cxas_model, img_path, saved_path, args) 

def update_config(args):
    #### IU xray
    config.iu_xray_caption = join(args.iu_xray_path, "annotation.json")
    config.iu_xray_image = join(args.iu_xray_path, "images")
    config.iu_xray_saved_path = join(args.data_path,"IU_xray_segmentation")
    #### Mimic-cxr
    config.mimic_cxr_caption = join(args.mimic_cxr_path, "annotation.json")
    config.mimic_cxr_image = join(args.mimic_cxr_path, "images")
    config.mimic_cxr_saved_path = join(args.data_path,"mimic_cxr_segmentation")

def main(args):
    cxas_model = create_cxas_model()
    update_config(args)
    if args.dataset_name == "iu_xray":
        caption_data = get_caption(config.iu_xray_caption)
        # Segmentation & saved visulization image
        segmentation_cxas_iu_xray(cxas_model, caption_data, args)
        # generate mask pickle
        saved_mask_pickle_iu_xray()
    else:
        caption_data = get_caption(config.mimic_cxr_caption)
        # Segmentation & saved visulization image
        segmentation_cxas_mimic_cxr(cxas_model, caption_data, args)
        # generate mask pickle
        # saved_mask_pickle() => due to it's too large will cost so much  storage space


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    project_directory = Path(__file__).parent.parent
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--data_path', type=str, default=join(project_directory, "data"),
                        help='the dataset path of total data')
    parser.add_argument('--iu_xray_path', type=str, default=join(project_directory, "data", "iu_xray"),
                        help='the dataset path of iu_xray.')
    parser.add_argument('--mimic_cxr_path', type=str, default=join(project_directory, "data", "mimic_cxr"),
                        help='the dataset path of mimic_xray.')
    args = parser.parse_args()
    main(args)