# COMG Model: Complex Organism Mask Guided Radiology Report Generation Model

This is the implementation of COMG Model: Complex Organism Mask Guided Radiology Report Generation Model

## Abstract
The goal of automatic report generation is to generate a clinically accurate and coherent phrase from a single given X-ray image, which could alleviate the workload of traditional radiology reporting. However, in a real world scenario, radiologists frequently face the challenge of producing extensive reports derived from numerous medical images, thereby medical report generation from multi-image perspective is needed. In this paper, we propose the Complex Organ Mask Guided (termed as COMG) report generation model, which incorporates masks from multiple organs (e.g., bones, lungs, heart, and mediastinum), to provide more detailed information and guide the model’s attention to these crucial body regions. Specifically, we leverage prior knowledge of the disease corresponding to each organ in the fusion process to enhance the disease identification phase during the report generation process. Additionally, cosine similarity loss is introduced as target function to ensure the convergence of cross-modal consistency and facilitate model optimization. Experimental results on two public datasets show that COMG achieves a 11.4% and 9.7% improvement in terms of BLEU4 scores over the SOTA model KiUT on IU-Xray and MIMIC, respectively.

## Requirements

```bash
conda env create -f environment.yaml # method 1
pip install -r requirements.txt # method 2
```

### pycocoevalcap
```bash
Download evaluation metrics
https://github.com/tylin/coco-caption/tree/master/pycocoevalcap
```
## Datasets

We use two datasets(IU X-Ray and MIMIC-CXR) in our paper

For `IU X-Ray`, you can download the dataset from [here](https://openi.nlm.nih.gov/faq) and then put the files in `data`.

For `MIMIC-CXR`, you can download the dataset from [here](https://physionet.org/content/mimic-cxr/2.0.0/) and then put the files in `data`.

### Generate Mask

Generate Mask by using [Chest X-Ray Anatomy Segmentation model](https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation/)

```bash
cd COMG_model
cd preprocess_mask
bash generate_mask.sh # almost 30 min(IU-xray) + 4h(mimic_cxr)
```

## Train & test - The first stage

```bash
# IU xray
cd COMG_model
bash train_iu_xray.sh
bash test_iu_xray.sh
# mimic-cxr
bash train_mimic_cxr.sh
bash test_mimic_cxr.sh
# visualization - mimic-cxr
bash plot_mimic_cxr.sh
```

## Train & test - RL - The second stage

```bash
# IU-xray
# train
cd COMG_model_RL 
bash scripts/iu_xray/run_rl.sh
# test
cd ../COMG_model
bash test_iu_xray.sh
# MIMIC-CXR
# train
cd COMG_model_RL 
bash scripts/mimic_cxr/run_rl.sh
# test
cd ../COMG_model
bash test_mimic_cxr.sh
```

## Checkpoints Download
future update

## Acknowledgement

Our project references the codes in the following repos. Thanks for their works and sharing.

* [R2GenCMN](https://github.com/zhjohnchan/R2GenCMN)

* [R2GenRL](https://github.com/synlp/R2GenRL)