python main_test.py \
    --image_dir data/mimic_cxr/images \
    --ann_path data/mimic_cxr/annotation_disease.json \
    --dataset_name mimic_cxr \
    --max_seq_length 100 \
    --threshold 10 \
    --epochs 30 \
    --batch_size 16 \
    --lr_ve 1e-4 \
    --lr_ed 5e-4 \
    --step_size 3 \
    --gamma 0.8 \
    --num_layers 3 \
    --topk 32 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --seed 7580 \
    --beam_size 3 \
    --save_dir results/mimic_cxr/ \
    --log_period 1000 \
    --load /home/tiancheng/Downloads/2526_WACV_COMG_Supplementary_material/Supplementary_material/COMG_model/results_weights/mimic_cxr_best/model_best.pth
