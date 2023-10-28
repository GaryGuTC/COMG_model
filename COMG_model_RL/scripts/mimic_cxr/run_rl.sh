seed=${RANDOM}

mkdir -p results/mimic_cxr/base_cmn_rl/
mkdir -p records/mimic_cxr/base_cmn_rl/

python train_rl.py \
    --image_dir ../COMG_model/data/mimic_cxr/images/ \
    --ann_path ../COMG_model/data/mimic_cxr/annotation_disease.json \
    --dataset_name mimic_cxr \
    --max_seq_length 100 \
    --threshold 10 \
    --batch_size 6 \
    --epochs 50 \
    --save_dir results/mimic_cxr/ \
    --record_dir records/mimic_cxr/base_cmn_rl/ \
    --step_size 1 \
    --gamma 0.8 \
    --seed ${seed} \
    --topk 32 \
    --sc_eval_period 3000 \
    --resume ../COMG_model/results/mimic_cxr/model_best.pth \
    --early_stop 50 \
    --log_period 1000
