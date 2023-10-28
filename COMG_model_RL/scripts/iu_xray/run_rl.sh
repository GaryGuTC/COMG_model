seed=${RANDOM}
noamopt_warmup=1000

RESUME=${1}

python train_rl.py \
    --image_dir ../COMG_model/data/IU_xray/images/ \
    --ann_path ../COMG_model/data/IU_xray/annotation_disease.json \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --batch_size 6 \
    --epochs 250 \
    --save_dir results/iu_xray/ \
    --step_size 1 \
    --gamma 0.8 \
    --seed ${seed} \
    --topk 32 \
    --beam_size 3 \
    --log_period 100 \
    --resume ../COMG_model/results/iu_xray/model_best.pth
