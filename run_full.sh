#!/bin/bash
k=-1
input_file=$1
output_file=$2
PYTHONPATH=/home/aistudio/external-libraries /opt/conda/envs/python35-paddle120-env/bin/python -u predict.py \
    --src_vocab_fpath data/wmt18_zhen_data_bpe/vocab_all.zh.bpe.16000 \
    --src_bpe_dict data/wmt18_zhen_data_bpe/bpe.zh.16000 \
    --trg_vocab_fpath data/wmt18_zhen_data_bpe/vocab_all.en.bpe.16000 \
    --special_token '<s>' '<e>' '<unk>' \
    --init_from_params models/zh-en-full-sent \
    --predict_file $input_file \
    --batch_size 128 \
    --beam_size 1 \
    --max_out_len 255 \
    --weight_sharing False \
    --waitk $k \
    --stream True \
    --only_src True \
    --output_file $output_file
