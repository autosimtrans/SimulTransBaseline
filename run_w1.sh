#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0
export CUDA_PATH=/tools/cuda_10.0.130
export CUDNN_PATH=/tools/cudnn/cuda-10.0/cudnn-v7.6.4
export NCCL_PATH=/tools/nccl/nccl_2.4.8-1+cuda10.0_x86_64
export PATH=${PATH}:${CUDA_PATH}/bin:${CUDNN_PATH}
export CPATH=${CUDA_PATH}/include:${CUDNN_PATH}/include:$CPATH
export LIBRARY_PATH=${CUDA_PATH}/lib64:${CUDNN_PATH}/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=${NCCL_PATH}/lib:${CUDA_PATH}/lib64:${CUDA_PATH}/extras/CUPTI/lib64:${CUDNN_PATH}/lib64:$LD_LIBRARY_PATH

k=1
n=114
input_file=$1
output_file=$2
python3 -u predict.py \
    --src_vocab_fpath /mnt/scratch/zrenj/Project/paddle-smt/transformer/get_data/wmt18_zhen_data_bpe/vocab_all.zh.bpe.16000 \
    --src_bpe_dict /mnt/scratch/zrenj/Project/paddle-smt/transformer/get_data/wmt18_zhen_data_bpe/bpe.zh.16000 \
    --trg_vocab_fpath /mnt/scratch/zrenj/Project/paddle-smt/transformer/get_data/wmt18_zhen_data_bpe/vocab_all.en.bpe.16000 \
    --special_token '<s>' '<e>' '<unk>' \
    --init_from_params /mnt/scratch/zrenj/Project/paddle-smt/transformer/models/ft_wmt18_w${k}/step_${n}000 \
    --predict_file $input_file \
    --batch_size 128 \
    --beam_size 1 \
    --max_out_len 255 \
    --weight_sharing False \
    --waitk $k \
    --stream True \
    --output_file $output_file
