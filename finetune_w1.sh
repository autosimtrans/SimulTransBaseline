python train.py \
     --save_model ./models/ft-zh-en-w1/ \
     --epoch 30 \
     --src_vocab_fpath data/wmt18_zhen_data_bpe/vocab_all.zh.bpe.16000 \
     --trg_vocab_fpath data/wmt18_zhen_data_bpe/vocab_all.en.bpe.16000 \
     --special_token '<s>' '<e>' '<unk>' \
     --training_file ../Zh-En/train/train.all.clean.bpe.zh-en \
     --init_from_checkpoint models/zh-en-w1 \
     --validation_file data/wmt18_zhen_data_bpe/newstest2017.tc.bpe.16000.zh-en \
     --batch_size 500 \
     --warmup_steps 16000 \
     --max_length 999 \
     --print_step 10 \
     --use_cuda True \
     --weight_sharing False \
     --waitk 1 \
     --save_step 500
 
