#!/bin/bash
k=1
file_nums=(3913 105 6634 2 111 4093 3075 2956 108 3 42 48 27 67 107 3063)
# file_nums=(3913)

if [ -e decode/zh-en.asr.w${k}.all ]; then
    rm decode/zh-en.asr.w${k}.all
fi
touch decode/zh-en.asr.w${k}.all

for fn in ${file_nums[*]}; do
    bash run_w1.sh data/Zh-En/dev/streaming_asr/${fn}.wav.txt decode/zh-en.asr.w${k}.${fn}
    cat decode/zh-en.asr.w${k}.${fn} >> decode/zh-en.asr.w${k}.all
done

python3 latency.py decode/zh-en.asr.w${k}.all /mnt/scratch/zrenj/Project/challenge/transformer/data/zh-en.dev.asr.zh.json
bash /mnt/scratch/zrenj/Project/challenge/transformer/data/Zh-En/dev/reference_eval/eval_scripts/demo_mteval.sh decode/zh-en.asr.w${k}.all.merge

