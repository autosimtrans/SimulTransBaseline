#!/bin/bash
k=-1
file_nums=(3913 105 6634 2 111 4093 3075 2956 108 3 42 48 27 67 107 3063)

if [ -e decode/zh-en.w${k}.all ]; then
    rm decode/zh-en.w${k}.all
fi
touch decode/zh-en.w${k}.all

for fn in ${file_nums[*]}; do
     bash run_w-1.sh ../Zh-En/dev/streaming_transcription/${fn}.txt decode/zh-en.w${k}.${fn}
     cat decode/zh-en.w${k}.${fn} >> decode/zh-en.w${k}.all
done

python3 latency.py decode/zh-en.w${k}.all ../Zh-En/zh-en.dev.zh.json
bash ../Zh-En/dev/reference_eval/eval_scripts/demo_mteval.sh decode/zh-en.w${k}.all.merge

