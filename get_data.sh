#! /usr/bin/env bash

set -e

OUTPUT_DIR=$PWD/data

OUTPUT_DIR_DATA="${OUTPUT_DIR}/wmt18_zhen_data"
ORIGIN_DIR_DATA="${OUTPUT_DIR_DATA}/corpus"
OUTPUT_DIR_BPE_DATA="${OUTPUT_DIR}/wmt18_zhen_data_bpe"
LANG1="zh"
LANG2="en"

mkdir -p $OUTPUT_DIR_DATA $OUTPUT_DIR_BPE_DATA

if [ ! -e ${ORIGIN_DIR_DATA} ]; then
    echo "Download WMT18 Preprocessed Training Data"
    wget -O ${OUTPUT_DIR_DATA}/corpus.gz http://data.statmt.org/wmt18/translation-task/preprocessed/zh-en/corpus.gz
    gunzip ${OUTPUT_DIR_DATA}/corpus.gz
    awk -F"\\t" '{print $1}' ${ORIGIN_DIR_DATA} > ${ORIGIN_DIR_DATA}.${LANG1}
    awk -F"\\t" '{print $2}' ${ORIGIN_DIR_DATA} > ${ORIGIN_DIR_DATA}.${LANG2}
fi

if [ ! -e ${OUTPUT_DIR_DATA}/newsdev2017.tc.en ]; then
    echo "Download WMT18 Preprocessed Dev Data"
    wget -O ${OUTPUT_DIR_DATA}/dev.tgz http://data.statmt.org/wmt18/translation-task/preprocessed/zh-en/dev.tgz
    tar zxvf ${OUTPUT_DIR_DATA}/dev.tgz -C $OUTPUT_DIR_DATA/
fi



# Clone mosesdecoder
if [ ! -d ${OUTPUT_DIR}/mosesdecoder ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git ${OUTPUT_DIR}/mosesdecoder
fi

# Clone subword-nmt and generate BPE data
if [ ! -d ${OUTPUT_DIR}/subword-nmt ]; then
  git clone https://github.com/rsennrich/subword-nmt.git ${OUTPUT_DIR}/subword-nmt
fi


# Generate BPE data and vocabulary
num_operations=16000
if [ ! -e ${OUTPUT_DIR_BPE_DATA}/bpe.en.${num_operations} ]; then
  echo "Learn BPE with ${num_operations} merge operations"
  cat ${ORIGIN_DIR_DATA}.${LANG1} | \
      ${OUTPUT_DIR}/subword-nmt/learn_bpe.py -s $num_operations > ${OUTPUT_DIR_BPE_DATA}/bpe.${LANG1}.${num_operations}
  cat ${ORIGIN_DIR_DATA}.${LANG2} | \
      ${OUTPUT_DIR}/subword-nmt/learn_bpe.py -s $num_operations > ${OUTPUT_DIR_BPE_DATA}/bpe.${LANG2}.${num_operations}
fi


for l in ${LANG1} ${LANG2}; do
  for f in `ls ${OUTPUT_DIR_DATA}/*.$l`; do
    f_base=${f%.*}  # dir/train.tok dir/train.tok.clean dir/newstest2016.tok
    f_base=${f_base##*/}  # train.tok train.tok.clean newstest2016.tok
    f_out=${OUTPUT_DIR_BPE_DATA}/${f_base}.bpe.${num_operations}.$l
    if [ ! -e $f_out ]; then
      echo "Apply BPE to "$f
      ${OUTPUT_DIR}/subword-nmt/apply_bpe.py -c ${OUTPUT_DIR_BPE_DATA}/bpe.$l.${num_operations} < $f > $f_out
    fi
  done
done


for l in ${LANG1} ${LANG2}; do
    if [ ! -e ${OUTPUT_DIR_BPE_DATA}/vocab.${l}.bpe.${num_operations} ]; then
      echo "Create vocabulary for BPE data"
      cat ${OUTPUT_DIR_BPE_DATA}/corpus.bpe.${num_operations}.${l} | \
        ${OUTPUT_DIR}/subword-nmt/get_vocab.py | cut -f1 -d ' ' > ${OUTPUT_DIR_BPE_DATA}/vocab.${l}.bpe.${num_operations}
    fi
done

# Adapt to the reader
for f in ${OUTPUT_DIR_BPE_DATA}/*.bpe.${num_operations}.${LANG1}; do
  f_base=${f%.*}  # dir/train.tok.clean.bpe.32000 dir/newstest2016.tok.bpe.32000
  f_out=${f_base}.${LANG1}-${LANG2}
  if [ ! -e $f_out ]; then
    paste -d '\t' $f_base.${LANG1} $f_base.${LANG2} > $f_out
  fi
done

for l in ${LANG1} ${LANG2}; do
    if [ ! -e ${OUTPUT_DIR_BPE_DATA}/vocab_all.${l}.bpe.${num_operations} ]; then
      sed '1i\<s>\n<e>\n<unk>' ${OUTPUT_DIR_BPE_DATA}/vocab.${l}.bpe.${num_operations} > ${OUTPUT_DIR_BPE_DATA}/vocab_all.${l}.bpe.${num_operations}
    fi
done


echo "All done."
