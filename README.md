# SimulTransBaseline

This is a sample code for AutoSimulTrans Workshop (https://autosimtrans.github.io) based
on PaddlePaddle(https://github.com/paddlepaddle/paddle) with dynamic graph.
This code implements Transformer based Wait-K training and decoding proposed in paper
STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency
(https://arxiv.org/abs/1810.08398).


The following is the code struture

```text
.
├── utils                # Utilities
├── gen_data.sh          # Scripts to download and bpe preprocessed WMT18 zh-en corpus
├── predict.py           # Inference code
├── reader.py            # Data reader
├── stream_reader.py     # Stream data reader
├── README.md            # Documentation
├── train.py             # Training
├── model.py             # Transformer model and beam (greedy) search
└── transformer.yaml     # configuration
```

## Quick Start

### Installation

1. Paddle

   This project depends on PaddlePaddle 1.7 develop version. Please refer to [Installation Manual](http://www.paddlepaddle.org/#quick-start) to install.

2. Download code

    克隆代码库到本地
    ```shell
    git clone https://github.com/PaddlePaddle/models.git
    cd models/dygraph/transformer
    ```
3. 环境依赖

   请参考PaddlePaddle[安装说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.6/beginners_guide/install/index_cn.html)部分的内容

### Data Preparation

We use official preprocessed WMT18 Chinese-to-English translation corpus

```text
.
├── wmt18_zhen_data              # WMT18 Chinese-to-English translation corpus
├── wmt18_zhen_data_bpe          # BPE encoded corpus
├── mosesdecoder                 # Moses MT toolkit, include Tokenize、BLEU scripts
└── subword-nmt                  # BPE encoding scripts
```

另外我们也整理提供了一份处理好的 WMT'16 EN-DE 数据以供[下载](https://transformer-res.bj.bcebos.com/wmt16_ende_data_bpe_clean.tar.gz)使用，其中包含词典（`vocab_all.bpe.32000`文件）、训练所需的 BPE 数据（`train.tok.clean.bpe.32000.en-de`文件）、预测所需的 BPE 数据（`newstest2016.tok.bpe.32000.en-de`等文件）和相应的评估预测结果所需的 tokenize 数据（`newstest2016.tok.de`等文件）。


自定义数据：如果需要使用自定义数据，本项目程序中可直接支持的数据格式为制表符 \t 分隔的源语言和目标语言句子对，句子中的 token 之间使用空格分隔。提供以上格式的数据文件（可以分多个part，数据读取支持文件通配符）和相应的词典文件即可直接运行。

### 单机训练

### 单机单卡

以提供的英德翻译数据为例，可以执行以下命令进行模型训练：

```sh
# setting visible devices for training
export CUDA_VISIBLE_DEVICES=0

python -u train.py \
  --epoch 30 \
  --src_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --trg_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --training_file gen_data/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de \
  --validation_file gen_data/wmt16_ende_data_bpe/newstest2014.tok.bpe.32000.en-de \
  --batch_size 4096
```

以上命令中传入了训练轮数（`epoch`）和训练数据文件路径（注意请正确设置，支持通配符）等参数，更多参数的使用以及支持的模型超参数可以参见 `transformer.yaml` 配置文件，其中默认提供了 Transformer base model 的配置，如需调整可以在配置文件中更改或通过命令行传入（命令行传入内容将覆盖配置文件中的设置）。可以通过以下命令来训练 Transformer 论文中的 big model：

```sh
# setting visible devices for training
export CUDA_VISIBLE_DEVICES=0

python -u train.py \
  --epoch 30 \
  --src_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --trg_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --training_file gen_data/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de \
  --validation_file gen_data/wmt16_ende_data_bpe/newstest2014.tok.bpe.32000.en-de \
  --batch_size 4096 \
  --n_head 16 \
  --d_model 1024 \
  --d_inner_hid 4096 \
  --prepostprocess_dropout 0.3
```

另外，如果在执行训练时若提供了 `save_model`（默认为 trained_models），则每隔一定 iteration 后（通过参数 `save_step` 设置，默认为10000）将保存当前训练的到相应目录（会保存分别记录了模型参数和优化器状态的 `transformer.pdparams` 和 `transformer.pdopt` 两个文件），每隔一定数目的 iteration (通过参数 `print_step` 设置，默认为100)将打印如下的日志到标准输出：





