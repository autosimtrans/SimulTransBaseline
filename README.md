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

    ```shell
    git clone https://github.com/PaddlePaddle/models.git
    cd models/dygraph/transformer
