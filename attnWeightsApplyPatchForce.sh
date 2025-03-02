#!/bin/bash

cp attn_weights/transformer.py /opt/conda/envs/env_dcase24_bert/lib/python3.11/site-packages/torch/nn/modules/transformer.py
echo "Copied patched attn_weights/transformer.py to $torchDir/nn/modules/transformer.py"

