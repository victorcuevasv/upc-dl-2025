#!/bin/bash

torchDir=$(python attn_weights/getTorchPath.py)
if [ ! -f $torchDir/nn/modules/transformer.py.bak ]; then
    mv $torchDir/nn/modules/transformer.py $torchDir/nn/modules/transformer.py.bak
    echo "Renamed $torchDir/nn/modules/transformer.py to $torchDir/nn/modules/transformer.py.bak"
    cp attn_weights/transformer.py $torchDir/nn/modules/transformer.py
    echo "Copied patched attn_weights/transformer.py to $torchDir/nn/modules/transformer.py"
fi
