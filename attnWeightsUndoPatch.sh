#!/bin/bash

torchDir=$(python attn_weights/getTorchPath.py)
if [ -f $torchDir/nn/modules/transformer.py.bak ]; then
    rm $torchDir/nn/modules/transformer.py
    echo "Deleted $torchDir/nn/modules/transformer.py"
    mv $torchDir/nn/modules/transformer.py.bak $torchDir/nn/modules/transformer.py
    echo "Restored $torchDir/nn/modules/transformer.py.bak"
fi
