#!/usr/bin/env bash

experiment () {
LATENT_N="$1"
echo "laten variable is: $1"
TIME=$(date +%H%M%S)
CHECKPOINT_PATH="./checkpoints/{prefix}-{date}-$TIME-$LATENT_N.pkl"
echo $CHECKPOINT_PATH
python -u train.py --latent-n=$LATENT_N --save=True --checkpoint-path=$CHECKPOINT_PATH --dashboard-server="http://localhost:8097" >> ./logs/latent-$LATENT_N.out
python -u generate.py --latent-n=$LATENT_N --checkpoint-path=$CHECKPOINT_PATH --output="./figures/{prefix}-{date}-$LATENT_N.png"
}



experiment 40
experiment 40
experiment 40
experiment 40
experiment 40
experiment 30
experiment 30
experiment 30
experiment 30
experiment 30
experiment 20
experiment 20
experiment 20
experiment 20
experiment 20
experiment 10
experiment 10
experiment 10
experiment 10
experiment 10
experiment 5
experiment 5
experiment 5
experiment 5
experiment 5
experiment 3
experiment 3
experiment 3
experiment 3
experiment 3
experiment 2
experiment 2
experiment 2
experiment 2
experiment 2
experiment 1
experiment 1
experiment 1
experiment 1
experiment 1
