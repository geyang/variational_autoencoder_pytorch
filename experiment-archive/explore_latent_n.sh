#!/usr/bin/env bash

python -u train.py --latent-n=20 --save=True --checkpoint-path="./checkpoints/{prefix}-{date}-20" --dashboard-server="http://localhost:8097" >> ./logs/latent-20.out
python -u generate.py --latent-n=20 --checkpoint-path="./checkpoints/{prefix}-{date}-20.pkl" --output="./figures/{prefix}-{date}-20.png"

python -u train.py --latent-n=10 --save=True --checkpoint-path="./checkpoints/{prefix}-{date}-10" --dashboard-server="http://localhost:8097" >> ./logs/latent-10.out
python -u generate.py --latent-n=10 --checkpoint-path="./checkpoints/{prefix}-{date}-10.pkl" --output="./figures/{prefix}-{date}-10.png"

python train.py --latent-n=2 --save=True --checkpoint-path="./checkpoints/{prefix}-{date}-2" --dashboard-server="http://localhost:8097" >> ./logs/latent-2.out
python generate.py --latent-n=2 --checkpoint-path="./checkpoints/{prefix}-{date}-2.pkl" --output="./figures/{prefix}-{date}-2.png"
