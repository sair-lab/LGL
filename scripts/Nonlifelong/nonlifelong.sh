#!/usr/bin/env bash

MODEL=attnplain
DEVICE=cuda:0
SEED=1
LR=0.01
OPT=SGD

DATASET=citeseer
BATCH=6
python non_lifelong.py --save loads/nonlifelong_${MODEL}_${DATASET}_${LR}_${OPT}_${SEED}_Drop53.model \
    --data-root ./data --dataset ${DATASET} --lr ${LR} --batch-size ${BATCH} --model ${MODEL} \
    --seed ${SEED} --device ${DEVICE} --eval ./Expnonlifelong/nonlifelong_${MODEL}_${DATASET}_${OPT} \
    --epochs 50 --patience 5 --optm ${OPT} --drop 0 0 --hidden 10 10

DATASET=cora
BATCH=10
python non_lifelong.py --save loads/nonlifelong_${MODEL}_${DATASET}_${LR}_${OPT}_${SEED}_Drop53.model \
    --data-root ./data --dataset ${DATASET} --lr ${LR} --batch-size ${BATCH} --model ${MODEL} \
    --seed ${SEED} --device ${DEVICE} --eval ./Expnonlifelong/nonlifelong_${MODEL}_${DATASET}_${OPT} \
    --epochs 50 --patience 5 --optm ${OPT} --drop 0 0 --hidden 10 10

DATASET=pubmed
BATCH=10
python non_lifelong.py --save loads/nonlifelong_${MODEL}_${DATASET}_${LR}_${OPT}_${SEED}_Drop53.model \
    --data-root ./data --dataset ${DATASET} --lr ${LR} --batch-size ${BATCH} --model ${MODEL} \
    --seed ${SEED} --device ${DEVICE} --eval ./Expnonlifelong/nonlifelong_${MODEL}_${DATASET}_${OPT} \
    --epochs 50 --patience 5 --optm ${OPT} --drop 0 0 --hidden 10 10
