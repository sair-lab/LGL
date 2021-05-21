# # This is run with normalization, but no 

MODEL=attnplain
DEVICE=cuda:0
SEED=1
SAMPLE=10
JUMP=1
MEMORY=500
OPT=SGD
MERGE=1

BATCH=6
DATASET=citeseer
for SEED in 1 2 3
do
    python lifelong.py --save loads/${MODEL}_class_${DATASET}_${OPT}_${MEMORY}_J${JUMP}_${SEED}.model --eval ./Exp_lifelongclass/${MODEL}/${MODEL}_class_${DATASET}_J${JUMP} --data-root ./data --dataset ${DATASET} --lr 0.01 --batch-size ${BATCH} --iteration 5 --memory ${MEMORY} --device $DEVICE --sample-rate ${SAMPLE} --seed $SEED --model ${MODEL} --jump ${JUMP} --optm ${OPT} --merge ${MERGE} --hidden 10 10 --drop 0 0
done

DATASET=pubmed
BATCH=10
for SEED in 1 2 3
do
    python lifelong.py --save loads/${MODEL}_class_${DATASET}_${OPT}_${MEMORY}_J${JUMP}_${SEED}.model --eval ./Exp_lifelongclass/${MODEL}/${MODEL}_class_${DATASET}_J${JUMP} --data-root ./data --dataset ${DATASET} --lr 0.01 --batch-size ${BATCH} --iteration 5 --memory ${MEMORY} --device $DEVICE --sample-rate ${SAMPLE} --seed $SEED --model ${MODEL} --jump ${JUMP} --optm ${OPT} --merge ${MERGE} --hidden 10 10 --drop 0 0
done

DATASET=cora
BATCH=10
for SEED in 1 2 3
do
    python lifelong.py --save loads/${MODEL}_class_${DATASET}_${OPT}_${MEMORY}_J${JUMP}_${SEED}.model --eval ./Exp_lifelongclass/${MODEL}/${MODEL}_class_${DATASET}_J${JUMP} --data-root ./data --dataset ${DATASET} --lr 0.01 --batch-size ${BATCH} --iteration 5 --memory ${MEMORY} --device $DEVICE --sample-rate ${SAMPLE} --seed $SEED --model ${MODEL} --jump ${JUMP} --optm ${OPT} --merge ${MERGE} --hidden 10 10 --drop 0 0
done
