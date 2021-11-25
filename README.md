# LGL
Lifelong Graph Learning

[Chen Wang](https://chenwang.site), [Yuheng Qiu](http://theairlab.org/team/yuheng/), [Dasong Gao](http://theairlab.org/team/dasongg/),  [Sebastian Scherer](http://theairlab.org/team/sebastian/). "[Lifelong Graph Learning](https://arxiv.org/pdf/2009.00647.pdf)." *arXiv preprint arXiv:2009.00647 (2021)*.

# Note 

This repo only contains source code for citation graphs.

Please go to [Human Action Recognition](https://github.com/wang-chen/lgl-action-recognition) and [Feature Matching](https://github.com/wang-chen/lgl-feature-matching) for other experiments.

# Dependencies

* Python 3
* [PyTorch v1.5+](https://pytorch.org/get-started)
* [DGL v0.4](https://www.dgl.ai/pages/start.html) (Only used for downloading graph datasets)

        conda install dgl=0.4

* [Open Graph Benchmark](https://ogb.stanford.edu/)

---     
## Training

* Citation datasets (Cora, Citeseer, and Pubmed) are automatically downloaded before training.

* Default dataset (download) location is '/data/datasets', you may change it via args '--data-root [data_location]'.

        python train.py --data-root [data_location] --config config/Regular/FGNRegularCoraPubmed.yaml
* To save your model during training

        python lifelong_data.py --data-root [data_location] --config config/Regular/FGNRegularCoraPubmed.yaml --save [model_file_location]

* To try some baselined model under `--model` with `GCN`, `GAT`, `MLP`, `SAGE` and `APP` 

        python lifelong_data.py --data-root [data_location] --config config/Regular/BaselineRegularCoraPubmed.yaml --save [model_file_location] --model [modle_name]

## LifeLong Learning
* Class-incremental Tasks

        python lifelong.py --data-root [data_location] --config config/ClassIncremental/FGNClassIncrementalCoraPubmed.yaml --save [model_file_location]

* Data-incremental Tasks

        python lifelong_data.py --data-root [data_location] --config config/ --config config/DataIncremental/FGNDatalifelongCoraPubmed.yaml --save [model_file_location]


## Testing

* download pretrained model from 
        https://github.com/wang-chen/LGL/releases/tag/pretrained. 
        Change the path of the pretrained model under `--load`, and change the `--config`.

        python train.py --data-root [data_location] --config config/Regular/FGNRegularOGB.yaml --load pretrained_model/Regular/nonlifelongFGNKTransCat_ogbn-arxiv.pt
        

## Some API for usage

        usage: lifelong.py [-h] [-c CONFIG] [--device DEVICE] [--data-root DATA_ROOT] [--dataset DATASET] [--model MODEL] [--load LOAD]
                        [--save SAVE] [--optm OPTM] [--lr LR] [--batch-size BATCH_SIZE] [--jump JUMP] [--iteration ITERATION]
                        [--memory-size MEMORY_SIZE] [--seed SEED] [-p] [--eval EVAL] [--sample-rate SAMPLE_RATE] [--k K]
                        [--hidden HIDDEN [HIDDEN ...]] [--drop DROP [DROP ...]] [--merge MERGE]

        optional arguments:
        -h, --help            show this help message and exit
        -c CONFIG, --config CONFIG
                                config file path
        --device DEVICE       cuda or cpu
        --data-root DATA_ROOT
                                dataset location
        --dataset DATASET     cora, citeseer, or pubmed
        --model MODEL         LGL or SAGE
        --load LOAD           load pretrained model file
        --save SAVE           model file to save
        --optm OPTM           SGD or Adam
        --lr LR               learning rate
        --batch-size BATCH_SIZE
                                minibatch size
        --jump JUMP           reply samples
        --iteration ITERATION
                                number of training iteration
        --memory-size MEMORY_SIZE
                                number of samples
        --seed SEED           Random seed.
        -p, --plot            increase output verbosity
        --eval EVAL           the path to eval the acc
        --sample-rate SAMPLE_RATE
                                sampling rate for test acc, if ogb datasets please set it to 200
        --k K                 the level of k hop.
        --hidden HIDDEN [HIDDEN ...]
        --drop DROP [DROP ...]
        --merge MERGE         Merge some class if needed.
        
 
# Citation

    @article{wang2021lifelong,
      title={Lifelong graph learning},
      author={Wang, Chen and Qiu, Yuheng and Gao, Dasong and Scherer, Sebastian},
      journal={arXiv preprint arXiv:2009.00647},
      year={2021}
    }
