# LGL
Lifelong Graph Learning

# Dependencies

* Python 3.7
* [PyTorch v1.5+](https://pytorch.org/get-started)
* [DGL v0.4+](https://www.dgl.ai/pages/start.html) (Only used for downloading graph datasets)

---     
# Training

* Citation datasets (Cora, Citeseer, and Pubmed) are automatically downloaded before training.

* Default dataset (download) location is '/data/datasets', you may change it via args '--data-root [data_location]'.

* Download Flickr dataset from the [v1.1 release page](https://github.com/wang-chen/LGL/releases/download/v1.1/flickr.zip) and put it in the [data_location] folder.

* Data-incremental Tasks

        python lifelong_data.py --lr 0.01 --batch-size 10 --dataset cora
        python lifelong_data.py --lr 0.01 --batch-size 10 --dataset citeseer
        python lifelong_data.py --lr 0.01 --batch-size 10 --dataset pubmed
        python lifelong_data.py --lr 0.001 --batch-size 10 --iteration 1 --memory-size 10 --optm Adam --dataset flickr --data-root [data_location]

* Class-incremental Tasks

        python lifelong.py --lr 0.01 --batch-size 10 --dataset cora
        python lifelong.py --lr 0.01 --batch-size 10 --dataset citeseer
        python lifelong.py --lr 0.01 --batch-size 10 --dataset pubmed
        python lifelong.py --lr 0.001 --batch-size 10 --iteration 1 --memory-size 10 --optm Adam --dataset flickr --data-root [data_location]

* To save your model during training

        python lifelong_data.py --lr 0.01 --batch-size 10 --dataset cora --save model_file_location

* More arguments you can speficy:

        usage: lifelong.py [-h] [--device DEVICE] [--data-root DATA_ROOT]
                           [--dataset DATASET] [--load LOAD] [--save SAVE]
                           [--optm OPTM] [--lr LR] [--batch-size BATCH_SIZE]
                           [--iteration ITERATION] [--memory-size MEMORY_SIZE]
                           [--seed SEED] [-p]

        Feature Graph Networks

        optional arguments:
          -h, --help            show this help message and exit
          --device DEVICE       cuda or cpu, default: cuda:0
          --data-root DATA_ROOT
                                dataset location
          --dataset DATASET     cora, citeseer, or pubmed
          --load LOAD           load pretrained model file
          --save SAVE           model file to save
          --optm OPTM           SGD or Adam
          --lr LR               learning rate
          --batch-size BATCH_SIZE
                                minibatch size
          --iteration ITERATION
                                number of training iteration
          --memory-size MEMORY_SIZE
                                number of examplers
          --seed SEED           Random seed.

---
# Reproduce the results from paper

* Download the [pre-trained models](https://github.com/wang-chen/LGL/releases/download/v1.1/loads.zip) and extract it into a folder named 'loads'.

* To reproduce all results, you may simply run:

        bash reproduce.sh

* For data-incremental tasks, e.g.

        python lifelong_data.py --load loads/data_incre_cora_memory100.model --dataset cora

* For class-incremetnal tasks, e.g.

        python lifelong.py --load loads/class_incre_cora_memory200.model --dataset cora

* You are expected to obtain the following performance.

* Data-incremental Tasks

     |     Memory    |         Cora  |  Citeseer     |    Pubmed     |
     | :-----------: | :-----------: | :-----------: | :-----------: |
     |       100     |     0.830     |     0.752     |     0.884     |
     |       500     |     0.890     |     0.740     |     0.882     |

* Class-incremental Tasks

     |     Memory    |         Cora  |  Citeseer     |    Pubmed     |
     | :-----------: | :-----------: | :-----------: | :-----------: |
     |       200     |     0.827     |     0.712     |     0.826     |
     |       500     |     0.880     |     0.746     |     0.876     |

* Flickr

     |     Memory       |         0     |        10     |        20     |
     | :-----------:    | :-----------: | :-----------: | :-----------: |
     | Data-incremental |     0.475     |     0.471     |     0.470     |
     | Class-incremental|     0.479     |     0.476     |     0.459     |

* Note that the performance reported in the paper is an average of 10 runs, while the above is only one trial.
