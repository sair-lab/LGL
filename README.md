# LGL
Lifelong Graph Learning

# Requirement

* PyTorch
* DGL

---     
# Training

* Datasets are automatically downloaded during training.

* Data-incremental Tasks

        python lifelong_data.py --lr 0.01 --batch-size 10 --dataset cora
        python lifelong_data.py --lr 0.01 --batch-size 10 --dataset citeseer
        python lifelong_data.py --lr 0.01 --batch-size 10 --dataset pubmed

To save your model during training

        python lifelong_data.py --lr 0.01 --batch-size 10 --dataset cora --save model_file_location

* Class-incremental Tasks

        python lifelong.py --lr 0.01 --batch-size 10 --dataset cora
        python lifelong.py --lr 0.01 --batch-size 10 --dataset citeseer
        python lifelong.py --lr 0.01 --batch-size 10 --dataset pubmed

---
# Reproduce the results from paper

* Download the pre-trained models from the [releases](https://github.com/wang-chen/LGL/releases/download/v1.0/loads.zip) Page.

* To reproduce all results, you may simply run:

        bash reproduce.sh

* For data-incremental tasks, e.g.

        python lifelong_data.py --load loads/data_incre_cora_memory100.model --dataset cora

* For class-incremetnal tasks, e.g.

        python lifelong.py --load loads/class_incre_cora_memory200.model --dataset cora

* You are expected to obtain the following performance.

Data-incremental Tasks

|     Memory    |         Cora  |  Citeseer     |    Pubmed     |
| ------------- | ------------- | ------------- | ------------- |
|       100     |     0.837     |     0.754     |     0.886     |
|       500     |     0.887     |     0.742     |     0.884     |


Class-incremental Tasks
|     Memory    |         Cora  |  Citeseer     |    Pubmed     |
| ------------- | ------------- | ------------- | ------------- |
|       200     |     0.823     |      0.712    |     0.830     |
|       500     |     0.883     |     0.746     |     0.876     |

* Note that the performan reported in the paper is an average of 10 runs, while the above is only one trial.
