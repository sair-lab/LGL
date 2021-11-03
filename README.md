# LGL
Lifelong Graph Learning

# Dependencies

* Python 3
* [PyTorch v1.5+](https://pytorch.org/get-started)
* [DGL v0.4](https://www.dgl.ai/pages/start.html) (Only used for downloading graph datasets)
* [Open Graph Benchmark](https://ogb.stanford.edu/)

---     
# Training

* Citation datasets (Cora, Citeseer, and Pubmed) are automatically downloaded before training.

* Default dataset (download) location is '/data/datasets', you may change it via args '--data-root [data_location]'.

* Data-incremental Tasks

        python lifelong_data.py --data-root [data_location]

* Class-incremental Tasks

        python lifelong.py --data-root [data_location]

* To save your model during training

        python lifelong_data.py --lr 0.01 --batch-size 10 --dataset cora --save model_file_location

* For data-incremental tasks, e.g.

        python lifelong_data.py --load loads/data_incre_cora_memory100.model --dataset cora

* For class-incremetnal tasks, e.g.

        python lifelong.py --load loads/class_incre_cora_memory200.model --dataset cora