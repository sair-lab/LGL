#!/usr/bin/env bash

python lifelong_data.py --load loads/data_incre_cora_memory100.model --dataset cora

python lifelong_data.py --load loads/data_incre_citeseer_memory100.model --dataset citeseer

python lifelong_data.py --load loads/data_incre_pubmed_memory100.model --dataset pubmed


python lifelong.py --load loads/class_incre_cora_memory200.model --dataset cora

python lifelong.py --load loads/class_incre_citeseer_memory200.model --dataset citeseer

python lifelong.py --load loads/class_incre_pubmed_memory200.model --dataset pubmed


python lifelong_data.py --load loads/data_incre_cora_memory500.model --dataset cora

python lifelong_data.py --load loads/data_incre_citeseer_memory500.model --dataset citeseer

python lifelong_data.py --load loads/data_incre_pubmed_memory500.model --dataset pubmed


python lifelong.py --load loads/class_incre_cora_memory500.model --dataset cora

python lifelong.py --load loads/class_incre_citeseer_memory500.model --dataset citeseer

python lifelong.py --load loads/class_incre_pubmed_memory500.model --dataset pubmed



python lifelong_data.py --dataset flickr --load loads/data_incre_flickr_memory0.model

python lifelong_data.py --dataset flickr --load loads/data_incre_flickr_memory10.model

python lifelong_data.py --dataset flickr --load loads/data_incre_flickr_memory20.model


python lifelong_data.py --dataset flickr --load loads/class_incre_flickr_memory0.model

python lifelong_data.py --dataset flickr --load loads/class_incre_flickr_memory10.model

python lifelong_data.py --dataset flickr --load loads/class_incre_flickr_memory20.model
