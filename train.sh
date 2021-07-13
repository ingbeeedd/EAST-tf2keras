#!/bin/sh

python train.py --gpu_list=0 --input_size=512 --batch_size=12 --nb_workers=6 --training_data_path=./data/ICDAR2013+2015/train_data/ --validation_data_path=./data/ICDAR2013+2015/test_data/ --checkpoint_path=tmp/icdar2015_east_resnet50/
