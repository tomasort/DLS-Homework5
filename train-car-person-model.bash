git clone https://github.com/qfgaohao/pytorch-ssd.git
wget -P models https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth
wget -P models https://storage.googleapis.com/models-hao/voc-model-labels.txt

python3 pytorch-ssd/open_images_downloader.py --root open_images --class_names "Airplane, Helicopter, Human body, Car, Girl, Vehicle, Dog, Window, Van, Umbrella, Helment, Boy, Bicycle" --num_workers 20

python3 pytorch-ssd/train_ssd.py --dataset_type open_images --datasets open_images --net mb1-ssd --pretrained_ssd models/mobilenet-v1-ssd-mp-0_675.pth --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 300 --base_net_lr 0.001  --batch_size 64
