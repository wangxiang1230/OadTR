# OadTR
Code for our ICCV2021 paper: "OadTR: Online Action Detection with Transformers" [["Paper"]](https://arxiv.org/pdf/2106.11149.pdf)

## Update

* July 28, 2021: Our Paper "OadTR: Online Action Detection with Transformers" was accepted by ICCV2021. At the same time, we released [THUMOS14-Kinetics features](https://zenodo.org/record/5140603#.YQDk8britPY).  

## Dependencies

* pytorch==1.6.0 
* json
* numpy
* tensorboard-logger
* torchvision==0.7.0


# Prepare
* Unzip the anno file "./data/anno_thumos.zip"
* Download the feature [THUMOS14-Anet feature](https://zenodo.org/record/5035147#.YNhWG7vitPY) (Note: [HDD](https://usa.honda-ri.com/hdd) and [TVSeries](https://homes.esat.kuleuven.be/psi-archive/rdegeest/TVSeries.html) are available by contacting the authors of the datasets and signing agreements due to the copyrights. You can use this [Repo](https://github.com/yjxiong/anet2016-cuhk) to extract features.)

# Training
```
python main.py --num_layers 3 --decoder_layers 5 --enc_layers 64 --output_dir models/en_3_decoder_5_lr_drop_1
```
# Validation
```
python main.py --num_layers 3 --decoder_layers 5 --enc_layers 64 --output_dir models/en_3_decoder_5_lr_drop_1 --eval --resume models/en_3_decoder_5_lr_drop_1/checkpoint000{}.pth
```

# Citing OadTR
Please cite our paper in your publications if it helps your research:

```BibTeX
@article{wang2021oadtr,
  title={OadTR: Online Action Detection with Transformers},
  author={Wang, Xiang and Zhang, Shiwei and Qing, Zhiwu and Shao, Yuanjie and Zuo, Zhengrong and Gao, Changxin and Sang, Nong},
  journal={arXiv preprint arXiv:2106.11149},
  year={2021}
}
```
