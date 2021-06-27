# OadTR
Code for our paper: "OadTR: Online Action Detection with Transformers" [["arxiv"]](https://arxiv.org/pdf/2106.11149.pdf)

## Dependencies

* pytorch==1.6.0 
* json
* numpy
* tensorboard-logger
* torchvision==0.7.0


# Prepare
* Unzip the anno file "./data/anno_thumos.zip"
* Download the feature [THUMOS14 feature](https://zenodo.org/record/5035147#.YNhWG7vitPY) (Note: [HDD](https://usa.honda-ri.com/hdd) and [TVSeries](https://homes.esat.kuleuven.be/psi-archive/rdegeest/TVSeries.html) are available by contacting the authors of the datasets and signing agreements due to the copyrights.)

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
