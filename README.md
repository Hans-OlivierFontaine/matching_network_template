# Matching Networks for One Shot Learning 
This repo provides a Pytorch implementation fo the [Matching Networks for One Shot Learning](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) paper.

## Installation of pytorch
The experiments needs installing [Pytorch](http://pytorch.org/)

## Data 
This projects classifies image data organised and labelled with csv files under the following manner:\
├── root_dir\
│   ├── images\
│   │   ├── img1.png\
│   │   ├── img2.png\
│   │   ├── img3.png\
│   │   ├── ...\
│   ├── train.csv\
│   ├── val.csv\
│   ├── test.csv\
│   ├── labels.csv\

In which case the train.csv, val.csv and test.csv files are organised as such:\
Filename, label\
img1.png, 0\
img2.png, 1\
img3.png, 4\
...


## Installation

    $ pip install -r requirements.txt
    $ python matching_main.py
    

## Acknowledgements
Special thanks to https://github.com/zergylord and https://github.com/AntreasAntoniou for their Matching Networks implementation. I intend to use some parts for this implementation. More details at https://github.com/zergylord/oneshot and https://github.com/AntreasAntoniou/MatchingNetworks

## Cite
```
@misc{Fontaine2023matching,
  author = {Fontaine, Hans-Olivier},
  title = {Matching network template project},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Hans-OlivierFontaine/matching_network_template}},
  commit = {TODO}
}
```

## Authors

* Hans-Olivier Fontiane (@aberenguel) [Webpage](https://www.linkedin.com/in/hans-olivier-fontaine-333b28195/)
