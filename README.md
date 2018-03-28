# LPS
Code for the paper ``Learning to Promote Saliency Detectors" accepted by CVPR 2018.

## Usage
* first, install [Pytorch](https://github.com/pytorch/pytorch)
* modify the prior map, input and output directory in ```test_once.py```, ```test_iterative.py```, and ```train.py```
* to download and use my trained model, run ```test_once.py``` or ```test_iterative.py```
* to train a new model, run ```train.py```

## Citation
```
@inproceedings{zeng2018learning,
    author = {Yu Zeng, Huchuan Lu, Lihe Zhang, Mengyang Feng, and Ali Borji},
    title = {Learning to Promote Saliency Detectors},
    booktitle = {CVPR},
    year = {2018}}
```