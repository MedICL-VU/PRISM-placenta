# PRISM-placenta
Interactive Segmentation Model for Placenta Segmentation from 3D Ultrasound Images

## News
[07/07/24] Repo is created.

## TL;DR
[PRISM](https://github.com/MedICL-VU/PRISM) is an **effective** and **efficient** 3D interactive model for placenta segmentation. 

## Effective 
Qualitative results with compared methods ![qualitative_results](figs/qualitative.png)

The quantitative results can be viewed in our [paper](https://arxiv.org/abs/2404.15028).

## Efficient
We consider a Dice score of 0.95 as a bar for success, which is higher than inter-rater variability (0.85-0.90)


![Efficient results](figs/efficient_github.png)

More **efficient** and **effective** results are included in our [paper](https://arxiv.org/abs/2404.15028), please check if you are interested.


## Datasets
in-house dataset, the details can be viewed in other paper from our group (you can find them from the **Reference** section in the paper)


## Get Started

**Installation**
```
conda create -n prism python=3.9
conda activate prism
sudo install git
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 # install pytorch
pip install git+https://github.com/facebookresearch/segment-anything.git # install segment anything packages
pip install git+https://github.com/deepmind/surface-distance.git # for normalized surface dice (NSD) evaluation
pip install -r requirements.txt
```


**Train**

```
python train.py --data ultrasound --data_dir your_data_directory --save_name your_save_name --multiple_outputs --dynamic --use_box --refine
```

add "--use_scribble" and "--efficient_scribble" if you want to train with scribbles.

**Train (Distributed Data Parallel)**

the only difference between this and above (train) command is the use of "--ddp".
```
python train.py --data ultrasound --data_dir your_data_directory --save_name your_save_name -multiple_outputs --dynamic --use_box --refine --ddp
```


**Test**

put downloaded pretrained model under the implementation directory
```
python test.py --data ultrasound --data_dir your_data_directory --split test --checkpoint best --save_name prism_pretrain --num_clicks 1 --iter_nums 11 --multiple_outputs --use_box --use_scribble --efficient_scribble --refine --refine_test
```


**FAQ**

if you got the error as AttributeError: module 'cv2' has no attribute 'ximgproc', please check [this](https://stackoverflow.com/questions/57427233/module-cv2-cv2-has-no-attribute-ximgproc) out

I haven't cleaned the code, and it has some arguments for past/future efficient or effective analyses.
