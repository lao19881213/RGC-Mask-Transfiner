# Radio Galaxy Classification based on Mask Transfiner -- HeTu-v2
HeTu-v2 is built on the pioneering work of [Transfiner](https://arxiv.org/abs/2111.13673) and [detectron2](https://github.com/facebookresearch/detectron2) to support high-quality segmentation and morphological classification of radio sources for modern radio continuum surveys.


Highlights
-----------------
- **Build morphological classification catalog:** 
- **Build accurate sky mode:** .
- **Automatic radio component association:** . 
- **Radio galaxy classification:** .



Introduction
-----------------


## Step-by-step Installation
```
# install python3.7 (need gcc > 5.0, cuda 11.0)
./Anaconda3-2020.02-Linux-x86_64.sh  
 
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

# on gznu 380
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Coco api and visualization dependencies
pip install ninja yacs cython matplotlib tqdm
pip install opencv-python==4.4.0.40
# Boundary dependency
pip install scikit-image
pip install kornia==0.5.11
 
export INSTALL_DIR=$PWD
 
# install pycocotools. Please make sure you have installed cython.
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
 
# install RGC-Mask-Transfiner
cd $INSTALL_DIR
git clone https://github.com/lao19881213/RGC-Mask-Transfiner.git
cd RGC-Mask-Transfiner/
python3 setup.py build develop
 
unset INSTALL_DIR
```

## Load dependences on CSRC-P 
```
source bashrc  
# bashrc includes as follows lines:
export PATH=/home/blao/anaconda3-py3.7/bin:$PATH  

module use /home/app/modulefiles  
module load gcc/cpu-7.3.0  

module load openblas/cpu-0.3.6-gcc-4.8.5  
module load hdf5/cpu-1.10.4 gcc/cpu-7.3.0  
#cuda 11.0  
export PATH=/usr/local/cuda-11.0/bin:$PATH  
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH  
export CPATH=/usr/local/cuda-11.0/include:$CPATH  

export PATH=/home/blao/anaconda3-py3.7/bin:$PATH  
export CPATH=/home/blao/anaconda3-py3.7/include/python3.7m:$CPATH  
export CPATH=/home/blao/anaconda3-py3.7/lib/python3.7/site-packages/numpy/core/include:$CPATH  


export HDF5_USE_FILE_LOCKING="FALSE"  

```


## Dataset Preparation
Prepare for [coco2017](http://cocodataset.org/#home) dataset and [Cityscapes](https://www.cityscapes-dataset.com) following [this instruction](https://github.com/facebookresearch/detectron2/tree/master/datasets).

```
  mkdir -p datasets/coco
  ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
  ln -s /path_to_coco_dataset/train2022 datasets/coco/train2022
  ln -s /path_to_coco_dataset/test2022 datasets/coco/test2022
  ln -s /path_to_coco_dataset/val2022 datasets/coco/val2022
```

Multi-GPU Training and Evaluation on Validation set
---------------
Refer to our [scripts folder](https://github.com/SysCV/transfiner/tree/main/scripts) for more traning, testing and visualization commands:
 
```
sbatch scripts/train_transfiner_3x_101.sh
```
Or
```
bash scripts/train_transfiner_1x_50.sh
```

Pretrained Models
---------------
Download the pretrained models from [detectron2 ImageNet Pretrained Models](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md): [R-50.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl) and [R-101.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl). 
```
  mkdir pre_trained_model
  #And put the downloaded pretrained models in this directory.
```

Testing on Test-dev
---------------
```
bash scripts/test_3x_transfiner_101.sh
```

Visualization
---------------
```
bash scripts/visual.sh
```
for swin-based model:
```
bash scripts/visual_swinb.sh
```

Batch Inference 
------------------
```
nohup ./batch_predict_FIRST_sourcefinding_gznu_part3.sh >> part3.out &
```

Citation
---------------
If you find HeTu-v2 useful in your research or refer to the provided baseline results, please consider citing :pencil::
```
```


