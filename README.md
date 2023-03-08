# Radio Sources Segmentation and Classification with Deep Learning -- HeTu-v2
HeTu-v2 is built on the pioneering works of [Transfiner](https://arxiv.org/abs/2111.13673) and [detectron2](https://github.com/facebookresearch/detectron2) to achieve the high-quality segmentation and morphological classification of radio sources for modern radio continuum surveys. HeTu-v2 can perform multiple astronomical tasks and has distinct advantages over other source detectors:
- **Build morphological classification catalog:** The main function of HeTu-v2 is to build a catalog from radio image. Each source in the catalog is described by the detected information (class name, mask, bounding box, and score) and standard properties (flux density and location) of the source. The morphological classification covers five categories, namely CS, FRI, FRII, Head-Tail (HT), and CJ sources.
- **Build accurate sky mode:** HeTu-v2 can produce high-quality segmentation, which can be used to build a sky model that is typically used in the radio data processing pipeline like [profund](https://arxiv.org/abs/1802.00937) and [caesar](https://arxiv.org/abs/1605.01852), and avoids the Gaussian over-fit when modeling the extended emission. More importantly, HeTu-v2 can automatically associate the separated radio lobes to build a complete model, which is more accurate than profund and caesar.
- **Automatic radio component association:** HeTu-v2 can be extended to find the radio components associated with a radio source in an automated way from the components-based catalog. The association method is based on the predicted segmentation mask of the radio source that is generated by HeTu-v2, and therefore the result is more accurate than the methods based on the bounding box. 
- **Radio galaxy classification:** If there are a number of input radio images and each input image is considered to have exactly one interesting source, it is troublesome to morphologically classify the source in each image manually through visual inspection when using traditional source finders. HeTu-v2 can, however, be used as a classifier to automatically classify compact and extended radio galaxies into the five classes mentioned above.


## Step-by-step Installation
```
# install python3.7 (need gcc > 5.0, cuda 11.0)
./Anaconda3-2020.02-Linux-x86_64.sh  

# install pytorch
## for cuda 11.0
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

## for cuda 11.7 (gznu 380) 
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

## Load dependences  
```
source bashrc_gznu  
# bashrc includes as follows lines:
export PATH=/usr/bin
export LD_LIBRARY_PATH=/usr/lib64
export CPATH=/usr/include

export PATH=/home/data0/lbq/gcc/bin:$PATH
export CPATH=/home/data0/lbq/gcc/include:$CPATH
export LD_LIBRARY_PATH=/home/data0/lbq/gcc/lib:/home/data0/lbq/gcc/lib64:$LD_LIBRARY_PATH

export CC=/home/data0/lbq/gcc/bin/gcc
export CXX=/home/data0/lbq/gcc/bin/g++

#hdf5
export PATH=/home/data0/lbq/hdf5/bin:$PATH
export LD_LIBRARY_PATH=/home/data0/lbq/hdf5/lib:$LD_LIBRARY_PATH
export CPATH=/home/data0/lbq/hdf5/include:$CPATH

#cuda 11.0
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cuda/include:$CPATH

export PATH=/home/data0/lbq/anaconda3-2022/bin:$PATH
export CPATH=/home/data0/lbq/anaconda3-2022/include/python3.7m:$CPATH
export CPATH=/home/data0/lbq/anaconda3-2022/lib/python3.7/site-packages/numpy/core/include:$CPATH

export PATH=/home/data0/lbq/software/openmpi/bin:$PATH
export CPATH=/home/data0/lbq/software/openmpi/include:$CPATH
export LD_LIBRARY_PATH=/home/data0/lbq/software/openmpi/lib:$LD_LIBRARY_PATH

export HDF5_USE_FILE_LOCKING="FALSE"
```



## Dataset Preparation
Prepare for [train]() dataset and [val]() following [this instruction](https://github.com/facebookresearch/detectron2/tree/master/datasets).

```
  mkdir -p datasets/coco
  ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
  ln -s /path_to_coco_dataset/train2022 datasets/coco/train2022
  ln -s /path_to_coco_dataset/val2022 datasets/coco/val2022
```

Multi-GPU Training and Evaluation on Validation set
---------------
Refer to our [scripts folder](https://github.com/lao19881213/RGC-Mask-Transfiner/tree/main/scripts) for more traning, testing and visualization commands:
 
```
bash scripts/train_transfiner_3x_101_deform_plain_gznu_3gpu.sh
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
bash scripts/test_3x_transfiner_101_deform_gznu.sh
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

Batch Inference example 
------------------
```
nohup ./batch_predict_FIRST_sourcefinding_gznu_part3.sh >> part3.out &
```

Citation
---------------
If you find HeTu-v2 useful in your research or refer to the provided baseline results, please consider citing :pencil::
```

```


