# Radio Galaxy Classification based on Mask Transfiner -- HeTu2
HeTu2 is built on the pioneering work of [Transfiner](https://arxiv.org/abs/2111.13673) and [detectron2](https://github.com/facebookresearch/detectron2).


Highlights
-----------------
- **Transfiner:** High-quality instance segmentation with state-of-the-art performance and extreme details.
- **Novelty:** An efficient transformer targeting for high-resolution instance masks predictions based on the quadtree structure.
- **Efficacy:** Large mask and boundary AP improvements on three instance segmentation benchmarks, including COCO, Cityscapes and BDD100k. 
- **Simple:** Small additional computation burden compared to standard transformer and easy to use.



Introduction
-----------------
Two-stage and query-based instance segmentation methods have achieved remarkable results. However, their segmented masks are still very coarse. In this paper, we present Mask Transfiner for high-quality and efficient instance segmentation. Instead of operating on regular dense tensors, our Mask Transfiner decomposes and represents the image regions as a quadtree. Our transformer-based approach only processes detected error-prone tree nodes and self-corrects their errors in parallel. While these sparse pixels only constitute a small proportion of the total number, they are critical to the final mask quality. This allows Mask Transfiner to predict highly accurate instance masks, at a low computational cost. Extensive experiments demonstrate that Mask Transfiner outperforms current instance segmentation methods on three popular benchmarks, significantly improving both two-stage and query-based frameworks by a large margin of +3.0 mask AP on COCO and BDD100K, and +6.6 boundary AP on Cityscapes. 


## Step-by-step Installation
```
# install python3.7 (need gcc > 5.0, cuda 11.0)
./Anaconda3-2020.02-Linux-x86_64.sh  
 
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
 
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
 
# install transfiner
cd $INSTALL_DIR
git clone https://github.com/lao19881213/RGC-Mask-Transfiner.git
cd transfiner/
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
  ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
  ln -s /path_to_coco_dataset/test2017 datasets/coco/test2017
  ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017
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
Download the pretrained models from [detectron2 ImageNet Pretrained Models](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl): 
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

Citation
---------------
If you find Mask Transfiner useful in your research or refer to the provided baseline results, please star :star: this repository and consider citing :pencil::
```
@inproceedings{transfiner,
    author={Ke, Lei and Danelljan, Martin and Li, Xia and Tai, Yu-Wing and Tang, Chi-Keung and Yu, Fisher},
    title={Mask Transfiner for High-Quality Instance Segmentation},
    booktitle = {CVPR},
    year = {2022}
}  
```
Related Links
---------------
Related NeurIPS 2021 Work on multiple object tracking & segmentation: [PCAN](https://github.com/SysCV/pcan)

Related CVPR 2021 Work on occlusion-aware instance segmentation: [BCNet](https://github.com/lkeab/BCNet)

Related ECCV 2020 Work on partially supervised instance segmentation: [CPMask](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530375.pdf)


