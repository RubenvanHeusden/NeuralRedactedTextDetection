# NeuralRedactedTextDetection

![repo_example](https://github.com/RubenvanHeusden/NeuralRedactedTextDetection/blob/main/repo_example.png?raw=true)
This repository contains the code and experiments associated with the paper "Redacted Text Detection Using Neural Image Segmentation
Methods", currently under submission at the International Journal of Document Analysis and Recognition (IJDAR).

## Directory Structure

```
model configs
│      └─── detectron2_configs
│      └─── mask2former_configs
│
training_scripts
│      └─── mask2former_train_net.py
│      └─── maskrcnn_train_net.py
│
model_outputs
│      └─── Morphology
│      └─── Mask-RCNN
│      └─── Mask2Former
│
notebooks
│      └─── DatasetExploration.ipynb
│      └─── EdactRayOnScansBaseline.ipynb
│      └─── NeuralModelExperiments.ipynb
│      └─── OCRandMorphologyBaseline.ipynb
│      └─── evaluation.py
│      └─── utils.py
```

## Installation

The repository includes both a requirements.txt and a environment.yml file for installing the required dependencies.
To install the required dependencies via `pip`, please run the following command:

This repository uses the [Mask2Former](https://github.com/facebookresearch/Mask2Former/tree/main) and [Detectron2](https://github.com/facebookresearch/detectron2) libraries from Meta for the experiments. Although the requirements file contains versions of these packages, if you want to run the Mask2Former model on your own GPU, you will have to compile the MSDeformAttn CUDA operation yourself for your own system. You can do this by following the installation instructions [here](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md).

## Trained models
- The trained Mask2Former and Mask R-CNN models are included in the repository to use for your own redaction detection.
- If you want to re-train the models of this paper you will need the [Swin-T model trained on COCO] from the Mask2Former model zoo and the [Mask R-CNN ResNext] model from the Detectron2 model zoo. Both of these models can be downloaded by running the `download_mask2former_model.sh` script in the root folder of the directory, which will create the `pretrained_models` directory in this repository, so that they are in the correct place for the config files.

  To train the Mask2Former model, you can run the following command from the `training_scripts` folders
```
python mask2former_train_net.py \
--config-file ../model_configs/mask2former_configs/maskformer2_swin_tiny_bs16_50ep.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.0001 SOLVER.MAX_ITER 5000 OUTPUT_DIR ../Mask2FormerOutput \
  DATALOADER.FILTER_EMPTY_ANNOTATIONS False DATASETS.TRAIN '("classic_train",)'  DATASETS.TEST'("classic_test",)' \
MODEL.WEIGHTS ../resources/model_binaries/pretrained/model_final_86143f.pkl
```
To train the Mask RCNN folder, use the following command:
```
python maskrcnn_train_net.py \
--config-file ../model_configs/maskrcnn_configs/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
  SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.0001 SOLVER.MAX_ITER 5000 OUTPUT_DIR ../MaskRCNNOutput \
  DATALOADER.FILTER_EMPTY_ANNOTATIONS False DATASETS.TRAIN '("classic_train",)'  DATASETS.TEST '("classic_test",)'
```
If you don't have a GPU you can still train the model, but you will have to remove the `--num-gpus` command and specify
`MODEL.DEVICE "cpu"`.

For both the training scripts, the datasets you want to use for training can be given to the models as shown above. 
They can also be specified in the config file itself, but this makes changes it for different runs more involved. 
For the experiments, the following datasets have been registered in the Detectron2 library, and can be used as train 
and test dataset:
- classic_train, classic_test
- extended_train, extended_test
- train10, train20, train40, train60, train80



## Data & Models
The dataset is encoded in the standard COCO format, with train and test folders containing the images and a JSON file in the COCO annotation style. Please note that although the Mask2Former library works with RLE masks of the annotations, the Detectron2 library does not, and therefore we have included annotations in both formats, with instructions of how to run everything for both models. 

Both the dataset as well as the trained models and the model outputs can be downloaded through Zenodo on the following link: https://zenodo.org/records/10805206
Before you run the training scripts for the models, you should run `download_pretrained_models.sh` to get the Detectron2 and Mask2Former models from the model zoos.
