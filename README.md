# NeuralRedactedTextDetection

This repository contains the code and experiments associated with the paper "XXXX", currently under submission at IJDAR.

## Installation

The repository includes both a requirements.txt and a environment.yml file for installing the required dependencies. This reposotiry uses the Mask2Former and Detectron2 libraries from Meta for the experiments. Although the requirements file contains versions of these packages, if you want to run the Mask2Former model on your own GPU, you will have to compile the 'XX' CUDA operation yourself for your own system. You can do this by following the installation instructions HERE.

## Trained models
- The trained Mask2Former and Mask R-CNN models are included in the repository to use for your own redaction detection.
- If you want to re-train the models of this paper you will need the [Swin-T model trained on COCO] from the Mask2Former modell zoo and the [Mask R-CNN ResNext] model from the Detectron2 model zoo. Both of these models can be downloaded by running the `download_detectron_models.sh` script in the root folder of the directory, which will create the `pretrained_models` directory in this repository.

### Training
To train the Mask2Former and Mask R-CNN models, the training scripts from Detectron2 and Mask2Former have been adapted for our dataset, and can be run using the following commands.

```
dddd
```

```
dddd
```

