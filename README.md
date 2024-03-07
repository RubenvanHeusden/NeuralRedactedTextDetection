# NeuralRedactedTextDetection

![repo_example](https://github.com/RubenvanHeusden/NeuralRedactedTextDetection/blob/main/repo_example.png?raw=true)
This repository contains the code and experiments associated with the paper "Redacted Text Detection Using Neural Image Segmentation
Methods", currently under submission at the International Journal of Document Analysis and Recognition (IJDAR).

## Installation

The repository includes both a requirements.txt and a environment.yml file for installing the required dependencies.
To install the required dependencies via `pip`, please run the following command:

This repository uses the [Mask2Former](https://github.com/facebookresearch/Mask2Former/tree/main) and [Detectron2](https://github.com/facebookresearch/detectron2) libraries from Meta for the experiments. Although the requirements file contains versions of these packages, if you want to run the Mask2Former model on your own GPU, you will have to compile the 'XX' CUDA operation yourself for your own system. You can do this by following the installation instructions HERE.

## Trained models
- The trained Mask2Former and Mask R-CNN models are included in the repository to use for your own redaction detection.
- If you want to re-train the models of this paper you will need the [Swin-T model trained on COCO] from the Mask2Former modell zoo and the [Mask R-CNN ResNext] model from the Detectron2 model zoo. Both of these models can be downloaded by running the `download_mask2former_model.sh` script in the root folder of the directory, which will create the `pretrained_models` directory in this repository.


## Data
The dataset is encoded in the standard COCO format, with train and test folders containing the images and a JSON file in the COCO annotation style. Please note that although the Mask2Former library works with RLE masks of the annotations, the Detectron2 library does not, and therefore we have included annotations in both formats, with instructions of how to run everything for both models. 

