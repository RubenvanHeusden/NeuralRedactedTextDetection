# This script contains a demo of the redaction detection algorithm using
# The mask R-CNN model ran on a CPU, on a sample document contain different types of redaction.

import os
import argparse
import torch.cuda
import numpy as np
from tqdm import tqdm
from PIL import Image
from pdf2image import convert_from_path
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from torch import jit

# local imports
from notebooks.evaluation import *


def main(arguments):
    # The first step is to load the model that we will be using, which is the extended Mask R-CNN model

    cfg = get_cfg()  # the default config
    cfg.merge_from_file("model_configs/maskrcnn_configs/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    # data config
    cfg.DATALOADER.NUM_WORKERS = 2  # this will alter the speed of the training, my gpu could only handle 2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False  # we also want the model to train on documents without redactions
    cfg.INPUT.RANDOM_FLIP = "none"  # we don't add any random flips as data augmentation

    model = jit.load('msc/output/model.ts')

    # Run on CPU
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    # model config
    cfg.MODEL.WEIGHTS = "resources/model_binaries/finetuned/extended/maskrcnn/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    predictor = DefaultPredictor(cfg)

    input_images = convert_from_path(arguments.input_file)

    for i, input_image in tqdm(enumerate(input_images)):
        model_output = predictor(np.array(input_image))
        output_masks = model_output['instances'].pred_masks
        RLE_masks = []
        for item in output_masks:
            converted_mask = np.asfortranarray(item.numpy())
            RLE_masks.append(encode(converted_mask))

        # Because the evaluation code requires a slightly different format, we convert this
        annotations = [{'segmentation': segmentation, 'score': score} for segmentation, score in zip(RLE_masks, model_output['instances'].scores)]
        filtered_masks = non_overlapping_filtering(annotations, iou_score=0.5, confidence_score=0.5)
        decode_func = decode if args.vis_mode == 'masks' else toBbox

        objects = [decode_func(item['segmentation']) for i, item in enumerate(filtered_masks)]

        output_image = torch.tensor(np.array(input_image))

        # If we detected an image we will add the segmentation masks, otherwise
        # we will just return the image without annotation masks.
        if objects:
            objects = np.stack(objects)
            colors = ['black' for _ in range(len(objects))]
            if args.vis_mode == 'masks':
                output_image = draw_segmentation_masks(torch.tensor(np.array(output_image)).permute(2, 0, 1),
                                                       torch.tensor(objects).bool(), colors=colors).permute(1, 2, 0)
            else:
                all_boxes = box_convert(torch.tensor(objects), in_fmt='xywh', out_fmt='xyxy')
                output_image = draw_bounding_boxes(torch.tensor(np.array(output_image)).permute(2, 0, 1),
                                                   all_boxes, width=20)

        path = os.path.join('samples', '%d.png' % i)

        Image.fromarray(output_image.numpy()).save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--vis_mode', type=str, default='masks')
    args = parser.parse_args()
    main(args)