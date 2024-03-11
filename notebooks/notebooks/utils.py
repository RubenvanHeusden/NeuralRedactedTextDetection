import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from evaluation import *
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
from pycocotools.mask import decode, encode, merge, area, frPyObjects, toBbox 

def read_json(file_name: str):
    with open(file_name, 'r') as json_file:
        return json.load(json_file)


def load_image(image_path: str) -> np.ndarray:
    """
    Function that loads an image from a path.
    :param image_path: string specifying the path to the image
    :return: Numpy array with the image in BGR format.
    """
    image = cv2.imread(image_path)
    return image


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


 

def visualize_annotations(image_filename: str, gold_standard_dict: dict, prediction_dict: dict,
    vis_mode: str ='masks', iou_score: float = 0.5, confidence_score: float = 0.5):
    """
    This function implements a simple visualization function hat can be used to show the predicted annotations
    and gives different colors based on whether the prediction is a True Positive, False Positive or False Negative. 
    image_filename: string that specifies where the image is located
    gold_standard_dict: dictionary containing the all the annotations, with the image field included
    prediction_dict: dictionary containing all the predicted annotations, as saved by all models,
    so not included the images fields, this will be filtered by the function.
    """
    
    assert vis_mode in ['masks', 'boxes']
    
    # set-up the image for the visualization
    plt.rcParams["figure.figsize"] = (20, 20)
    image_path = os.path.join('../resources/dataset/test/images/', image_filename)
    im_arr = torch.from_numpy(np.array(load_image(image_path))).permute(2, 0, 1)
    image_id = [item for item in gold_standard_dict['images'] if item['file_name'] == image_filename][0]['id']
    gold_annotations = [annot for annot in gold_standard_dict['annotations'] if annot['image_id'] == image_id]
    predicted_annotations = [annot for annot in prediction_dict if annot['image_id'] == image_id]
    filtered_predictions = non_overlapping_filtering(predicted_annotations,
        iou_score=iou_score, confidence_score=confidence_score)
    
    scores = compare_ground_truth_and_predictions(gold_annotations, filtered_predictions,
        iou_score=iou_score, confidence_score=confidence_score)
    
    # Set the right decode function, depending on whether we want to show masks are boxes
    decode_func = decode if vis_mode == 'masks' else toBbox

    FP_objects = [decode_func(item['segmentation']) for i, item in enumerate(filtered_predictions) if i in scores['FP']]
    FN_objects = [decode_func(item['segmentation']) for i, item in enumerate(gold_annotations) if i in scores['FN']]
    TP_objects = [decode_func(item['segmentation']) for i, item in enumerate(filtered_predictions) if i in [it[1] for it in scores['TP']]]
    
    if FP_objects:
        FP_objects = np.stack(FP_objects)
    if FN_objects:
        FN_objects = np.stack(FN_objects)
    if TP_objects:
        TP_objects = np.stack(TP_objects)
    
    all_objects = np.concatenate([item for item in [FP_objects, FN_objects, TP_objects] if isinstance(item, np.ndarray)])
    colors = ['red' for _ in range(len(FP_objects))] +  ['yellow' for _ in range(len(FN_objects))] + ['green' for _ in range(len(TP_objects))]
    if vis_mode == 'masks':
        show(draw_segmentation_masks(torch.flip(im_arr, [0]), torch.tensor(all_objects).bool(), colors=colors))
    else:
        all_boxes = box_convert(torch.tensor(all_objects), in_fmt='xywh', out_fmt='xyxy')
        show(draw_bounding_boxes(torch.flip(im_arr, [0]), all_boxes, colors=colors, width=5))
        
    plt.show()