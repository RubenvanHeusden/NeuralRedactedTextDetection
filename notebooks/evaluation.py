"""
This script contains the code that can be used to evaluate any algorithm that outputs COCO instances predictions
against a COCO ground truth json file. It includes calculations over dataframes, and the possibility to split the predicted scores
into the scores for the various redaction types. Although most of the algorithm is optimized to work with RLE masks
The official PQ implementation used by kirilov adds partial masks to predictions to make them non-overlapping
which requires numpy arrays for the calculation, making the evaluation less efficient.
"""

import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from torchvision.ops import nms
from torchvision.ops import box_convert
from pycocotools.mask import iou, toBbox, area, decode, encode, merge


def non_overlapping_filtering(annotations: List[dict], iou_score: float = 0.5, confidence_score: float = 0.5):
    """
    This function implements the approach taken by Kirilov et al. to filter and 
    manipulate predicted predictions so that they are never overlapping, which is a prerequisite
    for the correct calculation of the PQ metric.
    annotations: List of annotations in the coco format, with the masks being
    encoded in the RLE format.
    iou_score: The iou score below which masks that overlap will be rejected.
    confidence score: float specifying the minimal confidence score a prediction need to be considered in the evaluation. If 
    the algorithm used to create the predictions does not implement a function that assigns confidence, set the parameter to None
    """

    assert 0 < iou_score <= 1
    assert (confidence_score == None) or (0 < confidence_score <= 1)

    if confidence_score:
        # Filter out empty annotations that are below the `confidence_score` threshold
        confident_annotations = [item for item in annotations if item['score'] > confidence_score]
        # We now sort these predictions on their confidence, with the most confident anntation first.
        confident_annotations = sorted(confident_annotations, key=lambda x : x['score'])[::-1]
    else:
        confident_annotations = annotations

    # In case we removed all predictions, we return an empty list
    if not confident_annotations:
        return []
    # Identical to kirilov, we now loop through the predictions, and iteratively add non-overlapping predictions
    non_overlapping_masks = [confident_annotations[0]] # start with the most confident predictions and always keep this
    
    # Keep track of the of all the previous predictions to calculate overlaps
    merged_masks = confident_annotations[0]['segmentation']
    # Loop over all but the first prediction
    for mask in confident_annotations[1:]:
        # Calculate overlap with previous masks
        mask_overlap = merge([merged_masks, mask['segmentation']], intersect=False)
        # Now calculate get the area only occupied by the current mask
        mask_only = np.logical_and((decode(mask['segmentation']) == 1), (decode(merged_masks) == 0))
        # Now threshold portions where the overlap is small enough
        if (mask_only.sum() / area(mask['segmentation'])) > iou_score:
            non_overlapping_mask = encode(mask_only)
            mask['segmentation'] = non_overlapping_mask
            merged_masks = mask_overlap
            non_overlapping_masks.append(mask)

    return non_overlapping_masks


def compare_ground_truth_and_predictions(list_of_ground_truth_annotations: List[dict], list_of_predicted_annotations: List[dict],
                                        do_filtering: bool=True, iou_score: float=0.5, confidence_score: float=0.5):
    """
    list_of_ground_truth_annotations: list of ground truth annotations in the coco format, with the annotations in LRE format and 
    stores with the 'segmentation' key.

    list_of_predicted_annotations: List of predicted annotations in the coco format, with the same 'image_id' as in the ground truth annotations.
    do_filtering: whether to perform the non_overlapping filtering or not. As some methods always output non overlapping predictions, or don't
    implement a 'score' function, this can be turned off to save computation time.
    iou_score: The iou score below which masks that overlap will be rejected.
    confidence score: float specifying the minimal confidence score a prediction need to be considered in the evaluation.
    """

    # Define the TP and IOU values
    TPs = []
    IOU = 0
    
    # The first step is to filter the predictions, which we do via non maximum suppression
    if do_filtering:
        list_of_predicted_annotations = non_overlapping_filtering(list_of_predicted_annotations, iou_score=iou_score, confidence_score=confidence_score)        

    # If there are no predictions, we only have False Positives, so we return this
    if not len(list_of_predicted_annotations):
        return {'TP': [], 'FP': [], 'FN': list(range(len(list_of_ground_truth_annotations))), 'IOU': 0}

    # Loop trough all ground truth and predicted pairs
    for i, ground_truth_annot in enumerate(list_of_ground_truth_annotations):
        for j, predicted_annot in enumerate(list_of_predicted_annotations):
            iou_value = iou([ground_truth_annot['segmentation']], [predicted_annot['segmentation']], [False]).item()
            if iou_value > 0.5:
                TPs.append((i, j))
                IOU+=iou_value
                
    # After this we can also calculate FP and FN
    FPs = [i for i in range(len(list_of_predicted_annotations)) if i not in [item[1] for item in TPs]]
    FNs = [i for i in range(len(list_of_ground_truth_annotations)) if i not in [item[0] for item in TPs]]

    return {'TP': TPs, 'FP': FPs, 'FN': FNs, 'IOU': IOU}



def PQ_calculation(dataframe):
    '''
    The metric calculations as done in https://github.com/irlabamsterdam/TPDLTextRedaction/blob/main/notebooks/Experiments.ipynb
    @param  pd.DataFrame    The dataframe for one class with the following columns { IOU, TP, FN, FP }
                            where the IOU is the sum of IOU scores and the others a total count.
    @return dict            The metric scores for this class
    '''
    
    SQ = dataframe['IOU'].sum() / dataframe['TP'].sum() if dataframe['TP'].sum() > 0 else 0
    RQ = dataframe['TP'].sum() / (dataframe['TP'].sum() + 0.5*dataframe['FN'].sum() + 0.5*dataframe['FP'].sum())
    
    PQ = SQ*RQ
    
    P = dataframe['TP'].sum() / (dataframe['TP'].sum() + dataframe['FP'].sum())
    R = dataframe['TP'].sum() / (dataframe['TP'].sum() + dataframe['FN'].sum())
    
    return {'SQ': round(SQ, 3), 'F1': round(RQ, 3), 'P': round(P, 3), 'R': round(R, 3), 'PQ': round(PQ, 3)}



def evaluate_predictions(list_of_ground_truth_annotations: list, list_of_predicted_annotations: list,
                        do_filtering: bool=True, iou_score: float=0.5, confidence_score=0.5,
                                                            count_empty_pages: bool=False):
    """
    This function loops through all annotations, gathers the ground truth and predictions for each image,
    and calculates the PQ metric for each of them.
    """
    dataset_scores = []
    ground_truth_df = pd.DataFrame(list_of_ground_truth_annotations)
    predictions_df = pd.DataFrame(list_of_predicted_annotations)
        
    for _, ground_truth_annots in tqdm(ground_truth_df.groupby('image_id')):
        image_id = ground_truth_annots['image_id'].unique()[0]
        page_type = ground_truth_annots['type'].unique()[0]
        predicted_annots = predictions_df[predictions_df['image_id'] == image_id]
        image_scores = compare_ground_truth_and_predictions(ground_truth_annots.to_dict(orient='records'),
                                                            predicted_annots.to_dict(orient='records'),
                                                           iou_score=iou_score,
                                                            confidence_score=confidence_score,
                                                            do_filtering=do_filtering)
        
        image_scores['TP'] = len(image_scores['TP'])
        image_scores['FP'] = len(image_scores['FP'])
        image_scores['FN'] = len(image_scores['FN'])
    
        image_scores['redaction_type'] = page_type
        image_scores['image_id'] = image_id
        image_scores['sizes'] = 0
        dataset_scores.append(image_scores)
    
    # This will skip pages without annotations, so we will have to redo those
    # but here the number of predictions is just the number of valse positives
    if count_empty_pages:
        unmatched_predictions = set(predictions_df['image_id']) - set(ground_truth_df['image_id'])
        unmatched_annots = predictions_df[predictions_df['image_id'].isin(unmatched_predictions)]
        for _, page in tqdm(unmatched_annots.groupby('image_id')):
            image_id = page['image_id'].unique()[0]
            preds = page.to_dict(orient='records')
            if do_filtering:
                filtered_preds = non_overlapping_filtering(preds, iou_score=iou_score, confidence_score=confidence_score)        
            else:
                filtered_preds = preds
            image_scores = {'TP': 0, 'FP': len(filtered_preds), 'FN': 0, 'IOU': 0,
                           'image_id': image_id, 'redaction_type': 'no_annotation', 'sizes': sum([area(item['segmentation']) for item in filtered_preds])}
            dataset_scores.append(image_scores)

    return pd.DataFrame(dataset_scores)


def calculate_color_specific_metrics(dataframe):
    scores = dataframe.groupby('redaction_type').apply(lambda x: pd.Series(PQ_calculation(x))).T
    scores['total'] = PQ_calculation(dataframe)
    return scores
    
