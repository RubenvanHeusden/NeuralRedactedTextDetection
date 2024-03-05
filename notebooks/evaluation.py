"""
This script contains the code that can be used to evaluate any algorithm that outputs COCO instances predictions
against a COCO ground truth json file. It includes calculations over dataframes, and the possibility to split the predicted scores
into the scores for the various redaction types.
"""

import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.ops import nms
from pycocotools.mask import iou, toBbox, area, decode, encode
from torchvision.ops import box_convert

def read_json(file_name):
    with open(file_name, 'r') as json_file:
        return json.load(json_file)

def compare_ground_truth_and_predictions(list_of_ground_truth_annotations: list, list_of_predicted_annotations: list,
                                        do_filtering: bool=True, iou_score: float=0.5, confidence_score: float=0.8):
    """
    This function takes as input a list of ground truth annotations in COCO format for a single image
    and a list of predictions for that image, and calculates the TP, FP, FN and total IOU scores for 
    calculation of the PQ score.
    """
    # Define the TP and IOU values
    TPs = []
    IOU = 0
    
    if not len(list_of_predicted_annotations):
    	return {'TP': 0, 'FP': 0, 'FN': len(list_of_ground_truth_annotations), 'IOU': 0}

    # The first step is to filter the predictions, which we do via non maximum suppression
    if do_filtering:
        list_of_predicted_annotations = [item for item in list_of_predicted_annotations if area(item['segmentation']) > 0 and (item['score'] > confidence_score)]
        sorted_predictions = sorted(list_of_predicted_annotations, key=lambda x : x['score'])[::-1]
        if len(sorted_predictions):
            filtered_results = [sorted_predictions[0]]
            for i in range(1, len(sorted_predictions[1:]), 2):
                mask = sorted_predictions[i]
                next_mask = sorted_predictions[i+1]
                if iou([mask['segmentation']], [next_mask['segmentation']], [False]) > iou_score:
                    bin_mask, bin_next_mask = decode(mask['segmentation']), decode(next_mask['segmentation'])
                    overlap = bin_mask & bin_next_mask
                    new_bin_mask, new_bin_next_mask = bin_mask-overlap, bin_next_mask - overlap
                    new_mask, new_next_mask = encode(new_bin_mask), encode(new_bin_next_mask)
                    mask['segmentation'] = new_mask
                    next_mask['segmentation'] = new_next_mask
                filtered_results.extend([mask, next_mask])
            list_of_predicted_annotations = filtered_results

        else:
            return {'TP': 0, 'FP': 0, 'FN': len(list_of_ground_truth_annotations), 'IOU': 0}

        

    # Loop trough all ground truth and predicted pairs
    for i, ground_truth_annot in enumerate(list_of_ground_truth_annotations):
        for j, predicted_annot in enumerate(list_of_predicted_annotations):
            iou_value = iou([ground_truth_annot['segmentation']], [predicted_annot['segmentation']], [False]).item()
            if iou_value > 0.5:
                TPs.append((i, j))
                IOU+=iou_value
                
    # After this we can also calculate FP and FN
    FPs = len([i for i in range(len(list_of_predicted_annotations)) if i not in [item[1] for item in TPs]])
    FNs = len([i for i in range(len(list_of_ground_truth_annotations)) if i not in [item[0] for item in TPs]])
    
    return {'TP': len(TPs), 'FP': FPs, 'FN': FNs, 'IOU': IOU}



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
    
    return {'SQ': round(SQ, 2), 'F1': round(RQ, 2), 'P': round(P, 2), 'R': round(R, 2)}



def evaluate_predictions(list_of_ground_truth_annotations: list, list_of_predicted_annotations: list,
                        do_filtering: bool=True, iou_score: float=0.5, confidence_score=0.7,
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
    
        image_scores['redaction_type'] = page_type
        image_scores['image_id'] = image_id
        dataset_scores.append(image_scores)
    
    # This will skip pages without annotations, so we will have to redo those
    # but here the number of predictions is just the number of valse positives
    unmatched_predictions = set(predictions_df['image_id']) - set(ground_truth_df['image_id'])
    unmatched_annots = predictions_df[predictions_df['image_id'].isin(unmatched_predictions)]
    if count_empty_pages:
        for _, page in unmatched_annots.groupby('image_id'):
            image_id = page['image_id'].unique()[0]
            preds = len(page.to_dict(orient='records'))
            if do_filtering:
                filtered_preds = [item for item in page.to_dict(orient='records') if area(item['segmentation']) > 0 and (item['score'] > confidence_score)]
            else:
                filtered_preds = page
            image_scores = {'TP': 0, 'FP': len(filtered_preds), 'FN': 0, 'IOU': 0,
                           'image_id': image_id, 'redaction_type': 'no_annotation'}
            dataset_scores.append(image_scores)

    return pd.DataFrame(dataset_scores)


def calculate_color_specific_metrics(dataframe):
    scores = dataframe.groupby('redaction_type').apply(lambda x: pd.Series(PQ_calculation(x))).T
    scores['total'] = PQ_calculation(dataframe)
    return scores
    
