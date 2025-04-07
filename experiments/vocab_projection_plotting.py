import click
import os
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error
from experiments.grounding_utils.common import VocabProjectionTracking
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

@click.command()
@click.option("--dataset", type=str, required=True, help="Dataset name")
@click.option("--vlm", type=str, required=True, help="VLM name")
def visualize_vocab_projection(dataset, vlm):
    pass


def recollect_projection(dataset, vlm, run_variant, parameters=None):
    """
    Recollect the vocab projection data for a given dataset and VLM.
    This is useful if the data has been corrupted or lost.
    """
    if parameters is None:
        parameters = load_parameters()
    vocab_projection_tracker = VocabProjectionTracking(dataset, vlm, run_variant, parameters)
    # Load the checkpoint
    vocab_projection_tracker.load_checkpoint()
    # Check if the data is empty
    if vocab_projection_tracker.kl_divergence == {}:
        log_error(parameters["logger"], f"No vocab projection data found for {dataset} {vlm}.")
        return
    return vocab_projection_tracker.kl_divergence, vocab_projection_tracker.projection_prob

def separate_by_metric(dict_array, results_df, metric, parameters=None):
    if  parameters is None:
        parameters = load_parameters()
    trues = []
    falses = []
    for i, row in results_df.iterrows():
        if row[metric] is None:
            continue
        else:
            if i not in dict_array:
                log_error(parameters["logger"], f"Index {i} not found in dict_array.")
            if row[metric] == True:
                trues.append(dict_array[i])
            elif row[metric] == False:
                falses.append(dict_array[i])
            else:
                log_error(parameters["logger"], f"Invalid value for {metric}: {row[metric]}. Must be True or False or None.")
    return np.array(trues), np.array(falses)

def plot_by_metric(dict_array, results_df, metric, parameters=None):
    if parameters is None:
        parameters = load_parameters()
    trues, falses = separate_by_metric(dict_array, results_df, metric, parameters)
    # unsure but I think trues shape is (n_datapoints, n_layers-1) we want a line plot of n_layers-1
    # TODO: Figure this one out when you can. 
    # Do KL of layer i, of the two different kinds of full language and vision language impoverished for linking failure and success. 
    # This is different from what you were doing of just doing i vs i+1, do that also. 

    
