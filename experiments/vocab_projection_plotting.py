import click
import os
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error
from experiments.grounding_utils.common import VocabProjectionTracking
from inference.vlms import kl_divergence
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_style("whitegrid")

@click.command()
@click.option("--dataset", type=str, required=True, help="Dataset name")
@click.option("--vlm", type=str, help="VLM name", default="llava-v1.6-vicuna-7b-hf")
@click.option("--run_variants", type=click.Choice(["identification", "full_information", "image_reference", "trivial_black_full_information", "trivial_black_image_reference"]), multiple=True, default=["image_reference", "full_information", "trivial_black_image_reference"], help="Run variants to visualize")
@click.option("--metric", type=str, default="two_way_inclusion", help="Metric to visualize")
@click.pass_obj
def visualize_vocab_projection(parameters, dataset, vlm, run_variants, metric):
    results_df = get_results_df(dataset, vlm, parameters)
    total_projections = {}
    for run_variant in run_variants:
        kl_div, proj_prob, total_projection = recollect_projection(dataset, vlm, run_variant, parameters)
        metric_col = f"{metric}_{run_variant}_response"
        true_kl_divs, false_kl_divs = separate_by_metric(kl_div, results_df, metric_col, parameters)
        true_proj_probs, false_proj_probs = separate_by_metric(proj_prob, results_df, metric, parameters)
        true_total_projections, false_total_projections = separate_by_metric(total_projection, results_df, metric, parameters)
        total_projections[run_variant] = (true_total_projections, false_total_projections)
        lineplot(true_kl_divs, false_kl_divs, title=f"{dataset} {vlm} {run_variant} KL Divergence")
        lineplot(true_proj_probs, false_proj_probs, title=f"{dataset} {vlm} {run_variant} Projection Probability")
    if "full_information" in run_variants and "image_reference" in run_variants:
        pass



def get_results_df(dataset, vlm, parameters=None):
    if parameters is None:
        parameters = load_parameters()
    results_dir = os.path.join(parameters["results_dir"], dataset, vlm, "final_results.csv")
    if not os.path.exists(results_dir):
        log_error(parameters["logger"], f"Results file {results_dir} does not exist.")
    results_df = pd.read_csv(results_dir)
    return results_df



def recollect_projection(dataset, vlm, run_variant, parameters=None):
    """
    Recollect the vocab projection data for a given dataset and VLM.
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
    return vocab_projection_tracker.kl_divergence, vocab_projection_tracker.projection_prob, vocab_projection_tracker.total_projection

def separate_by_metric(dict_array, results_df, metric_col, parameters=None):
    if  parameters is None:
        parameters = load_parameters()
    trues = []
    falses = []
    for i, row in results_df.iterrows():
        if np.isnan(row[metric_col]) or row[metric_col] is None:
            continue
        else:
            if i not in dict_array:
                log_error(parameters["logger"], f"Index {i} not found in dict_array.")
            if row[metric_col] == True:
                trues.append(dict_array[i])
            elif row[metric_col] == False:
                falses.append(dict_array[i])
            else:
                log_error(parameters["logger"], f"Invalid value for {metric_col}: {row[metric_col]}. Must be True or False or None.")
    return np.array(trues), np.array(falses)

def lineplot(trues, falses, title, ylabel):
    layer_idx = None # This is a placeholder, you need to define how to get the layer index
    data_df = None # You basically need to form a dataframe with the data you want to plot
    sns.lineplot(layer_idx, trues, label="Success", color="blue") 
    sns.lineplot(layer_idx, falses, label="Failure", color="red")
    plt.title(title)
    plt.xlabel("Layer Index")
    plt.ylabel(ylabel)
    plt.legend()
    show()


def show():
    #plt.show()
    pass

def contrast_plot(full_information_trues, image_reference_trues, image_reference_falses, title):
    layer_idx = None
    data_df = None






    
