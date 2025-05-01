import click
import os
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error
from experiments.grounding_utils.common import VocabProjectionTracking
from experiments.hidden_state_predictor import get_hidden_states
from inference.vlms import kl_divergence
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd

sns.set_style("whitegrid")

@click.command()
@click.option("--dataset", type=str, required=True, help="Dataset name")
@click.option("--vlm", type=str, help="VLM name", default="llava-v1.6-vicuna-7b-hf")
@click.option("--run_variants", type=click.Choice(["identification", "full_information", "image_reference", "trivial_black_full_information", "trivial_black_image_reference"]), multiple=True, default=["image_reference", "full_information", "trivial_black_image_reference"], help="Run variants to visualize")
@click.option("--metric", type=str, default=None, help="Metric to visualize")
@click.option("--remove_trivial_success", is_flag=True, default=False, help="Remove trivial success from the results")
@click.pass_obj
def visualize_vocab_projection(parameters, dataset, vlm, run_variants, metric, remove_trivial_success):
    if metric is None:
        if "_mcq" in dataset:
            metric = "mcq_correct"
        else:
            metric = "two_way_inclusion"
    results_df = get_results_df(dataset, vlm, parameters)
    total_projections = {}
    for run_variant in run_variants:
        kl_div, proj_prob, total_projection = recollect_projection(dataset, vlm, run_variant, parameters)
        true_kl_divs, false_kl_divs = separate_by_metric(kl_div, results_df, metric, run_variant, parameters, remove_trivial_success=remove_trivial_success)
        true_proj_probs, false_proj_probs = separate_by_metric(proj_prob, results_df, metric, run_variant, parameters, remove_trivial_success=remove_trivial_success)
        true_total_projections, false_total_projections = separate_by_metric(total_projection, results_df, metric, run_variant, parameters, dict_return=True, remove_trivial_success=remove_trivial_success)
        total_projections[run_variant] = (true_total_projections, false_total_projections)
        #lineplot(true_kl_divs, false_kl_divs, "KL Divergence w Prev Layer", f"{dataset}_{vlm}_{run_variant}_kl_divergence")
        lineplot(true_proj_probs, false_proj_probs, "Probability of Token", f"{dataset}/{vlm}/projection_probability/{run_variant}")
    if "full_information" in run_variants and "image_reference" in run_variants:
        plot_contrast_kl(total_projections["full_information"][0], total_projections["image_reference"][0], total_projections["image_reference"][1], f"{dataset}/{vlm}/remove_trivial_{remove_trivial_success}/kl_divergence/full_information_vs_image_reference", parameters)
    if "trivial_black_image_reference" in run_variants and "image_reference" in run_variants:
        total_projections["trivial_black_image_reference"][0].update(total_projections["trivial_black_image_reference"][1]) # This is a hack to get the same format as the other ones so that the hidden states can be matched, but the trivial black here are not the truths. 
        plot_contrast_kl(total_projections["trivial_black_image_reference"][0], total_projections["image_reference"][0], total_projections["image_reference"][1], f"{dataset}/{vlm}/remove_trivial_{remove_trivial_success}/kl_divergence/trivial_black_vs_image_reference", parameters)
    del total_projections

    hidden_states = {}
    if "image_reference" in run_variants:
        true_hidden, false_hidden = get_hidden_dict(results_df, dataset, vlm, metric, "image_reference", parameters, remove_trivial_success=remove_trivial_success)
        hidden_states["image_reference"] = (true_hidden, false_hidden)
        if "full_information" in run_variants:
            true_hidden, false_hidden = get_hidden_dict(results_df, dataset, vlm, metric, "full_information", parameters, remove_trivial_success=remove_trivial_success)
            hidden_states["full_information"] = (true_hidden, false_hidden)
            plot_contrast_cosine(hidden_states["full_information"][0], hidden_states["image_reference"][0], hidden_states["image_reference"][1], f"{dataset}/{vlm}/remove_trivial_{remove_trivial_success}/cosine_similarity/full_information_vs_image_reference", parameters)
        if "trivial_black_image_reference" in run_variants:
            true_hidden, false_hidden = get_hidden_dict(results_df, dataset, vlm, metric, "trivial_black_image_reference", parameters, remove_trivial_success=remove_trivial_success)
            hidden_states["trivial_black_image_reference"] = (true_hidden, false_hidden)
            plot_contrast_cosine(hidden_states["trivial_black_image_reference"][0], hidden_states["image_reference"][0], hidden_states["image_reference"][1], f"{dataset}/{vlm}/remove_trivial_{remove_trivial_success}/cosine_similarity/trivial_black_vs_image_reference", parameters)
    return 



def get_results_df(dataset, vlm, parameters=None):
    if parameters is None:
        parameters = load_parameters()
    results_dir = os.path.join(parameters["results_dir"], dataset, vlm, "final_results.csv")
    if not os.path.exists(results_dir):
        log_error(parameters["logger"], f"Results file {results_dir} does not exist.")
    results_df = pd.read_csv(results_dir)
    return results_df


def get_hidden_dict(results_df, dataset, model, metric, run_variant, parameters, token_kind="input", remove_trivial_success=False):
    _, hidden_tracker, metric_col = get_hidden_states(dataset, model, metric, run_variant, parameters)
    dict_array = hidden_tracker.hidden_states
    trues = {}
    falses = {}
    trivial_col = f"{metric}_trivial_mode_image_reference_response"
    for i, row in results_df.iterrows():
        if np.isnan(row[metric_col]) or row[metric_col] is None:
            continue
        else:
            if i not in dict_array:
                parameters['logger'].warn(f"Index {i} not found in dict_array.")
                continue
                #log_error(parameters["logger"], f"Index {i} not found in dict_array.")
            if remove_trivial_success and row[trivial_col] == True:
                continue
            internal_dict = {}
            for key in dict_array[i]:
                layer, pos, token_pos_item = key.split("_")
                layer = int(layer)
                if token_pos_item == token_kind:
                    internal_dict[layer] = dict_array[i][key]
            if row[metric_col] == True:
                trues[i] = internal_dict
            elif row[metric_col] == False:
                falses[i] = internal_dict
            else:
                log_error(parameters["logger"], f"Invalid value for {metric_col}: {row[metric_col]}. Must be True or False or None.")
    return trues, falses



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
        log_error(parameters["logger"], f"No vocab projection data found for {dataset} {vlm} {run_variant}.")
        return
    return vocab_projection_tracker.kl_divergence, vocab_projection_tracker.projection_prob, vocab_projection_tracker.total_projection


def separate_by_metric(dict_array, results_df, metric, run_variant, parameters=None, dict_return=False, remove_trivial_success=False):
    if  parameters is None:
        parameters = load_parameters()
    if dict_return:
        trues = {}
        falses = {}
    else:
        trues = []
        falses = []
    metric_col = f"{metric}_{run_variant}_response"
    trivial_col = f"{metric}_trivial_mode_image_reference_response"
    for i, row in results_df.iterrows():
        if np.isnan(row[metric_col]) or row[metric_col] is None:
            continue
        else:
            if remove_trivial_success and row[trivial_col] == True:
                continue
            if i not in dict_array:
                parameters['logger'].warn(f"Index {i} not found in dict_array.")
                continue
                #log_error(parameters["logger"], f"Index {i} not found in dict_array.")
            if row[metric_col] == True:
                if dict_return:
                    trues[i] = dict_array[i]
                else:
                    trues.append(dict_array[i])
            elif row[metric_col] == False:
                if dict_return:
                    falses[i] = dict_array[i]
                else:
                    falses.append(dict_array[i])
            else:
                log_error(parameters["logger"], f"Invalid value for {metric_col}: {row[metric_col]}. Must be True or False or None.")
    if dict_return:
        return trues, falses
    else:
        return np.array(trues), np.array(falses)

def lineplot(trues, falses, ylabel, save_name):
    columns = ["Layer Index", ylabel,"Linking Status"]
    data = []
    for item in trues:
        for layer_idx, layer in enumerate(item):
            data.append([layer_idx, layer, "Success"])
    for item in falses:
        for layer_idx, layer in enumerate(item):
            data.append([layer_idx, layer, "Failure"])
    data_df = pd.DataFrame(data, columns=columns)
    data_df["Layer Index"] = data_df["Layer Index"].astype(int)    
    sns.lineplot(data=data_df, x="Layer Index", y=ylabel, hue="Linking Status", palette=["green", "red"], linewidth=2.5, errorbar="sd") 
    plt.title("")
    plt.legend()
    show(save_name, data_df=data_df)


def show(save_path, parameters=None, data_df=None):
    if parameters is None:
        parameters = load_parameters()
    figure_path = parameters["results_dir"] + f"/figures/{save_path}"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    if data_df is not None:
        data_df.to_csv(f"{figure_path}.csv", index=False)
    #plt.show()
    plt.savefig(f"{figure_path}.pdf")
    plt.clf()

    pass

def plot_contrast_kl(reference_truths, candidate_truths, candidate_falses, save_name, parameters):
    """
    Compute the KL divergence between the reference and candidate distributions.
    """
    columns = ["Layer Index", "KL Divergence", "Linking Status"]
    data = []
    for idx in tqdm(candidate_truths, desc="Computing KL Divergence", total=len(candidate_truths)):
        if idx not in reference_truths:
            parameters["logger"].warn(f"Index {idx} not found in reference_truths.")
            continue
            log_error(parameters["logger"], f"Index {idx} not found in reference_truths.")
        reference_truth = reference_truths[idx]
        candidate_truth = candidate_truths[idx]
        n_layers, vocab_size = reference_truth.shape
        for layer in range(n_layers):
            kl_div = kl_divergence(reference_truth[layer], candidate_truth[layer])
            data.append([layer, kl_div, "Success"])
    for idx in tqdm(candidate_falses, desc="Computing KL Divergence", total=len(candidate_falses)):
        if idx not in reference_truths:
            parameters["logger"].warn(f"Index {idx} not found in reference_truths.")
            continue
        reference_truth = reference_truths[idx]
        candidate_false = candidate_falses[idx]
        n_layers, vocab_size = reference_truth.shape
        for layer in range(n_layers):
            kl_div = kl_divergence(reference_truth[layer], candidate_false[layer])
            data.append([layer, kl_div, "Failure"])
    data_df = pd.DataFrame(data, columns=columns)
    data_df["Layer Index"] = data_df["Layer Index"].astype(int)
    sns.lineplot(data=data_df, x="Layer Index", y="KL Divergence", hue="Linking Status", palette=["green", "red"], linewidth=2.5, errorbar="sd")
    plt.title("")
    plt.legend()
    show(save_name, parameters, data_df=data_df)

def plot_contrast_cosine(reference_truths, candidate_truths, candidate_falses, save_name, parameters):
    columns = ["Layer Index", "Cosine Similarity", "Linking Status"]
    data = []
    for idx in tqdm(candidate_truths, desc="Computing Cosine Similarity", total=len(candidate_truths)):
        if idx not in reference_truths:
            parameters["logger"].warn(f"Index {idx} not found in reference_truths.")
            continue
            log_error(parameters["logger"], f"Index {idx} not found in reference_truths.")
        reference_truth = reference_truths[idx]
        candidate_truth = candidate_truths[idx]
        for layer in candidate_truth:
            if layer not in reference_truth:
                parameters["logger"].warn(f"Layer {layer} not found in reference_truth.")
                continue
            reference_truth_layer = reference_truth[layer]
            candidate_truth_layer = candidate_truth[layer]
            cosine_sim = np.dot(reference_truth_layer, candidate_truth_layer) / (np.linalg.norm(reference_truth_layer) * np.linalg.norm(candidate_truth_layer))
            data.append([layer, cosine_sim, "Success"])

    for idx in tqdm(candidate_falses, desc="Computing Cosine Similarity", total=len(candidate_falses)):
        if idx not in reference_truths:
            parameters["logger"].warn(f"Index {idx} not found in reference_truths.")
            continue
        reference_truth = reference_truths[idx]
        candidate_false = candidate_falses[idx]
        for layer in candidate_false:
            if layer not in reference_truth:
                parameters["logger"].warn(f"Layer {layer} not found in reference_truth.")
                continue
            reference_truth_layer = reference_truth[layer]
            candidate_false_layer = candidate_false[layer]
            cosine_sim = np.dot(reference_truth_layer, candidate_false_layer) / (np.linalg.norm(reference_truth_layer) * np.linalg.norm(candidate_false_layer))
            data.append([layer, cosine_sim, "Failure"])
    data_df = pd.DataFrame(data, columns=columns)
    data_df["Layer Index"] = data_df["Layer Index"].astype(int)
    sns.lineplot(data=data_df, x="Layer Index", y="Cosine Similarity", hue="Linking Status", palette=["green", "red"], linewidth=2.5, errorbar="sd")
    plt.title("")
    plt.legend()
    show(save_name, parameters, data_df=data_df)
    return


    







    
