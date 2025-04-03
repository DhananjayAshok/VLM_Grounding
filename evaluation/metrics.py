from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings
import tqdm
import pandas as pd
import os
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error



def inclusion(candidate, reference):
    return str(reference).lower().strip() in str(candidate).lower()

def two_way_inclusion(candidate, reference):
    return inclusion(candidate, reference) or inclusion(reference, candidate)

def exact_match(candidate, reference):
    return str(candidate).lower().strip() == str(reference).lower().strip()

def bleu(candidate, reference):
    reference_tokens = [str(reference).split()]  # BLEU expects a list of lists for references
    candidate_tokens = candidate.split()
    # suppress UserWarning
    score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=SmoothingFunction().method0, weights=(1, 0, 0, 0))
    return score


metric_map = {
    "inclusion": inclusion,
    "two_way_inclusion": two_way_inclusion,
    "exact_match": exact_match,
    "bleu": bleu
}

def sequential_compute_metric(metric_fn, candidates, references):
    assert len(candidates) == len(references)
    scores = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        for candidate, reference in tqdm.tqdm(zip(candidates, references)):
            if candidate is None or reference is None:
                scores.append(None)
            else:
                scores.append(metric_fn(candidate, reference))
    return scores


def df_compute_metric_fn(metric_fn, df, candidate_column, reference_column, save=True, output_column=None, output_path=None, parameters=None):
    if parameters is None:
        parameters = load_parameters()
    if candidate_column not in df.columns:
        log_error(parameters["logger"], f"Column {candidate_column} not found in df with columns {df.columns}.")
    if reference_column not in df.columns:
        log_error(parameters["logger"], f"Column {reference_column} not found in df with columns {df.columns}.")
    scores = sequential_compute_metric(metric_fn, df[candidate_column].values, df[reference_column].values)
    if not save and not output_column:
        return scores
    elif not save:
        parameters["logger"].warning(f"Column {output_column} already found in df. Overwriting ...")
        df[output_column] = scores
        return df
    if output_column is None:
        log_error(parameters["logger"], "output_column must be specified if save is True.")
    if output_path is None:
        log_error(parameters["logger"], "output_path must be specified if save is True.")
    if output_column in df.columns:
        parameters["logger"].warning(f"Column {output_column} already found in df. Overwriting ...")
    df[output_column] = scores
    new_file = output_path
    if os.path.exists(new_file):
        parameters["logger"].warning(f"File {new_file} already exists. Overwriting ...")
    df.to_csv(new_file, index=False)
    return df


def df_compute_metric_str(metric_str, df, candidate_column, reference_column, save=True, output_column=None, output_path=None, parameters=None):
    if metric_str not in metric_map:
        log_error(parameters["logger"], f"Metric {metric_str} not found in metric_map {metric_map.keys()}.")
    metric_fn = metric_map[metric_str]
    return df_compute_metric_fn(metric_fn, df, candidate_column, reference_column, save, output_column, output_path, parameters)

    

def file_compute_metric_str(metric_str, csv_file, candidate_column, reference_column, save=True, output_column=None, save_suffix="_evaluated", parameters=None):
    if parameters is None:
        parameters = load_parameters()
    if not os.path.exists(csv_file):
        log_error(parameters["logger"], f"File {csv_file} not found.")
    df = pd.read_csv(csv_file)
    return df_compute_metric_str(metric_str, df, candidate_column, reference_column, save, output_column, csv_file.replace(".csv", f"{save_suffix}.csv"), parameters)

