import click
import pandas as pd
from inference.vlms import LlaVaInference
from data import get_dataset
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from experiments.hidden_modeling_utils.model import Linear
from experiments.hidden_modeling_utils.metrics import compute_metrics, compute_conformal_metrics, compute_threshold_metrics
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error
from utils.hash_handling import write_meta, hash_meta_dict
from experiments.grounding_utils.identification import HiddenStateTracking, VocabProjectionTracking

def get_xydfs(dataset, model, layer, parameters, run_variant="image_reference", metric="two_way_inclusion"):
    hidden_tracker = HiddenStateTracking(dataset, model, run_variant, parameters)
    hidden_tracker.load_checkpoint()
    if hidden_tracker.hidden_states == {}:
        log_error(parameters["logger"], f"Hidden states not found for {dataset} {model} {run_variant}. Please run the model first.")
        return
    results_df = parameters["results_dir"] + f"/{dataset}/{model}/final_results.csv"
    if not os.path.exists(results_df):
        log_error(parameters["logger"], f"Results file not found at {results_df}. Please run the model first.")
        return
    results_df = pd.read_csv(results_df)
    label_col = f"{metric}_{run_variant}_response"
    if label_col not in results_df.columns:
        log_error(parameters["logger"], f"Label column {label_col} not found in results_df with columns {results_df.columns}.")
        return
    nonnans = results_df[results_df[label_col].notna()]
    if len(nonnans) == 0:
        log_error(parameters["logger"], f"No non-nan values found in {label_col}.")
        return
    # get the idx's of the non-nan values
    idxs = nonnans.index.tolist()
    # hidden_tracker.hidden_states format is {idx: {layer: hidden_state}}
    X = []
    # first get the ordered list of idx's from the results_df
    for idx in idxs:
        if idx not in hidden_tracker.hidden_states:
            log_error(parameters["logger"], f"Hidden state for index {idx} not found. Please run the model first.")
            continue
        if layer not in hidden_tracker.hidden_states[idx]:
            log_error(parameters["logger"], f"Layer {layer} not found in hidden states for {idx}.")
        X.append(hidden_tracker.hidden_states[idx][layer])
    X = np.array(X)
    y = nonnans[label_col].values
    df = nonnans.reset_index(drop=True)
    return X, y, df


def split_dataset(X, y, df, train_size=0.8):
    idxs = range(len(y))
    np.random.shuffle(idxs)
    n_train = int(len(y) * train_size)
    train_idxs = idxs[:n_train]
    test_idxs = idxs[n_train:]
    X_train = X[train_idxs]
    y_train = y[train_idxs]
    X_test = X[test_idxs]
    y_test = y[test_idxs]
    df_train = df.iloc[train_idxs]
    df_test = df.iloc[test_idxs]
    return X_train, y_train, X_test, y_test, df_train, df_test

def split_ood_dataset(data_dict, test_dataset=None):
    # data_dict is a dictionary with X, y and df, test_dataset is a key in this dict
    X_train = []
    y_train = []
    df_train = []
    X_test = None
    y_test = None
    df_test = None
    for key in data_dict:
        if key == test_dataset:
            X_test = data_dict[key]["X"]
            y_test = data_dict[key]["y"]
            df_test = data_dict[key]["df"]
        else:
            X_train.append(data_dict[key]["X"])
            y_train.append(data_dict[key]["y"])
            df_train.append(data_dict[key]["df"])
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    df_train = pd.concat(df_train, ignore_index=True)
    return X_train, y_train, X_test, y_test, df_train, df_test



def compute_perc(array, lower, upper):
    return round((1 - ((lower <= array) & (array <= upper)).mean())*100, 2)


def print_base_rate(arr, verbose=False, parameters=None):
    if parameters is None:
        parameters = load_parameters()
    classes = range(len(set(arr)))
    class_props = []
    for class_label in classes:
        class_prop = round((arr == class_label).mean()*100, 2)
        if verbose:
            parameters["logger"].info(f"{class_label}: {class_prop}")
        class_props.append(class_prop)
    return max(class_props)


def safe_length(x):
    if not isinstance(x, str):
        return None
    return len(x.split(" "))


def do_model_fit(model, X_train, y_train, X_test, y_test, verbose=True, prediction_dir=None, validation_split=0.15, parameters=None):
    if parameters is None:
        parameters = load_parameters()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=1-validation_split, random_state=42)
    model.fit(X_train, y_train)
    train_pred = model.predict_proba(X_train)
    val_pred = model.predict_proba(X_val)
    test_pred = model.predict_proba(X_test)
    test_acc, test_prec, test_recall, test_f1, test_auc = compute_metrics(y_test, test_pred)
    if verbose:
        parameters["logger"].info(f"Base Rate: ")
        print_base_rate(y_test, verbose=verbose, parameters=parameters)
        parameters["logger"].info(f"Total Test Accuracy: {test_acc}")
        parameters["logger"].info(f"Test Precision: {test_prec}, Test Recall: {test_recall}, Test F1: {test_f1}, Test AUC: {test_auc}")
    threshold = 0.95
    perc_selected, accuracy, precision, recall, f1, auc = compute_threshold_metrics(y_test, test_pred, threshold)
    if verbose:
        parameters["logger"].info(f"With threshold {threshold}: Predicts on {round(perc_selected*100, 2)} % of samples (Test)")
        parameters["logger"].info(f"Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, AUC: {auc}")
    perc_selected, accuracy, val_acc, quartile, selected = compute_conformal_metrics(y_true_val=y_val, y_pred_proba_val=val_pred, y_true_test=y_test, y_pred_proba_test=test_pred, confidence=0.91)
    if verbose:
        if selected is None:
            parameters["logger"].info(f"Unable to find conformal quartile")
        else:
            parameters["logger"].info(f"Conformal Predicts on {round(perc_selected*100, 2)} % of samples (Test)")
            # within selected columns:
            parameters["logger"].info(f"Distribution of probe target in selected columns:")
            print_base_rate(y_test[selected], verbose=True)
            parameters["logger"].info(f"Test Accuracy: {accuracy}, Val Accuracy: {val_acc}, Quartile: {quartile}")
        

    if prediction_dir is not None:
        meta = {"random_seed": parameters["random_seed"]}
        meta_hash = hash_meta_dict(meta)
        np.save(f"{prediction_dir}/{meta_hash}/train_pred.npy", train_pred)
        np.save(f"{prediction_dir}/{meta_hash}/test_pred.npy", test_pred)
        model.save(f"{prediction_dir}/{meta_hash}/")
        write_meta(f"{prediction_dir}/", meta, parameters["logger"])
    return train_pred, test_pred, test_acc


@click.command()
@click.option("--datasets", multiple=True, required=True)
@click.option("--vlm", required=True, type=click.Choice(["llava-v1.6-mistral-7b-hf", "llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf", "instructblip-vicuna-7b", "instructblip-vicuna-13b"]), help="The model whose hidden states to use")
@click.option('--layer', type=int, default=10, help='The layer of the model to use')
@click.option("--run_variant", type=click.Choice(["identification","image_reference", "full_information", "trivial_black"]), default="image_reference", help="The variant of the prompt to use")
@click.option("--metric", type=click.Choice(["two_way_inclusion", "exact_match"]), default="two_way_inclusion", help="The metric to use as the label")
@click.pass_obj
def main(parameters, datasets, vlm, layer, run_variant, metric):
    np.random.seed(parameters["random_seed"])
    if len(datasets) == 1:
        parameters["logger"].info(f"Only one dataset {datasets[0]} found. Using that for training and testing.")
        results_dir = parameters["results_dir"] + f"/{dataset}/{vlm}/hidden_states/{run_variant}/layer_{layer}"
        X, y, df = get_xydfs(datasets[0], vlm, layer, parameters, run_variant=run_variant, metric=metric)
        X_train, y_train, X_test, y_test, df_train, df_test = split_dataset(X, y, df)
        prediction_dir = results_dir
        model = Linear()
        do_model_fit(model, X_train, y_train, X_test, y_test, verbose=True, prediction_dir=prediction_dir, parameters=parameters)
    else:
        parameters["logger"].info(f"Multiple datasets found {datasets}. Conducting OOD experiment and saving a single weight when trained on all.")
        xydfs = {}
        for dataset in datasets:
            X, y, df = get_xydfs(dataset, vlm, layer, parameters, run_variant=run_variant, metric=metric)
            xydfs[dataset] = {"X": X, "y": y, "df": df}
        for dataset in datasets:
            model = Linear()
            X_train, y_train, X_test, y_test, df_train, df_test = split_ood_dataset(xydfs, dataset)
            do_model_fit(model, X_train, y_train, X_test, y_test, verbose=True, parameters=parameters)
        X_train, y_train, X_test, y_test, df_train, df_test = split_ood_dataset(xydfs)
        results_dir = parameters["results_dir"] + f"all/{vlm}/hidden_states/{run_variant}/layer_{layer}"
        model = Linear()
        do_model_fit(model, X_train, y_train, X_test, y_test, verbose=True, prediction_dir=results_dir, parameters=parameters)    

if __name__ == "__main__":
    main()
