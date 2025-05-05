import click
import pandas as pd
from experiments.grounding_utils.common import HiddenStateTracking
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
from experiments.grounding_utils.common import VocabProjectionTracking
import warnings
from sklearn.exceptions import ConvergenceWarning


def identify_layers(dataset, model, run_variant, parameters):
    hidden_tracker = HiddenStateTracking(dataset, model, run_variant, parameters)
    hidden_tracker.load_checkpoint()
    if hidden_tracker.hidden_states == {}:
        log_error(parameters["logger"], f"Hidden states not found for {dataset} {model} {run_variant}. Please run the model first.")
    items = hidden_tracker.hidden_states
    first = list(items.keys())[0]
    arrays = items[first]
    layers = []
    for key in arrays:
        layer, last, token_pos = key.split("_")
        if layer not in layers:
            layers.append(int(layer))
    layers.sort()
    return list(set(layers))

def get_hidden_states(dataset, model, metric, run_variant, parameters):
    hidden_tracker = HiddenStateTracking(dataset, model, run_variant, parameters)
    hidden_tracker.load_checkpoint()
    if hidden_tracker.hidden_states == {}:
        log_error(parameters["logger"], f"Hidden states not found for {dataset} {model} {run_variant}. Please run the model first.")
    results_df = parameters["results_dir"] + f"/{dataset}/{model}/final_results.csv"
    if not os.path.exists(results_df):
        log_error(parameters["logger"], f"Results file not found at {results_df}. Please run the model first.")
    results_df = pd.read_csv(results_df)
    label_col = f"{metric}_{run_variant}_response"
    if label_col not in results_df.columns:
        log_error(parameters["logger"], f"Label column {label_col} not found in results_df with columns {results_df.columns}.")
    nonnans = results_df[results_df[label_col].notna()]
    if len(nonnans) == 0:
        log_error(parameters["logger"], f"No non-nan values found in {label_col}.")
    return nonnans, hidden_tracker, label_col

def get_xydfs(dataset, model, layer, parameters, run_variant="image_reference", metric="two_way_inclusion", token_pos="input"):
    nonnans, hidden_tracker, label_col = get_hidden_states(dataset, model, metric, run_variant, parameters)
    # get the idx's of the non-nan values
    idxs = nonnans.index.tolist()
    # hidden_tracker.hidden_states format is {idx: {layer: hidden_state}}
    keep_idxs = []
    X = []
    # first get the ordered list of idx's from the results_df
    for idx in idxs:
        if idx not in hidden_tracker.hidden_states:
            parameters["logger"].warn(f"Hidden state for index {idx} not found. Please run the model first.")
            continue
        if f"{layer}_last_{token_pos}" not in hidden_tracker.hidden_states[idx]:
            log_error(parameters["logger"], f"Layer {layer} not found in hidden states for {idx} with keys {hidden_tracker.hidden_states[idx].keys()}.")
        X.append(hidden_tracker.hidden_states[idx][f"{layer}_last_{token_pos}"])
        keep_idxs.append(idx)
    X = np.array(X)
    y = nonnans.loc[keep_idxs, label_col].values.astype(int)
    X_perplexity = nonnans.loc[keep_idxs, f"{run_variant}_response_perplexity"].values.reshape(-1, 1)
    df = nonnans.loc[keep_idxs].reset_index(drop=True)
    return X, X_perplexity, y, df


def split_dataset(X, X_perplexity, y, df, train_size=0.8):
    idxs = list(range(len(y)))
    np.random.shuffle(idxs)
    n_train = int(len(y) * train_size)
    train_idxs = idxs[:n_train]
    test_idxs = idxs[n_train:]
    X_train = X[train_idxs]
    X_perplexity_train = X_perplexity[train_idxs]
    y_train = y[train_idxs]
    X_test = X[test_idxs]
    X_perplexity_test = X_perplexity[test_idxs]
    y_test = y[test_idxs]
    df_train = df.iloc[train_idxs]
    df_test = df.iloc[test_idxs]
    return X_train, X_perplexity_train, y_train, X_test, X_perplexity_test, y_test, df_train, df_test

def split_ood_dataset(data_dict, test_dataset=None):
    # data_dict is a dictionary with X, y and df, test_dataset is a key in this dict
    X_train = []
    X_perplexity_train = []
    y_train = []
    df_train = []
    X_test = None
    X_perplexity_test = None
    y_test = None
    df_test = None
    for key in data_dict:
        if key == test_dataset:
            X_test = data_dict[key]["X"]
            y_test = data_dict[key]["y"]
            X_perplexity_test = data_dict[key]["X_perplexity"]
            df_test = data_dict[key]["df"]
        else:
            X_train.append(data_dict[key]["X"])
            X_perplexity_train.append(data_dict[key]["X_perplexity"])
            y_train.append(data_dict[key]["y"])
            df_train.append(data_dict[key]["df"])
    X_train = np.concatenate(X_train)
    X_perplexity_train = np.concatenate(X_perplexity_train)
    y_train = np.concatenate(y_train)
    df_train = pd.concat(df_train, ignore_index=True)
    return X_train, X_perplexity_train, y_train, X_test, X_perplexity_test, y_test, df_train, df_test



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


def do_model_fit(model, X_train, X_perplexity_train, y_train, X_test, X_perplexity_test, y_test, verbose=True, prediction_dir=None, validation_split=0.15, parameters=None):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if parameters is None:
            parameters = load_parameters()
        #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=1-validation_split, random_state=42)

        model.fit(X_train, y_train)
        train_pred = model.predict_proba(X_train)
        #val_pred = model.predict_proba(X_val)
        test_pred = None
        test_pred_perp = None
        test_acc = None
        if X_test is not None:
            test_pred = model.predict_proba(X_test)
            test_acc, test_prec, test_recall, test_f1, test_auc = compute_metrics(y_test, test_pred)
            if verbose:
                parameters["logger"].info(f"Base Rate: ")
                print_base_rate(y_test, verbose=verbose, parameters=parameters)
                parameters["logger"].info(f"Total Test Accuracy: {test_acc}")
                parameters["logger"].info(f"Test Precision: {test_prec}, Test Recall: {test_recall}, Test F1: {test_f1}, Test AUC: {test_auc}")
        else:
            parameters["logger"].warning(f"Got no X_test in model fit. This should only happen if you are running OOD hidden state modeling.")
        #threshold = 0.95
        #perc_selected, accuracy, precision, recall, f1, auc = compute_threshold_metrics(y_test, test_pred, threshold)
        #if verbose:
        #    parameters["logger"].info(f"With threshold {threshold}: Predicts on {round(perc_selected*100, 2)} % of samples (Test)")
        #    parameters["logger"].info(f"Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, AUC: {auc}")
        #perc_selected, accuracy, val_acc, quartile, selected = compute_conformal_metrics(y_true_val=y_val, y_pred_proba_val=val_pred, y_true_test=y_test, y_pred_proba_test=test_pred, confidence=0.91)
        if verbose and False:
            selected = None
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
            savedir = f"{prediction_dir}/{meta_hash}/"
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            np.save(f"{savedir}//train_pred.npy", train_pred)
            if test_pred is not None:
                np.save(f"{savedir}//test_pred.npy", test_pred)
            model.save(f"{savedir}/")
            write_meta(f"{prediction_dir}/", meta, parameters["logger"])
        if X_perplexity_test is not None:
            parameters['logger'].info("This is compared to perplexity based decision making:")
            model.fit(X_perplexity_train, y_train)
            _ = model.predict_proba(X_perplexity_train)
            test_pred_perp = model.predict_proba(X_perplexity_test)
            test_acc_perp, _, _, _, _ = compute_metrics(y_test, test_pred_perp)
            if verbose:
                parameters["logger"].info(f"Perplexity based Test Accuracy: {test_acc_perp}")
        return test_pred, test_pred_perp, test_acc


def compute_coverage(estimator_pred, threshold):
    # estimator pred is shape (n_samples, 2)
    one_confidence = estimator_pred[:, 1]
    removed = one_confidence < threshold
    return (removed).mean() * 100

def compute_risk(estimator_pred, y_test, threshold):
    # estimator pred is shape (n_samples, 2)
    one_confidence = estimator_pred[:, 1]
    removed = one_confidence < threshold
    risk = (y_test[removed] == 1).mean() * 100
    return risk

def do_selective_prediction(y_test, test_pred, test_pred_perp, layer, parameters=None):
    if parameters is None:
        parameters = load_parameters()
    thresholds = [0.01 + i * 0.01 for i in range(100)]
    data = []
    columns = ["Method", "Threshold", "Coverage", "Risk"]
    for threshold in thresholds:
        coverage = compute_coverage(test_pred, threshold)
        risk = compute_risk(test_pred, y_test, threshold)
        data.append(["Probe", threshold, coverage, risk])
        coverage_perp = compute_coverage(test_pred_perp, threshold)
        risk_perp = compute_risk(test_pred_perp, y_test, threshold)
        data.append(["Perplexity", threshold, coverage_perp, risk_perp])
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f"{parameters['results_dir']}/okvqa_selective_prediction/{layer}.csv", index=False)
    return df


@click.command()
@click.option("--datasets", multiple=True, required=True)
@click.option("--vlm", default="llava-v1.6-vicuna-7b-hf", type=click.Choice(["llava-v1.6-mistral-7b-hf", "llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf", "instructblip-vicuna-7b", "instructblip-vicuna-13b"]), help="The model whose hidden states to use")
@click.option('--layer', type=int, default=None, help='The layer of the model to use')
@click.option("--run_variant", type=click.Choice(["identification","image_reference", "full_information", "trivial_black"]), default="image_reference", help="The variant of the prompt to use")
@click.option("--metric", type=click.Choice(["two_way_inclusion", "exact_match"]), default="two_way_inclusion", help="The metric to use as the label")
@click.option("--token_pos", type=click.Choice(["input", "output"]), default="output", help="The position of the token to use")
@click.pass_obj
def fit_hidden_state_predictor(parameters, datasets, vlm, layer, run_variant, metric, token_pos):
    np.random.seed(parameters["random_seed"])
    available_layers = identify_layers(datasets[0], vlm, run_variant, parameters)
    if layer is not None and layer not in available_layers:
        log_error(parameters["logger"], f"Layer {layer} not found in available layers {available_layers}.")
    if layer is not None:
        layers_to_do = [layer]
    else:
        layers_to_do = available_layers
    if len(datasets) == 1:
        dataset = datasets[0]
        parameters["logger"].info(f"Only one dataset {datasets[0]} found. Using that for training and testing.")
        columns = ["Layer", "Test Accuracy"]
        data = []
        results_dir_parent = parameters['results_dir'] + f"/{dataset}/{vlm}/hidden_states/{run_variant}/"
        for layer in layers_to_do:
            results_dir = results_dir_parent + f"/layer_{layer}"
            X, X_perplexity, y, df = get_xydfs(datasets[0], vlm, layer, parameters, run_variant=run_variant, metric=metric, token_pos=token_pos)
            X_train, X_perplexity_train, y_train, X_test, X_perplexity_test, y_test, df_train, df_test = split_dataset(X, X_perplexity, y, df)
            prediction_dir = results_dir
            model = Linear()
            parameters["logger"].info(f"Probes for layer {layer}")
            _, _, test_acc = do_model_fit(model, X_train, X_perplexity_train, y_train, X_test, X_perplexity_test, y_test, verbose=True, prediction_dir=prediction_dir, parameters=parameters)
            data.append([layer, test_acc])
        data = pd.DataFrame(data, columns=columns)
        data.to_csv(results_dir + "/iid_results.csv", index=False)
    else:
        parameters["logger"].info(f"Multiple datasets found {datasets}. Conducting OOD experiment and saving a single weight when trained on all.")
        xydfs = {}
        columns = ["Layer", "Dataset", "Test Accuracy"]
        data = []
        for layer in layers_to_do:            
            for dataset in datasets:
                X, X_perplexity, y, df = get_xydfs(dataset, vlm, layer, parameters, run_variant=run_variant, metric=metric, token_pos=token_pos)
                xydfs[dataset] = {"X": X, "y": y, "df": df, "X_perplexity": X_perplexity}
            for dataset in datasets:                
                model = Linear()
                X_train, X_perplexity_train, y_train, X_test, X_perplexity_test, y_test, df_train, df_test = split_ood_dataset(xydfs, dataset)
                parameters["logger"].info(f"Testing on {dataset} after training on all other datasets. Layer {layer}")
                test_pred, test_pred_perp, test_acc = do_model_fit(model, X_train, X_perplexity_train, y_train, X_test, X_perplexity_test, y_test, verbose=True, prediction_dir=None, parameters=parameters)
                data.append([layer, dataset, test_acc])
                if dataset == "okvqa":
                    do_selective_prediction(y_test, test_pred, test_pred_perp, layer, parameters=parameters)
            X_train, X_perplexity_train, y_train, X_test, X_perplexity_test, y_test, df_train, df_test = split_ood_dataset(xydfs)
            results_dir = parameters["results_dir"] + f"/all/{vlm}/hidden_states/{run_variant}/layer_{layer}/"
            model = Linear()
            do_model_fit(model, X_train, X_perplexity_train, y_train, X_test, X_perplexity_test, y_test, verbose=True, prediction_dir=results_dir, parameters=parameters)

        data = pd.DataFrame(data, columns=columns)
        results_dir = parameters["results_dir"] + f"/{dataset}/{vlm}/hidden_states/{run_variant}/"
        for dataset in datasets:
            subdf = data[data["Dataset"] == dataset].reset_index(drop=True)
            subdf.to_csv(results_dir + f"/ood_results.csv", index=False)
        return 

if __name__ == "__main__":
    fit_hidden_state_predictor()
