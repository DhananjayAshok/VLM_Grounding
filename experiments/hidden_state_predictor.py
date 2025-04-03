import click
import pandas as pd
from inference.vlms import LlaVaInference
from data import get_dataset
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from experiments.hidden_modeling_utilities.model import Linear
from experiments.hidden_modeling_utilities.metrics import compute_metrics, compute_conformal_metrics, compute_threshold_metrics


def get_xydfs(hidden, df):
    # TODO: IMplement
    return


def compute_perc(array, lower, upper):
    return round((1 - ((lower <= array) & (array <= upper)).mean())*100, 2)


def print_base_rate(arr, verbose=False):
    classes = range(len(set(arr)))
    class_props = []
    for class_label in classes:
        class_prop = round((arr == class_label).mean()*100, 2)
        if verbose:
            print(f"{class_label}: {class_prop}")
        class_props.append(class_prop)
    return max(class_props)


def safe_length(x):
    if not isinstance(x, str):
        return None
    return len(x.split(" "))


def do_model_fit(model, X_train, y_train, X_test, y_test, verbose=True, prediction_dir=None, validation_split=0.15):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=1-validation_split, random_state=42)
    model.fit(X_train, y_train)
    train_pred = model.predict_proba(X_train)
    val_pred = model.predict_proba(X_val)
    test_pred = model.predict_proba(X_test)
    test_acc, test_prec, test_recall, test_f1, test_auc = compute_metrics(y_test, test_pred)
    if verbose:
        print(f"Base Rate: ")
        print_base_rate(y_test, verbose=verbose)
        print(f"Total Test Accuracy: {test_acc}")
    threshold = 0.95
    perc_selected, accuracy, precision, recall, f1, auc = compute_threshold_metrics(y_test, test_pred, threshold)
    if verbose:
        print(f"With threshold {threshold}: Predicts on {round(perc_selected*100, 2)} % of samples (Test)")
        print(f"Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, AUC: {auc}")
    perc_selected, accuracy, val_acc, quartile, selected = compute_conformal_metrics(y_true_val=y_val, y_pred_proba_val=val_pred, y_true_test=y_test, y_pred_proba_test=test_pred, confidence=0.91)
    if verbose:
        if selected is None:
            print(f"Unable to find conformal quartile")
        else:
            print(f"Conformal Predicts on {round(perc_selected*100, 2)} % of samples (Test)")
            # within selected columns:
            print(f"Distribution of probe target in selected columns:")
            print_base_rate(y_test[selected], verbose=True)
            print(f"Test Accuracy: {accuracy}, Val Accuracy: {val_acc}, Quartile: {quartile}")
        

    if prediction_dir is not None:
        np.save(f"{prediction_dir}/train_pred.npy", train_pred)
        np.save(f"{prediction_dir}/test_pred.npy", test_pred)
        model.save(f"{prediction_dir}/")
    return train_pred, test_pred, test_acc


@click.command()
@click.option("--dataset", type=click.Choice(["mnist_math", "imagenette", "food101"]), required=True)
@click.option("--model", required=True, type=click.Choice(["llava-v1.6-mistral-7b-hf", "llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf", "instructblip-vicuna-7b", "instructblip-vicuna-13b"]), help="The model to evaluate")
@click.option('--layer', type=int, default=10)
@click.pass_obj
def main(parameters, dataset, model, layer, rerun):
    # Have an option to load all datasets. TODO
    #  find the results file in VLM_INVESTIGATION_699_RESULTS_DIR/dataset/model/results_visual_evaluated.csv
    results_dir = parameters["results_dir"]
    # TODO: Fix given new code
    X_train, X_test, y_train, y_test = get_xydfs(hidden, results_df)
    linear_model = Linear()
    do_model_fit(linear_model, X_train, y_train, X_test, y_test, verbose=True, prediction_dir=hidden_results_dir)
    

if __name__ == "__main__":
    main()
