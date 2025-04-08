import click
from data.data_holder import get_data_creator
from utils.log_handling import log_error
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def get_annotation_file_path(annotator=None):
    # get the current file path
    current_file_path = os.path.abspath(__file__)
    # get the grandparent directory
    grandparent_dir = os.path.dirname(os.path.dirname(current_file_path))
    annotation_file_path = os.path.join(grandparent_dir, "annotation", "annotations.csv")
    if annotator is not None:
        annotation_file_path = annotation_file_path.replace("annotations.csv", f"annotations_{annotator}.csv")
    return annotation_file_path

def get_annotation_file(parameters, annotator=None):
    filepath = get_annotation_file_path(annotator=annotator)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        if annotator is None:
            log_error(parameters["logger"], f"Annotation file not found at {filepath}. Let me create it first (do not do it yourself).")
        else:
            df = get_annotation_file(parameters)
            df["annotator_" + annotator] = None
            return df
    


def create_annotation_file(parameters, datasets_to_use=["cifar100", "food101", "landmarks"]):
    filepath = get_annotation_file_path()
    if os.path.exists(filepath):
        log_error(parameters["logger"], f"Annotation file already exists at {filepath}. You must delete it to create a new one, but check for data loss.")
    data = []
    columns = ["dataset", "text", "question", "answer"]
    for dataset in datasets_to_use:
        sampled_questions = get_sampled_qas(dataset, parameters)
        for idx, question in sampled_questions.iterrows():
            data.append([dataset, question["text"], question["question"], question["answer"], None, None, None])
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filepath, index=False)



def get_sampled_qas(dataset_name, parameters, n_sample=100, assume_vlm="llava-v1.6-vicuna-7b-hf"):
    results_path = os.path.join(parameters["results_dir"] + f"/{dataset_name}/{assume_vlm}/full_information_results_evaluated.csv")
    if not os.path.exists(results_path):
        log_error(parameters["logger"], f"No results found for {dataset_name} and {assume_vlm}. Please run the full information grounding experiment script first.")
    results_df = pd.read_csv(results_path)
    passing_questions = results_df.loc[results_df["full_information_pass"] == True, "question_str"].reset_index(drop=True)
    sampled_questions = passing_questions.sample(n=n_sample, random_state=parameters["random_seed"])
    return sampled_questions

def display_row(row, parameters):
    logger = parameters["logger"]
    logger.info(f"Dataset: {row['dataset']}")
    logger.info(f"Text: {row['text']}")
    logger.info(f"Question: {row['question']}")
    logger.info(f"Answer: {row['answer']}")
    return
    

def get_input(parameters):
    verdict_valid = False
    while not verdict_valid:
        parameters["logger"].info("Please enter your verdict (0: Incorrect, 1: Correct)")
        verdict = input("Verdict: ")
        try:
            verdict = int(verdict)
            if verdict in [0, 1]:
                verdict_valid = True
            else:
                parameters["logger"].info("Invalid input. Please enter 0 or 1.")
        except ValueError:
            parameters["logger"].info("Invalid input. Please enter 0 or 1.")
    return bool(verdict)

def save_annotation_file(df, annotator):
    filename = get_annotation_file_path(annotator=annotator)
    df.to_csv(filename, index=False)
    return

def handle_annotation(parameters, annotator_id):
    df = get_annotation_file(parameters, annotator=annotator_id)
    annotation_column = f"annotator_{annotator_id}"
    nans = df[df[annotation_column].isna()]
    if len(nans) == 0:
        parameters["logger"].info(f"All rows have been annotated by {annotator_id}.")
        return
    first_nan_idx = nans.index[0]
    for i in tqdm(range(first_nan_idx, len(df))):
        row = df.loc[i]
        display_row(row, parameters)
        verdict = get_input(parameters)
        df.loc[i, annotation_column] = verdict
        save_annotation_file(df, annotator_id)
    return

    

@click.command()
@click.option("--dataset_names", type=str, multiple=True, default=["cifar100", "food101", "landmarks"], help="The names of the datasets to use")
@click.pass_obj
def create_annotation_file_command(parameters, dataset_names):
    """
    Create an annotation file using the given datasets
    """
    np.random.seed(parameters["random_seed"])
    create_annotation_file(parameters, dataset_names)


@click.command()
@click.option("--annotator_id", type=click.Choice(["d", "h", "a"]), help="Unique ID of the annotator")
@click.pass_obj
def do_annotation(parameters, annotator_id):
    """
    Annotate the dataset with the given annotator ID
    """
    handle_annotation(parameters, annotator_id)


