import click
from utils.log_handling import log_error
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def get(df, i):
    row = df.loc[i]
    return row['class_name'], row['question_str'], row['answer_str']

@click.command()
@click.option("--model", default="llava-v1.6-vicuna-13b-hf", help="The VLM whose filtering we'll use", type=click.Choice(["llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf", "llava-v1.6-mistral-7b-hf", "instructblip-vicuna-7b", "instructblip-vicuna-13b"]))
@click.option("--metric", default="two_way_inclusion", help="The metric to use for filtering", type=click.Choice(["two_way_inclusion", "exact_match", "inclusion"]))
@click.pass_obj
def split_annotation_files(parameters, model, metric):
    datasets = ["cifar100", "food101", "landmarks"]
    n_points_per_dataset = 100 // len(datasets)        
    columns = ["dataset", "entity", "question", "answer"]
    data = []
    for dataset in datasets:
        results_path = os.path.join(parameters["results_dir"], dataset, model, "final_results.csv")
        df = pd.read_csv(results_path)
        df = df[df[f"{metric}_trivial_mode_image_reference_response"] == False].reset_index() # this should also only get the ones where full information passes
        # filter so that only unique question_ids remain
        df = df.drop_duplicates(subset=["question_id"], ignore_index=True)
        samples = df.sample(n_points_per_dataset, random_state=42).reset_index()
        for i in range(len(samples)):
            class_name, question_str, answer_str = get(samples, i)
            data.append([dataset, class_name, question_str, answer_str])
    df = pd.DataFrame(data, columns=columns)
    df["question_relevant"] = None
    df["answer_correct"] = None
    if not os.path.exists("annotation/"):
        os.makedirs("annotation/")
    df.to_csv("annotation/annotation_base.csv", index=False)

@click.command()
@click.pass_obj
def process_annotation_results(parameters):
    annotations_path = "annotation/annotation_"
    base_df = pd.read_csv(f"{annotations_path}0.csv")
    second_df = pd.read_csv(f"{annotations_path}1.csv")
    third_df = pd.read_csv(f"{annotations_path}2.csv")
    base_df["question_relevant_0"] = base_df["question_relevant"]
    base_df["answer_correct_0"] = base_df["answer_correct"]
    base_df["question_relevant_1"] = second_df["question_relevant"]
    base_df["answer_correct_1"] = second_df["answer_correct"]
    base_df["question_relevant_2"] = third_df["question_relevant"]
    base_df["answer_correct_2"] = third_df["answer_correct"]
    metrics = ["question_relevant", "answer_correct"]
    n_annotators = 3
    for metric in metrics:
        for i in range(n_annotators):
            avg = base_df[f"{metric}_{i}"].mean()
    

    

