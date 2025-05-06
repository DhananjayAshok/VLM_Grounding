import click
from utils.log_handling import log_error
import pandas as pd
import numpy as np
from nltk.metrics import AnnotationTask
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
    datasets = base_df["dataset"].unique().tolist() + [None]
    for dataset in datasets:
        parameters["logger"].info(f"Processing dataset {dataset}")
        if dataset is not None:
            dataset_df = base_df[base_df["dataset"] == dataset].reset_index(drop=True)
        for metric in metrics:
            true_avg = 0
            coders = ["coder0", "coder1", "coder2"]
            items = dataset_df.index
            labels = {}
            for i in range(n_annotators):
                metric_str = f"{metric}_{i}"
                avg = dataset_df[metric_str].mean()
                parameters["logger"].info(f"\t\tAverage {metric} for annotator {i}: {avg}")
                labels[f"coder{i}"] = (dataset_df[metric_str].tolist())
                true_avg += avg
            true_avg /= n_annotators       
            task_data = []
            for coder in coders:
                for item_i, item in enumerate(items):
                    task_data.append((coder, item, labels[coder][item_i]))
            task = AnnotationTask(task_data)
            kappa = task.kappa()
            parameters["logger"].info(f"\tAverage {metric} for all annotators: {true_avg}")
            parameters["logger"].info(f"\tKappa for {metric}: {kappa}")
            three_way_agreement = (dataset_df[metric + "_0"] == dataset_df[metric + "_1"]) & (dataset_df[metric + "_0"] == dataset_df[metric + "_2"])
            parameters["logger"].info(f"\tThree way agreement for {metric}: {three_way_agreement.mean()}")
            #enumerate over pairs of annotators:
            for i in range(n_annotators):
                for j in range(i + 1, n_annotators):
                    coder_i = f"coder{i}"
                    coder_j = f"coder{j}"
                    agreement = (dataset_df[metric + f"_{i}"] == dataset_df[metric + f"_{j}"])
                    parameters["logger"].info(f"\tAgreement between {coder_i} and {coder_j}: {agreement.mean()}")
    

