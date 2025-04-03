from experiments.grounding_utilities import do_full_information, do_image_reference, do_identification, do_trivial
from utils.log_handling import log_error
import os
import click
from evaluation.metrics import file_compute_metric_str, df_compute_metric_fn
from data import get_dataset
from inference.vlms import get_vlm


def do_checked_evaluation(vlm, filename, metric_str, candidate_column, reference_column, output_column, parameters):
    if not os.path.exists(filename):
        if "gpt" not in str(vlm):
            log_error(parameters["logger"], f"Model {vlm} has not been run yet. Please run the model first.")
        else:
            parameters["logger"].info(f"OpenAI model is either still running, just sent or there is a bug and so cannot evaluate yet. Run this script again once the batch script is done.")
            return
    file_compute_metric_str(metric_str, filename, candidate_column, reference_column, output_column=output_column, parameters=parameters)
    return     


click.command()
@click.option("--dataset_name", help="The name of the dataset(s) to use", default="mnist", options=["mnist", "cifar", "imagenet", "food", "landmarks"])
@click.option("--model", help="The VLM whose grounding ability is being tested", default="llava-v1.6-vicuna-13b-hf")
@click.option("--stage", help="The stage of the grounding process", default="full_information", options=["identification", "full_information", "image_reference", "trivial", "evaluation", "all",])
@click.option("--checkpoint_every", type=float, default=0.1, help="Checkpoint every x percent of the dataset")
@click.option("--variant", type=click.Choice(["default" ,"hidden_state", "vocab_projection"], help="The variant of the forward pass, controlling what information is stord.")
@click.pass_obj
def grounding_experiment(parameters, dataset_name, model, stage, checkpoint_every, variant):
    dataset = get_dataset(dataset_name)
    vlm = get_vlm(model)
    if "gpt" in str(vlm) and stage == "all":
        log_error(parameters["logger"], "OpenAI models do not support the 'all' stage. Please specify a specific stage.")
    if stage in ["identification", "all"]:
        filename = do_identification(dataset_name, vlm, parameters)
        do_checked_evaluation(vlm, filename, "two_way_inclusion", "identification_response", "class_name", "identification_pass", parameters)
        
    if stage in ["full_information", "all"]:
        filename = do_full_information(dataset_name, vlm, parameters)
        do_checked_evaluation(vlm, filename, "two_way_inclusion", "full_information_response", "full_information_question", "full_information_pass", parameters)

    if stage in ["image_reference", "all"]:
        filename = do_image_reference(dataset_name, vlm, parameters)


    if stage in ["trivial", "all"]:
        filename = do_trivial(dataset_name, vlm, parameters)

    if stage in ["evaluation", "all"]:
        filename = parameters["results_dir"] + f"/{dataset}/{vlm}/trivial_results.csv"

