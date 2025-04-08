from experiments.grounding_utils import do_full_information, do_image_reference, do_identification, do_trivial
from evaluation.grounding_evaluation import do_final_evaluation, log_final_evaluation
from utils.log_handling import log_error
import os
import click
import pandas as pd
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


@click.command()
@click.option("--dataset_name", help="The name of the dataset(s) to use", default="mnist")
@click.option("--model", help="The VLM whose grounding ability is being tested", type=click.Choice(["llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf", "llava-v1.6-mistral-7b-hf", "instructblip-vicuna-7b", "instructblip-vicuna-13b", "gpt-4o-mini", "gpt-4o"]), required=True)
@click.option("--stage", help="The stage of the grounding process", default="all", type=click.Choice(["identification", "full_information", "image_reference", "trivial", "evaluation", "all",]))
@click.option("--checkpoint_every", type=float, default=0.1, help="Checkpoint every x percent of the dataset")
@click.option("--variant", type=click.Choice(["default" ,"hidden_state", "vocab_projection", "hidden_state_vocab_projection"]), help="The variant of the forward pass, controlling what information is stored.", default="default")
@click.pass_obj
def grounding_experiment(parameters, dataset_name, model, stage, checkpoint_every, variant):
    """
    Run primary experiment for dataset
    """
    dataset = get_dataset(dataset_name, parameters)
    if stage != "evaluation":
        if variant == "default":
            vlm = get_vlm(model)
        elif variant == "hidden_state":
            vlm = get_vlm(model, hidden_state_tracking_mode=True)
        elif variant == "vocab_projection":
            vlm = get_vlm(model, vocab_projection_mode=True)
        elif variant == "hidden_state_vocab_projection":
            vlm = get_vlm(model, hidden_state_tracking_mode=True, vocab_projection_mode=True)
        else:
            log_error(parameters["logger"], f"Invalid variant: {variant}. Must be one of ['default', 'hidden_state', 'vocab_projection']")
    else:
        vlm = model # just need the string
    if "gpt" in str(vlm) and stage == "all":
        log_error(parameters["logger"], "OpenAI models do not support the 'all' stage. Please specify a specific stage.")
    if stage in ["identification", "all"]:
        filename = do_identification(dataset, vlm, variant, parameters, checkpoint_every=checkpoint_every)
        do_checked_evaluation(vlm, filename, "two_way_inclusion", "identification_response", "class_name", "identification_pass", parameters)
        
    if stage in ["full_information", "all"]:
        filename = do_full_information(dataset, vlm, variant, parameters, checkpoint_every=checkpoint_every)
        do_checked_evaluation(vlm, filename, "two_way_inclusion", "full_information_response", "answer_str", "full_information_pass", parameters)

    if stage in ["image_reference", "all"]:
        filename = do_image_reference(dataset, vlm, variant, parameters, checkpoint_every=checkpoint_every)


    if stage in ["trivial", "all"]:
        filename = do_trivial(dataset, vlm, variant, parameters, checkpoint_every=checkpoint_every)

    if stage in ["evaluation", "all"]:
        filename = parameters["results_dir"] + f"/{dataset}/{vlm}/trivial_results.csv"
        df = pd.read_csv(filename)
        df = do_final_evaluation(df, parameters)
        filename = filename.replace("trivial_results.csv", "final_results.csv")
        df.to_csv(filename, index=False)
        log_final_evaluation(df, parameters)


