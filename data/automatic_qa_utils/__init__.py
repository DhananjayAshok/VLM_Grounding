import click
from utils.log_handling import log_error
from inference.llms import get_llm_inference_class
from data.automatic_qa_utils.qa_validation import deduplicate_qas, validate_qas
from data.automatic_qa_utils.qa_generation import generate_all_qas
from data import get_data_creator
import os
import json

@click.command()
@click.option("--dataset_name", type=str, help="The name of the dataset(s) to use")
@click.option("--strong_llm", type=str, help="The name of the strong LLM to use", default="meta-llama/Meta-Llama-3.1-70B-Instruct")
@click.pass_obj
def generate_questions(parameters, dataset_name, strong_llm):
    """
    Generate questions for the given dataset using the specified LLMs.
    """
    data_creator = get_data_creator(dataset_name, parameters=parameters)
    class_labels = data_creator.load_validated_classes()
    llm = get_llm_inference_class(strong_llm)
    qas = generate_all_qas(class_labels, llm, parameters=parameters)
    storage_dir = parameters["storage_dir"]
    dataset_path = os.path.join(storage_dir, "processed_datasets", dataset_name)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    qas_path = os.path.join(dataset_path, "qas_generated.json")
    with open(qas_path, "w") as f:
        json.dump(qas, f)
    parameters["logger"].info(f"Generated {len(qas)} questions for {dataset_name} and saved to {qas_path}")
    return qas

@click.command()
@click.option("--dataset_name", type=str, help="The name of the dataset(s) to use")
@click.option("--strong_llm", type=str, help="The name of the strong LLM to use", default="meta-llama/Meta-Llama-3.1-70B-Instruct")
@click.pass_obj
def validate_questions(parameters, dataset_name, strong_llm):
    """
    Validate the generated questions for the given dataset using the specified LLMs.
    """
    llm = get_llm_inference_class(strong_llm)
    qas_path = os.path.join(parameters["storage_dir"], "processed_datasets", dataset_name, "qas_generated.json")
    if not os.path.exists(qas_path):
        log_error(parameters["logger"], f"QA file not found at {qas_path}. Please generate questions first.")
    qa_validation_path = qas_path.replace("qas_generated.json", "qas_validated.json")
    if os.path.exists(qa_validation_path):
        log_error(parameters["logger"], f"QA validation file already exists at {qa_validation_path}. Please delete it first.")
    qas = json.load(open(qas_path, "r"))
    validate_qas(qas, llm, parameters=parameters)
    with open(qa_validation_path, "w") as f:
        json.dump(qas, f)


@click.command()
@click.option("--dataset_name", type=str, help="The name of the dataset(s) to use")
@click.option("--weak_llm", type=str, help="The name of the weak LLM to use", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
@click.pass_obj
def deduplicate_questions(parameters, dataset_name, weak_llm):
    """
    Deduplicate the validated questions for the given dataset using the specified LLMs.
    """
    llm = get_llm_inference_class(weak_llm)
    qas_path = os.path.join(parameters["storage_dir"], "processed_datasets", dataset_name, "qas_validated.json")
    if not os.path.exists(qas_path):
        log_error(parameters["logger"], f"QA file not found at {qas_path}. Please validate questions first.")
    deduped_qas_path = qas_path.replace("qas_validated.json", "qas_deduplicated.json")
    if os.path.exists(deduped_qas_path):
        log_error(parameters["logger"], f"Deduplicated QA file already exists at {deduped_qas_path}. Please delete it first.")
    qas = json.load(open(qas_path, "r"))
    deduplicate_qas(qas, llm, parameters=parameters)
    with open(deduped_qas_path, "w") as f:
        json.dump(qas, f)









