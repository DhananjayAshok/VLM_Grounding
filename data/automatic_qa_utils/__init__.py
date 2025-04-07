import click
from utils.log_handling import log_error
from inference.llms import get_llm
from data.automatic_qa_utils.qa_validation import deduplicate_qas, validate_qas, log_status_count
from data.automatic_qa_utils.qa_generation import generate_all_qas, generate_mcqas
from data import get_data_creator
import os
import json

def handle_question_generation(parameters, dataset_name, strong_llm):
    """
    Handle question generation for a dataset
    """
    data_creator = get_data_creator(dataset_name, parameters=parameters)
    class_labels = data_creator.load_validated_classes()
    storage_dir = parameters["storage_dir"]
    dataset_path = os.path.join(storage_dir, "processed_datasets", dataset_name)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    qas_path = os.path.join(dataset_path, "qas_generated.json")
    if os.path.exists(qas_path):
        parameters["logger"].warning(f"QA file already exists at {qas_path}. Please delete it first to regenerate.")
        qas = json.load(open(qas_path, "r"))
    else:
        if isinstance(strong_llm, str):
            llm = get_llm(strong_llm)
        else:
            llm = strong_llm
        qas = generate_all_qas(class_labels, llm, parameters=parameters)
        with open(qas_path, "w") as f:
            json.dump(qas, f)
        parameters["logger"].info(f"Generated {len(qas)} questions for {dataset_name} and saved to {qas_path}")

def handle_question_validation(parameters, dataset_name, strong_llm):
    qas_path = os.path.join(parameters["storage_dir"], "processed_datasets", dataset_name, "qas_generated.json")
    if not os.path.exists(qas_path):
        log_error(parameters["logger"], f"QA file not found at {qas_path}. Please generate questions first.")
    qa_validation_path = qas_path.replace("qas_generated.json", "qas_validated.json")
    if os.path.exists(qa_validation_path):
        parameters["logger"].warning(f"QA validation file already exists at {qa_validation_path}. Please delete it first to revalidate.")
        qas = json.load(open(qa_validation_path, "r"))
    else:
        qas = json.load(open(qas_path, "r"))
        if isinstance(strong_llm, str):
            llm = get_llm(strong_llm)
        else:
            llm = strong_llm
        validate_qas(qas, llm)
        with open(qa_validation_path, "w") as f:
            json.dump(qas, f)
        parameters["logger"].info(f"Validated questions for {dataset_name} and saved to {qa_validation_path}")
    log_status_count(qas, parameters["logger"])



def handle_question_deduplication(parameters, dataset_name, weak_llm):
    qas_path = os.path.join(parameters["storage_dir"], "processed_datasets", dataset_name, "qas_validated.json")
    if not os.path.exists(qas_path):
        log_error(parameters["logger"], f"QA file not found at {qas_path}. Please validate questions first.")
    deduped_qas_path = qas_path.replace("qas_validated.json", "qas_deduplicated.json")
    if os.path.exists(deduped_qas_path):
        parameters["logger"].warning(f"QA deduplication file already exists at {deduped_qas_path}. Please delete it first to rededuplicate.")
        qas = json.load(open(deduped_qas_path, "r"))
    else:
        qas = json.load(open(qas_path, "r"))
        if isinstance(weak_llm, str):
            llm = get_llm(weak_llm)
        else:
            llm = weak_llm
        deduplicate_qas(qas, llm)
        with open(deduped_qas_path, "w") as f:
            json.dump(qas, f)
        parameters["logger"].info(f"Deduplicated questions for {dataset_name} and saved to {deduped_qas_path}")
    log_status_count(qas, parameters["logger"])

def handle_mcq_question_generation(parameters, dataset_name, weak_llm):
    qas_path = os.path.join(parameters["storage_dir"], "processed_datasets", dataset_name, "qas_deduplicated.json")
    if not os.path.exists(qas_path):
        log_error(parameters["logger"], f"QA file not found at {qas_path}. Please deduplicate questions first.")
    mcq_qas_path = qas_path.replace("qas_deduplicated.json", "mcqas_deduplicated.json")
    if os.path.exists(mcq_qas_path):
        parameters["logger"].warning(f"MCQA file already exists at {mcq_qas_path}. Please delete it first to regenerate.")
        return
    qas = json.load(open(qas_path, "r"))
    if isinstance(weak_llm, str):
        llm = get_llm(weak_llm)
    else:
        llm = weak_llm
    mcqas = generate_mcqas(qas, llm, parameters=parameters)
    with open(mcq_qas_path, "w") as f:
        json.dump(mcqas, f)
    parameters["logger"].info(f"Generated {len(mcqas)} MCQAs for {dataset_name} and saved to {mcq_qas_path}")
    return





@click.command()
@click.option("--dataset_name", type=str, help="The name of the dataset to use")
@click.option("--strong_llm", type=str, help="The name of the strong LLM to use", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
@click.pass_obj
def generate_questions(parameters, dataset_name, strong_llm):
    """
    Generate questions for dataset
    """
    handle_question_generation(parameters, dataset_name, strong_llm)


@click.command()
@click.option("--dataset_name", type=str, help="The name of the dataset to use")
@click.option("--strong_llm", type=str, help="The name of the strong LLM to use", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
@click.pass_obj
def validate_questions(parameters, dataset_name, strong_llm):
    """
    Validate generated questions for dataset
    """
    handle_question_validation(parameters, dataset_name, strong_llm)



@click.command()
@click.option("--dataset_name", type=str, help="The name of the dataset to use")
@click.option("--weak_llm", type=str, help="The name of the weak LLM to use", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
@click.pass_obj
def deduplicate_questions(parameters, dataset_name, weak_llm):
    """
    Deduplicate and reorganize validated questions for dataset.
    """
    handle_question_deduplication(parameters, dataset_name, weak_llm)


@click.command()
@click.option("--dataset_name", type=str, help="The name of the dataset to use")
@click.option("--weak_llm", type=str, help="The name of the weak LLM to use", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
@click.pass_obj
def generate_mcq_questions(parameters, dataset_name, weak_llm):
    """
    Deduplicate validated questions for dataset.
    """
    handle_question_deduplication(parameters, dataset_name, weak_llm)

@click.command()
@click.option("--dataset_name", type=str, help="The name of the dataset to use")
@click.option("--llm", type=str, help="The name of the LLM to use", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
@click.pass_obj
def full_qa_pipeline(parameters, dataset_name, llm):
    """
    Generate, validate and deduplicate questions for dataset
    """
    llm = get_llm(llm)
    handle_question_generation(parameters, dataset_name, llm)
    handle_question_validation(parameters, dataset_name, llm)
    handle_question_deduplication(parameters, dataset_name, llm)
    handle_mcq_question_generation(parameters, dataset_name, llm)
    












