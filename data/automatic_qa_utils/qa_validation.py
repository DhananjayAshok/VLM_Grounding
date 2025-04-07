from inference.llms import get_llm
from tqdm import tqdm
from utils.parameter_handling import load_parameters
from evaluation.metrics import two_way_inclusion
import numpy as np


def reject_simple_failures(qas):
    for qa in qas:
        question = qa["question"]
        answer = qa["answer"]
        entity_name = qa["entity_name"]
        if question.strip() == "" or answer.strip() == "":
            qa["status"] = "rejected_empty_question_answer"
            break
        if len(answer.split(" ")) > 7:
            qa["status"] = "rejected_answer_too_long"
            break
        if entity_name.lower() in answer.lower():
            qa["status"] = "rejected_answer_contains_entity"
            break
        if entity_name.lower() not in question.lower():
            qa["status"] = "rejected_question_does_not_contain_entity"
            break
        qa["status"] = "basic_accepted"

def reject_nonunique_answers(qas, llm):
    for qa in tqdm(qas, desc="Measuring uniqueness of answer to question"):
        question = qa["question"]
        text = qa["text"]
        if "rejected" not in qa["status"]:
            unique_answer = llm.perform_uniqueness_validation(question, text)
            if unique_answer is None:
                qa["status"] = "rejected_answer_not_unique"
            elif unique_answer:
                qa["status"] = "uniqueness_accepted"
            else:
                qa["status"] = "rejected_answer_not_unique"

def deduplicate_qas_inclusion(qas):
    for i, qa in enumerate(qas):
        if "rejected" in qa["status"]:
            continue
        if any(two_way_inclusion(qa["question"], deduped_qa["question"]) for deduped_qa in qas[:i]):
            qa["status"] = "rejected_duplicate_inclusion"
            continue
    
def deduplicate_qas_llm(qas, llm):
    for i, qa in tqdm(enumerate(qas), desc="Deduplicating QAs", total=len(qas)):
        if "rejected" in qa["status"]:
            continue
        for j, deduped_qa in tqdm(enumerate(qas[:i]), total=len(qas[:i]), desc="Deduplicating QAs", leave=False):
            if "rejected" in deduped_qa["status"]:
                continue
            if i == j:
                continue
            if qa["entity_name"] != deduped_qa["entity_name"]:
                continue
            if two_way_inclusion(qa["answer"], deduped_qa["answer"]): # then check whether it is a duplicate. Otherwise it is not
                if llm.perform_question_duplicate_evaluation(qa["question"], qa["answer"], deduped_qa["question"], deduped_qa["answer"]):
                    qa["status"] = "rejected_duplicate_llm"
                    break

def verify_qa(qas, llm, mcq=False, random_seed=42):
    if mcq:
        np.random.seed(random_seed)
    for qa in tqdm(qas, desc="Verifying Question Answerability"):
        if "rejected" in qa["status"]:
            continue
        if not mcq:
            output = llm.perform_question_answering(qa["question"])
        else:
            options = qa["options"]
            np.random.shuffle(options)
            option_str = ""
            for i, option in enumerate(options):
                option_str = option_str + f"Option {i + 1}: {option}"
                if i < len(options) - 1:
                    option_str = option_str + "\n"
            question = qa["question"] + "\n" + option_str
            output = llm.perform_question_answering(question)
        answer = qa["answer"]
        if two_way_inclusion(output, answer):
            qa["status"] = "accepted"
        else:
            qa["status"] = "rejected_answer_incorrect"
        qa["llm_answer"] = output
    return 


def reorganize_qas(qas):
    new_qas = {}
    for qa in qas:
        if qa["entity_name"] not in new_qas:
            new_qas[qa["entity_name"]] = [qa]
        else:
            new_qas[qa["entity_name"]].append(qa)
    return new_qas

def deduplicate_qas(qas, llm):
    deduplicate_qas_inclusion(qas)
    deduplicate_qas_llm(qas, llm)
    reorganized_qas = reorganize_qas(qas)
    return reorganized_qas


def validate_qas(qas, llm, mcq=False, random_seed=42):
    reject_simple_failures(qas)
    reject_nonunique_answers(qas, llm)
    verify_qa(qas, llm, mcq=mcq, random_seed=random_seed)
    return qas

def log_status_count(qas, parameters=None):
    if parameters is None:
        parameters = load_parameters()
    status_count = {}
    for qa in qas:
        status_count[qa["status"]] = status_count.get(qa["status"], 0) + 1
    parameters["logger"].info(f"QA status counts: \n\t{status_count}")
    n_qas = len(qas)
    for status in status_count:
        status_count[status] = 100*(status_count[status] / n_qas)
    parameters["logger"].info(f"QA status counts (normalized): \n\t{status_count}")