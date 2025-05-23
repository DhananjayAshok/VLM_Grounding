import click
from utils.log_handling import log_error
from experiments.grounding_utils.common import update_row, HiddenStateTracking, save
from evaluation.grounding_evaluation import do_final_evaluation, log_final_evaluation
from inference.vlms import get_vlm
import os
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image



okvqa_input_prompt = "Answer the questions with a short response. Do not state the name of the object in the image. \nWhat are swords made of?\nAnswer: steel [STOP]\n What is the capital of France?\nAnswer: Paris [STOP]\n"
okvqa_cot_input_prompt = "First identify the object in the image, then answer the question. \nWhat sort of vehicle uses this item?\nAnswer: The item is a fire hydrant. It is used by a firetruck [STOP]\nWhat days might I most commonly go to this building?\nAnswer: The building is a church. You are most likely to go on sunday [STOP]\n"

def process_okvqa(parameters):
    """
    Take the OKVQA dataset as downloaded from extracted zip files and put in easy to iterate format
    """
    okvqa_dir = os.path.join(parameters["data_dir"], "raw", "okvqa")
    expected_files = ["OpenEnded_mscoco_val2014_questions.json", "val2014", "mscoco_val2014_annotations.json"]
    for file in expected_files:
        file_path = os.path.join(okvqa_dir, file)
        if not os.path.exists(file_path):
            log_error(parameters["logger"], f"Missing file {file_path}. Please download the OKVQA dataset and place it in {okvqa_dir}.")
            return
    # Load the OKVQA dataset
    questions = json.load(open(os.path.join(okvqa_dir, "OpenEnded_mscoco_val2014_questions.json"), "r"))
    annotations = json.load(open(os.path.join(okvqa_dir, "mscoco_val2014_annotations.json"), "r"))
    data = []
    q_types = annotations["question_types"]
    for idx in range(len(questions["questions"])):
        annotation = annotations["annotations"][idx]
        question_dict = questions["questions"][idx]
        if annotation["image_id"] != question_dict["image_id"]:
            log_error(parameters["logger"], f"Image ID mismatch for index {idx}. Please check the dataset.")
            return
        if annotation["question_id"] != question_dict["question_id"]:
            log_error(parameters["logger"], f"Question ID mismatch for index {idx}. Please check the dataset.")
            return
        question_type = q_types[annotation["question_type"]]
        question = question_dict["question"]
        answers = [ans["raw_answer"] for ans in annotation["answers"]]
        # get the mode of answers 
        mode = max(set(answers), key=answers.count)
        image_path = os.path.join(okvqa_dir, "val2014", f"COCO_val2014_{str(question_dict['image_id']).zfill(12)}.jpg")
        data.append({
            "question": question,
            "answer_str": mode,
            "image_path": image_path,
            "question_id": question_dict["question_id"],
            "image_id": question_dict["image_id"],
            "all_answers": answers,
            "question_type": question_type
        })
    data_df = pd.DataFrame(data)
    okvqa_save_dir = os.path.join(parameters["storage_dir"], "processed_datasets", "okvqa")
    if not os.path.exists(okvqa_save_dir):
        os.makedirs(okvqa_save_dir, exist_ok=True)
    data_df.to_csv(os.path.join(okvqa_save_dir, "data.csv"), index=False)
    parameters["logger"].info(f"Processed OKVQA dataset and saved to {os.path.join(parameters['storage_dir'], 'processed_datasets', 'okvqa', 'data.csv')}")
    return

def run_okvqa(parameters, vlm, cot=False):
    """
    Run the inference on the OKVQA dataset using the specified VLM.
    """
    #Must save the results to a csv file with a format that the hidden_state_predictor script can read. 
    okvqa_path = os.path.join(parameters["storage_dir"], "processed_datasets", "okvqa")
    data_path = os.path.join(okvqa_path, "data.csv")
    results_file = f"/okvqa/{vlm}/final_results.csv" if not cot else f"/okvqa/{vlm}/final_results_cot.csv"
    results_path = parameters["results_dir"] + results_file
    if os.path.exists(results_path):
        parameters["logger"].warning(f"Results file already exists at {results_path}. Please delete it first to regenerate.")
        return
    if not os.path.exists(os.path.dirname(results_path)):
        os.makedirs(os.path.dirname(results_path))
    data_df = pd.read_csv(data_path)
    if not cot:
        hidden_state_tracker = HiddenStateTracking("okvqa", vlm, "image_reference", parameters)
    else:
        hidden_state_tracker = None
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Running OKVQA Inference"):
        image_path = row["image_path"]
        question_part = "\n" + row["question"] + "\nAnswer: "
        if not cot:
            question = okvqa_input_prompt + question_part
        else:
            question = okvqa_cot_input_prompt + question_part
        image = Image.open(image_path)
        response = vlm(image, question, max_new_tokens=10 if not cot else 50)
        update_row(data_df, idx, "image_reference", response, completed=True, hidden_state_tracker=hidden_state_tracker, projection_tracker=None)
    save(data_df, results_path, hidden_state_tracker, None)
    parameters["logger"].info(f"OKVQA inference completed and saved to {results_path}")


@click.command()
@click.pass_obj
def setup_okvqa(parameters):
    """
    Set up the OKVQA dataset for inference.
    """
    process_okvqa(parameters)


@click.command()
@click.option("--model", default="llava-v1.6-vicuna-7b-hf",help="The VLM whose grounding ability is being tested", type=click.Choice(["llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf", "llava-v1.6-mistral-7b-hf", "instructblip-vicuna-7b", "instructblip-vicuna-13b"]))
@click.option("--cot", is_flag=True, default=False, help="Use Chain of Thought (CoT) prompting")
@click.pass_obj
def okvqa_inference(parameters, model, cot):
    """
    Run inference on the OKVQA dataset using the specified VLM.
    """
    vlm = get_vlm(model, hidden_state_tracking_mode=not cot, vocab_projection_mode=False)
    run_okvqa(parameters, vlm, cot)


@click.command()
@click.option("--model", default="llava-v1.6-vicuna-7b-hf",help="The VLM whose grounding ability is being tested", type=click.Choice(["llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf", "llava-v1.6-mistral-7b-hf", "instructblip-vicuna-7b", "instructblip-vicuna-13b"]))
@click.option("--cot", is_flag=True, default=False, help="Use Chain of Thought (CoT) prompting")
@click.pass_obj
def evaluate_okvqa(parameters, model, cot):
    """
    Run inference on the OKVQA dataset using the specified VLM.
    """
    results_file = f"/okvqa/{model}/final_results_cot.csv" if cot else f"/okvqa/{model}/final_results.csv"
    results_path = parameters["results_dir"] + results_file
    if not os.path.exists(results_path):
        log_error(parameters["logger"], f"Results file {results_path} does not exist. Please run the inference first.")
        return
    results_df = pd.read_csv(results_path)
    results_df = do_final_evaluation(results_df, parameters, okvqa=True)
    results_df.to_csv(results_path, index=False)
    parameters["logger"].info(f"OKVQA evaluation completed and saved to {results_path}")
    log_final_evaluation(results_df, parameters, okvqa=True)



