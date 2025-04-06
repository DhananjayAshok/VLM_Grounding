import os
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error
import pandas as pd
from tqdm import tqdm
from PIL import Image
import random
from experiments.grounding_utils.identification import HiddenStateTracking, VocabProjectionTracking, update_row

def create_black_image(width, height):
    return Image.new("RGB", (width, height), "black")

def create_white_image(width, height):
    return Image.new("RGB", (width, height), "white")

def create_noise_image(width, height):
    image = Image.new("RGB", (width, height))
    pixels = image.load() # Pixel access object
    for i in range(width):
        for j in range(height):
            pixels[i, j] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return image

def create_trivial_images(width, height):
    return create_black_image(width, height), create_white_image(width, height), create_noise_image(width, height)

def get_starting_df(dataset, vlm, results_df_path, parameters):
    if os.path.exists(results_df_path):
        # restore checkpoint and start from there
        results_df = pd.read_csv(results_df_path)
    else:
        results_path = parameters["results_dir"] + f"/{dataset}/{vlm}/image_reference_results.csv"
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
            results_df["trivial_complete"] = False
            pass_row_idx = results_df[results_df["full_information_pass"]].index
            results_df.loc[pass_row_idx, "trivial_complete"] = True
        else:
            log_error(parameters["logger"], f"No Image Reference results found for {dataset} and {vlm}. Please run the image reference script first.")
    return results_df

def handle_openai(dataset, vlm, results_df_path, parameters):
    previous_check = "full_information_pass"
    variant = "trivial"
    image_texts = {}
    for trivial_kind in ["black", "white", "noise"]:
        image_texts[f"trivial_{trivial_kind}_full_information"] = []
        image_texts[f"trivial_{trivial_kind}_image_reference"] = []
    results_df = get_starting_df(dataset, results_df_path)
    if results_df[f"{variant}_complete"].any() and not results_df[f"{variant}_complete"].all():
        log_error(parameters["logger"], f"Found partial {variant} completes in {results_df_path}. This is a bug and shouldn't be happening.")
    if results_df[f"{variant}_complete"].all():
        return results_df    
    indexes = []
    for idx, row in results_df.iterrows():
        if previous_check is not None:
            check_bool = row[previous_check]
            if not check_bool:
                continue
        data = dataset[idx]
        image = data["image"]
        black, white, noise = create_trivial_images(image.width, image.height)
        for trivial_image, trivial_name in zip([black, white, noise], ["black", "white", "noise"]):
            image_texts[f"trivial_{trivial_name}_full_information"].append((trivial_image, data["full_information_question"]))
            image_texts[f"trivial_{trivial_name}_image_reference"].append((trivial_image, data["image_reference_question"]))
        indexes.append(idx)
    results = {}
    for key in image_texts:
        results[key] = vlm(image_texts[key], f"{vlm}_{dataset}_{key}", indexes)
    first_key = list(results.keys())[0]
    if results[first_key] is None:
        parameters["logger"].info(f"Using OpenAI, sent batch job. Try again when batch is completed to parse results")
        return
    for idx in indexes:
        for key in results:
            idx_row = results[key].loc[results[key]["idx"] == idx]
            if len(idx_row) != 1:
                log_error(parameters["logger"], f"Error in OpenAI results. Found {len(idx_row)} idx rows for idx {idx} in {key} results.")
            results_df.loc[idx, f"{key}_response"] = idx_row["response"].values[0]
    results_df.loc[f"{variant}_complete"] = True
    results_df.to_csv(results_df_path, index=False)

def save(results_df, results_df_path, full_hidden_state_trackers, full_projection_trackers, image_reference_hidden_state_trackers, image_reference_projection_trackers):
    results_df.to_csv(results_df_path, index=False)
    for trivial in ["black", "white", "noise"]:
            if full_hidden_state_trackers[trivial] is not None:
                full_hidden_state_trackers[trivial].save_checkpoint()
                image_reference_hidden_state_trackers[trivial].save_checkpoint()
            if full_projection_trackers[trivial] is not None:
                full_projection_trackers[trivial].save_checkpoint()
                image_reference_projection_trackers[trivial].save_checkpoint()


def do_trivial(dataset, vlm, variant="default",  parameters=None, checkpoint_every=0.1): # must consider OpenAI as well
    if parameters is None:
        parameters = load_parameters()
    random.seed(parameters["random_seed"])
    results_df_path = parameters["results_dir"] + f"/{dataset}/{vlm}/trivial_results.csv"

    if "gpt" in str(vlm): # handle openAI differently
        handle_openai(dataset, vlm, results_df_path, parameters)
    else:
        results_df = get_starting_df(dataset, vlm, results_df_path, parameters)
        if results_df["trivial_complete"].all():
            parameters["logger"].warning(f"Image reference script already completed for {dataset} and {vlm}. Returning file found at {results_df_path} ...")
            return results_df_path
        if checkpoint_every > 1 or checkpoint_every < 0:
            log_error(parameters["logger"], f"Invalid checkpoint_every value: {checkpoint_every}. Must be between 0 and 1.")


        full_hidden_state_trackers = {}
        full_projection_trackers = {}
        image_reference_hidden_state_trackers = {}
        image_reference_projection_trackers = {}
        for trivial in ["black", "white", "noise"]:
            if variant == "hidden_state":
                name = f"trivial_{trivial}_full_information"
                full_hidden_state_trackers[trivial] = HiddenStateTracking(dataset, vlm, name, parameters)
                full_hidden_state_trackers[trivial].load_checkpoint()
                name = f"trivial_{trivial}_image_reference"
                image_reference_hidden_state_trackers[trivial] = HiddenStateTracking(dataset, vlm, name, parameters)
                image_reference_hidden_state_trackers[trivial].load_checkpoint()
            elif variant == "vocab_projection":
                name = f"trivial_{trivial}_full_information"
                full_projection_trackers[trivial] = VocabProjectionTracking(dataset, vlm, name, parameters)
                full_projection_trackers[trivial].load_checkpoint()
                name = f"trivial_{trivial}_image_reference"
                image_reference_projection_trackers[trivial] = VocabProjectionTracking(dataset, vlm, name, parameters)
                image_reference_projection_trackers[trivial].load_checkpoint()
        else:
            full_hidden_state_trackers[trivial] = None
            full_projection_trackers[trivial] = None
            image_reference_hidden_state_trackers[trivial] = None
            image_reference_projection_trackers[trivial] = None

            

        checkpoint_every = int(checkpoint_every * len(results_df))
        for idx, row in tqdm(results_df.iterrows(), total=len(results_df), desc=f"Trivial for {dataset} on {vlm}"):
            if not row["trivial_complete"]:
                data = dataset[idx]
                image = data["image"]
                black, white, noise = create_trivial_images(image.width, image.height)
                full_information_question = data["full_information_question"]
                image_reference_question = data["image_reference_question"]
                for trivial_image, trivial_name in zip([black, white, noise], ["black", "white", "noise"]):
                    response = vlm(trivial_image, full_information_question)
                    update_row(results_df, idx, f"trivial_{trivial_name}_full_information", response, hidden_state_tracker=full_hidden_state_trackers[trivial_name], projection_tracker=full_projection_trackers[trivial_name], completed=False)
                    response = vlm(trivial_image, image_reference_question)
                    update_row(results_df, idx, f"trivial_{trivial_name}_image_reference", response, hidden_state_tracker=image_reference_hidden_state_trackers[trivial_name], projection_tracker=image_reference_projection_trackers[trivial_name], completed=False)
                results_df.loc[idx, "trivial_complete"] = True
                if idx % checkpoint_every == 0:  # This is okay because its sequential so it won't skip saving once it restarts
                    save(results_df, results_df_path, full_hidden_state_trackers, full_projection_trackers, image_reference_hidden_state_trackers, image_reference_projection_trackers)
        save(results_df, results_df_path, full_hidden_state_trackers, full_projection_trackers, image_reference_hidden_state_trackers, image_reference_projection_trackers)
    return results_df_path