
import os
import pandas as pd
from utils.log_handling import log_error
import pickle


def get_starting_df(dataset, vlm, results_df_path, parameters, run_variant="identification"):
    if os.path.exists(results_df_path):
        results_df = pd.read_csv(results_df_path)
        return results_df
    else:
        prev_files = {"identification": None, "full_information": "identification", 
                     "image_reference": "full_information"}
        prev_run = "identification" if run_variant == "full_information" else "full_information"
        suffix = "_results_evaluated" if run_variant in ["full_information", "image_reference"] else "_results"
        if run_variant == "identification":
            results_df = dataset.data_df.copy()
            results_df[f"{run_variant}_complete"] = False
        else:
            results_path = parameters["results_dir"] + f"/{dataset}/{vlm}/{prev_files[run_variant]}{suffix}.csv"
            if os.path.exists(results_path):
                results_df = pd.read_csv(results_path)
                results_df[f"{run_variant}_complete"] = True
                pass_row_idx = results_df[results_df[f"{prev_run}_pass"] == True].index
                results_df.loc[pass_row_idx, f"{run_variant}_complete"] = False
            else:
                log_error(parameters["logger"], f"Tried looking for prerequisite file: {results_path} but it does not exist. Please run the prerequisite script first. Will be either generation + eval or just generation")
        return results_df


def handle_openai(dataset, vlm, results_df_path, parameters, variant="identification", previous_check=None):
    image_texts = {f"{variant}": []}
    read_variant = variant
    if "trivial" in variant:
        read_variant = variant.split("trivial_")[1]
    results_df = get_starting_df(dataset, vlm, results_df_path, parameters, run_variant=variant)
    if results_df[f"{variant}_complete"].all():
        parameters["logger"].warning(f"{variant} script already completed for {dataset} and {vlm}. Returning file found at {results_df_path} ...")
        return 
    names = [f"{vlm}_{dataset}_{key}" for key in image_texts.keys()]
    statuses = [vlm.get_batch_status(name) for name in names]
    completed = all([status == 1 for status in statuses])
    running = any([status == 0 for status in statuses])
    not_sent = all([status == None for status in statuses])
    if running:
        parameters["logger"].info(f"Batch job is still running (or might have failed, just check it). Try again later to parse results.")
        return
    elif completed == 1:
        parameters["logger"].info(f"Batch job is completed. Parsing results.")
        reading = True
    else:
        if not not_sent:
            log_error(parameters["logger"], "Status of batch jobs is weird. Check this.")
        parameters["logger"].info(f"Batch job is not sent yet. Sending batch job now.")
        reading = False        
    indexes = []
    for idx, row in results_df.iterrows():
        if previous_check is not None:
            check_bool = row[previous_check]
            if not check_bool:
                continue
        if not reading:
            data = dataset[idx]
            if "trivial" in variant:
                log_error(parameters["logger"], f"Trivial experiment script not implemented for OpenAI in common.py you should be calling on the one in trivial.py. Please run the image reference script first.")
            else:
                image = data["image"]
            question = data[f"{read_variant}_question"]
            image_texts[variant].append((image, question))
        indexes.append(idx)
    results = {}
    for key in image_texts:
        image_text_input = image_texts[key] if not reading else None
        results[key] = vlm(image_text_input, f"{vlm}_{dataset}_{key}", indexes)
    if not reading:
        parameters["logger"].info(f"Using OpenAI, sent batch job. Try again when batch is completed to parse results")
        return
    first_key = list(results.keys())[0]
    if results[first_key] is None:
        log_error(parameters["logger"], f"Results for {first_key} are None. This is a bug and shouldn't be happening.")
        return
    fails = []
    for idx in indexes:
        for key in results:
            idx_row = results[key].loc[results[key]["idx"] == idx]
            if len(idx_row) != 1:
                fails.append(idx)
                continue
            results_df.loc[idx, f"{key}_response"] = idx_row["response"].values[0]
    results_df[f"{variant}_complete"] = True
    results_df.to_csv(results_df_path, index=False)
    if len(fails) > 0:
        parameters["logger"].info(f"Failed to parse results for {len(fails)} indexes out of {len(results[first_key])}: This is a bug and shouldn't be happening.")
    return


class HiddenStateTracking:
    def __init__(self, dataset_name, vlm_name, run_variant, parameters):
        self.dataset_name = dataset_name
        self.vlm_name = vlm_name
        self.run_variant = run_variant
        self.parameters = parameters
        self.hidden_states = {}
        self.save_path = parameters["storage_dir"] + f"/hidden_states/{dataset_name}/{vlm_name}/{run_variant}/hidden_states.pkl"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)


    def load_checkpoint(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, "rb") as f:
                self.hidden_states = pickle.load(f)
        else:
            self.hidden_states = {}

    def save_checkpoint(self):
        with open(self.save_path, "wb") as f:
            pickle.dump(self.hidden_states, f)

    def add_hidden_state(self, idx, hidden_states):
        if idx not in self.hidden_states:
            self.hidden_states[idx] = hidden_states
        else:
            self.parameters["logger"].warning(f"Hidden state for index {idx} already exists. Overwriting.")
            self.hidden_states[idx] = hidden_states


class VocabProjectionTracking:
    def __init__(self, dataset_name, vlm_name, run_variant, parameters):
        self.dataset_name = dataset_name
        self.vlm_name = vlm_name
        self.run_variant = run_variant
        self.parameters = parameters
        self.kl_divergence = {}
        self.projection_prob = {}
        self.total_projection = {}
        self.save_path = parameters["storage_dir"] + f"/vocab_projections/{dataset_name}/{vlm_name}/{run_variant}/"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)


    def load_checkpoint(self):
        if os.path.exists(self.save_path+f"/kl_divergence.pkl"):
            with open(self.save_path+f"/kl_divergence.pkl", "rb") as f:
                self.kl_divergence = pickle.load(f)
        else:
            self.kl_divergence = {}
        if os.path.exists(self.save_path+f"/projection_prob.pkl"):
            with open(self.save_path+f"/projection_prob.pkl", "rb") as f:
                self.projection_prob = pickle.load(f)
        else:
            self.projection_prob = {}
        if os.path.exists(self.save_path+f"/total_projection.pkl"):
            with open(self.save_path+f"/total_projection.pkl", "rb") as f:
                self.total_projection = pickle.load(f)
        else:
            self.total_projection = {}
        if len(self.kl_divergence) != len(self.projection_prob) or len(self.kl_divergence) != len(self.total_projection):
            log_error(self.parameters["logger"], f"Mismatch between kl_divergence and projection_prob lengths. {len(self.kl_divergence)} vs {len(self.projection_prob)} vs {len(self.total_projection)}. This is a bug and shouldn't be happening.")
            raise ValueError("Mismatch between kl_divergence and projection_prob lengths.")

    def save_checkpoint(self):
        with open(self.save_path+f"/kl_divergence.pkl", "wb") as f:
            pickle.dump(self.kl_divergence, f)
        with open(self.save_path+f"/projection_prob.pkl", "wb") as f:
            pickle.dump(self.projection_prob, f)
        with open(self.save_path+f"/total_projection.pkl", "wb") as f:
            pickle.dump(self.total_projection, f)

    def add_projection(self, idx, kl_divergence, projection_prob, total_projection):
        if idx not in self.kl_divergence:
            self.kl_divergence[idx] = kl_divergence
            self.projection_prob[idx] = projection_prob
            self.total_projection[idx] = total_projection
        else:
            self.parameters["logger"].warning(f"Projection for index {idx} already exists. Overwriting.")
            self.kl_divergence[idx] = kl_divergence
            self.projection_prob[idx] = projection_prob
            self.total_projection[idx] = total_projection


def save(results_df, results_df_path, hidden_state_tracker=None, projection_tracker=None):
    if hidden_state_tracker is not None:
        hidden_state_tracker.save_checkpoint()
    if projection_tracker is not None:
        projection_tracker.save_checkpoint()
    results_df.to_csv(results_df_path, index=False)


def update_row(results_df, idx, item_name, response, completed=True, hidden_state_tracker=None, projection_tracker=None):
    results_df.loc[idx, f"{item_name}_response"] = response["text"]
    results_df.loc[idx, f"{item_name}_response_perplexity"] = response["perplexity"]
    if hidden_state_tracker is not None:
        hidden_state_tracker.add_hidden_state(idx, response["hidden_states"])
    if projection_tracker is not None:
        projection_tracker.add_projection(idx, response["kl_divergence"], response["projection_prob"], response["total_projection"])
    if completed:
        results_df.loc[idx, f"{item_name}_complete"] = True
    return