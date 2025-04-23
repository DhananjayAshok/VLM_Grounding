
import os
import pandas as pd
from utils.log_handling import log_error
import pickle


def get_starting_df(dataset, vlm, results_df_path, parameters, run_variant="identification"):
    if os.path.exists(results_df_path):
        results_df = pd.read_csv(results_df_path)
        return results_df
    else:
        prev_runs = {"identification": None, "full_information": "identification_results_evaluated", 
                     "image_reference": "full_information_results_evaluated", "trivial_full_information": "image_reference_results",
                     "trivial_image_reference": "trivial_full_information_results"}
        if run_variant == "identification":
            results_df = dataset.data_df.copy()
            results_df[f"{run_variant}_complete"] = False
        else:
            results_path = parameters["results_dir"] + f"/{dataset}/{vlm}/{prev_runs[run_variant]}.csv"
            if os.path.exists(results_path):
                results_df = pd.read_csv(results_path)
                results_df[f"{run_variant}_complete"] = True
                pass_row_idx = results_df[results_df[f"{prev_runs[run_variant]}_pass"] == True].index
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
    indexes = []
    for idx, row in results_df.iterrows():
        if previous_check is not None:
            check_bool = row[previous_check]
            if not check_bool:
                continue
        data = dataset[idx]
        if "trivial" in variant:
            image = None
        else:
            image = data["image"]
        question = data[f"{read_variant}_question"]
        image_texts[variant].append((image, question))
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
    results_df[f"{variant}_complete"] = True
    results_df.to_csv(results_df_path, index=False)
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