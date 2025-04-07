
import os
import pandas as pd
from utils.log_handling import log_error
import pickle


def get_starting_df(dataset, vlm, results_df_path, parameters, run_variant="identification"):
    if os.path.exists(results_df_path):
        results_df = pd.read_csv(results_df_path)
        return results_df
    else:
        prev_runs = {"identification": None, "full_information": "identification", 
                     "image_reference": "full_information"}
        if run_variant == "identification":
            results_df = dataset.data_df.copy()
            results_df[f"{run_variant}_complete"] = False
        else:
            results_path = parameters["results_dir"] + f"/{dataset}/{vlm}/{prev_runs[run_variant]}_results_evaluated.csv"
            un_evaluated_results_path = parameters["results_dir"] + f"/{dataset}/{vlm}/{prev_runs[run_variant]}_results.csv"
            if os.path.exists(results_path):
                results_df = pd.read_csv(results_path)
                results_df[f"{run_variant}_complete"] = True
                pass_row_idx = results_df[results_df[f"{prev_runs[run_variant]}_pass"] == True].index
                results_df.loc[pass_row_idx, f"{run_variant}_complete"] = False
            elif os.path.exists(un_evaluated_results_path):
                log_error(parameters["logger"], f"Un-evaluated identification results found for {dataset} and {vlm}. Please evaluate them first.")
            else:
                log_error(parameters["logger"], f"No identification results found for {dataset} and {vlm}. Please run the identification script first.")
        return results_df


def handle_openai(dataset, vlm, results_df_path, parameters, variant="identification", previous_check=None):
    image_texts = {f"{variant}": []}
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
        image = data["image"]
        question = data[f"{variant}_question"]
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
    results_df.loc[f"{variant}_complete"] = True
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
        if len(self.kl_divergence) != len(self.projection_prob):
            log_error(self.parameters["logger"], f"Mismatch between kl_divergence and projection_prob lengths. {len(self.kl_divergence)} vs {len(self.projection_prob)}. This is a bug and shouldn't be happening.")
            raise ValueError("Mismatch between kl_divergence and projection_prob lengths.")

    def save_checkpoint(self):
        with open(self.save_path+f"/kl_divergence.pkl", "wb") as f:
            pickle.dump(self.kl_divergence, f)
        with open(self.save_path+f"/projection_prob.pkl", "wb") as f:
            pickle.dump(self.projection_prob, f)

    def add_projection(self, idx, kl_divergence, projection_prob):
        if idx not in self.kl_divergence:
            self.kl_divergence[idx] = kl_divergence
            self.projection_prob[idx] = projection_prob
        else:
            self.parameters["logger"].warning(f"Projection for index {idx} already exists. Overwriting.")
            self.kl_divergence[idx] = kl_divergence
            self.projection_prob[idx] = projection_prob


def save(results_df, results_df_path, hidden_state_tracker=None, projection_tracker=None):
    if hidden_state_tracker is not None:
        hidden_state_tracker.save_checkpoint()
    elif projection_tracker is not None:
        projection_tracker.save_checkpoint()
    results_df.to_csv(results_df_path, index=False)


def update_row(results_df, idx, item_name, response, completed=True, hidden_state_tracker=None, projection_tracker=None):
    results_df.loc[idx, f"{item_name}_response"] = response["text"]
    results_df.loc[idx, f"{item_name}_response_perplexity"] = response["perplexity"]
    if hidden_state_tracker is not None:
        hidden_state_tracker.add_hidden_state(idx, response["hidden_states"])
    elif projection_tracker is not None:
        projection_tracker.add_projection(idx, response["kl_divergence"], response["projection_prob"])
    if completed:
        results_df.loc[idx, f"{item_name}_complete"] = True
    return