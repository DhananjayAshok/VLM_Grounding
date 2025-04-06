import os
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error
import pandas as pd
import pickle
from tqdm import tqdm

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


def get_starting_df(dataset, results_df_path):
    if os.path.exists(results_df_path):
        # restore checkpoint and start from there
        results_df = pd.read_csv(results_df_path)
    else:
        results_df = dataset.data_df.copy()
        results_df["identification_complete"] = False
    return results_df


def handle_openai(dataset, vlm, results_df_path, parameters, variant="identification", previous_check=None):
    image_texts = {f"{variant}": []}
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


def do_identification(dataset, vlm, variant="default", parameters=None, checkpoint_every=0.1): # must consider OpenAI as well
    if parameters is None:
        parameters = load_parameters()
    results_df_path = parameters["results_dir"] + f"/{dataset}/{vlm}/identification_results.csv"
    if not os.path.exists(os.path.dirname(results_df_path)):
        os.makedirs(os.path.dirname(results_df_path), exist_ok=True)
    if "gpt" in str(vlm): # handle openAI differently
        handle_openai(dataset, vlm, results_df_path, parameters, variant="identification")
    else:
        results_df = get_starting_df(dataset, results_df_path)
        if results_df["identification_complete"].all():
            parameters["logger"].warning(f"Identification script already completed for {dataset} and {vlm}. Returning file found at {results_df_path} ...")
            return results_df_path
        if checkpoint_every > 1 or checkpoint_every < 0:
            log_error(parameters["logger"], f"Invalid checkpoint_every value: {checkpoint_every}. Must be between 0 and 1.")

        hidden_state_tracker = None
        projection_tracker = None
        if variant == "hidden_state":
            hidden_state_tracker = HiddenStateTracking(dataset, vlm, "identification", parameters)
            hidden_state_tracker.load_checkpoint()
        elif variant == "vocab_projection":
            projection_tracker = VocabProjectionTracking(dataset, vlm, "identification", parameters)
            projection_tracker.load_checkpoint()
            

        checkpoint_every = int(checkpoint_every * len(results_df))
        for idx, row in tqdm(results_df.iterrows(), total=len(results_df), desc=f"Identification for {dataset} on {vlm}"):
            if not row["identification_complete"]:
                data = dataset[idx]
                image = data["image"]
                identification_question = data["identification_question"]
                response = vlm(image, identification_question)
                update_row(results_df, idx, "identification", response, hidden_state_tracker=hidden_state_tracker, projection_tracker=projection_tracker)
                if idx % checkpoint_every == 0:  # This is okay because its sequential so it won't skip saving once it restarts
                    save(results_df, results_df_path, hidden_state_tracker, projection_tracker)
        save(results_df, results_df_path, hidden_state_tracker, projection_tracker)

    return results_df_path
