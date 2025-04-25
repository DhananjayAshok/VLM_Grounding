import os
from experiments.grounding_utils.common import HiddenStateTracking, VocabProjectionTracking, save
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error
import pandas as pd
from tqdm import tqdm
from experiments.grounding_utils.common import update_row, handle_openai, get_starting_df


def do_full_information(dataset, vlm, variant="default",  parameters=None, checkpoint_every=0.1): # must consider OpenAI as well
    if parameters is None:
        parameters = load_parameters()
    results_df_path = parameters["results_dir"] + f"/{dataset}/{vlm}/full_information_results.csv"

    if "gpt" in str(vlm): # handle openAI differently
        handle_openai(dataset, vlm, results_df_path, parameters, variant="full_information", previous_check="identification_pass")
    else:
        results_df = get_starting_df(dataset, vlm, results_df_path, parameters, run_variant="full_information")
        if results_df["full_information_complete"].all():
            parameters["logger"].warning(f"Full information script already completed for {dataset} and {vlm}. Returning file found at {results_df_path} ...")
            return results_df_path
        if checkpoint_every > 1 or checkpoint_every < 0:
            log_error(parameters["logger"], f"Invalid checkpoint_every value: {checkpoint_every}. Must be between 0 and 1.")

        hidden_state_tracker = None
        projection_tracker = None
        if "hidden_state" in variant:
            hidden_state_tracker = HiddenStateTracking(dataset, vlm, "full_information", parameters)
            hidden_state_tracker.load_checkpoint()
        if "vocab_projection" in variant:
            projection_tracker = VocabProjectionTracking(dataset, vlm, "full_information", parameters)
            projection_tracker.load_checkpoint()
            

        checkpoint_every = int(checkpoint_every * len(results_df))
        for idx, row in tqdm(results_df.iterrows(), total=len(results_df), desc=f"Full information for {dataset} on {vlm}"):
            if not row["full_information_complete"]:
                data = dataset[idx]
                image = data["image"]
                full_information_question = data["full_information_question"]
                response = vlm(image, full_information_question, entity=data["class_name"])
                update_row(results_df, idx, "full_information", response, hidden_state_tracker=hidden_state_tracker, projection_tracker=projection_tracker)
                if idx % checkpoint_every == 0:  # This is okay because its sequential so it won't skip saving once it restarts
                    save(results_df, results_df_path, hidden_state_tracker, projection_tracker)
        save(results_df, results_df_path, hidden_state_tracker, projection_tracker)
    return results_df_path