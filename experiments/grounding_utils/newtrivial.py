import os
from experiments.grounding_utils.common import HiddenStateTracking, VocabProjectionTracking, save, update_row, handle_openai, get_starting_df
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error
import pandas as pd
from tqdm import tqdm


def do_trivial_full_information(dataset, vlm, variant="default",  parameters=None, checkpoint_every=0.1):
    do_trivial(dataset, vlm, "full_information", variant=variant, parameters=parameters, checkpoint_every=checkpoint_every)

def do_trivial_image_reference(dataset, vlm, variant="default",  parameters=None, checkpoint_every=0.1):
    do_trivial(dataset, vlm, "image_reference", variant=variant, parameters=parameters, checkpoint_every=checkpoint_every)


def do_trivial(dataset, vlm, trivial_variant, variant="default", parameters=None, checkpoint_every=0.1):
    if parameters is None:
        parameters = load_parameters()
    run_variant = f"trivial_{trivial_variant}"
    results_df_path = parameters["results_dir"] + f"/{dataset}/{vlm}/{run_variant}_results.csv"

    if "gpt" in str(vlm): # handle openAI differently
        handle_openai(dataset, vlm, results_df_path, parameters, variant=run_variant, previous_check="full_information_pass")
    else:
        results_df = get_starting_df(dataset, vlm, results_df_path, parameters, run_variant=run_variant)

        if results_df[f"{run_variant}_complete"].all():
            parameters["logger"].warning(f"Trivial script for variant {trivial_variant} already completed for {dataset} and {vlm}. Returning file found at {results_df_path} ...")
            return results_df_path
        if checkpoint_every > 1 or checkpoint_every < 0:
            log_error(parameters["logger"], f"Invalid checkpoint_every value: {checkpoint_every}. Must be between 0 and 1.")

        hidden_state_tracker = None
        projection_tracker = None
        if "hidden_state" in variant:
            hidden_state_tracker = HiddenStateTracking(dataset, vlm, run_variant, parameters)
            hidden_state_tracker.load_checkpoint()
        if "vocab_projection" in variant:
            projection_tracker = VocabProjectionTracking(dataset, vlm, run_variant, parameters)
            projection_tracker.load_checkpoint()
            

        checkpoint_every = int(checkpoint_every * len(results_df))
        for idx, row in tqdm(results_df.iterrows(), total=len(results_df), desc=f"Trivial {trivial_variant} for {dataset} on {vlm}"):
            if not row[f"{run_variant}_complete"]:
                data = dataset[idx]
                image = None
                question = data[f"{trivial_variant}_question"]
                response = vlm(image, question)
                update_row(results_df, idx, run_variant, response, hidden_state_tracker=hidden_state_tracker, projection_tracker=projection_tracker)
                if idx % checkpoint_every == 0:  # This is okay because its sequential so it won't skip saving once it restarts
                    save(results_df, results_df_path, hidden_state_tracker, projection_tracker)
        save(results_df, results_df_path, hidden_state_tracker, projection_tracker)
    return results_df_path