import os
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error
import pandas as pd
from tqdm import tqdm
from experiments.grounding_utilities.identification import handle_openai


def get_starting_df(dataset, vlm, results_df_path, parameters):
    if os.path.exists(results_df_path):
        # restore checkpoint and start from there
        results_df = pd.read_csv(results_df_path)
    else:
        # look for the identification results for the dataset, vlm
        results_path = parameters["results_dir"] + f"/{dataset}/{vlm}/full_information_results_evaluated.csv"
        un_evaluated_results_path = parameters["results_dir"] + f"/{dataset}/{vlm}/full_information_results.csv"
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
            results_df["image_reference_complete"] = False
            pass_row_idx = results_df[results_df["full_information_pass"]].index
            results_df.loc[pass_row_idx, "image_reference_complete"] = True
        elif os.path.exists(un_evaluated_results_path):
            log_error(parameters["logger"], f"Un-evaluated full information results found for {dataset} and {vlm}. Please evaluate them first.")
        else:
            log_error(parameters["logger"], f"No full information results found for {dataset} and {vlm}. Please run the full information script first.")
    return results_df


def do_image_reference(dataset, vlm, variant="default",  parameters=None, checkpoint_every=0.1): # must consider OpenAI as well
    if parameters is None:
        parameters = load_parameters()
    results_df_path = parameters["results_dir"] + f"/{dataset}/{vlm}/image_reference_results.csv"

    if "gpt" in str(vlm): # handle openAI differently
        handle_openai(dataset, vlm, results_df_path, parameters, variant="image_reference", previous_check="full_information_pass")
    else:
        results_df = get_starting_df(dataset, vlm, results_df_path, parameters)
        if results_df["image_reference_complete"].all():
            parameters["logger"].warning(f"Image reference script already completed for {dataset} and {vlm}. Returning file found at {results_df_path} ...")
            return results_df
        if checkpoint_every > 1 or checkpoint_every < 0:
            log_error(parameters["logger"], f"Invalid checkpoint_every value: {checkpoint_every}. Must be between 0 and 1.")

        checkpoint_every = int(checkpoint_every * len(results_df))
        for idx, row in tqdm(results_df, total=len(results_df)):
            if not row["image_reference_complete"]:
                data = dataset[idx]
                image = data["image"]
                image_reference_question = data["image_reference_question"]
                response = vlm(image, image_reference_question)
                results_df.loc[idx, "image_reference_response"] = response
                results_df.loc[idx, "image_reference_complete"] = True
                if idx % checkpoint_every == 0:  # This is okay because its sequential so it won't skip saving once it restarts
                    results_df.to_csv(results_df_path, index=False)
        results_df.to_csv(results_df_path, index=False)
    return results_df_path