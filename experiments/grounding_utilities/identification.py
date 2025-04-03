import os
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error
import pandas as pd
from tqdm import tqdm


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



def do_identification(dataset, vlm, variant="default", parameters=None, checkpoint_every=0.1): # must consider OpenAI as well
    if parameters is None:
        parameters = load_parameters()
    results_df_path = parameters["results_dir"] + f"/{dataset}/{vlm}/identification_results.csv"
    if "gpt" in str(vlm): # handle openAI differently
        handle_openai(dataset, vlm, results_df_path, parameters, variant="identification")
    else:
        results_df = get_starting_df(dataset, results_df_path)
        if results_df["identification_complete"].all():
            parameters["logger"].warning(f"Identification script already completed for {dataset} and {vlm}. Returning file found at {results_df_path} ...")
            return results_df
        if checkpoint_every > 1 or checkpoint_every < 0:
            log_error(parameters["logger"], f"Invalid checkpoint_every value: {checkpoint_every}. Must be between 0 and 1.")

        checkpoint_every = int(checkpoint_every * len(results_df))
        for idx, row in tqdm(results_df, total=len(results_df)):
            if not row["identification_complete"]:
                data = dataset[idx]
                image = data["image"]
                identification_question = data["identification_question"]
                response = vlm(image, identification_question)
                results_df.loc[idx, "identification_response"] = response
                results_df.loc[idx, "identification_complete"] = True
                if idx % checkpoint_every == 0:  # This is okay because its sequential so it won't skip saving once it restarts
                    results_df.to_csv(results_df_path, index=False)
        results_df.to_csv(results_df_path, index=False)

    return results_df_path
