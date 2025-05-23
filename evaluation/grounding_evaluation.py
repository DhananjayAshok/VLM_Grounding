from utils.log_handling import log_error
from utils.parameter_handling import load_parameters
from evaluation.metrics import df_compute_metric_str
from experiments.grounding_utils.trivial import all_trivials
import numpy as np
from tqdm import tqdm


def stringify(element, parameters=None):
    if parameters is None:
        parameters = load_parameters()
    if isinstance(element, float):
        return str(int(element))
    elif isinstance(element, str) or isinstance(element, int):
        return str(element)
    else:
        log_error(parameters["logger"], f"Unknown type {type(element)} for element {element}.")

        

def safe_isnan(x):
    if not isinstance(x, float):
        return False
    else:
        return np.isnan(x)

def column_mode(row):
    columns = row.index
    cleaned = [stringify(row[col]).strip() for col in columns if not safe_isnan(row[col])]
    if len(cleaned) == 0:
        return None
    mode = max(set(cleaned), key=cleaned.count)
    return mode

def get_column_mode(df, columns):
    modes = []
    for i, row in df.iterrows():#, total=len(df), desc="Computing column mode"):
        mode = column_mode(row[columns])
        modes.append(mode)
    return modes



metrics_to_use = ["two_way_inclusion", "exact_match", "bleu"] # + ["inclusion"]


def do_final_evaluation(df, parameters, verbose=False, okvqa=False, mcq=False):
    reference_column = "answer_str"
    if not okvqa:
        variants = ["full_information", "image_reference"]
        for trivial in all_trivials:
            variants.append(f"trivial_{trivial}_full_information")
            variants.append(f"trivial_{trivial}_image_reference")
        candidate_columns = [f"{variant}_response" for variant in variants]
    else:
        candidate_columns = ["image_reference_response"]
    metrics = metrics_to_use
    if mcq:
        metrics.append("mcq_correct")
    for metric in metrics:#, desc="Computing metrics"):
        for candidate_column in candidate_columns:
            output_column = f"{metric}_{candidate_column}"
            give_ref = reference_column if metric != "mcq_correct" else "mcq_answer"
            df = df_compute_metric_str(metric, df, candidate_column, give_ref, output_column=output_column, save=False, parameters=parameters, verbose=verbose)
    if okvqa:
        return df
    for metric in metrics:#, desc="Computing trivial aggregation metrics"):
        for variant in ["full_information", "image_reference"]:
            output_columns = []
            candidate_columns = []
            for trivial in all_trivials:
                candidate_column = f"trivial_{trivial}_{variant}_response"
                candidate_columns.append(candidate_column)
                output_column = f"{metric}_{candidate_column}"
                output_columns.append(output_column)
            # take the min over these columns
            df[f"trivial_mode_{variant}_response"] = get_column_mode(df, candidate_columns)#df[candidate_columns].apply(column_mode, axis=1)
            give_ref = reference_column if metric != "mcq_correct" else "mcq_answer"
            df = df_compute_metric_str(metric, df, f"trivial_mode_{variant}_response", give_ref, output_column=f"{metric}_trivial_mode_{variant}_response", save=False, parameters=parameters, verbose=verbose)
            df[f"{metric}_trivial_min_{variant}_response"] = df[output_columns].min(axis=1)
            # take the max over these columns
            df[f"{metric}_trivial_max_{variant}_response"] = df[output_columns].max(axis=1)
            # take the mean over these columns
            df[f"{metric}_trivial_mean_{variant}_response"] = df[output_columns].mean(axis=1)

    return df


def show_float(x):
    return f"{x*100:.2f}"




def log_final_evaluation(df, parameters, okvqa=False):
    logger = parameters["logger"]
    for column in df.columns:
        if column.endswith("_pass") and column == "full_information_pass":
            logger.info(f"{column}: {df[column].sum()}/{len(df)}")
    if not okvqa:
        response_cols = ["full_information_response", "image_reference_response"]
        candidate_cols = response_cols.copy()
        trivials = ["mode"] #+ ["max"] #+ all_trivials
        for trivial in trivials:
            for variant in response_cols:
                candidate_cols.append(f"trivial_{trivial}_{variant}")
        slice_cols = ["trivial_mode_image_reference_response"] #+ ["trivial_mode_full_information_response", "image_reference_response", "trivial_max_image_reference_response", "trivial_max_full_information_response"]
    else:
        candidate_cols = ["image_reference_response"]
        slice_cols = []
    metrics = metrics_to_use + ["mcq_correct"]
    for log_metric in metrics:
        logger.info(f"Metric: {log_metric}")
        for candidate_col in candidate_cols:
            column = f"{log_metric}_{candidate_col}"
            if column in df.columns:
                if candidate_col == "full_information_response":
                    nan_col = "image_reference_response"
                elif "trivial_mode" in candidate_col:
                    nan_col = candidate_col.replace("mode", "black")
                elif "trivial_max" in candidate_col:
                    nan_col = candidate_col.replace("max", "black")
                else:
                    nan_col = candidate_col
                nonnan = df[df[nan_col].notna()]
                if candidate_col in ["full_information_response", "trivial_mode_image_reference_response"]:
                    logger.info(f"\t{candidate_col}: {show_float(nonnan[column].mean())}")
                if candidate_col not in ["full_information_response", "trivial_mode_image_reference_response"]:
                    for slice_col in slice_cols:
                        slice_metric_col = f"{log_metric}_{slice_col}"
                        if column == slice_metric_col:
                            continue
                        for boolval in [False]:
                            slice_df = df[df[slice_metric_col] == boolval]
                            nonnan = slice_df[slice_df["image_reference_response"].notna()]
                            #logger.info(f"\t{column} when {slice_col} is {boolval}: {nonnan[column].mean()}")
                            logger.info(f"\t{candidate_col} Controlled: {show_float(nonnan[column].mean())}")