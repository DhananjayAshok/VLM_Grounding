from utils.log_handling import log_error
from evaluation.metrics import df_compute_metric_str


def do_final_evaluation(df, parameters, verbose=False, okvqa=False, mcq=False):
    reference_column = "answer_str"
    if not okvqa:
        variants = ["full_information", "image_reference"]
        for trivial in ["black", "white", "noise"]:
            variants.append(f"trivial_{trivial}_full_information")
            variants.append(f"trivial_{trivial}_image_reference")
        candidate_columns = [f"{variant}_response" for variant in variants]
    else:
        candidate_columns = ["image_reference_response"]
    metrics = ["inclusion", "two_way_inclusion", "exact_match", "bleu"]
    if mcq:
        metrics.append("mcq_correct")
    for metric in metrics:
        for candidate_column in candidate_columns:
            output_column = f"{metric}_{candidate_column}"
            give_ref = reference_column if metric != "mcq_correct" else "mcq_answer"
            df = df_compute_metric_str(metric, df, candidate_column, give_ref, output_column=output_column, save=False, parameters=parameters, verbose=verbose)
    if okvqa:
        return df
    for metric in metrics:
        for variant in ["full_information", "image_reference"]:
            output_columns = []
            for trivial in ["black", "white", "noise"]:
                candidate_column = f"trivial_{trivial}_{variant}_response"
                output_column = f"{metric}_{candidate_column}"
                output_columns.append(output_column)
            # take the min over these columns
            df[f"{metric}_trivial_min_{variant}_response"] = df[output_columns].min(axis=1)
            # take the max over these columns
            df[f"{metric}_trivial_max_{variant}_response"] = df[output_columns].max(axis=1)
            # take the mean over these columns
            df[f"{metric}_trivial_mean_{variant}_response"] = df[output_columns].mean(axis=1)
    return df



def log_final_evaluation(df, parameters, okvqa=False):
    logger = parameters["logger"]
    for column in df.columns:
        if column.endswith("_pass"):
            logger.info(f"{column}: {df[column].sum()}/{len(df)}")
    if not okvqa:
        response_cols = ["full_information_response", "image_reference_response"]
        candidate_cols = response_cols.copy()
        trivials = ["min", "max"]
        for trivial in trivials:
            for variant in response_cols:
                candidate_cols.append(f"trivial_{trivial}_{variant}")
    else:
        candidate_cols = ["image_reference_response"]
    for log_metric in ["two_way_inclusion", "bleu", "inclusion", "exact_match"]:
        for candidate_col in candidate_cols:
            column = f"{log_metric}_{candidate_col}"
            if column in df.columns:
                nonnan = df[df["image_reference_response"].notna()]
                logger.info(f"{column}: {nonnan[column].mean()}")