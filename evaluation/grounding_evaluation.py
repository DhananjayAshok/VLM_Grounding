from utils.log_handling import log_error
from evaluation.metrics import df_compute_metric_str


def do_final_evaluation(df, parameters, verbose=False, okvqa=False, mcq=False):
    reference_column = "answer_str"
    if not okvqa:
        variants = ["full_information_response", "image_reference_response"]
        candidate_columns = variants.copy()
        for variant in variants:
            candidate_columns.append(f"trivial_{variant}")
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
    return df



def log_final_evaluation(df, parameters, okvqa=False):
    logger = parameters["logger"]
    for column in df.columns:
        if column.endswith("_pass"):
            logger.info(f"{column}: {df[column].sum()}/{len(df)}")
    if not okvqa:
        response_cols = ["full_information_response", "image_reference_response"]
        candidate_cols = response_cols.copy()
        for variant in response_cols:
            candidate_cols.append(f"trivial_{variant}")
    else:
        candidate_cols = ["image_reference_response"]
    for log_metric in ["two_way_inclusion", "bleu", "inclusion", "exact_match", "mcq_correct"]:
        for candidate_col in candidate_cols:
            column = f"{log_metric}_{candidate_col}"
            if column in df.columns:
                nonnan = df[df[candidate_col].notna()]
                logger.info(f"{column}: {nonnan[column].mean()}")
                if "trivial" not in candidate_col:
                    trivialmax_column = f"{log_metric}_trivial_{candidate_col}"
                    if trivialmax_column in df.columns:
                        nonan_slice = df[df[candidate_col].notna() & (df[trivialmax_column] == False)]
                        if len(nonan_slice) > 0:
                            logger.info(f"{column} with trivial successes removed: {nonan_slice[column].mean()}")
                        else:
                            logger.info(f"{column} with trivial successes removed: No remaining entries")
