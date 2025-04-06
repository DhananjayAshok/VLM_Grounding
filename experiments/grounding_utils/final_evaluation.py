from utils.log_handling import log_error
from evaluation.metrics import df_compute_metric_str


def do_final_evaluation(df, parameters):
    reference_column = "answer_str"
    variants = ["full_information", "image_reference"]
    for trivial in ["black", "white", "noise"]:
        variants.append(f"trivial_{trivial}_full_information")
        variants.append(f"trivial_{trivial}_image_reference")
    candidate_columns = [f"{variant}_response" for variant in variants]
    metrics = ["inclusion", "two_way_inclusion", "exact_match", "bleu"]
    for metric in metrics:
        for candidate_column in candidate_columns:
            output_column = f"{metric}_{candidate_column}"
            df = df_compute_metric_str(metric, df, candidate_column, reference_column, output_column=output_column, save=False, parameters=parameters)
    for metric in metrics:
        for variant in ["full_information", "image_reference"]:
            output_columns = []
            for trivial in ["black", "white", "noise"]:
                candidate_column = f"trivial_{trivial}_{variant}_response"
                output_column = f"{metric}_{candidate_column}"
                output_columns.append(output_column)
            # take the min over these columns
            df[f"{metric}_trivial_min_{variant}_response"] = df[[output_columns]].min(axis=1)
            # take the max over these columns
            df[f"{metric}_trivial_max_{variant}_response"] = df[[output_columns]].max(axis=1)
            # take the mean over these columns
            df[f"{metric}_trivial_mean_{variant}_response"] = df[[output_columns]].mean(axis=1)
    return df