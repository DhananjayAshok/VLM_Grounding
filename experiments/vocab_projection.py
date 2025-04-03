import click
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


def plot_dynamics(dynamics, corrects):
    # form the csv, 
    columns = ["Layer", "Answer Dynamics", "Output Dynamics", "Correct"]
    data = []
    n_layers = dynamics.shape[2]
    for i in range(len(dynamics)):
        for j in range(n_layers):
            data.append([j, dynamics[i, 0, j], dynamics[i, 1, j], corrects[i]])
    df = pd.DataFrame(data, columns=columns)
    df["Correct"] = df["Correct"].map({True: "Linking Success", False: "Linking Failure"})
    sns.lineplot(data=df, x="Layer", y="Answer Dynamics", hue="Correct")
    plt.ylabel(f"Probability of Correct Answer Token")
    plt.legend()
    plt.show()
    sns.lineplot(data=df, x="Layer", y="Output Dynamics", hue="Correct")
    plt.ylabel(f"Probability of Eventual Output Token")
    plt.legend()
    plt.show()

def plot_kls(kls, corrects):
    columns = ["Layer", "Forward KL", "Reverse KL", "Correct"]
    data = []
    n_layers = kls.shape[2]
    for i in range(len(kls)):
        for j in range(n_layers):
            data.append([j, kls[i, 0, j], kls[i, 1, j], corrects[i]])
    df = pd.DataFrame(data, columns=columns)
    df["Correct"] = df["Correct"].map({True: "Linking Success", False: "Linking Failure"})
    df["Layer"] = df["Layer"] + 1
    sns.lineplot(data=df, x="Layer", y="Forward KL", hue="Correct")
    plt.ylabel(f"Forward KL Divergence")
    plt.legend()
    plt.show()
    df["Layer"] = df["Layer"] - 1
    sns.lineplot(data=df, x="Layer", y="Reverse KL", hue="Correct")
    plt.ylabel(f"Reverse KL Divergence")
    plt.legend()
    plt.show()

def kl_divergence(p, q):
    return np.sum(p * (np.log(p) - np.log(q+0.0001)) )


def forward_kl(vocab_array):
    # vocab_array shape is n_layers, vocab_size
    kl_divs = [] # this is a n_layers - 1 length array
    for i in range(1, vocab_array.shape[0]):
        kl_divs.append(kl_divergence(vocab_array[i-1], vocab_array[i]))
    return np.array(kl_divs)

def reverse_kl(vocab_array):
    kl_divs = [] # this is a n_layers - 1 length array
    for i in range(1, vocab_array.shape[0]):
        kl_divs.append(kl_divergence(vocab_array[i], vocab_array[i-1]))
    return np.array(kl_divs)



@click.command() # arguments are dataset, model, reference_column
@click.option("--dataset", type=click.Choice(["mnist_math", "imagenette", "food101"]), required=True)
@click.option("--model", required=True, type=click.Choice(["llava-v1.6-mistral-7b-hf", "llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf", "instructblip-vicuna-7b", "instructblip-vicuna-13b"]), help="The model to evaluate")
@click.option("--metric", type=str, default="exact_match")
@click.option('--rerun', type=bool, default=False)
@click.pass_obj
def main(parameters, dataset, model, metric, rerun):
    results_dir = parameters["results_dir"]
    real_results_dir = os.path.join(results_dir, dataset, model)
    if os.path.exists(os.path.join(real_results_dir, f"{metric}_dynamics.npy")) and not rerun:
        dynamics = np.load(os.path.join(real_results_dir, f"{metric}_dynamics.npy"))
        corrects = np.load(os.path.join(real_results_dir, f"{metric}_corrects.npy"))
        kls = np.load(os.path.join(real_results_dir, f"{metric}_kls.npy"))
    else:
        from inference.vlms import LlaVaInference
        from data import get_dataset
        # results_visual_evaluated.csv
        results_df = pd.read_csv(os.path.join(real_results_dir, "results_visual_evaluated.csv"))
        metric_col = f"image_language_text_answer_{metric}"
        assert metric_col in results_df.columns
        # cast to bool
        results_df[metric_col] = results_df[metric_col].astype(bool)
        dset = get_dataset(dataset)
        data = []
        kls = []
        corrects = []
        llava = LlaVaInference(model, vocab_projection_mode=True)
        for i in tqdm(range(len(results_df))):
            row = results_df.iloc[i]
            datapoint_idx = row["idx"]
            datapoint = dset[datapoint_idx]
            answer = datapoint["answer"]
            output = row["image_language_text"]
            answer_tokens = llava.processor(text=answer, add_special_tokens=False)['input_ids'][0]
            output_tokens = llava.processor(text=output, add_special_tokens=False)['input_ids'][0]
            if len(output_tokens) > 1:
                print(f"Output: {output} has more than one token")
                #continue # just match the first
            if len(answer_tokens) > 1:
                print(f"Answer: {answer} has more than one token")
                # continue # just match the first
            answer_token = answer_tokens[0]
            output_token = output_tokens[0]
            image = datapoint["base_image"]
            text = datapoint["image_language_text"]
            vocab_array = llava(image, text) # shape is n_layers, vocab_size
            # get the dynamics of both the answer and output tokens
            answer_dynamics = vocab_array[:, answer_token]
            output_dynamics = vocab_array[:, output_token]
            data.append([answer_dynamics, output_dynamics, ])
            corrects.append(row[metric_col])
            forward = forward_kl(vocab_array)
            reverse = reverse_kl(vocab_array)
            kls.append([forward, reverse])

        dynamics = np.array(data) # shape, n_datapoints, 2, n_layers
        corrects = np.array(corrects)
        kls = np.array(kls)
        # save to a numpy file:
        np.save(os.path.join(real_results_dir, f"{metric}_dynamics.npy"), dynamics)
        np.save(os.path.join(real_results_dir, f"{metric}_corrects.npy"), corrects)
        np.save(os.path.join(real_results_dir, f"{metric}_kls.npy"), kls)
        print(f"Saved dynamics at {real_results_dir}/{metric}_dynamics.npy")
    plot_dynamics(dynamics, corrects)
    plot_kls(kls, corrects)




if __name__ == "__main__":
    main()
