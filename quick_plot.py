from utils.plot_utils import Plotter, sns, plt
import click
import pandas as pd
import os


errorbar = "se"

@click.command()
@click.pass_obj
def do_plot(parameters):
    """
    Run primary experiment for dataset
    """
    plotter = Plotter(parameters)
    plotter.set_size_parameters(legend_font_size=20)
    for dataset in ["cifar100", "food101"]:
        parameters["logger"].info(f"Plotting {dataset}")
        do_all(plotter, dataset)


def do_all(plotter, dataset, model="llava-v1.6-vicuna-7b-hf", remove_trivial=False):
    figure_dir = plotter.parameters["figure_dir"] + "/figures"
    dpath = os.path.join(figure_dir, dataset, model, f"remove_trivial_{remove_trivial}")
    prob_projection_df_path = os.path.join(dpath, "projection_probability/image_reference.csv")
    kl_divergence_df_path = os.path.join(dpath, "kl_divergence/full_information_vs_image_reference.csv")
    cosine_similarity_df_path = os.path.join(dpath, "cosine_similarity/full_information_vs_image_reference.csv")
    lineplot(plotter, prob_projection_df_path, dataset)
    #klplot(plotter, kl_divergence_df_path, dataset)
    #cosineplot(plotter, cosine_similarity_df_path, dataset)





def klplot(plotter, data_df_path, dataset):
    data_df = pd.read_csv(data_df_path)
    sns.lineplot(data=data_df, x="Layer Index", y="KL Divergence", hue="Linking Status", palette=["green", "red"], linewidth=2.5, errorbar=errorbar)
    plt.title(dataset[0].upper()+dataset[1:])
    plt.legend(loc='upper left', #bbox_to_anchor=(1, 1), 
               frameon=True, fancybox=True, shadow=True, fontsize=plotter.size_params["legend_font_size"],
               borderaxespad=1.0, borderpad=1.0,
               labelspacing=1.0)
    plotter.show()

def cosineplot(plotter, data_df_path, dataset):
    data_df = pd.read_csv(data_df_path)
    sns.lineplot(data=data_df, x="Layer Index", y="Cosine Similarity", hue="Linking Status", palette=["green", "red"], linewidth=2.5, errorbar=errorbar)
    plt.title(dataset[0].upper()+dataset[1:])
    plt.legend(loc='lower left', #bbox_to_anchor=(1, 1), 
               frameon=True, fancybox=True, shadow=True, fontsize=plotter.size_params["legend_font_size"],
               borderaxespad=1.0, borderpad=1.0,
               labelspacing=1.0)
    plotter.show()

def lineplot(plotter, data_df_path, dataset):
    data_df = pd.read_csv(data_df_path)
    data_df["Token Probability"] = data_df["Probability of Token"]
    sns.lineplot(data=data_df, x="Layer Index", y="Token Probability", hue="Linking Status", palette=["green", "red"], linewidth=2.5, errorbar=errorbar) 
    plt.title(dataset[0].upper()+dataset[1:])
    plt.legend(loc='upper left', #bbox_to_anchor=(1, 1), 
            frameon=True, fancybox=True, shadow=True, fontsize=plotter.size_params["legend_font_size"],
            borderaxespad=1.0, borderpad=1.0,
            labelspacing=1.0)
    plotter.show()
