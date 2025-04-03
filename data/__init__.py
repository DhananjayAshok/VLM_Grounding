import click
#from data.data_holder import DataHolder


click.command()
@click.option("--dataset_name", multiple=True, help="The name of the dataset(s) to use", default=["mnist"])
@click.option("--validation_vlm", default="llava-v1.6-vicuna-13b-hf", help="The VLM that is used to check if the VLM can identify labels from the dataset")
@click.pass_obj
def setup_data(parameters, dataset_name, validation_vlm):
    # calls on the individual functions to set up the datasets
    # it will first get the labels, images and then test the identification ability of the model
    # then it will read from these accepted labels and decide which ones to generate questions for
    # then it generates questions for the dataset and validates it
    # saves all to a common format
    

    pass


def get_dataset(dataset_name: str):
    """
    Returns the dataset object for the given dataset name
    """
    return None