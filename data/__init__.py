import click
#from data.data_holder import DataCreator
from data.mnist.setup_mnist import MNISTCreator


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
    if "mnist" in dataset_name:
        mnist_creator = MNISTCreator(parameters=parameters)
        mnist_creator.setup_data(validation_vlm)

    if "cifar10" in dataset_name:
        pass

    if "food101" in dataset_name:
        pass

    if "landmarks" in dataset_name:
        pass




def get_dataset(dataset_name: str):
    """
    Returns the dataset object for the given dataset name
    """
    return None