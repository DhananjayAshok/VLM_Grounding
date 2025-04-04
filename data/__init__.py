import click
from data.data_holder import DataHolder
from data.mnist.setup_mnist import MNISTCreator
from data.cifar100.setup_cifar import CIFAR100Creator


def get_data_creator(dataset_name, parameters=None):
    """
    Returns the data creator object for the given dataset name
    """
    if dataset_name == "mnist":
        return MNISTCreator(parameters=parameters)
    elif dataset_name == "cifar100":
        return CIFAR100Creator(parameters=parameters)
    elif dataset_name == "food101":
        pass
    elif dataset_name == "landmarks":
        pass
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


@click.command()
@click.option("--dataset_names", multiple=True, help="The name of the dataset(s) to use", default=["mnist"])
@click.option("--validation_vlm", default="llava-v1.6-vicuna-13b-hf", help="The VLM that is used to check if the VLM can identify labels from the dataset")
@click.pass_obj
def setup_data(parameters, dataset_names, validation_vlm):
    """
    Set up datasets for use in experiments
    """
    for dataset_name in dataset_names:
        creator = get_data_creator(dataset_name, parameters=parameters)
        creator.setup_data(validation_vlm)




def get_dataset(dataset_name: str, parameters):
    """
    Returns the dataset object for the given dataset name
    """
    return DataHolder(dataset_name, parameters=parameters)