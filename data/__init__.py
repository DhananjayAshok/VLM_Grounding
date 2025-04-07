import click
from data.data_holder import DataHolder, get_data_creator
from utils.parameter_handling import load_parameters

@click.command()
@click.option("--dataset_names", multiple=True, help="The name of the dataset(s) to use", default=["mnist"])
@click.option("--validation_vlm", default="llava-v1.6-vicuna-13b-hf", help="The VLM that is used to check if the VLM can identify labels from the dataset")
@click.option('--validation_threshold', type=float, default=0.5, help='Will only consider a class validated if the VLM can identify more than this percentage of instances of that class.')
@click.pass_obj
def validate_classes(parameters, dataset_names, validation_vlm, validation_threshold):
    """
    Validates only the classes that VLMs can identify
    """
    for dataset_name in dataset_names:
        creator = get_data_creator(dataset_name, parameters=parameters)
        creator.validate_classes(validation_vlm, validation_threshold=validation_threshold)


@click.command()
@click.option("--dataset_names", multiple=True, help="The name of the dataset(s) to use", default=["mnist"])
@click.option('--target_datapoints', type=int, default=1000, help='The number of datapoints to aim for in the dataset (lower bound, will overshoot).')
@click.option("--skip_mcq", default=False, help="Skip MCQ generation")
@click.pass_obj
def setup_data(parameters, dataset_names, target_datapoints, skip_mcq):
    """
    Set up datasets for use in experiments
    """
    for dataset_name in dataset_names:
        creator = get_data_creator(dataset_name, parameters=parameters)
        creator.create_validated_data(target_datapoints=target_datapoints)
        if not skip_mcq and dataset_name not in ["mnist"]: # some datasets are not supported for MCQ
            creator = get_data_creator(dataset_name, parameters=parameters, mcq=True)
            creator.create_validated_data(target_datapoints=target_datapoints)






def get_dataset(dataset_name: str, parameters=None):
    """
    Returns the dataset object for the given dataset name
    """
    if parameters is None:
        parameters = load_parameters()
    return DataHolder(dataset_name, parameters=parameters)