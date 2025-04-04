import click
from utils.log_handling import log_error


@click.command()
@click.option("--dataset_name", type=str, help="The name of the dataset(s) to use", default=["mnist"])
@click.pass_obj
def generate_questions(parameters, dataset_name):
