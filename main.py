from utils.parameter_handling import load_parameters, compute_secondary_parameters
import click
from experiments.grounding import grounding_experiment
from experiments.hidden_state_predictor import fit_hidden_state_predictor
from experiments.vocab_projection_plotting import visualize_vocab_projection
from experiments.okvqa_inference import setup_okvqa, okvqa_inference
from data.automatic_qa_utils import generate_questions, validate_questions, deduplicate_questions, full_qa_pipeline
from data import setup_data, validate_classes


loaded_parameters = load_parameters()

# Any parameter from your project that you want to be able to change from the command line should be added as an option here
@click.group()
@click.option("--storage_dir", default=loaded_parameters["storage_dir"], help="The directory where the data is stored")
@click.option("--random_seed", default=loaded_parameters["random_seed"], help="The random seed for the project")
@click.option("--log_file", default=loaded_parameters["log_file"], help="The file to log to")
@click.pass_context
def main(ctx, **input_parameters):
    loaded_parameters.update(input_parameters)
    compute_secondary_parameters(loaded_parameters)
    ctx.obj = loaded_parameters

# Implement custom commands as functions in a separate file in the following way:
"""
@click.command()
@click.option("--example_option")
@click.pass_obj
def example_command(parameters, example_option):
    # have access to parameters here with any additional arguments that are specific to the script
    pass
"""
# Then add the custom command to the main group like this:
main.add_command(validate_classes, name="validate_classes")
main.add_command(generate_questions, name="generate_questions")
main.add_command(validate_questions, name="validate_questions")
main.add_command(deduplicate_questions, name="deduplicate_questions")
main.add_command(full_qa_pipeline, name="full_qa_pipeline")
main.add_command(setup_data, name="setup_data")
main.add_command(grounding_experiment, name="grounding_experiment")
main.add_command(fit_hidden_state_predictor, name="fit_hidden_states")
main.add_command(visualize_vocab_projection, name="visualize_vocab_projection")
main.add_command(setup_okvqa, name="setup_okvqa")
main.add_command(okvqa_inference, name="run_okvqa")


if __name__ == "__main__":
    main()