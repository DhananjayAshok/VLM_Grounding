import click



def process_okvqa(parameters):
    """
    Take the OKVQA dataset as downloaded from extracted zip files and put in easy to iterate format
    """
    # essentially have it in a dataframe of form: 
        # question, answers, answer (the most common of the answers in the annotations), image_path 
    pass

def run_okvqa(parameters, vlm):
    """
    Run the inference on the OKVQA dataset using the specified VLM.
    """
    #Must save the results to a csv file with a format that the hidden_state_predictor script can read. 
    pass



@click.command()
@click.option("--model", help="The VLM whose grounding ability is being tested", type=click.Choice(["llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf", "llava-v1.6-mistral-7b-hf", "instructblip-vicuna-7b", "instructblip-vicuna-13b"]), required=True)
@click.pass_obj
def okvqa_inference(parameters, model):
    """
    Run inference on the OKVQA dataset using the specified VLM.
    """
