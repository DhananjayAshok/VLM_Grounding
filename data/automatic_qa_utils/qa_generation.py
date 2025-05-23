from tqdm import tqdm
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error
from data.automatic_qa_utils.wiki import get_wikipedia_texts, split_text

def generate_wiki(entity_name, llm, parameters=None):
    text = llm.perform_wiki_generation(entity_name, max_new_tokens=500) # Set max_new_tokens as high as possible given your GPU. 
    return split_text(entity_name, text, sentences_per=2, do_entity_check=False)


def generate_qas(entity_name, llm, parameters=None):
    if parameters is None:
        parameters = load_parameters()
    # Get Wikipedia text for the entity
    texts = get_wikipedia_texts(entity_name)
    if texts is None:
        parameters["logger"].warning(f"Could not find Wikipedia page for {entity_name}")
        return None
    lm_texts = generate_wiki(entity_name, llm, parameters)
    # Generate QAs from the text
    qas = []
    for text in tqdm(texts, desc=f"Generating QAs for {entity_name} from Wiki"):
        questions = llm.perform_question_extraction(entity_name, text)
        for question in questions:
            question["text"] = text
            question["entity_name"] = entity_name
            question["status"] = "generated"
            question["source"] = "Wikipedia"
            qas.append(question)
    for text in tqdm(lm_texts, desc=f"Generating QAs for {entity_name} from LLM"):
        questions = llm.perform_question_extraction(entity_name, text)
        for question in questions:
            question["text"] = text
            question["entity_name"] = entity_name
            question["status"] = "generated"
            question["source"] = "LLM"
            qas.append(question)

    return qas

def generate_all_qas(entity_list, llm, parameters=None):
    all_qas = []
    for entity_name in tqdm(entity_list, desc="Generating QAs"):
        qas = generate_qas(entity_name, llm, parameters)
        if qas is not None:
            all_qas.extend(qas)
    return all_qas


def generate_mcqas(qas, llm, parameters=None):
    if parameters is None:
        parameters = load_parameters()
    # Generate MCQAs from the QAs
    mcqas = {}
    for class_name in tqdm(qas.keys(), desc="Generating MCQAs"):
        mcqas[class_name] = []
        for qa in tqdm(qas[class_name], desc=f"Generating MCQAs for {class_name}", leave=False):
            if "accepted" not in qa["status"] or "text" not in qa:
                continue
            text = qa["text"]
            question = qa["question"]
            answer = qa["answer"]
            options = [answer]
            other_options = llm.perform_question_extraction_mcq(text, question, answer)   
            if len(other_options) == 0:     
                continue
            qa["options"] = options + other_options
            mcqas[class_name].append(qa)
    return mcqas
