import wikipediaapi
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error


parameters = load_parameters()
wiki_wiki = wikipediaapi.Wikipedia(user_agent=f"Vision Language Model Grounding Data Gathering ({parameters['personal_email']})", language='en')


def get_wikipedia_texts(entity_name):
    page_py = wiki_wiki.page(entity_name)
    if not page_py.exists():
        parameters["logger"].warning(f"Could not find wikipedia page for {entity_name}")
        return None
    else:
        text = page_py.text
    return split_text(entity_name, text)
    

def split_text(entity_name, text, sentences_per=2):
    # split into sentences that have entity_name in them
    split_text = text.split("\n\n") # split by section first
    text = []
    for text_para in split_text:
        options = text_para.split("\n") # split by paragraph
        for option in options:
            if len(option.split(" ")) < 5: # skip short paras
                continue
            text.append(option)
    split_text = []
    for text_para in text:
        options = text_para.split(". ") # split into sentences
        # join every sentences_per sentences into one text
        joined_text = []
        for i in range(0, len(options), sentences_per):
            joined_text.append(". ".join(options[i:i+sentences_per]))
        for option in joined_text:
            if entity_name.lower() in option.lower():
                split_text.append(option)
            else:
                pass
    return split_text
