## Some models might be gated so it is recommended to run `huggingface-cli login` if you do not have your huggingface access token saved on your device.
## Get access to LLama3.1 models at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct before running 

import transformers
import torch
import string
from inference.constants import prompts_dict

def get_llm(name):
    return LMInference(variant=name)

class LMInference:
    def __init__(self, variant="meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens=256, device="auto", role=None):
        
        model_id = f"{variant}"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=device,
            tokenizer=self.tokenizer
        )

        self.max_new_tokens = max_new_tokens
        self.prompts_dict = prompts_dict
        self.set_role(role)

    def set_role(self, role):
        assert role in self.prompts_dict.keys() or role is None, f"Role {role} not recognized"
        self.role = role

    def check_contain(self, text):
        if self.role is None:
            return
        check_strings = []
        if self.role in ["identification_evaluation", "correctness_evaluation"]:
            check_strings = ["Candidate: ", "Reference: "]
        elif self.role in ["question_extraction", "question_extraction_mcq"]:
            check_strings = ["Text:", "Entity:"]
        elif self.role in ["uniqueness_validation"]:
            check_strings = ["Text:", "Question:"]
        elif self.role in ["question_duplicate_evaluation"]:
            check_strings = ["Question:", "Answer:"]      
        elif self.role in ["question_answering", "question_answering_mcq"]:
            check_strings = ["Question:"]  
        for check_string in check_strings:
            if check_string not in text:
                assert f"{check_string} not found in text {text}. This is needed for the role {self.role}"

    def perform_identification(self, candidate, reference):
        assert candidate is not None
        assert reference is not None
        self.set_role("identification_evaluation")
        text = f"Candidate: {candidate}\nReference: {reference}\nExplanation: "
        return self(text)
    
    def perform_correctness(self, candidate, reference):
        assert candidate is not None
        assert reference is not None
        self.set_role("correctness_evaluation")
        text = f"Candidate: {candidate}\nReference: {reference}\nExplanation: "
        return self.parse(self(text))
    
    def perform_question_extraction(self, entity, text):
        assert entity is not None
        assert text is not None
        self.set_role("question_extraction")
        text = f"Entity: {entity}\nText: {text}\nRationale: "
        return self.parse(self(text))

    def perform_question_extraction_mcq(self, entity, text):
        assert entity is not None
        assert text is not None
        self.set_role("question_extraction_mcq")
        text = f"Entity: {entity}\nText: {text}\n"
        return self.parse(self(text))

    def perform_uniqueness_validation(self, question, text):
        assert question is not None
        assert text is not None
        self.set_role("uniqueness_validation")
        text = f"Text: {text}\nQuestion: {question}\nRationale: "
        return self.parse(self(text))

    def perform_question_duplicate_evaluation(self, question1, answer1, question2, answer2):
        assert question1 is not None
        assert answer1 is not None
        assert question2 is not None
        assert answer2 is not None
        self.set_role("question_duplicate_evaluation")
        text = f"Question: {question1}\nAnswer: {answer1}\nQuestion: {question2}\nAnswer: {answer2}\nRationale: "
        return self.parse(self(text))

    def perform_question_answering(self, question):
        assert question is not None
        self.set_role("question_answering")
        text = f"Question: {question}\nAnswer: "
        return self.parse(self(text))

    def parse_binary(self, output, true_word="pass", false_word="fail"):
        if true_word in output.lower():
            return True
        elif false_word in output.lower():
            return False
        else:
            return None
        
    def parse_question_extraction(self, output):
        sepsplits = output.split("[SEP]") # each of these should now contain rationale, question and answer
        qas = []
        for sep in sepsplits:
            try:
                relevant_part = sep.split("Question:")[-1].strip()
                question, answer = relevant_part.split("Answer:")
                qas.append({"question": question.strip(), "answer": answer.strip()})
            except:
                pass
        return qas
    
    def parse_question_extraction_mcq(self, output):
        pieces = output.split("\n")
        # the first line should have Question in it, the second should have Correct Option in it, the rest should have Options in them
        if "Question:" not in pieces[0]:
            return []
        if "Correct Option:" not in pieces[1]:
            return []
        n_other_options = 0
        for i in range(2, len(pieces)):
            if "Option:" in pieces[i]:
                n_other_options += 1
        if n_other_options <= 2:
            return []
        question = pieces[0].split("Question:")[-1].strip()
        correct_option = pieces[1].split("Correct Option:")[-1].strip()
        options = []
        for i in range(1, len(pieces)):
            if "Option:" in pieces[i]:
                option = pieces[i].split("Option:")[-1].strip()
                options.append(option)
        return {"question": question, "answer": correct_option, "options": options}
        
    def parse(self, output):
        if self.role is None:
            return output.strip()
        if self.role in ["identification", "correctness"]:
            return self.parse_binary(output)
        elif self.role in ["uniqueness_validation"]:
            return self.parse_binary(output, true_word="unique", false_word="multiple")
        elif self.role in ["question_duplicate_evaluation"]:
            return self.parse_binary(output, true_word="duplicate", false_word="unique")

        elif self.role in ["question_answering"]:
            output = output.replace("Answer:", "").strip()
            for punct in string.punctuation:
                output = output.strip(punct)
            return output
        elif self.role in ["question_extraction"]:
            return self.parse_question_extraction(output)
        elif self.role in ["question_answering_mcq"]:
            pass
        elif self.role in ["question_extraction_mcq"]:
            pass
        else:
            raise ValueError(f"Role {self.role} not recognized")

    def __call__(self, text):
        messages = []
        if self.role is not None:
            self.check_contain(text)
            prompts = self.prompts_dict[self.role]            
            messages.append({"role": "system", "content": prompts[0] + "Response in the following format:"})
            for prompt in prompts[1:]:
                messages.append({"role": "user", "content": prompt[0]})
                messages.append({"role": "assistant", "content": prompt[1].replace("[STOP]", self.tokenizer.eos_token)})
            messages.append({"role": "system", "content": prompts[0]})
        messages.append({"role": "user", "content": text})
        
        outputs = self.pipeline(
            messages,
            max_new_tokens=self.max_new_tokens, 
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            temperature=1.0,
            top_p=None,
            return_full_text=False) # do not return input text
        return (outputs[0]["generated_text"])



if __name__ == "__main__":
    # Example usage of Llama31Inference
    llama = LMInference(variant="meta-llama/Meta-Llama-3.1-8B-Instruct")
    print(llama("Hi how are you?"))



