## Some models might be gated so it is recommended to run `huggingface-cli login` if you do not have your huggingface access token saved on your device.
## Get access to LLama3.1 models at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct before running 

import transformers
import torch
import string
from inference.constants import IDENTIFICATION_PROMPTS, CORRECTNESS_PROMPTS, QUESTION_EXTRACTION_PROMPTS, QUESTION_ANSWER_PROMPTS, QUESTION_UNIQUE_ANSWER_PROMPTS

def get_llm_inference_class(name):
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
        self.set_role(role)
        self.prompts_dict = {"identification": IDENTIFICATION_PROMPTS, "correctness": CORRECTNESS_PROMPTS, "question_extraction": QUESTION_EXTRACTION_PROMPTS, "question_answering": QUESTION_ANSWER_PROMPTS, "question_validation": QUESTION_UNIQUE_ANSWER_PROMPTS}

    def set_role(self, role):
        assert role in ["identification", "correctness", "question_extraction", "question_answering", "question_validation", None]
        self.role = role

    def check_contain(self, text):
        if self.role is None:
            return
        check_strings = []
        if self.role in ["identification", "correctness"]:
            check_strings = ["Candidate: ", "Reference: "]
        elif self.role in ["question_extraction"]:
            check_strings = ["Text:", "Entity:"]
        elif self.role in ["question_answering"]:
            check_strings = ["Text:", "Question:"]
        elif self.role in ["question_validation"]:
            check_strings = ["Text:", "Question:"]
        for check_string in check_strings:
            if check_string not in text:
                assert f"{check_string} not found in text {text}. This is needed for the role {self.role}"


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
                qas.append((question.strip(), answer.strip()))
            except:
                pass
        return qas


    def parse(self, output):
        if self.role is None:
            return output.strip()
        if self.role in ["identification", "correctness"]:
            return self.parse_binary(output)
        elif self.role in ["question_validation"]:
            return self.parse_binary(output, true_word="unique", false_word="multiple")
        elif self.role in ["question_answering"]:
            output = output.replace("Answer:", "").strip()
            for punct in string.punctuation:
                output = output.strip(punct)
            return output
        elif self.role in ["question_extraction"]:
            return self.parse_question_extraction(output)
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



