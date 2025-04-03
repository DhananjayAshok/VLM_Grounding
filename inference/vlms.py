from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
import base64
import os
import json
import numpy as np
import pandas as pd
from openai import OpenAI


openai_tmp_file_dir = "openai_tmp_files"

# load all models and have a single function that you can call (with batch input) and get the output text

def get_vlm(name):
    if "llava" in name:
        return LlaVaInference(variant=name)
    elif "instructblip" in name:
        return BLIPInference(variant=name)
    elif "gpt" in name:
        return OpenAIInference(variant=name)
    else:
        raise ValueError(f"Model {name} not recognized.")


class LlaVaInference:
    def __init__(self, variant="llava-v1.6-mistral-7b-hf", vocab_projection_mode=False, hidden_state_tracking_mode=False, attention_tracking_mode=False):
        self.variant = variant
        self.processor = LlavaNextProcessor.from_pretrained(f"llava-hf/{variant}")
        self.vocab_projection_mode = vocab_projection_mode
        self.hidden_state_tracking_mode = hidden_state_tracking_mode
        self.attention_tracking_mode = attention_tracking_mode
        #self.processor.patch_size = 14
        #self.processor.vision_feature_selection_strategy = "default"
        self.model = LlavaNextForConditionalGeneration.from_pretrained(f"llava-hf/{variant}", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
        self.model.eval()
        # set pad_token_id to eos_token_id
        if self.model.config.pad_token_id is None:
            if self.model.config.eos_token_id is not None:
                self.model.config.pad_token_id = self.model.config.eos_token_id
            else:
                self.model.config.eos_token_id = self.processor.tokenizer.eos_token_id
                self.model.config.pad_token_id = self.model.config.eos_token_id
        if self.hidden_state_tracking_mode:
            n_layers = self.model.config.num_hidden_layers
            layers_to_track = list(range(1, n_layers-1), 4)
            self.layers_to_track = layers_to_track
        

    def compute_vocab_proj(self, hidden_state, unembedding_layer, device):
        scores = torch.nn.functional.softmax(unembedding_layer(hidden_state.to(device)), dim=-1).detach().cpu().numpy()
        return scores


    def __call__(self, image, text):
        conversation = [
            {
              "role": "user",
              "content": [
                  {"type": "text", "text": text},
                  {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=[image], text=prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]
        output = self.model.generate(**inputs, max_new_tokens=10, pad_token_id=self.model.config.eos_token_id, stop_strings=["[STOP]"], tokenizer=self.processor.tokenizer, output_hidden_states=True, return_dict_in_generate=True)
        output_tokens = output["sequences"][0, input_length:]
        output_text = self.processor.decode(output_tokens, skip_special_tokens=True)
        if self.vocab_projection_mode:
            # output['hidden_states'] is a tuple of tuples of torch tensors with overall shape: n_tokens_generated x n_layers x (1, either input_length or 1) x hidden_dim)
            # it is input_length only in the first i.e. token_generated = 0
            # Return should be in the form: n_tokens_generated x n_layers x |V|
            n_tokens_generated = len(output["hidden_states"])
            n_layers = len(output["hidden_states"][0])
            vocab_size = self.model.config.vocab_size
            unembedding_layer = self.model.language_model.lm_head
            hidden_dim = unembedding_layer.in_features
            # infer the device that eventually goes into the unembedding_layer by getting the device of the last tensor in the hidden states by layer
            device = output["hidden_states"][0][-1].device
            ret_array = np.zeros((n_layers, vocab_size)) # to start, only consider the first token predicted
            for i in range(1):
                for j in range(n_layers):
                    # ret_array[i, j] 
                    ret_array[j] = self.compute_vocab_proj(output["hidden_states"][i][j][0, -1], unembedding_layer, device)
            return output_text, ret_array
        elif self.hidden_state_tracking_mode:
            output_hidden_states = {}
            for i, layer in enumerate(self.layers_to_track):
                output_hidden_states[f"{layer}_last_input"] = output["hidden_states"][0][-1, layer].detach().cpu().numpy()  # TODO: This is certainly wrong. It tracks something else. Fix this line. 
                output_hidden_states[f"{layer}_last_output"] = output["hidden_states"][-1][0, layer].detach().cpu().numpy()  # TODO: This is certainly wrong. It tracks something else. Fix this line. 

            return output_text, output_hidden_states
        elif self.attention_tracking_mode:
            pass
        else:
            return output_text
    
    def __str__(self):
        return f"{self.variant}"
    

class BLIPInference:
    def __init__(self, variant="instructblip-vicuna-7b"):
        self.variant = variant
        self.processor = InstructBlipProcessor.from_pretrained(f"Salesforce/{variant}")
        self.model = InstructBlipForConditionalGeneration.from_pretrained(f"Salesforce/{variant}", device_map="auto")
        self.model.eval()

    def get_parallelized_model(self, variant):
        n_devices = torch.cuda.device_count()
        if n_devices == 0:
            raise ValueError("No GPU devices found.")
        if n_devices == 1:
           return InstructBlipForConditionalGeneration.from_pretrained(f"Salesforce/{variant}", device="cuda")
        else: # as a general rule put to vision_model on first device and split the language model across all other devices
            device_map = {
                "vision_model": 0,
                "qformer": 0,
                "language_projection": 0
            }
            # embed_tokens, norm, rotary_emb, lm_head
            distinct_modules = ["language_model.model.embed_tokens", "language_model.model.norm", "language_model.model.rotary_emb", "language_model.model.lm_head"]            
            n_layers = None
            if "7b" in variant:
                n_layers = 32
            elif "13b" in variant:
                n_layers = 40 # ?
            else:
                raise ValueError(f"Variant {variant} not recognized.")
            for i in range(n_layers):
                distinct_modules.append("language_model.model.layers." + str(i))
            remaining_devices = list(range(1, n_devices))
            # even split the layers across the remaining devices
            n_modules_per_device = len(distinct_modules) // len(remaining_devices)
            for i, module in enumerate(distinct_modules):
                device_map[module] = remaining_devices[i // n_modules_per_device]
            return InstructBlipForConditionalGeneration.from_pretrained(f"Salesforce/{variant}", device_map=device_map)




    def __call__(self, image, text):
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=100, stop_strings=["[STOP]"], tokenizer=self.processor.tokenizer)
        outputs = outputs[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(outputs[0], skip_special_tokens=True)
    
    def __str__(self):
        return f"{self.variant}"
    


class OpenAIInference:
    def __init__(self, variant="gpt-4o-mini"):
        assert variant in ["gpt-4o", "gpt-4o-mini"]
        self.client = OpenAI()
        self.variant = variant
        if not os.path.exists(openai_tmp_file_dir):
            os.makedirs(openai_tmp_file_dir)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    
    def convert_to_dict_line(self, image_text, max_tokens=10):
        # {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
        image, text = image_text
        image.save(os.path.join(openai_tmp_file_dir, "tmp_image.jpg"))
        image_base64 = self.encode_image(os.path.join(openai_tmp_file_dir, "tmp_image.jpg"))
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}, {"type": "text", "text": text}]}]
        d = {"method": "POST", "url": "/v1/chat/completions", "body": {"model": self.variant, "messages": messages, "max_tokens": max_tokens}}
        return d
    
    def read_batch_results(self, file):
        file_response = self.client.files.content(file)
        # each response will be a json string
        columns = ["idx", "response"]
        data = []
        for line in file_response.text.split("\n"):
            if line:
                d = json.loads(line)
                id = int(d["custom_id"].split("_")[1])
                response = d['response']['body']["choices"][0]["message"]["content"]
                if "[STOP]" in response:
                    response = response.split("[STOP]")[0].strip()
                data.append([id, response])
        df = pd.DataFrame(data, columns=columns)
        df = df.sort_values(by="idx").reset_index(drop=True)
        return df
    
    def get_batch_status(self, batch_name):
        if not os.path.exists(os.path.join(openai_tmp_file_dir, f"id_{batch_name}.txt")):
            return None
        with open(os.path.join(openai_tmp_file_dir, f"id_{batch_name}.txt"), "r") as f:
            batch_id = f.read()
        batch = self.client.batches.retrieve(batch_id)
        if batch.status == "completed":
            return 1
        else:
            return 0

    def get_batch_results(self, batch_name):
        with open(os.path.join(openai_tmp_file_dir, f"id_{batch_name}.txt"), "r") as f:
            batch_id = f.read()
        batch = self.client.batches.retrieve(batch_id)
        if batch.status == "completed":
            batch_file = batch.output_file_id
            return self.read_batch_results(batch_file)
        else:
            print(f"Batch {batch_name} is not completed. Returning None.")
            return None

    def __call__(self, image_texts, batch_name, ids=None):
        if os.path.exists(os.path.join(openai_tmp_file_dir, f"id_{batch_name}.txt")):
            print(f"Batch {batch_name} already exists. Returning results from file.")
            return self.get_batch_results(batch_name)
        # otherwise
        if image_texts is None:
            print(f"Got None for image_texts. Returning None.")
            return None
        if ids is not None:
            assert len(image_texts) == len(ids), "Length of image_texts and ids should be the same."
        requests = []
        for i, image_text in enumerate(image_texts):
            d = self.convert_to_dict_line(image_text)
            if ids is not None:
                d["custom_id"] = f"id_{ids[i]}"
            else:
                d["custom_id"] = f"id_{i}"
            requests.append(d)
        with open(os.path.join(openai_tmp_file_dir, f"{batch_name}.json"), "w") as f:
            for i, request_dict in enumerate(requests):
                json_string = json.dumps(request_dict)
                if i != len(requests) - 1:
                    f.write(json_string + '\n')
                else:
                    f.write(json_string)
        batch_input_file = self.client.files.create(
            file=open(os.path.join(openai_tmp_file_dir, f"{batch_name}.json"), "rb"),
            purpose="batch"
        )
         # run the batch
        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"VLM Inference Batch {batch_name}"
            }
        )
        # write the batch id to tmp
        with open(os.path.join(openai_tmp_file_dir, f"id_{batch_name}.txt"), "w") as f:
            f.write(batch.id)
        return None
    
    def __str__(self):
        return f"{self.variant}"





    
if __name__ == "__main__":
    pass

