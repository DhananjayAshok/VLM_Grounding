from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration, AutoTokenizer
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error
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

def get_vlm(name, hidden_state_tracking_mode=False, vocab_projection_mode=False, attention_tracking_mode=False, parameters=None):
    if parameters is None:
        parameters = load_parameters()
    if "llava" in name:
        return LlaVaInference(variant=name, vocab_projection_mode=vocab_projection_mode, hidden_state_tracking_mode=hidden_state_tracking_mode, attention_tracking_mode=attention_tracking_mode)
    elif "instructblip" in name:
        return BLIPInference(variant=name)
    elif "gpt" in name:
        if hidden_state_tracking_mode or vocab_projection_mode or attention_tracking_mode:
            log_error(parameters['logger'], "OpenAI does not support hidden state tracking, vocab projection or attention tracking.")
        return OpenAIInference(variant=name, parameters=parameters)
    else:
        raise ValueError(f"Model {name} not recognized.")

def compute_vocab_proj(hidden_state, unembedding_layer, device):
    scores = unembedding_layer(hidden_state.to(device))
    return scores

def compute_softmax(logits):
    probs = torch.nn.functional.softmax(torch.Tensor(logits), dim=-1).detach().cpu().numpy()
    return probs


def kl_divergence(logits_p, logits_q):
    dist1 = torch.distributions.Categorical(logits=torch.Tensor(logits_p))
    dist2 = torch.distributions.Categorical(logits=torch.Tensor(logits_q))
    kl_div = torch.distributions.kl.kl_divergence(dist1, dist2)
    return kl_div.item()


def forward_kl(vocab_array):
    # vocab_array shape is n_layers, vocab_size
    kl_divs = [] # this is a n_layers - 1 length array
    for i in range(1, vocab_array.shape[0]):
        kl_divs.append(kl_divergence(vocab_array[i-1], vocab_array[i]))
    return np.array(kl_divs)


class HuggingFaceInference:    
    def generate(self, inputs, entity=None, max_new_tokens=10):
        input_length = inputs["input_ids"].shape[1]
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.model.config.eos_token_id, stop_strings=["[STOP]"], tokenizer=self.processor.tokenizer, output_hidden_states=True, return_dict_in_generate=True, output_scores=True)
        output_tokens = output["sequences"][0, input_length:]
        output_text = self.processor.decode(output_tokens, skip_special_tokens=True)
        if "blip" in str(self.variant):
            perplexity = None
        else:
            transition_scores = self.model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
            perplexity = transition_scores.mean().item()
        response = {
            "text": output_text,
            "perplexity": perplexity,
        }
        if self.vocab_projection_mode:
            # output['hidden_states'] is a tuple of tuples of torch tensors with overall shape: n_tokens_generated x n_layers x (1, either input_length or 1) x hidden_dim)
            # it is input_length only in the first i.e. token_generated = 0
            # Return should be in the form: n_tokens_generated x n_layers x |V|
            n_tokens_generated = len(output["hidden_states"])
            n_layers = len(output["hidden_states"][0])
            vocab_size = self.model.config.vocab_size
            unembedding_layer = self.model.language_model.lm_head # TODO: Check that this is the same for InstructBLIP
            hidden_dim = unembedding_layer.in_features
            # infer the device that eventually goes into the unembedding_layer by getting the device of the last tensor in the hidden states by layer
            device = output["hidden_states"][0][-1].device
            ret_array = np.zeros((n_layers, vocab_size)) # to start, only consider the first token predicted
            for i in range(1):
                for j in range(n_layers):
                    # ret_array[i, j] 
                    ret_array[j] = compute_vocab_proj(output["hidden_states"][i][j][0, -1], unembedding_layer, device).detach().cpu().numpy()
            # compute the KL Divergence and the probabilities here itself. 
            div_array = forward_kl(ret_array)
            # compute the softmax
            probs = compute_softmax(ret_array)
            first_token_id = output["sequences"][0, input_length].item()
            # get the probs for the first token
            first_token_probs = probs[:, first_token_id]
            response["kl_divergence"] = div_array
            response["projection_prob"] = first_token_probs
            response["total_projection"] = probs
            
        if self.hidden_state_tracking_mode:
            # look for the word object and the entity word in the input (this is from image reference questions is the assumption) and get the vocab distribution for that:
            look_for_words = ["object", "image", "entity"]
            default_start_len = len(self.processor.tokenizer.encode(""))
            look_indexes = {}
            input_ids = inputs["input_ids"][0]
            for kind in look_for_words:
                if entity is None and kind == "entity":
                    look_indexes["entity"] = None
                    continue
                word = kind if kind != "entity" else str(entity)
                encoded_tokens = self.processor.tokenizer.encode(word)[default_start_len:]
                # get the last token id of the word in the input
                track_token = encoded_tokens[-1] # not really sure about this.
                # find the index of the first occurance of this in the input:
                index_int = (input_ids == track_token).nonzero(as_tuple=True)[0]
                if len(index_int) > 0:
                    index = index_int[0].item()
                else:
                    index = None
                look_indexes[kind] = index
            output_hidden_states = {}
            for i, layer in enumerate(self.layers_to_track):
                output_hidden_states[f"{layer}_last_input"] = output["hidden_states"][0][layer][0, -1].detach().cpu().numpy()
                output_hidden_states[f"{layer}_last_output"] = output["hidden_states"][-1][layer][0, -1].detach().cpu().numpy()
                for kind in look_indexes:
                    if look_indexes[kind] is not None:
                        output_hidden_states[f"{layer}_{look_indexes[kind]}_{kind}"] = output["hidden_states"][0][layer][0, look_indexes[kind]].detach().cpu().numpy() # TODO: Need to check this
                    else:
                        output_hidden_states[f"{layer}_None_{kind}"] = None
            response["hidden_states"] = output_hidden_states
        elif self.attention_tracking_mode:
            pass
        return response


class LlaVaInference(HuggingFaceInference):
    def __init__(self, variant="llava-v1.6-mistral-7b-hf", vocab_projection_mode=False, hidden_state_tracking_mode=False, attention_tracking_mode=False):
        super().__init__()
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
            n_layers = self.model.language_model.config.num_hidden_layers
            layers_to_track = list(range(1, n_layers-1, 3))
            self.layers_to_track = layers_to_track
        

    def __call__(self, image, text, entity=None, max_new_tokens=10):
        conversation = [
            {
              "role": "user",
              "content": [
                  {"type": "text", "text": text},
                ],
            },
        ]
        if image is not None:
            conversation[0]['content'].append({"type": "image"})
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        if image is None:
            images=None
        else:
            images = [image]
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(self.model.device)
        return self.generate(inputs, entity=entity, max_new_tokens=max_new_tokens)
    
    def __str__(self):
        return f"{self.variant}"
    

class BLIPInference(HuggingFaceInference):
    def __init__(self, variant="instructblip-vicuna-7b", vocab_projection_mode=False, hidden_state_tracking_mode=False, attention_tracking_mode=False):
        super().__init__()
        self.variant = variant
        self.processor = InstructBlipProcessor.from_pretrained(f"Salesforce/{variant}")
        self.model = InstructBlipForConditionalGeneration.from_pretrained(f"Salesforce/{variant}", device_map="auto")
        self.model.eval()
        subvariant_name = "-".join(variant.split("-")[1:])
        self.lm_tokenizer = AutoTokenizer.from_pretrained(f"lmsys/{subvariant_name}-v1.5")
        self.vocab_projection_mode = vocab_projection_mode
        self.hidden_state_tracking_mode = hidden_state_tracking_mode
        self.attention_tracking_mode = attention_tracking_mode
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


    def __call__(self, image, text, entity=None, max_new_tokens=10):
        if image is None:
            if self.vocab_projection_mode or self.hidden_state_tracking_mode:
                raise ValueError("Image must be provided for vocab projection and hidden state tracking. Also just don't try this with instructblip but whatever")
            input_text = self.lm_tokenizer(text, return_tensors="pt").to(self.model.device)
            text_output = self.model.language_model.generate(**input_text, max_new_tokens=max_new_tokens)
            output_text = self.lm_tokenizer.decode(text_output[0], skip_special_tokens=True)
            response = {
                "text": output_text,
                "perplexity": None,
            }
            return response
        else:
            #q_max = self.model.qformer.config.max_position_embeddings  # usually 512
            #inputs = self.processor(images=image, text=text, truncation=True, padding="max_length", return_tensors="pt", max_length=q_max).to(self.model.device)
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.model.device)
            return self.generate(inputs, entity=entity, max_new_tokens=max_new_tokens)
    
    def __str__(self):
        return f"{self.variant}"
    


class OpenAIInference:
    def __init__(self, variant="gpt-4o-mini", parameters=None):
        assert variant in ["gpt-4o", "gpt-4o-mini"]
        if parameters is None:
            parameters = load_parameters()
        self.parameters = parameters
        self.client = OpenAI()
        self.variant = variant
        if not os.path.exists(openai_tmp_file_dir):
            os.makedirs(openai_tmp_file_dir)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    
    def convert_to_dict_line(self, image_text, max_new_tokens=10):
        # {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
        image, text = image_text
        if image is not None:
            image.save(os.path.join(openai_tmp_file_dir, "tmp_image.jpg"))
            image_base64 = self.encode_image(os.path.join(openai_tmp_file_dir, "tmp_image.jpg"))
            messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}, {"type": "text", "text": text}]}]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        d = {"method": "POST", "url": "/v1/chat/completions", "body": {"model": self.variant, "messages": messages, "max_tokens": max_new_tokens}}
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
        try:
            batch = self.client.batches.retrieve(batch_id)
            if batch.status == "completed":
                return 1
            else:
                return 0
        except:
            return None

    def get_batch_results(self, batch_name):
        with open(os.path.join(openai_tmp_file_dir, f"id_{batch_name}.txt"), "r") as f:
            batch_id = f.read()
        batch = self.client.batches.retrieve(batch_id)
        if batch.status == "completed":
            batch_file = batch.output_file_id
            return self.read_batch_results(batch_file)
        else:
            self.parameters['logger'].warning(f"Batch {batch_name} is not completed. Returning None.")
            return None

    def __call__(self, image_texts, batch_name, ids=None, max_new_tokens=10):
        if os.path.exists(os.path.join(openai_tmp_file_dir, f"id_{batch_name}.txt")):
            self.parameters['logger'].info(f"Batch {batch_name} already exists. Returning results from file.")
            return self.get_batch_results(batch_name)
        # otherwise
        if image_texts is None:
            self.parameters['logger'].warning(f"Got None for image_texts. Returning None.")
            return None
        if ids is not None:
            assert len(image_texts) == len(ids), "Length of image_texts and ids should be the same."
        requests = []
        for i, image_text in enumerate(image_texts):
            d = self.convert_to_dict_line(image_text, max_new_tokens=max_new_tokens)
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

