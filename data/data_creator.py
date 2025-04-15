from PIL import Image
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error
from inference.vlms import get_vlm
from evaluation.metrics import inclusion


alph_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
def alph_to_num(alph):
    return alph_list.index(alph)

def num_to_alph(num):
    return alph_list[num]


class DataCreator():
    def __init__(self, dataset_name: str, all_class_names=None, parameters=None, mcq=False):
        self.all_class_names = all_class_names
        self.dataset_name = dataset_name.lower()
        self.qas = None
        self.validated_classes = None
        if parameters is None:
            self.parameters = load_parameters()
        else:
            self.parameters = parameters
        self.object_str = "object"
        self.mcq = mcq
        storage_dir = parameters["storage_dir"]
        self.dataset_path = os.path.join(storage_dir, "processed_datasets", self.dataset_name+"_mcq" if self.mcq else self.dataset_name)
        

    def get_random_images(self, class_name, n=10):
        """
        Returns a list of at most n random images labelled as class_name
        """
        raise NotImplementedError
    

    def get_class_samples(self, n_samples=10):
        """
        Returns a dictionary of the form:
        {
            "class_name": [sample1: Image, sample2: Image, ...]
        }
        """
        class_samples = {}
        for label in self.all_class_names:
            images = self.get_random_images(label, n_samples)
            class_samples[str(label)] = []
            for image in images:
                if not isinstance(image, Image.Image):
                    log_error(self.parameters["logger"], f"Image is not a PIL image. Image: {image}, type: {type(image)}")
                class_samples[str(label)].append(image)
        return class_samples
    

    def get_question_prefix(self, class_name: str = None):
        """
        Returns the instruction tuning or fewshot example prefix for the dataset. 
        For MCQ datasets this should be a randomized function to ensure that 
        there is no systematic bias in the correct option in the the fewshot examples

        Args:
        class_name (str): The class name to generate the prefix for. This is optional if you want to customize the prefix for each class

        There is a default value here, but you should override this function to provide a more dataset specific prefix

        Return:
        str: The prefix for the dataset
        """
        if not self.mcq:
            prefix = "Answer the questions with a short response. Do not state the name of the object in the image. \nWhat are swords made of?\nAnswer: steel [STOP]\n What is the capital of France?\nAnswer: Paris [STOP]\n"
        else:
            prefix = "Answer the questions with a short response. Do not state the name of the object in the image"
            answer_inds = []
            prompt_dets = []
            q = "What are swords made of?"
            correct_answer = "Steel"
            options_answers = ["Clay", "Plastic", "Paper", correct_answer]
            prompt_dets.append((q, correct_answer, options_answers))

            q = "What is the capital of France?"
            correct_answer = "Paris"
            options_answers = ["London", "Berlin", correct_answer, "Rome"]
            prompt_dets.append((q, correct_answer, options_answers))

            for q, correct_answer, options_answers in prompt_dets:
                while options_answers.index(correct_answer) in answer_inds:
                    np.random.shuffle(options_answers)
                options = [f"{num_to_alph(i)}: {option}" for i, option in enumerate(options_answers)]
                options_str = "\n".join(options)
                answer_ind = options_answers.index(correct_answer)
                answer = f"{num_to_alph(answer_ind)}: {correct_answer}"
                answer_inds.append(answer_ind)
                prefix = prefix + f"\nQuestion: {q}\n{options_str}\nAnswer: {answer} [STOP]\n"
        return prefix

    
    def get_explicit_stating_question_prefix(self, class_name: str = None):
        """
        Returns the instruction tuning or fewshot example prefix for the dataset.
        This is the prefix for the explicit stating answer type i.e. 
        """
        prefix = "First identify the object in the image, and then answer the question. "
        return prefix


    def get_identification_prefix(self, class_name):
        """
        Returns a string to identify the image. Can be inherited and adapted to have class_name specific prompt
        """
        return f"Identify the object in the image. \nAnswer: "

    
    def validate_classes(self, vlm_name="llava-v1.6-vicuna-13b-hf", validation_threshold=0.2, limited_sample_warning=10):
        """
        Uses a VLM to judge whether the class can be identified from the images in the dataset. 
        If less than validation_threshold of the images are identified correctly, the class is not validated.
        We use this as a check to avoid using classes that seem like they will fail our eventual checks later on (to save compute)

        If you want to bypass this, you can inherit and override this function to just set self.validated_classes to all the classes.
        """
        dataset_path = os.path.join(self.parameters["storage_dir"], "processed_datasets", self.dataset_name) # This one isn't different for MCQ so its ok
        validated_classes_path = os.path.join(dataset_path, "validated_classes.pkl")
        if os.path.exists(validated_classes_path):
            return self.load_validated_classes()
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        vlm = get_vlm(vlm_name)
        class_samples = self.get_class_samples(n_samples=10)
        self.validated_classes = []
        for class_name in tqdm(class_samples):
            identification_prompt = self.get_identification_prefix(class_name)
            success = []
            if len(class_samples[class_name]) < limited_sample_warning:
                self.parameters["logger"].warning(f"On dataset {self}, class {class_name} has less than {limited_sample_warning} samples. Validation may not be accurate.")
            for sample in tqdm(class_samples[class_name], desc=f"Running validation for class {class_name}"): 
                response = vlm(sample, identification_prompt)
                success.append(inclusion(response["text"], class_name))
            self.parameters["logger"].info(f"Class {class_name} has success rate {(100*np.mean(success))}% success rate.")
            if np.mean(success) > validation_threshold:
                self.validated_classes.append(class_name)
        with open(validated_classes_path, "wb") as f:
            pickle.dump(self.validated_classes, f)
        return self.validated_classes

    def check_class_validation(self):
        if self.validated_classes is None:
            validated_classes_path = os.path.join(self.parameters["storage_dir"], "processed_datasets", self.dataset_name, "validated_classes.pkl")
            if os.path.exists(validated_classes_path):
                self.validated_classes = self.load_validated_classes()
            else:
                self.parameters["logger"].warning("Classes not validated. Running class validation now. This will take a long time and may not work if you are running without a GPU")
                self.validate_classes()


    def load_validated_classes(self):
        """
        Loads the validated classes from the validated_classes.pkl file
        """
        dataset_path = os.path.join(self.parameters["storage_dir"], "processed_datasets", self.dataset_name)
        validated_classes_path = os.path.join(dataset_path, "validated_classes.pkl")
        if not os.path.exists(validated_classes_path):
            log_error(self.parameters["logger"], f"Validated classes for {self.dataset_name} do not exist. Run validate_classes to generate them.")
        with open(validated_classes_path, "rb") as f:
            self.validated_classes = pickle.load(f)
        return self.validated_classes

    def get_qa_path(self):
        """
        Returns the qa path to the question answer pairs, errors out if dne
        """
        parameters = self.parameters
        storage_dir = parameters["storage_dir"]
        dataset_path = os.path.join(storage_dir, "processed_datasets", self.dataset_name) # This one isn't different for MCQ so its ok
        qa_path = os.path.join(dataset_path, "qas_deduplicated.json")
        generated_qa_path = os.path.join(dataset_path, "qas_generated.json")
        validated_qa_path = os.path.join(dataset_path, "qas_validated.json")
        mcqa_path = os.path.join(dataset_path, "mcqas_deduplicated.json")
        paths = {}
        paths["deduplicated"] = (qa_path)
        paths["generated"] = (generated_qa_path)
        paths["validated"] = (validated_qa_path)
        paths["mcqa"] = (mcqa_path)
        file_exists = False
        if self.mcq:
            file_exists = os.path.exists(mcqa_path)
        else:
            file_exists = os.path.exists(qa_path)
        if not file_exists:
            if self.mcq and os.path.exists(qa_path):
                log_error(parameters["logger"], f"QA pairs for {self.dataset_name} MCQ variant do not exist, but deduplicated data does exist at {qa_path}. Run the generate_mcq_questions command first.")
            else:
                if os.path.exists(validated_qa_path):
                    log_error(parameters["logger"], f"QA pairs for {self.dataset_name} do not exist, but validated data does exist at {validated_qa_path}. Run the generate_questions command first.")
                elif os.path.exists(generated_qa_path):
                    log_error(parameters["logger"], f"QA pairs for {self.dataset_name} do not exist, but generated data does exist at {generated_qa_path}. Run the validate_questions command first.")
                else:
                    log_error(parameters["logger"], f"QA pairs for {self.dataset_name} do not exist, validated data and generated_data does not exist either. Run generate_questions to generate them and then validate_questions to validate them and finally deduplicate_questions to deduplicate them.")
        
        if self.mcq:
            return mcqa_path
        else:
            return qa_path
    


    def load_qas(self):
        """
        Internally loads the validated qa pairs for the dataset in the form:
        {
            "class_name": [
                {
                'question': question1,
                'options': None or [option1, option2, option3, option4...],
                'answer': answer1 (equal to one of the options if options is not None),
                'status': status (approved, unique, not unique, etc.),
                'source': source (where the question was generated from, e.g. manual)
                }, ...
            ]
        }

        question is either a short form or MCQ question with the options 
        """
        self.check_class_validation()
        qa_path = self.get_qa_path()
        with open(qa_path, "r") as f:
            qas = json.load(f)
        # qas is a dictionary of the form 
        # {"class_name": [{"question": question1, "options": None or option_list, "answer": answer1, "status": status, "source": source}, ...]}
        # we want to select only ones where the status is approved
        self.qas = {}
        for class_name, qa_list in qas.items():
            if class_name not in self.validated_classes:
                continue
            self.qas[class_name] = [qa for qa in qa_list if qa["status"] == "accepted"]
        return self.qas


    def get_qa_strings(self, class_name):
        """
        To be used in the dataset creation for randomization of MCQ choice and question order. Will add the suffix to the question and options.

        If the dataset does not have MCQ questions, this will just return the question answer pairs. 
        Returns 
        [   
            {'question': question1, 'answer': answer1, 'source': source1},
            ...
        ]
        """
        np.random.seed(self.parameters["random_seed"])
        if self.qas is None:
            self.load_qas()
        class_qas = self.qas[class_name]
        qa_strings = []
        for qa in class_qas:
            qa_string = qa["question"]
            answer = qa["answer"]
            source = qa["source"]            
            if "options" not in qa:
                qa_string = qa_string + "\nAnswer: "
            else:
                qa_string = qa_string + "\nOptions: "
                options = qa["options"]
                np.random.shuffle(options)
                answer_index = options.index(answer)
                for i, option in enumerate(options):
                    qa_string = qa_string + f"\n{num_to_alph(i)}: {option}"
                qa_string = qa_string + "\nAnswer: "
            qa_strings.append({"question": qa_string, "answer": answer, "source": source})
        return qa_strings
    

    def save_data(self, target_datapoints=1000):
        """
        Saves the dataset to the processed_datasets folder

        target_datapoints is a variable that controls how many images we load in per question per class. 
        This will adjust to keep classes balanced in the benchmark, but will always overshoot. 
        """
        self.check_class_validation()
        parameters = load_parameters()
        dataset_path = self.dataset_path # This one is different for MCQ and normal QA
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        image_path = os.path.join(dataset_path, "images")
        if not os.path.exists(image_path):
            os.makedirs(image_path)  
        # saves a csv with columns: class_name, question_str, answer_str, question_source
        columns = ["class_name", "question_str", "answer_str", "question_source", "image_path"]
        data = []
        n_classes = len(self.validated_classes) 
        failed_classes = []
        for class_name in self.validated_classes:
            n_questions = len(self.get_qa_strings(class_name))
            if n_questions == 0:
                failed_classes.append(class_name)
        if len(failed_classes) > 0:
            parameters["logger"].warning(f"Some classes have no questions. These classes will be skipped: {failed_classes}")
        real_n_classes = n_classes - len(failed_classes)
        if real_n_classes == 0:
            log_error(parameters["logger"], f"All classes have failed. Please check the validation process.")
        target_datapoints_per_class = (target_datapoints // real_n_classes) + 1
        image_counter = 0
        for class_name in tqdm(self.validated_classes):
            qa_strings = self.get_qa_strings(class_name)
            n_questions = len(qa_strings)
            if n_questions == 0:
                continue
            images_per_question = (target_datapoints_per_class // n_questions) + 1
            for qa in qa_strings:
                question = qa["question"]
                answer = qa["answer"]
                source = qa["source"]
                image_samples = self.get_random_images(class_name, images_per_question)
                for image in image_samples:
                    image_file_path = os.path.join(image_path, f"{image_counter}.png")
                    data.append([class_name, question, answer, source, image_file_path])
                    image.save(image_file_path)
                    image_counter += 1
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join(dataset_path, "data.csv"), index=False)


    def create_validated_data(self, target_datapoints=1000):
        """
        Creates the dataset in the format that the model can use. 
        """
        self.check_class_validation()
        self.load_qas()
        self.save_data(target_datapoints=target_datapoints)
        self.parameters["logger"].info(f"Dataset {self.dataset_name} created and saved to processed_datasets folder.")
        return
    
    def get_data_df(self):
        """
        Returns the data csv for the dataset
        """
        dataset_path = self.dataset_path
        data_csv_path = os.path.join(dataset_path, "data.csv")
        if not os.path.exists(data_csv_path):
            log_error(self.parameters["logger"], f"Data csv for {self.dataset_name} does not exist.")
        return pd.read_csv(data_csv_path)


    def __str__(self):
        return self.dataset_name
    

    def __repr__(self):
        return f"DataCreator({self.dataset_name})"
    

