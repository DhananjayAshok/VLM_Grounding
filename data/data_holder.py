
from PIL import Image
import os
import json
import numpy as np
import pandas as pd
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
    def __init__(self, dataset_name: str, parameters=None):
        self.dataset_name = dataset_name.lower()
        self.qas = None
        self.validated_classes = None
        if parameters is None:
            self.parameters = load_parameters()
        else:
            self.parameters = parameters
        

    def get_class_samples(self):
        """
        Returns a dictionary of the form:
        {
            "class_name": [sample1: Image, sample2: Image, ...]
        }
        class_names should be every class / label in the dataset (in string form, for identification by VLM)

        You should try to have at least 10 samples. 
        """
        raise NotImplementedError # TODO: Implement this function in your individual datasets
    

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
        prefix = "Answer the questions with a short response. Do not state the name of the object in the image. \nWhat are swords made of?\nAnswer: steel [STOP]\n What is the capital of France?\nAnswer: Paris [STOP]\n"
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

        If you want to bypass this, you can inherit and override this function to just set self.validated_classes to all the classes
        """
        vlm = get_vlm(vlm_name)
        class_samples = self.get_class_samples()
        self.validated_classes = []
        for class_name in class_samples:
            identification_prompt = self.get_identification_prefix(class_name)
            success = []
            if len(class_samples[class_name]) < limited_sample_warning:
                self.parameters["logger"].warning(f"On dataset {self}, class {class_name} has less than {limited_sample_warning} samples. Validation may not be accurate.")
            for sample in class_samples[class_name]:
                response = vlm(sample, identification_prompt)
                success.append(inclusion(response, class_name))
            if np.mean(success) > validation_threshold:
                self.validated_classes.append(class_name)
        return self.validated_classes

    def check_class_validation(self):
        if self.validated_classes is None:
            self.parameters["logger"].warning("Classes not validated. Running class validation now. This will take a long time and may not work if you are running without a GPU")
            self.validate_classes()


    def get_random_images(self, class_name, n=5):
        """
        Returns a list of at most n random images labelled as class_name
        """
        raise NotImplementedError
    

    def load_qas(self):
        """
        Internally loads the validated qa pairs for the dataset in the form:
        {
            "class_name": [
                {
                'question': question1,
                'options': None or [option1, option2, option3, option4...],
                'answer': answer1 (equal to one of the options if options is not None),
                }, ...
            ]
        }

        question is either a short form or MCQ question with the options 
        """
        self.check_class_validation()
            
        parameters = load_parameters()
        storage_dir = parameters["storage_dir"]
        dataset_path = os.path.join(storage_dir, "processed_datasets", self.dataset_name)
        qa_path = os.path.join(dataset_path, "qa_validated.json")
        unvalidated_qa_path = os.path.join(dataset_path, "qa_unvalidated.json")
        if not os.path.exists(qa_path):
            if os.path.exists(unvalidated_qa_path):
                log_error(parameters["logger"], f"Validated QA pairs for {self.dataset_name} do not exist but unvalidated qa data exists.") # TODO: Add a message explaining how to validate the data
            log_error(parameters["logger"], f"QA pairs for {self.dataset_name} do not exist, validated data does not exist either. Run setup_data to generate them.")

        with open(qa_path, "r") as f:
            qas = json.load(f)
        # qas is a dictionary of the form 
        # {"class_name": [{"question": question1, "options": None or option_list, "answer": answer1, "status": status, "source": source}, ...]}
        # we want to select only ones where the status is approved
        self.qas = {}
        for class_name, qa_list in qas.items():
            if class_name not in self.validated_classes:
                continue
            self.qas[class_name] = [qa for qa in qa_list if qa["status"] == "approved"]
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
            options = qa["options"]
            if options is None:
                qa_string = qa_string + "\nAnswer: "
            else:
                np.random.shuffle(options)
                answer_index = options.index(answer)
                for i, option in enumerate(options):
                    qa_string = qa_string + f"{num_to_alph(i)}: {option}"
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
        storage_dir = parameters["storage_dir"]
        dataset_path = os.path.join(storage_dir, "processed_datasets", self.dataset_name)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)        
        # saves a csv with columns: class_name, question_str, answer_str, question_source
        columns = ["class_name", "question_str", "answer_str", "question_source", "image_path"]
        data = []
        n_classes = len(self.validated_classes) 
        target_datapoints_per_class = (target_datapoints // n_classes) + 1
        image_counter = 0
        for class_name in self.validated_classes:
            qa_strings = self.get_qa_strings(class_name)
            n_questions = len(qa_strings)
            images_per_question = (target_datapoints_per_class // n_questions) + 1
            for qa in qa_strings:
                question = qa["question"]
                answer = qa["answer"]
                source = qa["source"]
                image_samples = self.get_random_images(class_name, images_per_question)
                for image in image_samples:
                    image_path = os.path.join(dataset_path, "images", f"{image_counter}.jpg")
                    data.append([class_name, question, answer, source, image_path])
                    image.save(image_path)
                    image_counter += 1
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join(dataset_path, "data.csv"), index=False)


    def create_validated_data(self, target_datapoints=1000):
        """
        Creates the dataset in the format that the model can use. 
        """
        self.validate_classes()
        self.load_qas()
        self.save_data(target_datapoints=target_datapoints)
        self.parameters["logger"].info(f"Dataset {self.dataset_name} created and saved to processed_datasets folder.")
        return


    def __str__(self):
        return self.dataset_name
    

    def __repr__(self):
        return f"DataCreator({self.dataset_name})"
    

class DataHolder:
    def __init__(self, dataset_name: str, parameters=None):
        self.dataset_name = dataset_name
        if parameters is None:
            self.parameters = load_parameters()
        else:
            self.parameters = parameters
        data_df_path = os.path.join(parameters["storage_dir"], "processed_datasets", dataset_name, "data.csv")
        self.data_df = pd.read_csv(data_df_path)
        self.data_creator = DataCreator(dataset_name, parameters)

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.loc[idx]
        image = Image.open(row["image_path"])
        question_prefix = self.data_creator.get_question_prefix(row["class_name"])
        explicit_prefix = self.data_creator.get_explicit_stating_question_prefix(row["class_name"])
        identification_prefix = self.data_creator.get_identification_prefix(row["class_name"])
        full_information_question = question_prefix + " " + row["question_str"]
        reference_question_str = row["question_str"].replace(row["class_name"], "the object in the image")
        explicit_question = explicit_prefix + " " + reference_question_str
        image_reference_question = question_prefix + " " + reference_question_str

        data = {
            "image": image,
            "class_name": row["class_name"],
            "answer": row["answer_str"],
            "source": row["question_source"],
            "full_information_question": full_information_question,
            "image_reference_question": image_reference_question,
            "explicit_question": explicit_question,
            "identification_question": identification_prefix
        }
        return data
    
    def __str__(self):
        return self.dataset_name
    
    def __repr__(self):
        return str(self)