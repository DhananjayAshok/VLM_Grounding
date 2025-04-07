
from PIL import Image
import os
import pandas as pd
from utils.parameter_handling import load_parameters

from data.mnist.setup_mnist import MNISTCreator
from data.cifar100.setup_cifar import CIFAR100Creator


def get_data_creator(dataset_name, parameters=None):
    """
    Returns the data creator object for the given dataset name
    """
    if "mcq" in dataset_name:
        dataset_name, mcq_str = dataset_name.split("_")
        mcq = True
    else:
        mcq = False
    if dataset_name == "mnist":
        return MNISTCreator(parameters=parameters, mcq=mcq)
    elif dataset_name == "cifar100":
        return CIFAR100Creator(parameters=parameters, mcq=mcq)
    elif dataset_name == "food101":
        pass
    elif dataset_name == "landmarks":
        pass
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


class DataHolder:
    def __init__(self, dataset_name: str, parameters=None, mcq=False):
        self.dataset_name = dataset_name
        if parameters is None:
            self.parameters = load_parameters()
        else:
            self.parameters = parameters
        data_df_path = os.path.join(parameters["storage_dir"], "processed_datasets", dataset_name + "_mcq" if mcq else dataset_name, "data.csv")
        self.data_df = pd.read_csv(data_df_path)
        self.data_creator = get_data_creator(dataset_name, parameters, mcq=mcq)
        self.mcq = mcq

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.loc[idx]
        image = Image.open(row["image_path"])
        question_prefix = self.data_creator.get_question_prefix(row["class_name"])
        explicit_prefix = self.data_creator.get_explicit_stating_question_prefix(row["class_name"])
        identification_prefix = self.data_creator.get_identification_prefix(row["class_name"])
        full_information_question = question_prefix + " " + row["question_str"]
        reference_question_str = row["question_str"].replace(str(row["class_name"]), f"the {self.data_creator.object_str} in the image")
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
        return self.dataset_name + "_mcq" if self.mcq else self.dataset_name
    
    def __repr__(self):
        return str(self)