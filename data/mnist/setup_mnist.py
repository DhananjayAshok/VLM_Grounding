import torchvision
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error
from data.data_holder import DataCreator
import numpy as np
import json
import os


def load_mnist(raw_data_path):
    dset = torchvision.datasets.MNIST(root=f"{raw_data_path}/mnist", download=True, train=False, transform=torchvision.transforms.ToTensor())
    labels = []
    for i in range(len(dset)):
        img, label = dset[i]
        labels.append(label)
    labels = np.array(labels)
    return dset, labels



def get_random_question(class_name):
    random_operation = np.random.choice(["+", "*"])
    random_other_number = np.random.randint(0, 100)
    question = f"What is {class_name} {random_operation} {random_other_number}?"
    answer = None
    if random_operation == "+":
        answer = int(class_name) + random_other_number
    elif random_operation == "*":
        answer = int(class_name) * random_other_number
    return question, answer


def create_mnist_qas(n_questions_per_class=10, parameters=None):
    if parameters is None:
        parameters = load_parameters()
    np.random.seed(parameters["random_seed"])
    storage_dir = parameters["storage_dir"]
    class_qas = {}
    for class_name in range(10):
        qas = []
        for i in range(n_questions_per_class):
            question, answer = get_random_question(class_name)
            d = {
                "question": question, 
                "answer": answer, 
                "status": "approved",
                "source": "manual"
                }
            qas.append(d)
            dataset_path = os.path.join(storage_dir, "processed_datasets", "mnist")
            qa_path = os.path.join(dataset_path, "qa_validated.json")
        class_qas[str(class_name)] = qas
    with open(qa_path, "w") as f:
        json.dump(class_qas, f)
    return class_qas
            

class MNISTCreator(DataCreator):
    def __init__(self):
        parameters = load_parameters()
        super().__init__("mnist", parameters)
        self.dset, self.labels = load_mnist(raw_data_path=parameters["data_dir"]+"/raw/")

    def get_class_samples(self):
        class_samples = {}
        all_labels = list(set(self.labels))
        for label in all_labels:
            label_samples = np.where(self.labels == label)[0]
            label_samples = np.random.choice(label_samples, size=10, replace=False)
            images = []
            for sample_ind in label_samples:
                image = self.dset[sample_ind][0]
                images.append(image)
            class_samples[str(label)] = images
        return class_samples


    def get_question_prefix(self, class_name: str = None):
        """
        For MNIST specifically there is a risk of error because the example might have the class name digit in it. 
        You most likely will not have to worry about this in other datasets.
        """
        example_1 = "What is 2 + 4\nAnswer: 6"
        example_2 = "What is 3 + 5\nAnswer: 8"
        example_3 = "What is 0 + 7\nAnswer: 7"
        prefix = "Answer the questions with a short response. Do not state the name of the digit in the image."
        example_contaminated = [str(class_name) in example for example in [example_1, example_2, example_3]]
        if sum(example_contaminated) > 1:
            log_error(self.parameters["logger"], f"Bro wat {class_name}, {example_contaminated}")
        else:
            counter = 0
            for i, example in enumerate([example_1, example_2, example_3]):
                if example_contaminated[i] or counter >= 2:
                    pass
                else:
                    counter += 1
                    prefix = prefix + f"{example} [STOP]\n"
        return prefix
    
    def get_explicit_stating_question_prefix(self, class_name: str = None):
        """
        For MNIST I specify that the object is a digit, but be careful not to help the model too much here. 
        """
        prefix = "First identify the digit in the image, and then answer the question. "
        return prefix
    
    def get_identification_prefix(self, class_name):
        return f"Identify the digit in the image. \nAnswer: "
    

if __name__ == "__main__":
    create_mnist_qas(n_questions_per_class=10)