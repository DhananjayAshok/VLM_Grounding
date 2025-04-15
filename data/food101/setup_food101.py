import torchvision
import numpy as np

from data.food101.label_mapping import fine_labels
from data.data_creator import DataCreator
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error


def load_food101(raw_data_path):
    dset = torchvision.datasets.Food101(root=f"{raw_data_path}/food101", download=True, split="test")
    labels = []
    for i in range(len(dset)):
        img, label = dset[i]
        index = fine_labels[label]
        if index == -1:
            raise ValueError(f"Label {label} not found in fine_labels")
        labels.append(index)
    labels = np.array(labels)
    return dset, labels

            

class Food101Creator(DataCreator):
    def __init__(self, parameters=None, mcq=False):
        if parameters is None:
            parameters = load_parameters()
        super().__init__("food101", all_class_names=fine_labels, parameters=parameters, mcq=mcq)
        self.dset, self.labels = load_food101(raw_data_path=parameters["data_dir"]+"/raw/")

    def get_identification_prefix(self, class_name):
        return f"There are many food items in the world (e.g. pupusa, kimchi jjiggae, scrambled eggs) the object in the image is a food item, what is its name? \nAnswer: "

    
    def get_random_images(self, class_name, n=10):
        """
        Get n random images from the dataset
        """
        label_samples = np.where(self.labels == class_name)[0]
        label_samples = np.random.choice(label_samples, size=n, replace=(n > len(label_samples)))
        images = []
        for sample_ind in label_samples:
            image = self.dset[sample_ind][0]
            images.append(image)
        return images    