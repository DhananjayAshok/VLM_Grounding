import torchvision
import numpy as np

from data.landmarks.label_mapping import label_spaces
from data.data_creator import DataCreator
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error


def load_landmarks(raw_data_path):
    dset = torchvision.datasets.ImageFolder(root=f"{raw_data_path}/landmark_images", transform=torchvision.transforms.ToTensor())
    labels = [label[3:] for label in dset.classes]

    return dset, labels

class LandmarksCreator(DataCreator):
    def __init__(self, parameters=None, mcq=False):
        if parameters is None:
            parameters = load_parameters()
        super().__init__("landmarks", all_class_names=label_spaces, parameters=parameters, mcq=mcq)
        self.dset, self.labels = load_landmarks(raw_data_path=parameters["data_dir"]+"/raw/")

    def get_identification_prefix(self, class_name):
        return f"There are many famous landmarks in the world (e.g. Statue of Liberty, Yosemite National Park, Mount Fuji). What landmark is pictured in the image? \nAnswer: "

    
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

