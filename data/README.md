# Adding a New Image Classification Dataset

This guide outlines the process for integrating a new image classification dataset into the experiment framework. Follow these steps carefully to ensure compatibility.

---

## Process Overview

The process involves creating a dedicated data handler for your new dataset and then registering it with the central data factory function.

---

## Step-by-Step Instructions

1.  **Create a New Dataset Submodule:**
    * Navigate to the main `data` module directory within the project structure.
    * Create a new Python file (e.g., `your_dataset_name.py`) within the `data` directory. This file will contain the logic specific to your dataset.
    * Inside this new file, define a class for your dataset. This class **must inherit** from the `DataCreator` base class located in [`data/data_creator.py`](). 

For [example](data/cifar100/setup_cifar.py), CIFAR100 is set up as

```python
# ...
class  CIFAR100Creator(DataCreator):
	def  __init__(self, parameters=None, mcq=False):
		if  parameters  is  None:
		    parameters  =  load_parameters()
	    super().__init__("cifar100",  all_class_names=fine_labels,  parameters=parameters, mcq=mcq)
		self.dset,  self.labels  =  load_cifar100(raw_data_path=parameters["data_dir"]+"/raw/")
		
	def  get_random_images(self, class_name : str, n=10):
		"""
		Get n random images from the dataset
		"""
		label_samples  =  np.where(self.labels  ==  class_name)[0]
		label_samples  =  np.random.choice(label_samples,  size=n,  replace=(n  >  len(label_samples)))
		images  =  []
		for  sample_ind  in  label_samples:
		    image  =  self.dset[sample_ind][0]
		    images.append(image)
		return  images
```

2.  **Register the New Data Creator:**
    * Open the `data/data_holder.py` file. This file acts as a factory or entry point for accessing different data creators.
    * **Import** the new class you just created (`YourDatasetNameDataCreator`) at the top of `data/__init__.py`.

    * Locate the function responsible for providing the correct data creator instance (`get_data_creator`).
    * **Modify** this function to include logic for instantiating your `YourDatasetNameDataCreator`. This involves checking an input argument (like a dataset name string) and returning the corresponding creator instance.

Continuing the CIFAR100 example, it is added to the `data/data_holder.py` file like so:

```python
# data/data_holder.py
from  data.cifar100.setup_cifar  import  CIFAR100Creator

def get_data_creator(dataset_name, parameters=None, mcq=False):
    """
    Returns the data creator object for the given dataset name
    """
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

# ... rest of the file ...
```
---

## Class Validation and Question Answer Generation

Once these steps are completed, you can validate the classes and generate question-answer pairs for the validated classes using (cifar100 example again):

```bash
bash scripts/setup_dataset.sh cifar100
```
Note: This command uses VLMs and LLMs for validation, and so should only be run with GPU access. 