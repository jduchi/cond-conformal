# cifar_processing.py
#
# Some methods that transform a CIFAR100 dataset into a numpy array,
# and then allow it to be loaded into a DataLoader object for playing
# around with PyTorch (for simplicity) or other optimization toolkits.
#
# Author: John Duchi (jduchi@stanford.edu)

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np

from importlib import reload
from typing import Callable


# CIFAR-100 Python API Documentation has the mapping
superclass_mapping = {
    "aquatic mammals": [0, 1, 2, 3, 4],
    "fish": [5, 6, 7, 8, 9],
    "flowers": [10, 11, 12, 13, 14],
    "food containers": [15, 16, 17, 18, 19],
    "fruit and vegetables": [20, 21, 22, 23, 24],
    "household electrical devices": [25, 26, 27, 28, 29],
    "insects": [30, 31, 32, 33, 34],
    "large carnivores": [35, 36, 37, 38, 39],
    "large man-made outdoor Things": [40, 41, 42, 43, 44],
    "medium-sized mammals": [45, 46, 47, 48, 49],
    "natural outdoor scenes": [50, 51, 52, 53, 54],
    "people": [55, 56, 57, 58, 59],
    "reptiles": [60, 61, 62, 63, 64],
    "small mammals": [65, 66, 67, 68, 69],
    "trees": [70, 71, 72, 73, 74],
    "vehicles 1": [75, 76, 77, 78, 79],
    "vehicles 2": [80, 81, 82, 83, 84],
    "watercraft": [85, 86, 87, 88, 89],
    "wild animals": [90, 91, 92, 93, 94],
    "domestic animals": [95, 96, 97, 98, 99]
}

def class_to_superclass():
    """Creates a reverse mapping from class index to superclass
    """
    class_to_superclass = {}
    for superclass, classes in superclass_mapping.items():
        for class_index in classes:
            class_to_superclass[class_index] = superclass
    return class_to_superclass

def process_cifar_into_vectors(float_type = np.float32):
    """Loads CIFAR-100 datasets and saves them as numpy arrays

    Taking an input datatype (usually one of the np.float* datatypes),
    loads the CIFAR-100 datasets, then processes them through a
    ResNet50 model. Extracts the inputs to the last layer of the
    ResNet50 model (i.e., the second-to-last outputs, which is of size
    2048) and stores them in large matrices.

    Saves four files: two data files (cifar100_train_features.npy and
    cifar100_test_features.npy) and two label files
    (cifar100_train_labels.npy and cifar100_test_labels.npy)
    corresponding to the datasets.

    """
    # Load the CIFAR-100 datasets
    transform = transforms.Compose(
        [transforms.Resize(224),  # Resize images for ResNet
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]
    )
    # Load the CIFAR-100 test dataset
    cifar100_test = datasets.CIFAR100(root='./data', train = False,
                                      download = True, transform = transform)
    test_loader = DataLoader(cifar100_test, batch_size=64,
                             shuffle = False, num_workers=2)

    cifar100_train = datasets.CIFAR100(root='./data', train = True,
                                       download = True, transform = transform)
    train_loader = DataLoader(cifar100_train, batch_size=64, shuffle = False,
                              num_workers = 2)
    d_string = ("mps" if torch.backends.mps.is_available() else
                "cuda" if torch.cuda.is_available() else
                "cpu")
    print("Setting backend device to use " + d_string)
    device = torch.device(d_string)
    model = models.resnet50(weights = models.resnet.ResNet50_Weights.DEFAULT)
    model = model.to(device)
    
    print("*** Extracting Train Features ***")
    extract_features(model, train_loader, device,
                     "cifar100_train_features.npy",
                     "cifar100_train_labels.npy", float_type)

    print("*** Extracting Test Features ***")
    extract_features(model, test_loader, device,
                     "cifar100_test_features.npy",
                     "cifar100_test_labels.npy", float_type)

def extract_features(model, data_loader, device,
                     data_filename = "foo.npy",
                     label_filename = "foo_labels.npy",
                     float_type = np.float32):
    """Extracts the features from a given model on a given dataset

    This method saves the data in the given data_loader into numpy
    arrays, saving the results at data_filename (the features) and
    label_filename (the labels). See process_cifar_into_vectors for
    a description of the actual extractions.

    """
    data_size = len(data_loader.dataset)
    model.eval()  # Set the model to evaluation mode

    # Modify the model to get features from the last convolutional
    # layer by removing the fully connected layer (the last layer)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    # Store the outputs in a numpy array so we don't sit around
    # allocating stupid matrices. Assumes final layer of model is
    # model.fc (true for ResNet50)
    in_features = model.fc.in_features
    features = np.zeros((data_size, in_features), dtype = np.float32)
    labels_array = np.zeros(data_size, dtype = np.int32)
    curr_data_ind = 0;

    with torch.no_grad():
        for (images, labels) in data_loader:
            (images, labels) = (images.to(device), labels.to(device))
            outputs = feature_extractor(images)
            # Flatten the output to shape (batch_size, in_dimension), then
            # put them on CPU in the features matrix
            batch_size = outputs.size(0)
            print(f"\tExtracting datapoints {curr_data_ind + 1} through " +
                  f"{curr_data_ind + batch_size} (of {data_size})")
            outputs = outputs.view(batch_size, -1)
            features[curr_data_ind:(curr_data_ind + batch_size), :] = \
                outputs.cpu().numpy().astype(float_type)
            labels_array[curr_data_ind:(curr_data_ind + batch_size)] = \
                labels.cpu().numpy().astype(np.int32)
            curr_data_ind += batch_size
    np.save(data_filename, features)
    np.save(label_filename, labels_array)
    
class NumpyDataset(Dataset):
    """Simple dataset manipulation from NumPy arrays for use with PyTorch

    Constructs a PyTorch dataset from a given matrix X and set of
    labels y, where these are assumed to allow indexing.

    """
    def __init__(self, X, y):
        if ((not isinstance(X, np.ndarray)) or
            (not isinstance(y, np.ndarray))):
            raise TypeError("Expected numpy arrays")
        self.data = X
        self.labels = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors. Convert to float32s
        # in the sampling bit
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        # Change to Long (i.e. int64) for class label compatibility
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return sample, label

def load_numpy_into_data(feature_file = None,
                         label_file = None):
    """Returns a NumpyDataset object from the given features and labels
    """
    X = np.load(feature_file)
    y = np.load(label_file)
    return NumpyDataset(X, y)

class FilteredDataset(Dataset):
    """A (view) of a dataset filtered by a desired set of classes

    D = FilteredDataset(dataset, desired_classes)

    constructs a FilteredDataset object, where the examples in D
    consist only of those whose labels in the original dataset belong
    to the collection desired_classes.

    """
    def __init__(self, original_dataset, desired_classes):
        self.original_dataset = original_dataset
        self.desired_classes = desired_classes
        self.filtered_indices = \
            [ii for ii in range(len(original_dataset))
             if original_dataset[ii][1] in self.desired_classes]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        original_idx = self.filtered_indices[idx]
        return self.original_dataset[original_idx]

class CollapsedLabelDataset(Dataset):
    """A dataset with labels collapsed into a smaller number of labels

    cld = CollapsedLabelDataset(original_dataset, class_mapping)

    returns a dataset whose new labels are specified by class_mapping,
    which is a dictionary mapping from a key (typically a string)
    representing a superclass to a list or array of base class indices
    belonging to the superclass. The new dataset has labels
    corresponding to the superclasses to which each example belongs.

    """

    def __init__(self, original_dataset, class_mapping : dict[str, list]):
        self.original_dataset = original_dataset
        (class_to_superclass, superclass_indices) = \
            self._from_classes_to_super(class_mapping)
        relabel = lambda tensor_ind : \
            superclass_indices[class_to_superclass[tensor_ind.item()]]
        self.labels = torch.tensor(
            [relabel(ind) for ind in original_dataset[:][1]])
        self.num_classes = (torch.max(self.labels) + 1).item()
        
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # Get len(idx)-sized tensor of samples, an (idx x dim) tensor,
        # and len(idx)-length set of labels
        (sample, label) = self.original_dataset[idx] 
        new_label = self.labels[idx]
        return (sample, new_label)

    def _from_classes_to_super(self, class_mapping : dict):
        """Constructs a mapping from base classes to superclasses as
        well as superclasses to integers

        Returns pair (class_to_superclass, superclass_indices) of
        dictionaries, where class_to_superclass[i] gives the
        superclass name to which the base class i belongs;
        superclass_indices[key] gives the integer index (new class
        label) for the given key. So

        superclass_indices[class_to_superclass[y]]

        gives the new label to which the original base label y is mapped.

        """
        class_to_superclass = {}
        for superclass, classes in superclass_mapping.items():
            for class_index in classes:
                class_to_superclass[class_index] = superclass
        superclass_indices = {}
        ind = 0
        for key in class_mapping.keys():
            superclass_indices[key] = ind
            ind += 1
        return (class_to_superclass, superclass_indices)
        
class FunctionFilteredDataset(Dataset):
    """A (view) of a dataset filtered by a function acting

    D = FunctionFilteredDataset(dataset, f)

    constructs a FunctionFilteredDataset object, consisting of those
    examples in dataset for which f(dataset[i]) returns True.

    """

    def __init__(self, original_dataset, filter : Callable[[tuple], bool]):
        self.original_dataset = original_dataset
        self.filter_ = filter
        self.filtered_indices = \
            [ii for ii in range(len(original_dataset))
             if self.filter_(original_dataset[ii])]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        original_idx = self.filtered_indices[idx]
        return self.original_dataset[original_idx]

def split_into_train_and_validation(dataset, prop_train):
    """Splits a given PyTorch dataset into two, randomly

    Returns two datasets, where the first contains a prop_train
    proportion of the data, the second a (1 - prop_train) proportion of the
    data.
    """
    full_size = len(dataset)
    train_size = round(full_size * prop_train)
    if (train_size <= 0 or train_size >= full_size):
        raise ValueError("Expected training proportion in [0, 1], but got " +
                         f"{prop_train}")
    val_size = full_size - train_size
    # Split the dataset into training and validation sets

    (train_subset, val_subset) = \
        torch.utils.data.random_split(dataset, [train_size, val_size])
    return (train_subset, val_subset)
