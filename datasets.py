import pickle

import pandas as pd
from PIL import Image

import torch as th
from torchvision import transforms
from torch.utils.data import Dataset

Tensor = th.Tensor
grayscale = transforms.Grayscale()

class DatasetBase(Dataset):
    """Base class for storing datasets that can be readily used for training.

    Attributes:
        inputs: A list that contains the networks' inputs, ready to be called by
            the model. All pre-treatment must be done beforehand or during class
            initialization.
        output: A list that contains the expected outputs for training or
            validation. They should also be ready to be read by the loss
            function to be used.

    """

    def __init__(self, *args, **kwargs):
        super(DatasetBase, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.inputs = []
        self.output = []
        self.device = th.device("cpu")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, item) -> tuple[Tensor, Tensor]:
        inputs = self.inputs[item]
        output = self.output[item]
        return inputs, output

    def __str__(self) -> str:
        return self.report()

    def to(self, device: th.device):
        """Method for changing the input and output tensors' device.

        Args:
            device: A th.device.
        """

        if isinstance(self.inputs, list):
            for index in range(len(self.inputs)):
                self.inputs[index] = self.inputs[index].to(device)
        else:
            self.inputs = self.inputs.to(device)

        if isinstance(self.output, list):
            for index in range(len(self.output)):
                self.output[index] = self.output[index].to(device)
        else:
            self.output = self.output.to(device)

        self.device = device

    def report(self) -> str:
        """See :func:`report`."""
        string = ''

        if not self.args == ():
            string += f"Arguments:\n"
            for arg in self.args:
                string += f"\t{arg}\n"

        if not self.kwargs == {}:
            string += f"Keyword Arguments:\n"
            for key in self.kwargs:
                string += f"\t{key}: {self.kwargs[key]}\n"

        string += f"Number of data points: {len(self)}\n"
        return string


class HASYv2Dataset(DatasetBase):
    """Dataset class for preparing HASYv2Dataset data for network training.

    Note that initializing the class won't load the data into it just yet. One
    of the specialized methods needs to be called for that.
    """

    default_paths = {
        "base": './/_Data/HASYv2',
        "fold": './/_Data/HASYv2/classification-task',
    }

    id_dict = {}
    latex_dict = {}
    try:
        symb_csv = pd.read_csv(f"{default_paths['base']}/symbols.csv")
        for index, symbol_id in enumerate(symb_csv['symbol_id']):
            id_dict[int(symbol_id)] = index
            latex_dict[index] = symb_csv['latex'][index]

    except FileNotFoundError:
        id_dict = None
        latex_dict = None

    def __init__(self, **kwargs):
        super(HASYv2Dataset, self).__init__()
        self.path = HASYv2Dataset.default_paths
        self.path.update(kwargs)

        if HASYv2Dataset.id_dict is None:
            self.id = {}
            self.latex = {}
            symb_csv = pd.read_csv(f"{self.path['base']}/symbols.csv")
            for index, symbol_id in enumerate(symb_csv['symbol_id']):
                self.id[int(symbol_id)] = index
                self.latex[index] = symb_csv['latex'][index]
        else:
            self.id = HASYv2Dataset.id_dict
            self.latex = HASYv2Dataset.latex_dict

        self.inputs = []
        self.output = []
        self.fold = None

    def cross_val(self, fold: int, train: bool, dataset: "HASYv2Dataset" = None):
        """Method for loading data from one fold to the dataset class.

        Args:
            fold: The number of the fold (1 to 10).
            train: Whether we want to load the training or the validation (test)
                data from the corresponding fold.
            dataset: Another :class:`HASYv2Dataset` object containing the
                entirety of the data (generated through the
                :func:`~HASYv2Dataset.for_colab` method). This is needed when
                the dataset's individual images are not locally available (like
                in GoogleDrive). This method will then use this base dataset
                instead of trying to find the files locally.

        """

        self.fold = fold
        self.inputs = []
        self.output = []

        if train:
            string = 'train'
        else:
            string = 'test'

        df = pd.read_csv(f"{self.path['fold']}/fold-{fold}/{string}.csv")

        if dataset is None:
            # Load actual images
            for index, path in enumerate(df['path']):
                path = path[6:]
                tsr = transforms.Grayscale(
                      transforms.ToTensor(
                          Image.open(
                    f"{self.path['base']}/{path}"
                )))
                tsr = tsr - 0.5     # So data is comprised between -0.5 and 0.5
                label = th.LongTensor([
                    self.id[int(df['symbol_id'][index])]
                ])
                self.inputs.append(tsr)
                self.output.append(label)
        else:
            # Load data from larger dataset
            for index, path in enumerate(df['path']):
                path = path[-10:-4].replace('-', '')
                tsr = dataset.inputs[int(path)]
                label = dataset.output[int(path)]
                self.inputs.append(tsr)
                self.output.append(label)

    def for_colab(self):
        """Loads the entirety of the dataset into one single class instance.

        This is useful for passing the complete dataset around without having to
        move all of the 160,000+ image files. This is of course particularly
        useful for training models in Google Colab, since uploading the raw
        dataset to Google Drive has failed several times.

        This function will automatically save the full :class:`HASYv2Dataset`
        class in the Dataset's base directory (``self.path["base"]``) as
        "colab_dataset.pkl", a pickle.
        """

        self.inputs = []
        self.output = []

        data_csv = pd.read_csv(f"{self.path['base']}/hasy-data-labels.csv")
        for index, path in enumerate(data_csv['path']):
            tsr = transforms.Grayscale(
                  transforms.ToTensor(
                      Image.open(
                f"{self.path['base']}/{path}"
            )))

            tsr = tsr - 0.5     # So data is comprised between -0.5 and 0.5

            label = th.LongTensor([
                self.id[int(data_csv['symbol_id'][index])]
            ])

            self.inputs.append(tsr)
            self.output.append(label)

        with open(f"{self.path['base']}/colab_dataset.pkl", 'wb') as f:
            pickle.dump(self, f)





