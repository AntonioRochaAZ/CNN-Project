from training import *
import pandas as pd
from torchvision import transforms
from PIL import Image

img_to_tsr = transforms.ToTensor()
grayscale = transforms.Grayscale()

class HASYv2Dataset(DatasetBase):
    """Dataset class for preparing HASYv2Dataset data for network training.

    Note that initalizing the class won't load the data into it just yet. One
    of the specialized methods need to be called for that.
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

        for key in kwargs:
            if key in self.path:
                self.path[key] = kwargs[key]

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
            train: Whether or not we want to load the training or the validation
                (test) data from the corresponding fold.
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

            for index, path in enumerate(df['path']):
                path = path[6:]
                tsr = grayscale(img_to_tsr(Image.open(
                    f"{self.path['base']}/{path}"
                )))
                tsr = tsr - 0.5     # So data is comprised between -0.5 and 0.5
                label = th.LongTensor([
                    self.id[int(df['symbol_id'][index])]
                ])
                self.inputs.append(tsr)
                self.output.append(label)
        else:
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
            tsr = grayscale(img_to_tsr(Image.open(
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





