from main import *
import pandas as pd
from torchvision import transforms
from PIL import Image

img_to_tsr = transforms.ToTensor()
grayscale = transforms.Grayscale()

class HASYv2Dataset(DatasetBase):

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

    def cross_val(self, fold: int, train: bool, dataset: 'HASYv2Dataset' = None):

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

        return self

    def for_colab(self) -> 'HASYv2Dataset':

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

        with open('colab_dataset.pkl', 'wb') as f:
            pickle.dump(self, f)

        return self




