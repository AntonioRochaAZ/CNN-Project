import torch as th
import torch.nn as nn
import decorators
from typing import Tuple, Union
from types import GeneratorType
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import pickle
import re
from warnings import warn

# For typing:
Tensor = th.Tensor

# Functions:
@decorators.timer
def train_model(
        model: 'NetBase', nb_epochs: int, learning_rate: float,
        loss_fn: 'nn.Loss', loader_tuple: Tuple[DataLoader, DataLoader],
        params: Union[GeneratorType, str] = None, print_bool: bool = True,
        remove_bool: bool = True, comment: str = None):

    """Function for training a :class:`NetBase` model.

    This function will create a folder with reports about the training and the
    model, according to the information contained in the model's report manager
    (see :class:`ReportManager`) for more.

    The training can be stopped at any point by the user through a
    ``KeyboardInterrupt``, which will conclude the training correctly, without
    losing progress.

    Args:
        model: The actual network to be trained.
        nb_epochs: Number of epochs for training.
        learning_rate: The learning rate.
        loss_fn: The Loss Function to use.
        loader_tuple: (training, validation) DataLoader objects.
        params: Either the specific parameters that should be trained, if
            different from the whole network, or the ``params`` argument for
            the model's :func:`~NetBase.get_params` method. The latter use is
            recommended, since in this case the ``params`` information will be
            able to be integrated to the training report.
        print_bool: If we want to print training and validation errors during
            iterations.
        remove_bool: If we want to delete the Epoch files after the training
            is complete.
        comment: A comment to add to the training report.

    """

    if params is None:
        params = model.parameters()
    else:
        if not isinstance(params, GeneratorType):
            if comment is None:
                comment = ''
            else:
                comment += '\n\t\t\t'
            comment += f"Parameter argument: {params}, type: {type(params)}." \
                       f"\n\t\t\t"

            params = model.get_params(params)

    optimizer = th.optim.Adam(params, lr=learning_rate)

    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    model = model.to(device)
    train_loader, valid_loader = loader_tuple

    train_error = th.zeros(nb_epochs, 1)
    valid_error = th.zeros(nb_epochs, 1)
    train_accur = th.zeros(nb_epochs, 1)
    valid_accur = th.zeros(nb_epochs, 1)

    manager = model.trainclass.manager
    if len(model.trainclass) == 0:
        with manager('DataReport.txt', 'w') as f:
            f.write(
                f'Net Report:\n'
                f'{report(model)}\n'
            )

    with manager('DataReport.txt', 'a') as f:
        f.write(
            f'Training {len(model.trainclass)}: ----------------------\n'
            f'Training Dataset:\n'
            f'{report(loader_tuple[0].dataset)}\n'
            f'Validation Dataset:\n'
            f'{report(loader_tuple[1].dataset)}\n'
        )

    model.trainclass.add_training(
        nb_epochs, learning_rate, loss_fn, str(datetime.today())
    )
    if comment is not None:
        model.trainclass.train_list[-1].add_comment(comment)

    try:
        # TRAINING LOOP
        for epoch in range(nb_epochs):
            model.train()
            for data, label in train_loader:
                # forward:
                data = data.to(device)
                label = label.to(device)
                out = model.forward(data)

                # step
                optimizer.zero_grad()
                if model.is_classifier:
                    label = label.view(-1)
                loss = loss_fn(out, label)
                loss.backward()
                optimizer.step()

                # adding to the error
                # train_error[epoch] += abs(loss)
                train_error[epoch] += float(loss)

                if model.is_classifier:
                    dt_arg = th.argmax(out, dim=1)
                    train_accur[epoch] += (dt_arg == label).sum().cpu()

            train_error[epoch] = train_error[epoch] / len(train_loader)
            train_accur[epoch] = train_accur[epoch] / len(train_loader.dataset)

            if print_bool:
                print('{epoch_val}. Training loss: {error:.4g}'.format(
                    epoch_val=epoch+1, error=float(train_error[epoch])))
                if model.is_classifier:
                    print('\tTraining accuracy: {accur:.2f}%'.format(
                        accur=float(100 * train_accur[epoch])))

            with th.no_grad():
                model.eval()
                for data, label in valid_loader:
                    # forward:
                    data = data.to(device)
                    label = label.to(device)
                    out = model.forward(data)
                    if model.is_classifier:
                        label = label.view(-1)
                    loss = loss_fn(out, label)

                    # adding to the error
                    # valid_error[epoch] += abs(loss)
                    valid_error[epoch] += float(loss)

                    if model.is_classifier:
                        dt_arg = th.argmax(out, dim=1)
                        valid_accur[epoch] += (dt_arg == label).sum().cpu()

                valid_error[epoch] = valid_error[epoch] / len(valid_loader)
                valid_accur[epoch] = \
                    valid_accur[epoch] / len(valid_loader.dataset)

            if print_bool:
                print('{epoch_val}. Validation loss: {error:.4g}'.format(
                    epoch_val=epoch+1, error=float(valid_error[epoch])))
                if model.is_classifier:
                    print('\tValidation accuracy: {accur:.2f}%'.format(
                            accur=float(100*valid_accur[epoch])))

            # Also saving the state_dict at the epoch folder:
            th.save(model.state_dict(),
                    f'.//{manager.path}/Epochs/'
                    f'Epoch_{epoch + 1}.stdict')

            model.trainclass.train_list[-1].add_epoch(
                (train_error[epoch], valid_error[epoch]),
                (train_accur[epoch], valid_accur[epoch]), model.state_dict(),
            )

    except KeyboardInterrupt:

        if train_error[epoch] or valid_error[epoch] == 0:
            epoch2 = epoch - 1
        else:
            epoch2 = epoch


        model.trainclass.train_list[-1].add_epoch(
            (train_error[epoch2], valid_error[epoch2]),
            (train_accur[epoch2], valid_accur[epoch2]), model.state_dict()
        )
        model.trainclass.train_list[-1].add_comment(
            f'Pruned at epoch {epoch} by the user.\n\t\t\t'
        )

    # Finding and loading the best epoch:
    model.trainclass.finish_training()
    model.load_state_dict(model.trainclass.train_list[-1].best_state)

    if remove_bool:
        manager.remove_epochs()

    with manager('TrainingReport.txt', 'w') as f:
        f.write(report(model.trainclass))

    model.trainclass.plot(save_bool=True, block=False, var='error')
    if model.is_classifier:
        model.trainclass.plot(save_bool=True, block=False, var='accur')

    model = model.to(th.device('cpu'))
    with open(f'{manager.path}/best_model.pkl', 'wb') as f:
        pickle.dump(model, f)

def report(cls_instance) -> str:
    """Function for creating a report about an object.

    Returns a string that should describe all of the information needed to
    initialize an identical class. It should also contain complementary
    information that may be useful to the user. This function in itself only
    calls the class's ``report`` method, but, if none is defined, it will return
    a string with information about the object.

    The classes' ``report`` method shouldn't have any arguments.

    Args:
        cls_instance: The class instance we want to make a report of.

    Returns:
        A string with the report.

    """
    try:
        return cls_instance.report()
    except AttributeError:
        return f"{str(cls_instance)}\n{repr(cls_instance)}"

class NetBase(nn.Module):
    """Base class for neural networks.

    This class's definition includes general methods that should be inherited
    or redefined by the neural networks. It is also recommended that the child
    classes pass their ``*args`` and ``**kwargs`` when calling ``super``.

    Attributes:
        args: The model's initialization ``*args``.
        kwargs: The model's initialization ``**kwargs``.
        is_classifier: A ``bool`` stating whether or not the model is a
            classifier. This is useful for determining whether or not to
            calculate the model's accuracy and a few other things during
            training. It defaults to False to avoid raising errors in
            :func:`train_model`. It can, of course, be overridden either by the
            child class or by passing ``is_classifier=True`` as a
            keyword argument.
        trainclass: The model's associated :class:`TrainingClass`, which stores
            information about its training and report manager.
        manager: A shortcut for ``self.trainclass.manager``, which is the
            model's :class:`ReportManager`, that stores information about the
            report folder where the model's information is saved.

            .. note::
                **Why define the model's manager inside of its TrainingClass?**

                This is because a few of :class:`TrainingClass`'s methods use
                paths that are stored in the model's :class:`ReportManager`.
                If the ReportManager was outside it, these paths would have to
                be redefined inside of the training class, which is redundant,
                and may pose a problem if the user ever decides or has to change
                the report folder paths.

    """

    def __init__(self, *args, **kwargs):
        super(NetBase, self).__init__()
        self.args = args
        self.kwargs = kwargs
        if "is_classifier" in kwargs:
            is_classifier = kwargs["is_classifier"]
            if not isinstance(is_classifier, bool):
                raise TypeError(
                    f'Illegal type for  "is_classifier" attribute: received '
                    f'{type(is_classifier)}, expected bool.')
            else:
                self.is_classifier = kwargs["is_classifier"]
        else:
            self.is_classifier = False
        self.trainclass = TrainingClass(**kwargs)

    def __str__(self):
        return report(self)

    def report(self) -> str:
        """Creates a little report about the network.

        Returns a string that should describe all of the information
        needed to reproduce the neural network's initialization, as well
        as its representation (``.__repr__``) which is defined by PyTorch's
        ``nn.Module``.

        This function shouldn't have any arguments and can be called through
        the :func:`report` function.
        """
        string = ''
        if not self.args == ():
            string += f"Arguments:\n"
            for arg in self.args:
                string += f"\t{arg}\n"

        if not self.kwargs == {}:
            string += f"Keyword Arguments:\n"
            for key in self.kwargs:
                string += f"\t{key}: {self.kwargs[key]}\n"

        string += f"Representation:\n"
        string += f"{repr(self)}"

        return string

    def reset(self):
        """Resets the neural network's parameters.
        This method can be useful when training the same network multiple times.
        """
        self.__init__(*self.args, **self.kwargs)

    def forward(self, inputs: Tensor) -> Tensor:
        """The model's forward pass."""
        raise NotImplementedError(
            "This net's forward pass hasn't been implemented.")

    def get_params(self, params: str) -> GeneratorType:
        """Placeholder method for fetching a subset of the net's parameters.

        It should take an argument that can specify which parameters to select
        and return a ``GeneratorType`` object. See the example below for an
        example of implementation.

        Examples:
            .. code-block::

                class ConvModel(NetBase):
                    def __init__(self, **kwargs):
                        super(ConvModel, self).__init__(**kwargs)

                        # net 1
                        self.conv_net_1 = nn.Sequential(
                            nn.Conv2d(1, 32, (3, 3)),
                            nn.MaxPool2d((2, 2), (2, 2))
                        )
                        self.lin_net_1 = nn.Sequential(
                            nn.Linear(32 * 15 * 15, 369),
                            nn.Softmax(dim=1)
                        )

                        # net 2
                        self.conv_net_2 = nn.Sequential(
                            nn.Conv2d(1, 64, (3, 3)),
                            nn.MaxPool2d((2, 2), (2, 2))
                        )
                        self.lin_net_2 = nn.Sequential(
                            nn.Linear(64 * 15 * 15, 369),
                            nn.Softmax(dim=1)
                        )

                    def get_params(self, params: str = None) -> GeneratorType:
                        if params == '1':
                            for parameter in self.conv_net_1.parameters():
                                yield parameter
                            for parameter in self.lin_net_1.parameters():
                                yield parameter
                        elif params == '2':
                            for parameter in self.conv_net_2.parameters():
                                yield parameter
                            for parameter in self.lin_net_2.parameters():
                                yield parameter
                        else:
                            return self.parameters()

        Args:
            params: A value that can help specify which parameters to select.

        Returns:
            A GeneratorType object with the parameters.

        """
        return self.parameters()

    @property
    def manager(self) -> 'ReportManager':
        # See the class's docstring for information.
        return self.trainclass.manager

class DatasetBase(Dataset):
    """Base class for storing datasets that can be readily used for training.

    Attributes:
        inputs: A list that contains the networks' inputs, ready to be called by
            the model. All pre-treatment must done before-hand or during class
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

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        inputs = self.inputs[item]
        output = self.output[item]
        return inputs, output

    def __str__(self):
        return self.report()

    def report(self):
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

class TrainingClass:
    """Class for storing training information.

    Attributes:
        train_list: List of :class:`TrainingClass.TrainData` objects with the
            data for each individual training.
        nb_epochs: Total number of epochs completed (int).
        train_error: Training Error for each epoch (Tensor).
        valid_error: Validation Error for each epoch (Tensor).
        train_accur: Training Accuracy for each epoch (Tensor).
        valid_accur: Validation Accuracy for each epoch (Tensor).
        best_epoch: The number of the best epoch (int).
        best_state: The best state_dict out of all training epochs
            (collections.OrderedDict).
        delete_state_dicts: Whether or not to keep state_dicts from previous
            **training loops**. Its value indicates how many trainings back we
            delete. For example: 0 means we don't delete any; 1 we delete the
            last (meaning we won't keep any); 2 means we delete the second last
            (we will always keep the last, most recent); 3 means we delete the
            third last (we'll keep the two most recent) and so on. This is
            defined by the following lines of code, from this class's
            :func:`~TrainingClass.finish_training` method:

            .. code-block::

                if del_dicts:
                    try:
                        self.train_list[-del_dicts].state_dict_list = []
                    except IndexError:
                        pass

            .. attention::
                Note that this **will not** delete the :class:`TrainData` class
                for the respective training loop, all of its information is
                safe. It must also be kept in mind that picking a value
                :math:`x` greater than one means that all *epochs* from the
                :math:`x-1` most recent training loops will be kept, not only
                the last :math:`x-1` epochs from the most recent training loop.

        manager: The :class:`ReportManager` class that manages the model's
            report folder.

    """

    def __init__(self, delete_state_dicts: int = 1, **kwargs):
        """Class initialization"""

        self.train_list = []
        self.nb_epochs = 0
        self.train_error = th.Tensor([])
        self.valid_error = th.Tensor([])
        self.train_accur = th.Tensor([])
        self.valid_accur = th.Tensor([])
        self.best_epoch = None
        self.best_state = None
        self.delete_state_dicts = delete_state_dicts
        self.manager = ReportManager(**kwargs)

    def __getitem__(self, item) -> 'TrainingClass.TrainData':
        return self.train_list[item]

    def __len__(self) -> int:
        return len(self.train_list)

    def __str__(self) -> str:
        return self.report()

    def add_training(self, nb_epochs: int, learning_rate: float,
                     loss_fn: 'nn.Loss', file_name: str):
        """Adds a :class:`TrainingClass.TrainData` object to
            the train_list attribute.

        Args:
            nb_epochs: Number of epochs for training.
            learning_rate: Learning Rate.
            loss_fn: The Loss Function used.
            file_name: The file_name picked for the training.

        """
        if file_name is not None:
            warn('"file_name" may become deprecated in the future',
                 FutureWarning)
        self.train_list.append(
            self.TrainData(nb_epochs, learning_rate, loss_fn, file_name))

    def finish_training(self):
        """Method for adapting the class's attributes after training.

        This adds the last training's data to the rest of the training
        data, updating the best epoch and state_dict.

        """
        self.train_list[-1].finish_training()
        self.nb_epochs += len(self.train_list[-1])

        self.train_error = th.cat([self.train_error,
                                   self.train_list[-1].train_error]).detach()
        self.valid_error = th.cat([self.valid_error,
                                   self.train_list[-1].valid_error]).detach()
        self.train_accur = th.cat([self.train_accur,
                                   self.train_list[-1].train_accur]).detach()
        self.valid_accur = th.cat([self.valid_accur,
                                   self.train_list[-1].valid_accur]).detach()
        self.best_epoch = self.train_list[-1].best_epoch + \
            self.nb_epochs - len(self.train_list[-1].train_error)

        self.best_state = self.train_list[-1].best_state
        del_dicts = self.delete_state_dicts
        if del_dicts:
            try:
                self.train_list[-del_dicts].state_dict_list = []
            except IndexError:
                pass

    def plot(self, save_bool: bool = False, block: bool = False,
             var: str = 'error'):
        """Plots training or accuracy graphs for all trainings.

        Args:
            save_bool: Whether or not to save the graph as a file.
            block: Whether or not the plotting of the graph should stop
                the code from continuing.
            var: Which graph to plot ("error" for the error or "accur" for
                the accuracy).

        """

        sns.set()
        plt.figure()
        if var == 'error':
            plt.plot(range(1, len(self.train_error) + 1), self.train_error,
                     label='Training Loss')
            plt.plot(range(1, len(self.valid_error) + 1), self.valid_error,
                     label='Validation Loss')
            plt.ylabel('Loss value')
            plt.title('Loss')
            title = 'loss'

        elif var == 'accur':
            plt.plot(range(1, len(self.train_accur) + 1),
                     100 * self.train_accur, label='Training Accuracy')
            plt.plot(range(1, len(self.valid_accur) + 1),
                     100 * self.valid_accur, label='Validation Accuracy')
            plt.ylabel('Accuracy (%)')
            plt.title('Accuracy')
            title = 'accur'
        else:
            raise ValueError(f"Invalid variable name: {var}.")

        plt.xlabel('Epoch number')
        plt.legend()
        if save_bool:
            plt.savefig(f'{self.manager.path}/Images/'
                        f'{self.train_list[-1].file_name}_{title}.png')
        if block:
            plt.show(block=block)
            plt.pause(5)
            plt.close()
        else:
            plt.show()
        sns.reset_orig()

    def report(self) -> str:
        """Creates a report that describes overall training.

        Returns:
            A string with a little report, with information about the
            trainings.

        """

        string = f'Overall Training Report: {self.manager.dirname}\n' \
                 f'\tNumber of Trainings: {len(self.train_list)}\n' \
                 f'\tTotal Number of Epochs: {self.nb_epochs}\n' \
                 f'\tBest Epoch: {self.best_epoch}\n\n' \
                 f'Training Information:\n'

        for idx, train_data in enumerate(self.train_list):
            string += f'\tTraining {idx}: ------------------------\n'
            string += f'{report(train_data)}\n'

        return string

    class TrainData:
        """An object for storing information of one particular training.

        .. note::
            Some of the attributes listed below are defined during
            initialization as lists, but become Tensors when the
            :func:`~TrainingClass.TrainData.finish_training` method is called.

        Attributes:
            train_error: Training Error for each epoch (Tensor).
            valid_error: Validation Error for each epoch (Tensor).
            train_accur: Training Accuracy for each epoch (Tensor).
            valid_accur: Validation Accuracy for each epoch (Tensor).
            state_dict_list: A list with the state_dict of each epoch.
            comment: A string with a possible comment to be added to the
                training report. In particular, when the training is
                pruned by the user through a ``KeyboardInterrupt``.
            best_epoch: The number of the best epoch (int).
            best_state: The best state_dict (collections.OrderedDict).
            nb_epochs: Number of epochs for training (int).
            learning_rate: Learning Rate (float).
            loss_fn: The Loss Function used.
            file_name: The file_name picked for the training.

        """

        def __init__(self, nb_epochs: int, learning_rate: float,
                     loss_fn: 'nn.Loss', file_name: str):

            """Initializes the class.

            Args:
                nb_epochs: Number of epochs for training.
                learning_rate: Learning Rate.
                params: The network to train.
                loss_fn: The Loss Function used.
                file_name: The file_name picked for the training.

            """
            self.nb_epochs = nb_epochs
            self.learning_rate = learning_rate,
            self.loss_fn = loss_fn
            self.file_name = file_name

            self.train_error = []
            self.valid_error = []
            self.train_accur = []
            self.valid_accur = []
            self.state_dict_list = []
            self.best_epoch = None
            self.best_state = None
            self.comment = None

        def __len__(self) -> int:
            return len(self.train_error)

        def __str__(self) -> str:
            return self.report()

        def add_epoch(self, error_tuple: Tuple[Tensor, Tensor],
                      accur_tuple: Tuple[Tensor, Tensor], state_dict: dict,
                      comment: str = None) -> None:
            """Method for adding an epoch's information to the class.

            Args:
                error_tuple: training and validation errors.
                accur_tuple: training and validation accuracies
                    (Tuple[Tensor, Tensor]).
                state_dict: The model's current state_dict.
                comment: A string with a possible comment to be added to
                    the  training report.

            """
            train_error, valid_error = error_tuple
            train_accur, valid_accur = accur_tuple
            self.train_error.append(train_error)
            self.valid_error.append(valid_error)
            self.train_accur.append(train_accur)
            self.valid_accur.append(valid_accur)
            self.state_dict_list.append(
                {k: v.cpu() for k, v in state_dict.items()})
            if comment is not None:
                self.comment += str(comment)    # str() just to avoid problems.

        def finish_training(self) -> None:
            """Method for adapting the class's attributes after training.

            This turns training and validation errors and accuracies into a
            tensor (originally lists). As well as defining the best epoch in
            this training and separating it's state dict.

            """
            self.train_error = th.cat(self.train_error)
            self.valid_error = th.cat(self.valid_error)
            self.train_accur = th.cat(self.train_accur)
            self.valid_accur = th.cat(self.valid_accur)
            best_epoch = th.argmin(self.valid_error).item()
            self.best_epoch = best_epoch
            self.best_state = self.state_dict_list[best_epoch]

        def report(self) -> str:
            """Creates a report that describes the training.

            Returns:
                A string with a little report, with information about the
                training.

            """
            string = f'\t\tNumber of Epochs: {len(self.train_error)}\n' \
                     f'\t\tLearning Rate: {self.learning_rate}\n' \
                     f'\t\tLoss Function: {self.loss_fn}\n' \
                     f'\t\tFile Name: {self.file_name}\n' \
                     f'\t\tBest Epoch: {self.best_epoch}\n'
            if self.comment is not None:
                string += \
                     f'\t\tComments: {self.comment}\n'

            return string

        def add_comment(self, comment: str):
            if self.comment is None:
                self.comment = comment
            else:
                self.comment += comment


class ReportManager:
    """Class for managing a model's report folder and report files.

    This class is instantiated when a :class:`NetBase` class is initalized. It's
    stored in the model's :class:`TrainingClass` class (``trainclass``
    attribute), although it can be called through the model's ``manager``
    attribute (property).

    The class creates the model's report folder, where information about its
    training will be stored (such as its epochs' state dictionaries, images,
    reports and the best iteration's model), and manages the actual text
    reports.

    Args:
        dirname: The name of the model's report directory. Defaults to the
            current date and time (``datetime.today()``) if nothing is passed.

            If the specified dirname already exists, the current date and time
            will be added to its end to distinguish the reports. The class will
            print a note on that.

        report_dir: The name of the Reports folder (where the individual model
            report directories are stored). Default is ``"_Report"``.
        complete path: If the user wants to use a base folder different from the
            current one, it can specify a path and the report manager will enter
            it. The model's reports will be stored in
            ``complete_path/report_dir/dirname``. If nothing is specified, the
            current directory is used as the complete path.

            .. warning::
                Picking a different ``complete_path`` will result in a change
                of the current directory (to the specified ``complete_path``)
                which is *not* reverted.

    Attributes:
        report_dir: The specified report_dir
        base_path: The specified complete_path (if none is specified, then it's
            set as the current directory).
        dirname: The final dirname picked for the model's report folder.
        path: f".//{report_dir}/{dirname}"
        files: A set containing the text report files' names. It is a set
            because during multiple trainings, the same files may be edited
            multiple times, which would result in multiple entries with the same
            name.

    """
    # TODO: add the possibility of changing the report dir etc.

    def __init__(self, dirname: str = None, report_dir: str = "_Reports",
                 complete_path: str = None, **kwargs):

        self.report_dir = report_dir
        if dirname is None:
            date = str(datetime.today())
            date = date.replace(":", "_")
            dirname = date
        if complete_path is not None:
            os.chdir(complete_path)
            self.base_path = complete_path
        else:
            self.base_path = os.getcwd()

        try:
            if not os.path.exists(f".//{report_dir}/{dirname}"):
                os.mkdir(f".//{report_dir}/{dirname}")
                os.mkdir(f".//{report_dir}/{dirname}/Epochs")
                os.mkdir(f".//{report_dir}/{dirname}/Images")
            else:
                date = str(datetime.today())
                date = date.replace(":", "_")

                print(
                    f'Report "{dirname}" already exists, creating a new name: '
                    f'{dirname} - {date}')
                dirname = f'{dirname} - {date}'
                os.mkdir(f".//{report_dir}/{dirname}")
                os.mkdir(f".//{report_dir}/{dirname}/Epochs")
                os.mkdir(f".//{report_dir}/{dirname}/Images")
        except FileNotFoundError:
            # Related to the _Reports folder:
            print(f'Creating "{report_dir}" folder in the current directory '
                  f'({os.getcwd()})...')
            os.mkdir(f"{report_dir}")
            os.mkdir(f".//{report_dir}/{dirname}")
            os.mkdir(f".//{report_dir}/{dirname}/Epochs")
            os.mkdir(f".//{report_dir}/{dirname}/Images")
        self.dirname = dirname
        self.path = f".//{report_dir}/{dirname}"
        self.files = set()

    def __call__(self, filename: str, method: str):
        """Returns a :class:`ReportManager.File` object that creates a txt file.

        This is done so the ``with`` statement can be called with a
        :class:`ReportManager` instance for creating a text file in the correct
        report folder.

        Examples:
            >>> manager = ReportManager(dirname='ConvModel')
            >>> with manager("Report.txt", 'w') as f:
            ...     f.write("Hello There")
            ...
            >>> # The "Report.txt" file was written directly in
            >>> # './/_Reports/ConvModel'
            >>> # Another example:
            >>> model = NetBase(dirname='ConvModel')  # Toy model
            Report "ConvModel" already exists, creating a new name: ConvModel - 2021-11-10 12_43_10.517134
            >>> with model.manager('Report.txt', 'w') as f:
            ...     f.write("General Kenobi")
            ...
            >>> # "Report.txt" written directly at
            >>> # './/_Reports/ConvModel - 2021-11-10 12_43_10.517134'

        Args:
            filename: The text file's name.
            method: The method for editing the text file.

        Returns:
            An instance of a :class:`ReportManager.File` object.

        """
        if filename.find('.') == -1:
            filename += '.txt'
        self.files.add(filename)
        return self.File(filename, method, self.report_dir, self.dirname)

    def remove_epochs(self):
        """Removes the epoch state-dicts in the Epochs folder after training.
        """
        # Deleting the Epochs folder:
        files = os.listdir(f'.//{self.path}/Epochs/')
        for file in files:
            # Only take into account the Epoch ones (just to be safe)
            match = re.search(r'Epoch_\d{1}', file)
            if match is None:
                continue
            else:
                os.remove(f'.//{self.path}/Epochs/{file}')
        # Remove the empty directory altogether:
        # os.rmdir(f'.//{self.path}/Epochs')

    class File:
        """Class for creating text report files when entered.

        This class is sneakily entered when :class:`ReportManager` is
        called.

        .. code-block::

            >>> manager = ReportManager(dirname='ConvModel')
            >>> with manager("Report.txt", 'w') as f:
            ...     # It looks like we are using manager.__enter__ because
            ...     # of the 'while' statement. However, ReportManager doesn't
            ...     # have a .__enter__ method defined.
            ...     # manager("Report.txt", 'w') is actually
            ...     # manager.__call__("Report.txt", 'w') which actually returns
            ...     # ReportManager.File object
            ...     # which is then entered because of the 'with' statement
            ...     f.write("Hello There")
            ...

        Check the examples below. It then creates the desired text
        report file directly in the model's report directory.

        Examples:
            >>> manager = ReportManager(dirname='ConvModel')
            >>> with manager("Report.txt", 'w') as f:
            ...     f.write("Hello There")
            ...
            >>> # The "Report.txt" file was written directly in
            >>> # './/_Reports/ConvModel'
            >>> # Another example:
            >>> model = NetBase(dirname='ConvModel')  # Toy model
            Report "ConvModel" already exists, creating a new name: ConvModel - 2021-11-10 12_43_10.517134
            >>> with model.manager("Report.txt", 'w') as f:
            ...     f.write("General Kenobi")
            ...
            >>> # "Report.txt" written directly at
            >>> # './/_Reports/ConvModel - 2021-11-10 12_43_10.517134'

        """
        def __init__(self, filename, method, base_dir, dirname):
            self.filename = filename
            self.method = method
            self.base_dir = base_dir
            self.dirname = dirname
            self.path = f".//{base_dir}/{dirname}/{filename}"
            # Maybe check about the file here, instead of doing it in the
            # __enter__ method.
            self.test_path()

        def test_path(self):
            """Tests if the model's report directory can be accessed.

            If it can't, it will prompt the user with a few questions in order
            to find out whether or not the problem can be fixed.

            """
            if not os.path.exists(f".//{self.base_dir}/{self.dirname}"):
                print(f"Report file .//{self.base_dir}/{self.dirname} "
                      f"not found.\nCurrent Directory: {os.getcwd()}.")

                value1 = False
                while value1 not in ['y', 'n']:
                    value1 = input("Create new directory? [y]/[n]")

                if value1 == 'y':
                    try:
                        os.mkdir(f"{self.base_dir}")
                    except FileExistsError:
                        pass
                    os.mkdir(f".//{self.base_dir}/{self.dirname}")
                elif value1 == 'n':
                    value2 = False
                    while value2 not in ['y', 'n']:
                        value2 = input("Change directory path? [y]/[n]\n "
                                       "([n] will raise FileNotFoundError).")
                    if value2 == 'y':
                        new_path = input("Type in the new path.")
                        os.chdir(new_path)
                        self.test_path()

                    elif value2 == 'n':
                        raise FileNotFoundError("Report Folder does not exist.")

        def __enter__(self):
            self.file = open(
                f".//{self.base_dir}/{self.dirname}/{self.filename}",
                self.method)

            return self.file

        def __exit__(self, exc_type, exc_val, exc_tb):
            # print(exc_type, exc_val, exc_tb)
            self.file.close()
            # If we get an exception that can be handled, we can add
            # "return True" for that case.



if __name__ == '__main__':
    from datasets import *
    from nets import *

    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    with open('.//_Data/colab_dataset.pkl', 'rb') as f:
        full_dataset = pickle.load(f)

    fold = 1
    tds = HASYv2Dataset()
    vds = HASYv2Dataset()
    tds = tds.cross_val(fold, True, full_dataset)
    vds = vds.cross_val(fold, False, full_dataset)
    del full_dataset  # In order to avoid using all of the available memory
    model = TwoLayer(dirname='TwoLayerTest')
    tdl = DataLoader(tds, batch_size=1, shuffle=True)
    vdl = DataLoader(vds, batch_size=10000)
    train_model(model, 10, 1e-4, nn.NLLLoss(), (tdl, vdl))