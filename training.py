import torch as th
import torch.nn as nn
import decorators
from typing import Union, OrderedDict
from types import GeneratorType
from torch.utils.data import DataLoader
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
        loss_fn: 'nn.Loss', loader_tuple: tuple[DataLoader, DataLoader],
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

    train_loader, valid_loader = loader_tuple
    # device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    device = train_loader.dataset.device
    if device != valid_loader.dataset.device:
        valid_loader.dataset.to(device)

    cuda_nb = th.cuda.device_count()
    if cuda_nb > 1:
        print(f"Number of GPUs: {cuda_nb}.")
        model = nn.DataParallel(model)

    model = model.to(device)

    train_error = th.zeros(nb_epochs, 1)
    valid_error = th.zeros(nb_epochs, 1)
    train_accur = th.zeros(nb_epochs, 1)
    valid_accur = th.zeros(nb_epochs, 1)

    # Reporting:
    if len(model.trainclass) == 0:
        with model.manager('ModelReport.txt', 'w') as f:
            f.write(
                f'Net Report:\n'
                f'{report(model)}\n'
            )
    model.trainclass.add_training(
        nb_epochs, learning_rate, loss_fn, str(datetime.today()), loader_tuple,
        comment
    )

    try:
        # TRAINING LOOP
        for epoch in range(nb_epochs):
            model.train()
            for data, label in train_loader:
                # forward:
                # data = data.to(device)        # data and label devices are
                # label = label.to(device)      # now changed beforehand.
                out = model.forward(data)

                # step
                optimizer.zero_grad()
                if model.is_classifier:
                    label = label.view(-1)
                loss = loss_fn(out, label)
                loss.backward()
                optimizer.step()

                # adding to the error
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
                    # data = data.to(device)    # data and label devices are
                    # label = label.to(device)  # now changed beforehand.
                    out = model.forward(data)
                    if model.is_classifier:
                        label = label.view(-1)
                    loss = loss_fn(out, label)

                    # adding to the error
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
                    f'.//{model.manager.path}/Epochs/'
                    f'Epoch_{epoch + 1}.stdict')

            model.trainclass.train_list[-1].add_epoch(
                (train_error[epoch], valid_error[epoch]),
                (train_accur[epoch], valid_accur[epoch]), model.state_dict(),
            )

    except KeyboardInterrupt:

        # noinspection PyUnboundLocalVariable
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
    model.trainclass.finish_training(remove_bool, model.is_classifier)
    # Loading best state dict and saving it (in cpu device).
    model.load_state_dict(model.trainclass.train_list[-1].best_state)
    model = model.to(th.device('cpu'))
    with open(f'{model.manager.path}/best_model.pkl', 'wb') as f:
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
            del kwargs['is_classifier'] # So it isn't passed to TrainingClass.
        else:
            self.is_classifier = False
        self.trainclass = TrainingClass(**kwargs)

    def __str__(self) -> str:
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
    def manager(self) -> "ReportManager":
        # See the class's docstring for information.
        return self.trainclass.manager

class TrainingClass:
    """Class for storing training information.

    This class was created with the intent of storing multiple training
    information in case the model is fine-tuned (and thus trained multiple
    times, possibly with different data and training conditions).

    Attributes:
        train_list: List of :class:`TrainingClass.TrainData` objects with the
            data for each individual training.
        nb_epochs: Total number of epochs completed (int).
        train_error: Training Error for each epoch (Tensor).
        valid_error: Validation Error for each epoch (Tensor).
        train_accur: Training Accuracy for each epoch (Tensor).
        valid_accur: Validation Accuracy for each epoch (Tensor).
        train_epoch_idx: The index of the epochs reported in the train_error
            and train_accur tensors. This has been added because the pytorch
            lightning implementation may lead to float values (at least in
            the case of the valid_epoch_idx).
        valid_epoch_idx: The index of the epochs reported in the valid_error
            and valid_accur tensors. This has been added because the pytorch
            lightning implementation may lead to float values.
        best_epoch: The number of the best epoch (int).
        best_state: The best state_dict out of all training epochs
            (collections.OrderedDict).
        delete_state_dicts: Whether to keep state_dicts from previous
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

    def __init__(self, delete_state_dicts: int = 1, **report_manager_kwargs):
        """Class initialization"""

        self.train_list = []
        self.nb_epochs = 0
        self.train_error = th.Tensor([])
        self.valid_error = th.Tensor([])
        self.train_accur = th.Tensor([])
        self.valid_accur = th.Tensor([])
        self.train_epoch_idx = th.Tensor([])
        self.valid_epoch_idx = th.Tensor([])
        self.best_epoch = None
        self.best_state = None
        self.delete_state_dicts = delete_state_dicts
        self.manager = ReportManager(**report_manager_kwargs)

    def __getitem__(self, item) -> "TrainingClass.TrainData":
        return self.train_list[item]

    def __len__(self) -> int:
        return len(self.train_list)

    def __str__(self) -> str:
        return self.report()

    def add_training(self, nb_epochs: int, learning_rate: float,
                     loss_fn: 'nn.Loss', file_name: str,
                     loader_tuple: tuple[DataLoader], comment: str = None):
        """Adds a :class:`TrainingClass.TrainData` object to
            the train_list attribute.

        Args:
            nb_epochs: Number of epochs for training.
            learning_rate: Learning Rate.
            loss_fn: The Loss Function used.
            file_name: The file_name picked for the training.
            .. warning::
                ``file_name`` might become deprecated in the future.
            loader_tuple: (training, validation) DataLoader objects.

        """
        if file_name is not None:
            warn('"file_name" may become deprecated in the future',
                 FutureWarning)
        self.train_list.append(
            self.TrainData(nb_epochs, learning_rate, loss_fn, file_name))

        if len(self) == 1:
            method = "w"
        else:
            method = "a"

        if loader_tuple is not None:
            with self.manager('DataReport.txt', method) as f:
                f.write(
                    f'Training {len(self)}: ----------------------\n'
                    f'Training Dataset:\n'
                    f'{report(loader_tuple[0].dataset)}\n'
                    f'Validation Dataset:\n'
                    f'{report(loader_tuple[1].dataset)}\n'
                )

        # Adding the comment:
        self.train_list[-1].add_comment(comment)

    def finish_training(self, remove_bool: bool = False, plot_accuracy: bool = False):
        """Method for adapting the class's attributes after training.

        This adds the last training's data to the rest of the training
        data, updating the best epoch and state_dict.

        """
        self.train_list[-1].finish_training()

        self.train_error = th.cat([self.train_error,
                                   self.train_list[-1].train_error]).detach()
        self.valid_error = th.cat([self.valid_error,
                                   self.train_list[-1].valid_error]).detach()
        self.train_accur = th.cat([self.train_accur,
                                   self.train_list[-1].train_accur]).detach()
        self.valid_accur = th.cat([self.valid_accur,
                                   self.train_list[-1].valid_accur]).detach()
        # Train idxs:
        updated_idxs = self.train_list[-1].train_epoch_idx
        for i in range(len(updated_idxs)):
            updated_idxs[i] = updated_idxs[i] + self.nb_epochs
        self.train_epoch_idx = th.cat([self.train_epoch_idx,
                                       updated_idxs]).detach()
        # Validation idxs:
        updated_idxs = self.train_list[-1].valid_epoch_idx
        for i in range(len(updated_idxs)):
            updated_idxs[i] = updated_idxs[i] + self.nb_epochs
        self.valid_epoch_idx = th.cat([self.valid_epoch_idx,
                                       updated_idxs]).detach()
        
        self.nb_epochs += len(self.train_list[-1])
        self.best_epoch = self.train_list[-1].best_epoch + \
            self.nb_epochs - len(self.train_list[-1].train_error)

        self.best_state = self.train_list[-1].best_state
        del_dicts = self.delete_state_dicts
        if del_dicts:
            try:
                self.train_list[-del_dicts].state_dict_list = []
            except IndexError:
                pass

        if remove_bool:
            self.manager.remove_epochs()

        with self.manager('TrainingReport.txt', 'w') as f:
            f.write(report(self))

        self.plot(save_bool=True, block=False, var='error')
        if plot_accuracy:
            self.plot(save_bool=True, block=False, var='accur')

    def plot(self, save_bool: bool = False, block: bool = False,
             var: str = 'error'):
        """Plots training or accuracy graphs for all trainings.

        Args:
            save_bool: Whether to save the graph as a file.
            block: Whether the plotting of the graph should stop
                the code from continuing.
            var: Which graph to plot ("error" for the error or "accur" for
                the accuracy).

        """

        if self.nb_epochs == 1:
            plot_fn = plt.scatter   # Otherwise the point will not be visible.
        else:
            plot_fn = plt.plot

        sns.set()
        plt.figure()
        if var == 'error':
            plot_fn(self.train_epoch_idx, self.train_error, label='Training Loss')
            plot_fn(self.valid_epoch_idx, self.valid_error, label='Validation Loss')
            plt.ylabel('Loss value')
            plt.title('Loss')
            title = 'loss'

        elif var == 'accur':
            plot_fn(self.train_epoch_idx, 100 * self.train_accur, label='Training Accuracy')
            plot_fn(self.valid_epoch_idx, 100 * self.valid_accur, label='Validation Accuracy')
            plt.ylabel('Accuracy (%)')
            plt.title('Accuracy')
            title = 'accur'
        else:
            raise ValueError(f"Invalid variable name: {var}.")

        plt.xlabel('Epoch number')
        plt.legend()
        if save_bool:
            plt.savefig(
                f'{self.manager.path}/Images/'
                f'{title}-training_nb_{len(self.train_list)}'
                f'-{self.train_list[-1].file_name}.png'
            )
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
        """An object for storing information of one particular training loop.

        .. note::
            Some of the attributes listed below are defined during
            initialization as lists, but become Tensors when the
            :func:`~TrainingClass.TrainData.finish_training` method is called.

        Args:
            nb_epochs: Number of epochs for training.
            learning_rate: Learning Rate.
            loss_fn: The Loss Function used.
            file_name: The file_name picked for the training.
            .. warning::
                ``file_name`` might become deprecated in the future.

        Attributes:
            train_error: Training Error for each epoch (Tensor).
            valid_error: Validation Error for each epoch (Tensor).
            train_accur: Training Accuracy for each epoch (Tensor).
            valid_accur: Validation Accuracy for each epoch (Tensor).
            train_epoch_idx: The index of the epochs reported in the train_error
                and train_accur tensors. This has been added because the pytorch
                lightning implementation may lead to float values (at least in
                the case of the valid_epoch_idx).
            valid_epoch_idx: The index of the epochs reported in the valid_error
                and valid_accur tensors. This has been added because the pytorch
                lightning implementation may lead to float values.
            state_dict_list: A list with the state_dict of each epoch. It may
                be reset after training if the model's associated
                :class:`TrainingClass` has a ``delete_state_dicts`` attribute
                with a value different from 0. Check the class's documentation
                for more information on how this works.
            comment: A string with a possible comment to be added to the
                training report (through the
                :func:`~TrainingClass.TrainData.add_comment` method). In
                particular, when the training is pruned by the user through a
                ``KeyboardInterrupt``.
            best_epoch: The number of the best epoch (int).
            best_state: The best state_dict (collections.OrderedDict).

        """

        def __init__(self, nb_epochs: int, learning_rate: float,
                     loss_fn: 'nn.Loss', file_name: str):

            """Initializes the class.

            Args:
                nb_epochs: Number of epochs for training.
                learning_rate: Learning Rate.
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
            self.train_epoch_idx = []
            self.valid_epoch_idx = []
            self.state_dict_list = []
            self.best_epoch = None
            self.best_state = None
            self.comment = None

        def __len__(self) -> int:
            return len(self.train_error)

        def __str__(self) -> str:
            return self.report()

        def add_train_epoch(self, error: Tensor, accur: Tensor,
                            state_dict: OrderedDict, comment: str = None,
                            epoch_nb: int = None) -> None:
            if epoch_nb is None:
                epoch_nb = len(self.train_error)    # That way epoch_nb starts at 0
            self.train_error.append(error)
            self.train_accur.append(accur)
            self.state_dict_list.append(
                {k: v.cpu() for k, v in state_dict.items()})
            self.add_comment(comment)
            self.train_epoch_idx.append(epoch_nb)

        def add_valid_epoch(self, error: Tensor, accur: Tensor,
                            comment: str = None, epoch_nb: int = None) -> None:
            """Method used to add validation epoch loss information.

            This is called by the pytorch lightning custom model's
            :meth:`lightning_objects.LitConvNet.on_validation_epoch_end`.

            Args:
                error: validation error.
                accur: validation accuracy.
                comment: A string with a comment to be added to the training
                    report.
                epoch_nb: The epoch number, which may be a float if the
                    lightning's `Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_
                    flag `val_check_interval <https://lightning.ai/docs/pytorch/stable/common/trainer.html#val-check-interval>`_
                    is a float.

            """
            if epoch_nb is None:
                epoch_nb = len(self.valid_error)
            self.valid_error.append(error)
            self.valid_accur.append(accur)
            self.add_comment(comment)
            self.valid_epoch_idx.append(epoch_nb)

        def add_epoch(self, error_tuple: tuple[Tensor, Tensor],
                      accur_tuple: tuple[Tensor, Tensor],
                      state_dict: OrderedDict, comment: str = None) -> None:
            """Method for adding a training + validation epoch's information to
            the class.

            This is used by the :func:`~train_model` function for the training
            of pytorch's Module models. For pytorch lightning models, the
            :meth:`~add_train_epoch` and :meth:`~add_valid_epoch` methods will
            be called directly.

            Args:
                error_tuple: training and validation errors.
                accur_tuple: training and validation accuracies.
                state_dict: The model's current state_dict.
                comment: A string with a comment to be added to the training
                    report.

            """
            train_error, valid_error = error_tuple
            train_accur, valid_accur = accur_tuple
            # Only passing the comment once, so that it is not repeated
            self.add_train_epoch(train_error, train_accur, state_dict, comment)
            self.add_valid_epoch(valid_error, valid_accur)

        def finish_training(self):
            """Method for adapting the class's attributes after training.

            This turns training and validation errors and accuracies into a
            tensor (originally lists). As well as defining the best epoch in
            this training and separating its state dict.

            """
            self.train_error = th.Tensor(self.train_error)
            self.valid_error = th.Tensor(self.valid_error)
            self.train_accur = th.Tensor(self.train_accur)
            self.valid_accur = th.Tensor(self.valid_accur)
            self.train_epoch_idx = th.Tensor(self.train_epoch_idx)
            self.valid_epoch_idx = th.Tensor(self.valid_epoch_idx)
            best_epoch = int(self.valid_epoch_idx[int(th.argmin(self.valid_error))])
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
            """Adds a comment to the class's comment attribute.

            Args:
                comment: The comment we want to add.

            """
            if self.comment is None:
                self.comment = comment
            else:
                if comment is not None:
                    self.comment += str(comment)

class ReportManager:
    """Class for managing a model's report folder and report files.

    This class is instantiated when a :class:`NetBase` class is initialized. It's
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
        complete_path: If the user wants to use a base folder different from the
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

    def __init__(self, dirname: str = None, report_dir: str = "_Reports",
                 complete_path: str = None):

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

        def _make_dirs(rep_dir, dir_name):
            # Avoiding code repetition
            # Makes sure rep_dir gets created in case it isn't already:
            os.makedirs(f".//{rep_dir}/{dir_name}/Epochs")
            os.mkdir(f".//{rep_dir}/{dir_name}/Images")

        if not os.path.exists(f".//{report_dir}/{dirname}"):
            _make_dirs(report_dir, dirname)
        else:
            date = str(datetime.today())
            date = date.replace(":", "_")

            print(
                f'Report "{dirname}" already exists, creating a new name: '
                f'{dirname} - {date}')
            dirname = f'{dirname} - {date}'
            _make_dirs(report_dir, dirname)

        self.dirname = dirname
        self.path = f".//{report_dir}/{dirname}"
        self.files = set()

    def __call__(self, filename: str, method: str) -> "ReportManager.File":
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
        if "." not in filename:
            filename += '.txt'
        self.files.add(filename)
        return self.File(filename, method, self.report_dir, self.dirname)

    def chdir(self, path: str):
        """Changes the report directory according to the informed final path.

        This method may create the necessary directories to reach the path
        given by the user. If the number of directories that need to be created
        is greater than two, then the user will be prompted for confirmation.

        Args:
            path: The path to the final model report folder.

        TODO:
            Add "timed input" functionality, so the code will continue if the
            user is AFK and doesn't see the prompt.

        """
        abs_path = os.path.abspath(path)
        path_list = abs_path.split("/")
        dirname = path_list[-1]
        report_dir = path_list[-2]
        if os.path.exists(path):
            os.chdir(abs_path.removesuffix(f"/{report_dir}/{dirname}"))
            complete_path = os.getcwd()
        else:
            complete_path = ""
            for idx in range(len(path_list) - 2):
                complete_path += f"{path_list[idx]}/"
            complete_path = complete_path.removesuffix("/")
            if os.path.exists(complete_path):
                os.makedirs(abs_path)
                os.chdir(complete_path)
            else:
                val = ''
                while val not in ['y', 'n']:
                    val = input(f"Base path {complete_path} does not exist, "
                                f"make directories anyway? [y/n]\n"
                                f"([n] will raise FileNotFoundError).")

                if val == 'y':
                    os.makedirs(abs_path)
                    os.chdir(complete_path)
                else:
                    raise FileNotFoundError("Report Folder does not exist.")

        self.dirname = dirname
        self.report_dir = report_dir
        self.base_path = complete_path
        self.path = f".//{report_dir}/{dirname}"
        self.files = set()

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

        def __enter__(self) -> "open file":
            self.file = open(
                f".//{self.base_dir}/{self.dirname}/{self.filename}",
                self.method)

            return self.file

        def __exit__(self, exc_type, exc_val, exc_tb):
            # print(exc_type, exc_val, exc_tb)
            self.file.close()
            # If we get an exception that can be handled, we can add
            # "return True" for that case.
