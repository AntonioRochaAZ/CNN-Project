from datetime import datetime
import pickle
import os
import json
from warnings import warn

from nets import TwoLayer, ThreeLayer, FourLayer, TFCNN
import torch as th
import torch.nn as nn
from training import report
from lightning.pytorch import LightningModule, seed_everything, Trainer


def train_lightning_model(config, loader_tuple):
    """Function for training a Lightning model (more specifically a
    :class:`~LitConvNet` model).

    There are two possible ways of loading a pre-existing model:
    
    -   By loading the ``best_model.pkl``: if the user specified the 
        directory of the model's report with 
        ``config["training"]["trainer_kwargs"]["default_root_dir"]``.
        Note that this is preferable as it will also load the :class:`~training.TrainingClass`
        and :class:`~training.ReportManager` classes in their latest state,
        instead of recreating them from scratch.
    
    -   By loading a lightning checkpoint: if the user specified its
        path through the config entry
        ``config["training"]["fit_kwargs"]["ckpt_path"]``.

    .. note::
        If both options are specified in the config file,
        both the best model will be loaded and its checkpoints.

    Args:
        config: The config dictionary from which defined training and model
            hyperparameters.
        loader_tuple: A tuple with two torch.utils.data.DataLoader objects:
            for training and validation respectively.

    """

    # LOADING CONFIG
    # import json
    # with open("_configs/config.json", "r") as f:
    #     config = json.load(f)

    # DEFINING DATALOADERS:
    # loader_tuple = (None, None)     # for now

    # SETTING SEED:
    seed = config.get("seed", 74)
    seed_everything(seed)

    # SETTING THE MODEL UP:
    # Check if we must load a model.
    trainer_kwargs = config["training"].get("trainer_kwargs", dict())
    if "default_root_dir" in trainer_kwargs:
        # Load best model (since this will load trainclass information etc.)
        best_model_path = os.path.join(trainer_kwargs["default_root_dir"], "best_model.pkl")
        if os.path.exists(best_model_path):
            with open(best_model_path, "rb") as f:
                model = pickle.load(f)
        else:
            warn(f"COULD NOT FIND best_model.pkl in default_root_dir ({trainer_kwargs['default_root_dir']})."
                 " Creating new model instance. Checkpoint will be loaded if training.fit_kwargs.ckpt_path is specified.")
            model = LitConvNet(config)
        # For loading a specific checkpoint, one can pass ckpt_path in config["training"] (see below).
    else:
        model = LitConvNet(config)
        # trainer_kwargs["default_root_dir"] = model.manager.path

    # TRAINING:
    # Getting trainer:
    trainer = get_trainer(**trainer_kwargs)
    # Actual Training:
    train_loader, valid_loader = loader_tuple
    # Reporting data loader:
    with model.manager('DataReport.txt', "a") as f:
        f.write(
            f'Training {len(model.trainclass)}: ----------------------\n'
            f'Training Dataset:\n'
            f'{report(train_loader.dataset)}\n'
            f'Validation Dataset:\n'
            f'{report(valid_loader.dataset)}\n'
        )
    fit_kwargs = config["training"].get("fit_kwargs", dict())
    if not "ckpt_path" in fit_kwargs:
        fit_kwargs["ckpt_path"] = None
    trainer.fit(model, train_loader, valid_loader, ckpt_path=fit_kwargs["ckpt_path"])

def get_trainer(**user_trainer_kwargs):
    """Returns lightning's trainer object according to user-defined kwargs.
    Some default kwargs are defined in this function.

    Keyword Args:
        **user_trainer_kwargs: Any kwargs that can be passed to Lightning's 
            lightning.pytorch.Trainer object. Only one default kwarg is specified:
            ``deterministic = True``.

    """
    trainer_kwargs = {  # TODO: use Hydra config for default trainer kwargs
        "deterministic": True
    }
    trainer_kwargs.update(user_trainer_kwargs)
    return Trainer(**trainer_kwargs)

class LitConvNet(LightningModule):
    """Generic LightningModule wrapping Pytorch models defined in :py:mod:`nets`.

    Class Attributes:
        model_dict: dictionary relating names to the pytorch models defined in :py:mod:`nets`. Must
            be updated if new models are added.
            
            .. code-block:: python

                # Dictionary of available model names:
                model_dict = {
                    "twolayer":   TwoLayer,
                    "threelayer": ThreeLayer,
                    "fourlayer":  FourLayer,
                    "tfcnn":      TFCNN,
                }

    Args: 
        config: a dictionary containing all information required for model initialization and training. 
            See the example at the top of the page.

    Attributes:
        config: the config passed for ``__init__``.
        model_config: the ``"model"`` entry of the config.
        training_config: the ``"training"`` entry of the config.
        model: the actual PyTorch model (one of the models defined in :py:mod:`nets`).
        loss: a ``torch.nn`` loss function, used for calculating the loss metrirc.
        train_step_loss: a list containing each training step's loss. Cleared at the end of the epoch.
        valid_step_loss: a list containing each validation step's loss. Cleared at the end of the epoch.
        train_step_accur: a list containing each training step's accuracy. Cleared at the end of the epoch.
        valid_step_accur: a list containing each validation step's accuracy. Cleared at the end of the epoch.

    """

    # Dictionary of available model names:
    model_dict = {
        "twolayer":   TwoLayer,
        "threelayer": ThreeLayer,
        "fourlayer":  FourLayer,
        "tfcnn":      TFCNN,
    }

    def __init__(self, config) -> None:
        super(LitConvNet, self).__init__()

        # Saving configs:
        self.save_hyperparameters()
        self.config = config
        self.model_config = config.get("model")
        self.training_config = config.get("training")

        # Setting model information from config:
        model_name = self.model_config.get("model_name")
        if model_name.lower() not in LitConvNet.model_dict.keys(): # TODO: define default possibilities in hydra config? Idk if possiblle or logical
            raise ValueError(
                f'Received unexpected model name: {model_name}.\n'
                'Only names "twolayer", "threelayer", "fourlayer" and "tfcnn" '
                'are currently implemented'
            )
        # If we got here, then we have the model in the dictionary:
        self.model = LitConvNet.model_dict[model_name.lower()](
            *self.model_config.get("model_args", tuple()),
            **self.model_config.get("model_kwargs", dict())
        )

        # Setting training information from config:
        self.loss = getattr(
            nn, self.training_config.get("loss_fn", "NLLLoss")
        )()     # Extra parentheses instantiate the loss.

        # Setting up other attributes:
        self.train_step_loss = []
        self.valid_step_loss = []
        self.train_step_accur = []
        self.valid_step_accur = []

        with self.manager("config.json", "w") as f:
            json.dump(self.config, f)

    def configure_optimizers(self):
        params_name = self.training_config.get("params", None)
        learning_rate = self.training_config.get("learning_rate")
        params = self.model.get_params(params_name)
        return th.optim.Adam(params, lr=learning_rate)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model.forward(x)

    def on_fit_start(self) -> None:
        """[Custom documentation] Does some reporting with the model :class:`~TrainClass`
        :meth:`~TrainClass.add_training` method. If it is the model's first
        fit, then a ModelReport.txt is also created.
        """
        # Reporting
        if len(self.model.trainclass) == 0:
            with self.model.manager('ModelReport.txt', 'w') as f:
                f.write(
                    f'Net Report:\n'
                    f'{report(self.model)}\n'
                )
        comment = self.config.get("comment", None)
        if comment[-1] != "\n":
            comment += "\n"
        self.model.trainclass.add_training(
            self.trainer.max_epochs, self.training_config.get("learning_rate"),
            self.loss, str(datetime.today()),
            # loader_tuple, comment
            None, comment
        )

    def _common_step(self, batch: tuple[th.Tensor, th.Tensor], batch_idx: int) -> tuple[tuple[th.Tensor, th.Tensor], th.Tensor, th.Tensor]:
        """[Custom documentation] Calculates prediction, loss and accuracy. This is a generic 
        function used for both training and validation.

        Args:
            batch: a tuple containing the input and the label.

        Returns:
            A tuple containing:
            - A tuple with the loss and accuracy;
            - The model output (``y = self.model.forward(x)``);
            - The label, as passed in the batch argument.
        """
        x, label = batch
        y = self.model.forward(x)
        if self.model.is_classifier:
            label = label.view(-1)
            argmax = th.argmax(y, dim=1)
            accur = (argmax == label).sum().float()/len(label)
        else:
            accur = th.Tensor([0])
        loss = self.loss(y, label)
        return (loss, accur), y, label

    def training_step(self, batch: tuple[th.Tensor, th.Tensor], batch_idx: int) -> th.Tensor:
        loss_tuple, _, _ = self._common_step(batch, batch_idx)
        loss, accur = loss_tuple
        self.log("train_loss", loss)
        self.log("train_accur", accur)
        self.train_step_loss.append(loss)
        self.train_step_accur.append(accur)
        return loss

    def on_train_epoch_end(self) -> None:
        loss_mean = th.stack(self.train_step_loss).mean()
        accur_mean = th.stack(self.train_step_accur).mean()
        self.log("training_loss_epoch_mean", loss_mean)
        self.log("training_accur_epoch_mean", accur_mean)
        self.train_step_loss.clear()    # Reset list
        self.train_step_accur.clear()   # Reset list
        # Trainclass reporting for backwards compatibility:
        self.model.trainclass.train_list[-1].add_train_epoch(
            loss_mean, accur_mean, self.model.state_dict(), None, None
        )

    def validation_step(self, batch: tuple[th.Tensor, th.Tensor], batch_idx: int) -> th.Tensor:
        loss_tuple, _, _ = self._common_step(batch, batch_idx)
        loss, accur = loss_tuple
        self.log("val_loss", loss)
        self.log("val_accur", accur)
        self.valid_step_loss.append(loss)
        self.valid_step_accur.append(accur)
        return loss

    def on_validation_epoch_end(self) -> None:
        loss_mean = th.stack(self.valid_step_loss).mean()
        accur_mean = th.stack(self.valid_step_accur).mean()
        self.log("validation_loss_epoch_mean", loss_mean)
        self.log("validation_accur_epoch_mean", accur_mean)
        self.valid_step_loss.clear()    # Reset list
        self.valid_step_accur.clear()   # Reset list

        # Finally, add trainclass reporting for backwards compatibility:
        comment = None
        epoch_nb = None
        if self.trainer.check_val_every_n_epoch != 1:
            if isinstance(self.trainer.check_val_every_n_epoch, int):
                # In this case, we are logging every N training epochs.
                # For the loss plotting to still work, specify the epoch
                # number when calling add_valid_epoch
                nb_valid_epochs = len(self.model.trainclass.train_list[-1].valid_error)
                N = self.trainer.check_val_every_n_epoch
                epoch_nb = N * nb_valid_epochs + N
                comment = (f"Valid epochs {N * nb_valid_epochs + 1} to {N * nb_valid_epochs + N - 1}"
                           f" skipped because Trainer's check_val_every_n_epoch is set to {N}.\n")
                # try:
                #     last_error = self.model.trainclass.train_list[-1].valid_error[-1]
                #     last_accur = self.model.trainclass.train_list[-1].valid_accur[-1]
                #     comment = f"REPEATING RESULTS FROM VALIDATION EPOCH {nb_valid_epochs}."
                # except IndexError:
                #     # No validation epochs yet, let's get the training error/accuracy
                #     last_error = self.model.trainclass.train_list[-1].train_error[-1]
                #     last_accur = self.model.trainclass.train_list[-1].train_accur[-1]
                #     comment = f"REPEATING TRAINING RESULTS OF EPOCH {nb_train_epochs}."
                #
                # for i in range(N):
                #     self.model.trainclass.train_list[-1].add_valid_epoch(
                #         last_error, last_accur,
                #         f"VALIDATION EPOCH {nb_valid_epochs + 1 + i}: {comment}\n"
                #     )
            elif (
                self.trainer.check_val_every_n_epoch is None
                and (
                    isinstance(self.trainer.val_check_interval, float)
                    or isinstance(self.trainer.val_check_interval, int)
                )
            ):
                # In this case, we will have validation epochs during training
                # batches, in which case we must add some information so that
                # the plots still work.
                raise NotImplementedError("val_check_interval not implemented yet.")

            else:
                raise NotImplementedError("Unexpected val_check_interval case found in custom lightning module's on_validation_epoch_end")

        self.model.trainclass.train_list[-1].add_valid_epoch(
            loss_mean, accur_mean, comment, epoch_nb
        )

    def on_fit_end(self) -> None:
        """[Custom documentation] Calls the model TrainClass' :meth:`~TrainClass.finish_training`"""
        # Fix last validation epoch validation index, because lightning seems to
        # run a last validation epoch at the end of training:
        self.model.trainclass.train_list[-1].valid_epoch_idx[-1] = \
            self.model.trainclass.train_list[-1].train_epoch_idx[-1]
        # Now finish training:
        self.model.trainclass.finish_training(
            **self.training_config.get("finish_training_kwargs", dict()))
        # Loading best state dict and saving it (in cpu device).
        self.model.load_state_dict(self.model.trainclass.train_list[-1].best_state)
        self.model = self.model.to(th.device('cpu'))
        with open(f'{self.model.manager.path}/best_model.pkl', 'wb') as f:
            pickle.dump(self, f)

    # def predict_step(self, batch, batch_idx):

    @property
    def manager(self) -> "training.ReportManager":
        return self.model.manager

    @property
    def trainclass(self) -> "training.TrainClass":
        return self.model.trainclass

    def report(self) -> str:
        """TODO: Could eventually add more to this. Currently just returns ``self.model.report()`` 
        """
        return self.model.report()
