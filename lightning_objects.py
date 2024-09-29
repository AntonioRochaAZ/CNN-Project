from nets import TwoLayer, ThreeLayer, FourLayer, TFCNN
import torch as th
import torch.nn as nn
from training import report
from lightning.pytorch import LightningModule, seed_everything, Trainer
from datetime import datetime

class LitConvNet(LightningModule):

    # Dictionary of available model names:
    model_dict = {
        "twolayer":   TwoLayer,
        "threelayer": ThreeLayer,
        "fourlayer":  FourLayer,
        "tfcnn":      TFCNN,
    }

    def __init__(self, config) -> None:
        super(LitConvNet, self).__init__()

        # Saving logs:
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

    def configure_optimizers(self):
        params_name = self.training_config.get("params", None)
        learning_rate = self.training_config.get("learning_rate")
        params = self.model.get_params(params_name)
        return th.optim.Adam(params, lr=learning_rate)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model.forward(x)

    def training_step(self, batch: tuple[th.Tensor, th.Tensor], batch_idx: int) -> float:
        loss, y, label = self._common_step(batch, batch_idx)
        self.log("train_loss", loss)
        self.train_step_loss.append(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        epoch_mean = th.stack(self.train_step_loss).mean()
        self.log("training_epoch_mean", epoch_mean)
        # self.model.trainclass.train_list[-1].add_training_epoch(epoch_mean)
        self.train_step_loss.clear()  # Reset list

    def validation_step(self, batch: tuple[th.Tensor, th.Tensor], batch_idx: int) -> float:
        loss, y, label = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.valid_step_loss.append(loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        epoch_mean = th.stack(self.valid_step_loss).mean()
        self.log("validation_epoch_mean", epoch_mean)
        # self.model.trainclass.train_list[-1].add_validation_epoch(epoch_mean)
        self.valid_step_loss.clear()  # Reset list

    def _common_step(self, batch: tuple[th.Tensor, th.Tensor], batch_idx: int) -> tuple[float, th.Tensor, th.Tensor]:
        x, label = batch
        y = self.model.forward(x)
        if self.model.is_classifier:
            label = label.view(-1)
        loss = self.loss(y, label)
        # TODO: See how I can also add accuracy calculation/logging. --> training_epoch_end or sth
        return loss, y, label

    def on_fit_start(self) -> None:
        """Does some reporting with the model :class:`~TrainClass`
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
        # TODO: Uncomment next code with the correct variables
        self.model.trainclass.add_training(
            self.trainer.max_epochs, self.training_config.get("learning_rate"),
            self.loss, str(datetime.today()),
            # loader_tuple, self.config.get("comment", None)
            None, self.config.get("comment", None)
        )

    def on_fit_end(self) -> None:
        """Calls the model TrainClass' :meth:`~TrainClass.finish_training`"""
        # self.model.trainclass.finish_training() # Commented for now

    # def predict_step(self, batch, batch_idx):

    @property
    def manager(self) -> "training.ReportManager":
        return self.model.manager

    @property
    def trainclass(self) -> "training.TrainClass":
        return self.model.trainclass

    def report(self) -> str:
        # Could eventually add more to this
        return self.model.report()

def get_trainer(**user_trainer_kwargs):
    """Returns lightning's trainer object according to user-defined kwargs.
    Some default kwargs are defined in this function.
    """
    trainer_kwargs = {  # TODO: use Hydra config for default trainer kwargs
        "deterministic": True
    }
    trainer_kwargs.update(user_trainer_kwargs)
    return Trainer(**trainer_kwargs)

def run(config, loader_tuple):

    # LOADING CONFIG
    # import json
    # with open("_configs/config.json", "r") as f:
    #     config = json.load(f)

    # DEFINING DATALOADERS:
    # loader_tuple = (None, None)     # for now

    # SETTING SEED:
    seed = config.get("seed", 74)
    seed_everything(seed)

    # SETTING MODEL UP:
    model = LitConvNet(config)

    # TRAINING:
    # Getting trainer:
    trainer_kwargs = config.get("trainer_kwargs", dict())
    trainer = get_trainer(**trainer_kwargs)
    # Actual Training:
    train_loader, valid_loader = loader_tuple
    trainer.fit(model, train_loader, valid_loader)
