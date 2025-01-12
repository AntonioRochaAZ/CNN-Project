Lightning implementation
========================

To train a lightning model (more specifically a :class:`~lightning_objects.LitConvNet` object),
the :func:`~lightning_objects.train_lightning_model` function must be used. It is necessary to
pass a DataLoader tuple with the training and validation DataLoaders, as well as a configuration
dictionary.

The configuration dictionary contains information about the model itself ("model" entry), as well as the
training ("training" entry). It also allow specification of keyword arguments to lightning's `lightning.pytorch.Trainer <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#trainer>`_,
(through the "training.trainer_kwargs"). An example of a commented config dictionary is given below:

.. code-block:: python

   {
      "seed": 74, # Seed passed to lightning's "seed_everything".
      "comment": "First lightning test!", # A comment to add to the model report 
         # (this is done on_fit_start(), through training.TrainingClass.add_training()).
      "model": {
         "model_name": "fourlayer", # The name of the PyTorch model. 
            # See LitConvNet's class attribute "model_dict" for possible values.
         "model_args": [], # *args passed to the PyTorch model initialization.
         "model_kwargs": { # **kwargs passed to the PyTorch model initialization.
            "dirname": "FourLayer_lightning_test"
         }
      },
      "training": {
         "loss_fn": "NLLLoss", # Name of the loss function to be used.
            # must be the name of a function defined in the torch.nn module. 
            # We'll fetch it using getattr(torch.nn, config["training"]["loss_fn"])
         "params": "all", # See training.NetBase.get_params().
         "learning_rate": 1e-4,
         "trainer_kwargs": { 
            # kwargs passed directly to lightning's Trainer class 
            "max_epochs": 2,
            "limit_train_batches": 10,
            "limit_val_batches": 10,
            "accelerator": "cpu",
            "deterministic": True
         
         },
         """
         fit_kwargs: { 
            'ckpt_path' can be specified for loading a checkpoint. See train_lightning_model's
            docstring.
            'ckpt_path': path_to_checkpoint
         }
         """
         "finish_training_kwargs": {
            # kwargs passed to training.TrainingClass.finish_training() 
            # at the end of training.
            "remove_bool": True,
            "plot_accuracy": True
         }
      }
   }


Note that some of the documentation for :class:`~lightning_objects.LitConvNet` comes directly from Lightning's 
documentation (specifically, for :meth:`~lightning_objects.LitConvNet.configure_optimizers`, 
:meth:`~lightning_objects.LitConvNet.forward`, :meth:`~lightning_objects.LitConvNet.training_step`,
:meth:`~lightning_objects.LitConvNet.on_train_epoch_end`, :meth:`~lightning_objects.LitConvNet.validation_step`,
:meth:`~lightning_objects.LitConvNet.on_validation_epoch_end`). Lightning functions
which have custom documentation are noted with "\[Custom documentation\]" at the beginning of its
docstring.

Note that, since :class:`~lightning_objects.LitConvNet` is just a wrapper to a :class:`~nets.ConvNetBase`, it
also contains :class:`~training.TrainingClass` (dealt with in :meth:`~lightning_objects.LitConvNet.on_train_epoch_end`,
:meth:`~lightning_objects.LitConvNet.on_validation_epoch_end` and :meth:`~lightning_objects.LitConvNet.on_fit_end`) 
and :class:`~training.ReportManager` classes. These classes can also be directly accessed by the
property attributes ``manager`` and ``trainclass``.

.. automodule:: lightning_objects
   :members:
