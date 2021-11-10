ImageNet Project
================

Introduction
------------

Despite having previous experiences in developing CNNs in PyTorch, I've always
felt overwhelmed by the number of different of ways that one can combine
convolutional, max-pooling and linear layers - with different kernel sizes and
padding sizes, strides, dilation and feature numbers.

My objective is to develop skills in python and get used to using GitHub while
exploring a variety of CNN architectures to find in practice which ones work
best for different applications. This project will also serve as a portfolio.

.. note::
    This is a *work in progress*.

Dataset
-------

The first dataset I've chosen to use is the
`HASYv2 dataset <https://arxiv.org/abs/1701.08380>`_, because it has many more
classes and symbols than other symbol recognition datasets such as MNIST, and
the final models could possibly be adapted in the future for translating
handwritten equations (even if they are handwritten through a mouse pad of
sorts) into LaTeX equations.

This also inspires me to develop some kind of application where the user can
draw symbols in a 32x32 pixel grid with its mouse, and a trained net will try to
guess it at the same time. The user can then add it live to the LaTeX equation.
This application idea draws inspiration from Google's
`Quick, Draw! <https://quickdraw.withgoogle.com/>`_.

.. note::
    In order to work with the dataset on
    `GoogleColab <https://colab.research.google.com/>`_, I've tried uploading
    the data to GoogleDrive, which didn't work because it has trouble dealing
    with such a big amount of data (even unzipping the files directly from Colab
    didn't work). Because of this, I've organized them in a
    ``datasets.HASYv2Dataset`` class which can be found in the ``_Data`` folder.

Objectives
----------

Here's list of a few objectives I had in mind when starting this project. It
contains some things I have completed and others that I still want to complete.

- [DONE] Understand decorators.

    Although I understand how they work and how to implement them, I haven't yet
    found much use. Yet.

- [DONE] Understand context managers.

    Not only have I understood *how* they work, I've developed the
    ``main.ReportManager`` class specifically to deal with creating model
    reports, something I already used to do in a more manual way before.

- [WIP] Write a complete documentation with Sphinx.

    I have already worked with Sphinx in the past and personally loved it.
    This is a permanent work in progress, of course, but I'm currently testing a
    new theme (`Furo <https://github.com/pradyunsg/furo>`_) and haven't yet
    written a docstring for everything so it's particularly empty as of know.
    To access the documentation, you can use the shortcut file
    ``_documentation.html`` (you'll need to have the _Sphinx directory in the
    same folder).

- Try to use `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ and
  `PyTorch_Xla <https://github.com/pytorch/xla/>`_ for accelerating training
  using cloud multi-core TPUs (in GoogleColab).

    Despite knowing how to use GoogleColab's GPUs for accelerating PyTorch code,
    TPUs and specifically multi-core parallelism is something I don't (yet) know
    how to work with.

- Perhaps learn and use `Optuna <https://optuna.org/>`_ for selecting training
  and Neural Networks hyperparameters.
- Develop more CNNs for testing.
- Finish developing functions for evaluating trained model's performances on the
  HASYv2 dataset.

    Using the same parameters as the ones used in the
    `article <https://arxiv.org/abs/1701.08380>`_.
