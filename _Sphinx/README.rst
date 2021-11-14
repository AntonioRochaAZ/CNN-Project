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

Update 14/11/2021: I've found out through a friend that a website already exists
for this: `http://detexify.kirelabs.org/symbols.html`_. Still, my development
continues.

.. note::
    In order to work with the dataset on
    `GoogleColab <https://colab.research.google.com/>`_, I've tried uploading
    the data to GoogleDrive, which didn't work because it has trouble dealing
    with such a big amount of data (even unzipping the files directly from Colab
    didn't work). Because of this, I've organized them in a
    ``datasets.HASYv2Dataset`` class. However, it also can't be uploaded here
    because it exceeds 100MB. If you wish to use the codes presented here, you
    need to unzip the dataset (which can be found
    `here <https://zenodo.org/record/259444#.YYwmp73MLUJ>`_) in the
    ``_Data`` folder (creating a ``HASYv2`` folder with all of its contents).

Current State of Development
----------------------------

.. note::
    With the return of my classes, development will be slowed down
    significantly in the next few months.

Base classes and a few models have already been defined, but development is
hindered by the time it takes to train models. Currently, using Google Colab's
GPU acceleration, training on one fold takes several hours to complete (which
basically uses up the daily available GPU runtime). Training a model with all
10 folds would thus take up to 10 days for each model. Development should thus
shift to implementations with
`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ and
`PyTorch/XLA <https://github.com/pytorch/xla/>`_ (see :ref:`Objectives`),
that could allow for multicore TPU training in Colab, speeding up the process.

In terms of the Pygame implementation (see "pygame-tests" branch), much has yet
to be done and improved, but the base window is about what I had in mind. I have
understood a little bit better how Pygame works, which will help in the next
steps. The text box is still very limited, and I'll be working on it in the
future.

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

- [WIP] Switch to
  `Google's Style <https://google.github.io/styleguide/pyguide.html>`_

    Working on it!

- [WIP] Write a complete documentation with Sphinx.

    I have already worked with Sphinx in the past and personally loved it.
    This is a permanent work in progress, of course, but I'm currently testing a
    new theme (`Furo <https://github.com/pradyunsg/furo>`_) and haven't yet
    written a docstring for everything so it's particularly empty as of know.
    To access the documentation, you can use the shortcut file
    ``_documentation.html`` (you'll need to have the _Sphinx directory in the
    same folder).

- [WIP] Implement an interface for real-time drawing and prediction.

    Development has started using the Pygame module.

- Try to use `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ and
  `PyTorch/XLA <https://github.com/pytorch/xla/>`_ for accelerating training
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
