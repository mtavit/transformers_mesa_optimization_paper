# Uncovering mesa-optimization algorithms in Transformers (Codebase)

Here you can find the codebase for our paper, "Uncovering mesa-optimization algorithms in Transformers".

## Usage

You can use this codebase to replicate and extend our experiments with various Transformer-models.
The codebase contains:

    - src: Here you can find our implementation of various models, data-generators, finetuning modules, training routines and especially of all relevant experiments (in 'util') such as e.g. in the file 'sequenceloss_eval.py', which implements an analysis of models on a test-set of sequences and reports their performance at each time-step.

    - configs: Here you can find hyperparameters for the various experiments. For extensions and certain experiments, respective config fields have to be set or appended.

    - A sandbox jupyter notebook, containing the training logic for a simple 2-layer linear model trained on constructed tokens

Please note that we use the term 'epoch' in this codebase to actually refer to 100 train-steps in this codebase.
