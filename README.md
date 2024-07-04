# Uncovering mesa-optimization algorithms in Transformers (Code-Base)

Here you can find the codebase for our paper, "Uncovering mesa-optimization algorithms in Transformers".

## Usage

You can use this code-base to replicate and extend our experiments with various Transformer-models.
The codebase is split into four main parts:

    - src: Here you can find our implementation of various models, data-generators, finetuning modules, training routines and especially of all relevant experiments (in 'util') such as e.g. in the file 'sequenceloss_eval.py', which implements an analysis of models on a test-set of sequences and reports their performance at each time-step.

    - configs: Here you can find hyperparameters for the various experiments

    - jupyter-notebooks: Here you can find concrete instantations of the main experiments from the paper. Each notebook is designed such that it can be run individually in different modi: You may rerun the entire training-procedure from scratch (by setting the respective flag 'rerun_model'), only the experiments themselves (flag: 'rerun_analyses') or nothing and just generate the figures using our pre-computed results, which can be found in. At the end of the notebook you can find code to visualize the results as in the paper.

    - experiment_results: Here you can find files containing various model-parameters and experiment results to quickly replicate plots from the paper or omit rerunning certain procedures. Some of the files in here are used in different notebooks. However, each notebook is implemented such that the corresponding models/experiments can also directly be rerun in the same notebook, without having to first run another notebook.

Please note that we use the term 'epoch' to actually refer to 100 train-steps in this codebase.
