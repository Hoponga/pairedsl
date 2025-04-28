# PAIRED-CL 

Inspired from "Emergent Complexity and Zero-Shot Transfer via Unsupervised Environment Design" (Dennis et al, 2020), we show that a classifier trained on data sourced from an RL-trained adversary environment-generating policy (trained using the PAIRED algorithm) can achieve robust classification accuracy on a generalization of continual learning. 

Our PAIRED for continual learning package, "pairedcl," is inspired from UCL-Dark's implementation of PAIRED for minigrid/minihack environments. 

## Installation

cd into pairedcl directory and pip install -e . 

## Experiments 

scripts/mnist_train.py trains a classifier on a generalization of the permuted MNIST task, where evaluation tests the classifier's performance on batches of MNIST transformed by an arbitrary permutation in the total 784!-sized permutation space. 