# VIME_pytorch

A PyTorch implementation of VIME.

## Acknowledgement

This work is based on the paper by Jinsung Yoon, Yao Zhang, James Jordon, and Mihaela van der Schaar titled 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," presented at Neural Information Processing Systems (NeurIPS) in 2020. 

[Link to the paper](https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html)

## Motivation

The original codebase for "VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain" was implemented using outdated versions of Keras (2.3.1) and TensorFlow (1.15.0). Our aim is to modernize the implementation by transitioning to a more recent version of PyTorch.

## Requirements

- `sklearn`
- `pytorch`

## Test Run

To test the implementation, run the following command:

```python
python semi_pytorch_mnist_test.py
```
