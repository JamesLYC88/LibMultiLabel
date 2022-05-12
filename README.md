# LibMultiLabel — a Library for Multi-label Text Classification

LibMultiLabel is a simple tool with the following functionalities.

- end-to-end services from raw texts to final evaluation/analysis
- support for common neural network architectures and linear classsifiers
- easy hyper-parameter selection

This is an on-going development so many improvements are still being made. Comments are very welcome.

## Environments
- Python: 3.7+
- CUDA: 10.2 (if training neural networks by GPU)
- Pytorch 1.8+

If you have a different version of CUDA, follow the installation instructions for PyTorch LTS at their [website](https://pytorch.org/).

## Documentation
See the documentation here: https://www.csie.ntu.edu.tw/~cjlin/libmultilabel

## Retrain

### Train (without retraining)
```python3
python3 main.py --config config/EUR-Lex/kim_cnn.yml
```

### Fixed-epochs
```python3
python3 main.py --config config/EUR-Lex/kim_cnn.yml --retrain_alg fixed
```

### Optimal-epochs
```python3
python3 main.py --config config/EUR-Lex/kim_cnn.yml --retrain_alg optimal --train_checkpoint_dir runs/EUR-Lex_kim_cnn_train
```

### Function-based
```python3
python3 main.py --config config/EUR-Lex/kim_cnn.yml --retrain_alg function --train_checkpoint_dir runs/EUR-Lex_kim_cnn_train
```
