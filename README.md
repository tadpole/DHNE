# DHNE
This is an implementation of "[Structural Deep Embedding for Hyper-Networks](https://arxiv.org/abs/1711.10146)"(AAAI 2018).

### Requirements
```
python >= 2.7.0
scipy >= 0.19.1
numpy >= 1.13.1
tensorflow >= 1.0.0
Keras >= 2.0.8
```

### Usage
##### Example Usage
```
python src/hypergraph_embedding.py --data_path data/GPS --save_path result/GPS -s 16 16 16
```
##### Full Command List
```
usage: hyper-network embedding [-h] [--data_path DATA_PATH]
                               [--save_path SAVE_PATH]
                               [-s EMBEDDING_SIZE EMBEDDING_SIZE EMBEDDING_SIZE]
                               [--prefix_path PREFIX_PATH]
                               [--hidden_size HIDDEN_SIZE]
                               [-e EPOCHS_TO_TRAIN] [-b BATCH_SIZE]
                               [-lr LEARNING_RATE] [-a ALPHA]
                               [-neg NUM_NEG_SAMPLES] [-o OPTIONS]
                               [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Directory to load data.
  --save_path SAVE_PATH
                        Directory to save data.
  -s EMBEDDING_SIZE EMBEDDING_SIZE EMBEDDING_SIZE, 
                --embedding_size EMBEDDING_SIZE EMBEDDING_SIZE EMBEDDING_SIZE
                        The embedding dimension size
  --prefix_path PREFIX_PATH
                        .
  --hidden_size HIDDEN_SIZE
                        The hidden full connected layer size
  -e EPOCHS_TO_TRAIN, --epochs_to_train EPOCHS_TO_TRAIN
                        Number of epoch to train. Each epoch processes the
                        training data once completely
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Number of training examples processed per step
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        initial learning rate
  -a ALPHA, --alpha ALPHA
                        radio of autoencoder loss
  -neg NUM_NEG_SAMPLES, --num_neg_samples NUM_NEG_SAMPLES
                        Neggative samples per training example
  -o OPTIONS, --options OPTIONS
                        options files to read, if empty, stdin is used
  --seed SEED           random seed
```
### Cite
If you find this code useful, please cite our paper:
```
@article{tu2017structural,
  title={Structural Deep Embedding for Hyper-Networks},
  author={Tu, Ke and Cui, Peng and Wang, Xiao and Wang, Fei and Zhu, Wenwu},
  journal={arXiv preprint arXiv:1711.10146},
  year={2017}
}
```
