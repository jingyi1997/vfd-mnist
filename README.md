# Variational Feature Disentangling on MNIST

This repository contains a toy example (on MNIST dataset) of a few-shot learning method presented in [Variational Feature Disentangling for Fine-Grained Few-Shot Classification](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Variational_Feature_Disentangling_for_Fine-Grained_Few-Shot_Classification_ICCV_2021_paper.pdf) (ICCV 2021). This paper proposed a feature disentanglement framework that disentangles features into two components, i.e., class-discriminative features and intra-class variance features. Using this framework, the generated features exhibit large intra-class variance while inheriting crucial class-discriminative features. These generated features can be used to augment the training set in few-shot learning tasks.

## Usage

Use `python main.py --config <config_file> <params>` to train a model.

### Output

This will create a directory `output/<time-stamp>/` which will contain the saved checkpoints and the loss computed during training.

## Plot

After the model is trained, augmented samples will be generated and saved as 'visualize.png'. 

## Data

The used dataset is [MNIST](http://yann.lecun.com/exdb/mnist/). The dataset will be downloaded and will be stored in `data` for future uses.




