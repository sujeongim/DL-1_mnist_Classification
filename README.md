# mnist_classification
This repository holds the codes for the mnist classification project from lecture of Ki Hyun Kim (FastCampus)

Training

    python train.py --model_fn ./model.pth --train_ratio 0.8 --batch_size 512 --n_epochs 20 --n_layers 3 --verbose 1

|Name|Type|Description|
|------|---|---|
|model_fn|Required|When you save a model, the model_fn will be a filename|
|train_ratio|Optional|split ratio for train data and validation data|
|batch_size|Optional|batch size for stochastic gradient descent|
|n_epochs|Optional|number of iterations|
|n_layers|Optional|number of layers of the model|
|verbose|Optional|Decide the details of the explanation|
