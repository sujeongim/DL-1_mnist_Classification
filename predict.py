import argparse

import torch
import torch.nn

import sys
import numpy as np
import matplotlib.pyplot as plt

from model import ImageClassifier

from utils import load_mnist
from utils import split_data
from utils import get_hidden_sizes

def load(fn, device):
    d = torch.load(fn, map_location=device)  #  현재 device에 올리기 
    
    return d['model'], d['config']

def plot(x, y_hat):
    for i  in range(x.size(0)):
        img = (np.array(x[i].detach().cpu(), dtype='float')).reshape(28, 28)

        plt.imshow(img, cmap='gray')
        plt.show()
        print('Predict:', float(torch.argmax(y_hat[i], dim=-1)))

def test(model, x, y, to_be_shown=True):
    model.eval()
    
    # 데이터 너무 크면 split하고 진행하기
    with torch.no_grad():
        y_hat = model(x)

        correct_cnt = (y.squeeze() == torch.argmax(y_hat, dim=-1)).sum()
        total_cnt = float(x.size(0))
        
        accuracy = correct_cnt / total_cnt
        print("Accuracy: %.4e" % accuracy)

        if to_be_shown:
            plot(x, y_hat)

def define_argparser():
    p = argparse.ArgumentParser()

    # hyper-parameters
    p.add_argument("--model_fn", required=True)
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1)

    config = p.parse_args()

    return config


def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device("cuda:%d" % config.gpu_id)

    x, y = load_mnist(is_train=False, flatten=True)
    x, y = x.to(device), y.to(device)

    input_size =  int(x.shape[-1])
    output_size = int(max(y))+1

    model_dict, train_config = load(config.model_fn, device)

    model = ImageClassifier(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=get_hidden_sizes(input_size, output_size, train_config.n_layers),
        use_batch_norm=not train_config.use_dropout,
        dropout_p=train_config.dropout_p,
    ).to(device)

    model.load_state_dict(model_dict)
    test(model, x, y, to_be_shown=False)

    n_test = 20
    test(model, x[:n_test], y[:n_test], to_be_shown=True)


if __name__ == '__main__':
    config = define_argparser()
    main(config)