import torch

def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', 
        train=is_train, 
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.  # normalization
    y = dataset.targets

    if flatten:
        x=x.view(x.size(0),-1)

    return x, y

def split_data(x, y, device, train_ratio=.8):
    train_cnt = int(x.size(0)* train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0)).to(device)
    x = torch.index_select(
        x, 
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    y = torch.index_select(
        y, 
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    return x, y

def get_hidden_sizes(input_size, output_size, n_layers):
    step_size = int((input_size - output_size) / n_layers) # (784 - 10) / 5 

    hidden_sizes = []
    current_size = input_size
    for i in range(n_layers-1):  # layer size를 등차수열로 만들어주기
        hidden_sizes += [current_size - step_size]
        current_size = hidden_sizes[-1]

    return hidden_sizes
