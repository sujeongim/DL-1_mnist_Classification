from copy import deepcopy
from copy import deepcopy
import numpy as np

import torch

class Trainer():

    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        #super().__int__()

    #  첫 밑줄 하나 = private
    def _batchify(self, x, y, batch_size, random_split=True):  
        if random_split: # random shuffling  / x, y 똑같은 순서로 shuffle
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices)
            y = torch.index_select(y, dim=0, index=indices)

        x = x.split(batch_size, dim=0)
        y = y.split(batch_size, dim=0)

        return x, y


    def _train(self, x, y, config):
        self.model.train()

        x, y = self._batchify(x, y, config.batch_size)
        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x,y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze())

            # Initialize the gradients of the model.
            self.optimizer.zero_grad()
            loss_i.backward()

            self.optimizer.step()

            if config.verbose >= 2:
                print("Train Iteration(%d/%d): loss=%.4e" % (i+1, len(x), float(loss_i)))

            # Don't forget to detach to prevent memory leak.
            total_loss += float(loss_i)

        return total_loss/len(x)

    def _validate(self, x, y, config):
        # Turn evaluation mode on.
        self.model.eval()

        # Turn on the no_grad mode to make more efficiently.
        with torch.no_grad():
            x, y = self._batchify(x, y, config.batch_size, random_split=False)
            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                if config.verbose >=2:
                    print('Valid Iteration(%d/%d) : loss=%.4e' % (i+1, len(x), float(loss_i)))

                total_loss += float(loss_i)

        return total_loss/len(x)

    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)

            # You must use deep copy to take a snapshot of current best weights.
            if valid_loss < lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d): train_loss = %.4e valid_loss = %.4e lowest_loss = %.4e" % (
                epoch_index + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss
            ))

            # early stop 구현할거면 여기에 넣기. loss 더 내려갈 수도 있으므로 웬만하면 안해도 됨.

        # Restore to best model.
        self.model.load_state_dict(best_model)