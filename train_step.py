import torch
from loss_function import Tacotron2Loss

class Train_Step():
    def __init__(self, hparams, model):
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=hparams.learning_rate,
                                          weight_decay=hparams.weight_decay)
        self.criterion = Tacotron2Loss()
        self.model = model

    def step(self, x, y):
        self.model.train()

        self.model.zero_grad()

        y_pred = self.model(x)

        loss = self.criterion(y_pred, y)
        loss.backward()

        self.optimizer.step()

        self.model.eval()
        return y_pred, loss, self.model