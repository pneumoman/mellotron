import torch

class Cubic_Solver:
    # Assuming we know that the desired function is a polynomial of 2nd degree, we
    # allocate a vector of size 3 to hold the coefficients and initialize it with
    # random noise.
    def __init__(self):
        self.w = torch.randn([4, 1]).clone().detach().requires_grad_(True)
        # We use the Adam optimizer with learning rate set to 0.1 to minimize the loss.
        self.opt = torch.optim.Adam([self.w], 0.1)
        print(type(self.w))

    def calc(self, x):
        # We define yhat to be our estimate of y.
        p1 = torch.ones_like(x, dtype=torch.float)
        p2 = x*x
        p3 = x*p2
        #print(p1.shape, p2.shape, p3.shape,x.shape)
        f = torch.stack([p3, p2, x, p1], 1)
        yhat = torch.squeeze(f @ self.w, 1)
        return yhat

    def compute_loss(self, y, yhat):
        # The loss is defined to be the mean squared error distance between our
        # estimate of y and its true value.
        loss = torch.nn.functional.mse_loss(yhat, y)
        return loss

    def train_step(self, x, y):

        yhat = self.calc(x)
        loss = self.compute_loss(y, yhat)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss

    def train(self, x, y, limit=100):
        for _ in range(limit):
            loss = self.train_step(x, y)
            if loss < 5:
                break
        print("loss={}".format(loss.item()))

