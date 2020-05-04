import torch
from utils import to_gpu

class Cubic_Solver:
    # Assuming we know that the desired function is a polynomial of 2nd degree, we
    # allocate a vector of size 3 to hold the coefficients and initialize it with
    # random noise.
    def __init__(self, m=0):
        if torch.cuda.is_available():
            # self.w = torch.randn([4, 1], requires_grad=True, device=torch.device('cuda:0'))
            # self.w = torch.zeros([4, 1], requires_grad=True, device=torch.device('cuda:0'))
            w = torch.FloatTensor([4,1])
            self.w = w.new_tensor([[0.0], [0.0], [m], [0.0]],
                                  requires_grad=True,
                                  device=torch.device('cuda:0'))
        else:
            self.w = torch.randn([4, 1], requires_grad=True)
        # We use the Adam optimizer with learning rate set to 0.1 to minimize the loss.
        self.opt = torch.optim.Adam([self.w], 0.3, weight_decay=0.001)
        # print(self.w.is_leaf, self.w.requires_grad, self.w.is_cuda)

    def calc(self, x):
        # We define yhat to be our estimate of y.
        p1 = torch.ones_like(x, dtype=torch.float)
        p2 = x*x
        p3 = x*p2
        #p4 = x*p3
        # print(p1.is_cuda, p2.is_cuda, p3.is_cuda, x.is_cuda)
        f = torch.stack([p3, p2, x, p1], 1)
        if torch.cuda.is_available():
            f = f.to(device=torch.device('cuda:0'))
        yhat = torch.squeeze(f @ self.w, 1)
        return yhat

    def compute_loss(self, y, yhat):
        # The loss is defined to be the mean squared error distance between our
        # estimate of y and its true value.
        # print(y.is_cuda, yhat.is_cuda, y.requires_grad, yhat.requires_grad)
        # loss = torch.nn.functional.smooth_l1_loss(yhat, y)
        loss = torch.nn.functional.mse_loss(yhat, y)
        if torch.cuda.is_available():
            loss = loss.to(device=torch.device('cuda:0'))
        return loss

    def train_step(self, x, y):

        yhat = self.calc(x)
        loss = self.compute_loss(y, yhat)
        # print("cuda", loss.is_cuda, "leaf", loss.is_leaf, "grad", loss.requires_grad)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss

    def train(self, x, y, limit=100):
        loss_tmp = 5000

        for _ in range(limit):
            loss = self.train_step(x, y)
            # print(loss.item())
            if loss_tmp < 10 and loss > loss_tmp:
                self.w = save_weight
                break
            loss_tmp=loss
            save_weight = self.w
        print("loss={}".format(loss.item()))

