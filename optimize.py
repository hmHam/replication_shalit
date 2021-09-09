'''学習プロセスを
OptimizerはAdam
'''
import torch
from torch.optim import Adam
# TODO: (最後) GPUを駆使する実装を行う。
# TODO: (最後) CFR-MMD, CFR-Wasserstein x {IHDP, Jobs}を並行して解く。

class MMD(torch.nn.Module):
    '''emprical MMD
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y, sigma=10):
        # ガウスカーネルを採用する。
        # | x_i - x_j |^2 = x_i^T x_i + x_j^T x_j - 2x_i^T x_j
        # の値を算出する
        dx = torch.sum(x**2, axis=1)[:, None]
        dy = torch.sum(y**2, axis=1)[:, None]

        dxx = dx + dx.T - 2 * torch.mm(x, x.T)
        dyy = dy + dy.T - 2 * torch.mm(y, y.T)
        dxy = dx + dy.T - 2 * torch.mm(x, y.T)

        XX = torch.exp(-0.5*dxx/sigma)
        XY = torch.exp(-0.5*dxy/sigma)
        YY = torch.exp(-0.5*dyy/sigma)
        return XX.mean() + YY.mean() - 2 * XY.mean()


# TODO: Wassersteinをdifferentialbeに実装したい。
class Wasserstein(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, p, q):
        pass


def weighted_mse(yf_estimate, yf_true, w):
    return torch.mean(w * (yf_estimate - yf_true)**2)
    # return torch.mean((yf_estimate - yf_true)**2)


def train(cfr_net, train_loader, learning_rate=1e-2, alpha=0.1, seed=0, epoch=10):
    optimizer_w = Adam(cfr_net.representation.parameters(), lr=learning_rate)
    optimizer_v = Adam(cfr_net.hypothesis.parameters(), lr=learning_rate, weight_decay=0.9)

    mmd = MMD()
    def loss(yf_estimate, yf_true, w, r_control, r_treat):
        return weighted_mse(yf_estimate, yf_true, w) + alpha * mmd(r_control, r_treat)
        # return weighted_mse(yf_estimate, yf_true, w)

    mse = torch.nn.MSELoss()
    train_losses = []
    test_losses = []

    torch.random.manual_seed(seed)
    for e in range(epoch):
        for batch in train_loader:
            optimizer_w.zero_grad()
            optimizer_v.zero_grad()
            x_batch, t_batch, w_batch = batch['x'], batch['t'], batch['w']
            r_batch, yf_estimate_batch = cfr_net(x_batch, t_batch)

            # IPM項の勾配
            treat_idx = torch.where(t_batch == 1)
            control_idx = torch.where(t_batch == 0)
            # NOTE: Wassersteinを使うときはこの行を変更する。

            yf_batch = batch['yf']
            if yf_estimate_batch.dim() == 2:
                yf_batch = yf_batch[:, None]
            L = loss(yf_estimate_batch, yf_batch, w_batch, r_batch[control_idx], r_batch[treat_idx])
            L.backward()
            train_losses.append(L.item())
            optimizer_w.step()
            optimizer_v.step()
    return cfr_net, train_losses, test_losses


if __name__ == '__main__':
    train_D = IHDP_Dataset(train=True)
    train(train_D, )