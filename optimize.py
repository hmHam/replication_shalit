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


def train(CFR, train_D, batch_size=32, learning_rate=1e-2, seed=0):
    # TODO: 引数に必要なハイパーパラメータを追加する。
    train_loader = DataLoader(train_D, batch_size=batch_size, shuffle=True)
    # optimizer_wとoptimizer_vの学習率は、同じ(paperのAlgorithm1より)
    optimizer_w = Adam(
        CFR.representation.parameters(),
        lr=learning_rate
    )
    # TODO: ModuleDictは、そのパラメータを全てOptimizerに渡せるのか？確かめていない。
    # TODO: weight_decayのハイパラの大きさは、paparから正確な数字を取ってくる。
    optimizer_v = Adam(
        CFR.hypothesis.parameters(),
        lr=learning_rate,
        weight_decay=0.9
    )

    # TODO: compute u = 1/n sum_i=1^n ti
    # TODO: compute wi = ti/2u + 1-ti/2(1-u)
    # TODO: 収束性の判定？
    torch.random.manual_seed(seed)
    for batch in train_loader:
        optimizer_w.zero_grad()
        optimizer_v.zero_grad()
        
        # TODO: IPM項を算出する。
        #       (1) MMDのパターンと(2) Wassersteinのパターンがある。

        # TODO: 1/m \sum_j w_ij L(h_v(\Phi_w(x_ij), t_ij), yij)を算出する。
        # Lは、IHDPで二乗損失, Jobsでlog-loss
        
        # backward & optimizer.step()
        optimizer_w.step()
        optimizer_v.step()


if __name__ == '__main__':
    train_D = IHDP_Dataset(train=True)
    train(train_D, )