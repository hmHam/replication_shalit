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
        
    def forward(self, X, Y, sigma=10):
        # ガウスカーネルを採用する。
        # | x_i - x_j |^2 = x_i^T x_i + x_j^T x_j - 2x_i^T x_j
        # の値を算出する
        xx, xy, yy = torch.mm(X, X.T), torch.mm(X, Y.T), torch.mm(Y, Y.T)
        expand_diag_xx = xx.diag().expand_as(xx)
        expand_diag_yy = yy.diag().expand_as(yy)
        
        dxx = expand_diag_xx.T + expand_diag_xx - 2 * xx
        dxy = expand_diag_xx.T + expand_diag_yy - 2 * xy
        dyy = expand_diag_yy.T + expand_diag_yy - 2 * yy
        
        XX = torch.exp(-0.5*dxx/sigma)
        XY = torch.exp(-0.5*dxy/sigma)
        YY = torch.exp(-0.5*dyy/sigma)
        return torch.mean(XX + YY - 2 * XY)


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