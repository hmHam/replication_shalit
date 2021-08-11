'''学習プロセスを
OptimizerはAdam
'''
from torch.optim import Adam
# TODO: (最後) GPUを駆使する実装を行う。
# TODO: (最後) CFR-MMD, CFR-Wasserstein x {IHDP, Jobs}を並行して解く。

def train(CFR, learning_rate=1e-2, train_dataset):
    # TODO: 引数に必要なハイパーパラメータを追加する。

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

    N = len(train_dataset)
    # TODO: compute u = 1/n sum_i=1^n ti
    # TODO: compute wi = ti/2u + 1-ti/2(1-u)
    # TODO: 収束性の判定？
    for n in range(N):
        optimizer_w.zero_grad()
        optimizer_v.zero_grad()
        # TODO: IPM項を算出する。
        #       (1) MMDのパターンと(2) Wassersteinのパターンがある。

        # TODO: 1/m \sum_j w_ij L(h_v(\Phi_w(x_ij), t_ij), yij)を算出する。
        # Lは、IHDPで二乗損失, Jobsでlog-loss
        
        # backward & optimizer.step()
        optimizer_w.step()
        optimizer_v.step()