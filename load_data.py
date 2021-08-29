'''IHDP, Jobsのデータセットを読み込むインターフェース
'''
import numpy as np
import torch

class DataInterface(object):
    KEY_X = 'x'
    KEY_T = 't'
    KEY_YF = 'yf'
    KEY_YCF = 'ycf'
    KEY_MU1 = 'mu1'
    KEY_MU0 = 'mu0'
    KEYS = [KEY_X, KEY_T, KEY_YF, KEY_YCF, KEY_MU1, KEY_MU0]


class ArtificialData(DataInterface, torch.utils.data.Dataset):
    '''
    mode)
      0 -> 特徴量1dim, 全ての個体に対して介入後は定数の効果
      1 -> 特徴量1dim, 特徴量の大きさに対して線形の効果
      2 -> 特徴量1dim, 特徴量の大きさに対して2次の効果
    --------------
      3 -> 特徴量5dim, 全ての個体に対して介入後は定数の効果
      4 -> 特徴量5dim, 特徴量の大きさに対して線形の効果
      5 -> 特徴量5dim, 特徴量の大きさに対して2次の効果
    '''
    def __init__(self, mode=0, N=100, seed=0):
        self.N = 100  # データ数
        torch.manual_seed(seed)
        d = 1 if mode in [0, 1, 2] else 5
        if mode in [0, 3]:
            generate_func = self.generate_constant_effect
        elif mode in [1, 4]:
            generate_func = self.generate_linear_effect
        else:
            generate_func =self.generate_quadratic_effect
        self.x = torch.linspace(0, 1, self.N)[..., None]
        self.mu0 = torch.randn(self.N)
        self.mu1 = generate_func(self.x, self.mu0)
        self.t = torch.randint(0, 2, (self.N,))
        potential_outcomes = torch.stack([self.mu0, self.mu1], axis=1)
        factual_mask = self.t[..., None]
        self.yf = torch.gather(potential_outcomes, 1, factual_mask)[:, 0]
        counter_factual_mask = torch.logical_not(factual_mask).type(torch.int64)
        self.ycf = torch.gather(potential_outcomes, 1, counter_factual_mask)[:, 0]
        self.u = torch.mean(self.t.float(), axis=0)
        self.w = self.t / (2 * self.u) + (1 - self.t) / (2 * (1 - self.u))

    def generate_constant_effect(self, x, mu0):
        return mu0.clone() + 5  # テキトーな定数

    def generate_linear_effect(self, x, mu0):
        # FIXME: 5次元も対応？
        return mu0.clone() + x[:, 0]

    def generate_quadratic_effect(self, x, mu0):
        # FIXME: 5次元も対応？
        return mu0.clone() + 10 * x[:, 0]**2

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return {
            self.KEY_X: self.x[idx],
            self.KEY_T: self.t[idx],
            self.KEY_YF: self.yf[idx],
            self.KEY_YCF: self.ycf[idx],
            self.KEY_MU1: self.mu1[idx],
            self.KEY_MU0: self.mu0[idx],
            'w': self.w[idx],
        }


class IHDP_Dataset(DataInterface, torch.utils.data.Dataset):
    def __init__(self, train=True, mono=True, R=100):
        '''load IHDP dataset used in the paper

        (arguments)
        * train: train=True -> return train dataset, otherwise return test dataset
        * mono: mono=True -> return first one dataset from 100 datasets in `data/ihdp_npci_1-100.train.npz`.
        * R: N個のデータセットを作成する回数

        (data)
        the following is the meaning of keys in data dictionary
        {
            'x': feature,
            't': treatment,
            'yf': factual outcome,
            'ycf': counterfactual outcome, 
            'mu1': treated potential outcome,
            'mu0': non treated potential outcome,
        }
        '''
        path = f'data/ihdp_npci_1-{R}.{"train" if train else "test"}.npz'
        data = np.load(path, mmap_mode='r')

        try:
            data = {k: data[k] for k in self.KEYS}
        except KeyError:
            raise f"datasetに{self.KEYS}のキーがない。想定外"

        self.N = len(data['x'])

        # NOTE: monoは一旦排除して、
        # ひとまずR回生成されたデータのうち最初のもののみを用いる実装にする。
        # if mono:
        for key, value in data.items():
            # NOTE: xだけ多次元なので分岐する
            #       original x shape is (672, 25, R), the others are (672, R)
            if key == self.KEY_X:
                data[key] = value[:, :, 0]
            else:
                data[key] = value[:, 0]
            # NOTE: getitemで使いやすくするため
            value_as_tensor = torch.from_numpy(data[key]).to(dtype=torch.float)
            setattr(self, key, value_as_tensor)
        self.u = torch.mean(self.t.float(), axis=0)
        self.w = self.t / (2 * self.u) + (1 - self.t) / (2 * (1 - self.u))

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return {
            self.KEY_X: self.x[idx],
            self.KEY_T: self.t[idx],
            self.KEY_YF: self.yf[idx],
            self.KEY_YCF: self.ycf[idx],
            self.KEY_MU1: self.mu1[idx],
            self.KEY_MU0: self.mu0[idx],
            'w': self.w[idx],
        }
    

# TODO: 実装
def load_JOBS():
    pass