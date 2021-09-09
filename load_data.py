'''IHDP, Jobsのデータセットを読み込むインターフェース
'''
import numpy as np
import torch
import os

class DataInterface(object):
    KEY_X = 'x'
    KEY_T = 't'
    KEY_YF = 'yf'
    KEY_YCF = 'ycf'
    KEY_MU1 = 'mu1'
    KEY_MU0 = 'mu0'
    KEYS = [KEY_X, KEY_T, KEY_YF, KEY_YCF, KEY_MU1, KEY_MU0]


class ArtificialData(DataInterface, torch.utils.data.Dataset):
    def __init__(self, mode=0, N=100, seed=0):
        '''
        mode)
        0 -> 特徴量2dim, 定数の効果 RCT
        1 -> 特徴量2dim, 定数の効果 セレクションbiasあり
        '''
        self.N = N  # データ数
        self.d = 2  # 共変量の数
        torch.manual_seed(seed)
        np.random.seed(seed)
        if mode == 0:
            gen, assign_treatment = self.__generate_constant_effect, self.__rct_assign_treatment
        elif mode == 1:
            gen, assign_treatment = self.__generate_constant_effect, self.__bias_assign_treatment
        self.X = torch.randn(self.N, self.d)
        self.mu0 = gen(self.X, torch.zeros(self.N))
        self.mu1 = gen(self.X, torch.ones(self.N))
        # treatment
        assign_treatment()
        self.potential_outcomes = torch.stack([self.mu0, self.mu1], axis=1)
        factual_mask = self.t[..., None]
        self.yf = torch.gather(self.potential_outcomes, 1, factual_mask)[:, 0]
        counter_factual_mask = torch.logical_not(factual_mask).type(torch.int64)
        self.ycf = torch.gather(self.potential_outcomes, 1, counter_factual_mask)[:, 0]
        self.u = torch.mean(self.t.float(), axis=0)
        self.w = self.t / (2 * self.u) + (1 - self.t) / (2 * (1 - self.u))
        self._shape_check()

    def _shape_check(self):
        assert self.X.dim() == 2, f'{self.__class__} wrong with X.dim'
        assert self.mu0.dim() == 1, f'{self.__class__} wrong with mu0.dim'
        assert self.mu1.dim() == 1, f'{self.__class__} wrong with mu1.dim'
        assert self.t.dim() == 1, f'{self.__class__} wrong with t.dim'
        assert self.yf.dim() == 1, f'{self.__class__} wrong with yf.dim'
        assert self.ycf.dim() == 1, f'{self.__class__} wrong with ycf.dim'
        assert self.u.dim() == 0, f'{self.__class__} wrong with u.dim'
        assert self.w.dim() == 1, f'{self.__class__} wrong with w.dim'
        print('shape check has been done.')

    def __generate_constant_effect(self, X, t):
        '''定数の効果を'''
        return 0.2 * X[:, 0] + 0.5 * X[:, 1] + 0.3 * t

    def __rct_assign_treatment(self):
        self.t = torch.randint(0, 2, (self.N, ))

    def __bias_assign_treatment(self):
        p = torch.sigmoid(torch.sum(self.X, axis=1))  # propensity_score for treatment
        propensity_score = torch.stack([1-p, p], axis=1).numpy()
        tN = []
        for n in range(self.N):
            t = np.random.choice([0, 1], p=propensity_score[n])
            tN.append(t)
        self.t = torch.tensor(tN)

    def generate_linear_effect(self):
        '''特徴量に関して線形に効果が増加する構造方程式モデル
        '''
        self._shape_check()

    def generate_quadratic_effect(self):
        '''特徴量に関して2乗で効果が増加する構造方程式モデル
        '''
        self._shape_check()

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return {
            self.KEY_X: self.X[idx],
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
        data = np.load(os.path.join(os.path.dirname(__file__), path), mmap_mode='r')

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