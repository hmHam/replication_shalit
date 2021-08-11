'''IHDP, Jobsのデータセットを読み込むインターフェース
'''
import numpy as np
import torch

class IHDP_Dataset(torch.utils.data.Dataset):
    KEY_X = 'x'
    KEY_T = 't'
    KEY_YF = 'yf'
    KEY_YCF = 'ycf'
    KEY_MU1 = 'mu1'
    KEY_MU0 = 'mu0'
    KEYS = [KEY_X, KEY_T, KEY_YF, KEY_YCF, KEY_MU1, KEY_MU0]

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
        }
    

def load_JOBS():
    pass