import torch
from torch import nn


class CFR(nn.Module):
    TREAT_IDX = 0
    CONTROL_IDX = 1

    def __init__(self, feature_dim, representation_dim=200, hypothesis_dim=200):
        '''IHDP -> representation_dim=200, hypothesis_dim=200,
           Jobs -> representation_dim=200, hypothesis_dim=100,
        '''

        super().__init__()
        self.representation_dim = representation_dim
        self.hypothesis_dim = hypothesis_dim
        # self.representation match with \Phi in the paper.
        self.representation = nn.Sequential(
            nn.Linear(feature_dim, representation_dim),
            nn.ELU(),
            nn.Linear(representation_dim, representation_dim),
            nn.ELU(),
            nn.Linear(representation_dim, representation_dim),
            nn.ELU(),
        )

        self.hypothesis_treat = nn.Sequential(
            nn.Linear(representation_dim, hypothesis_dim),
            nn.ELU(),
            nn.Linear(hypothesis_dim, hypothesis_dim),
            nn.ELU(),
            nn.Linear(hypothesis_dim, hypothesis_dim),
            nn.ELU(),
        )

        self.hypothesis_control = nn.Sequential(
            nn.Linear(representation_dim, hypothesis_dim),
            nn.ELU(),
            nn.Linear(hypothesis_dim, hypothesis_dim),
            nn.ELU(),
            nn.Linear(hypothesis_dim, hypothesis_dim),
            nn.ELU(),
        )

        # https://pytorch.org/docs/stable/notes/modules.html
        # module dict
        # intervention variable is in {0, 1}
        self.hypothesis = nn.ModuleList([
            self.hypothesis_treat,
            self.hypothesis_control
        ])

    def forward(self, x, t):
        '''this is implementation of f(x, t)
        '''
        r = self.representation(x)
        if t.dim() == 0:
            return self.hypothesis[t](r)
        # (ミニ)バッチ処理
        B = t.shape[0]
        # 先にoutput用の配列を用意
        batch_out = torch.zeros((B, self.hypothesis_dim))
        
        # 処置グループを計算して、batch_outの配列に代入
        treat_batch_idx = torch.where(t == self.TREAT_IDX)
        h1_batch_out = self.hypothesis_treat(r[treat_batch_idx])
        batch_out[treat_batch_idx] = h1_batch_out
        
        # 統制グループを計算して、batch_outの配列に代入
        control_batch_idx = torch.where(t == self.CONTROL_IDX)
        h0_batch_out= self.hypothesis_control(r[control_batch_idx])
        batch_out[control_batch_idx] = h0_batch_out

        return batch_out

        
        