import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import config as C

class CycleS(nn.Module):
    def __init__(self):
        super(CycleS, self).__init__()
        self.mse = nn.MSELoss()
        self.delta = 0.5
        self.m = C.MARGIN
        self.epsilon = 0.1

    def pairwise_loss(self, all_S):
        loss_num = 0
        loss_sum = 0
        for i in range(len(all_S)):
            for j in range(len(all_S)):
                if i < j:
                    loss_num += 1
                    S = all_S[i][j]
                    if S.shape[0] < S.shape[1]:
                        S21 = S
                        S12 = S21.transpose(1, 0)
                    else:
                        S12 = S
                        S21 = S12.transpose(1, 0)

                    scale12 = np.log(self.delta / (1 - self.delta) * S12.size(1)) / self.epsilon
                    scale21 = np.log(self.delta / (1 - self.delta) * S21.size(1)) / self.epsilon
                    S12_hat = f.softmax(S12 * scale12, dim=1)
                    S21_hat = f.softmax(S21 * scale21, dim=1)
                    S1221_hat = torch.mm(S12_hat, S21_hat)
                    n = S1221_hat.shape[0]
                    I = torch.eye(n).cuda()
                    pos = S1221_hat * I
                    neg = S1221_hat * (1 - I)
                    loss = 0
                    loss += torch.sum(f.relu(torch.max(neg, 1)[0] + self.m - torch.diag(pos)))
                    loss += torch.sum(f.relu(torch.max(neg, 0)[0] + self.m - torch.diag(pos)))
                    loss /= 2 * n
                    loss_sum += loss
        return loss_sum / loss_num

    def triplewise_loss(self, all_S):
        loss_num = 0
        loss_sum = 0
        for i in range(len(all_S)):
            for j in range(len(all_S)):
                if i < j:
                    for k in range(len(all_S)):
                        if k != i and k != j :
                            loss_num += 1
                            S12_ = all_S[i][k]
                            S23_ = all_S[k][j]
                            S = torch.mm(S12_, S23_)
                            if S.shape[0] < S.shape[1]:
                                S21 = S
                                S12 = S21.transpose(1, 0)
                            else:
                                S12 = S
                                S21 = S12.transpose(1, 0)
                            scale12 = np.log(self.delta / (1 - self.delta) * S12.size(1)) / self.epsilon
                            scale21 = np.log(self.delta / (1 - self.delta) * S21.size(1)) / self.epsilon
                            S12_hat = f.softmax(S12 * scale12, dim=1)
                            S21_hat = f.softmax(S21 * scale21, dim=1)
                            S1221_hat = torch.mm(S12_hat, S21_hat)
                            n = S1221_hat.shape[0]
                            I = torch.eye(n).cuda()
                            pos = S1221_hat * I
                            neg = S1221_hat * (1 - I)
                            loss = 0
                            loss += torch.sum(f.relu(torch.max(neg, 1)[0] + self.m - torch.diag(pos)))
                            loss += torch.sum(f.relu(torch.max(neg, 0)[0] + self.m - torch.diag(pos)))
                            loss /= 2 * n
                            loss_sum += loss
        return loss_sum / loss_num

    def gen_X_S(self, feature_ls: list):
        norm_feature = [f.normalize(i, dim=-1) for i in feature_ls]
        all_blocks_S = []
        all_blocks_X = []
        for idx, x in enumerate(norm_feature):
            row_blocks_S = []
            row_blocks_X = []
            for idy, y in enumerate(norm_feature):
                S = torch.mm(x, y.transpose(0, 1))
                scale = np.log(self.delta / (1 - self.delta) * S.size(1)) / self.epsilon
                S_hat = f.softmax(S * scale, dim=1)
                row_blocks_X.append(S_hat)
                row_blocks_S.append(S)
            row_blocks_X = torch.cat(row_blocks_X, dim=1)
            all_blocks_S.append(row_blocks_S)
            all_blocks_X.append(row_blocks_X)
        all_blocks_X = torch.cat(all_blocks_X, dim=0)
        return all_blocks_S, all_blocks_X

    def forward(self, feature_ls):
        S, X = self.gen_X_S(feature_ls)
        loss = 0
        if 'pairwise' in C.LOSS:
            loss += self.pairwise_loss(S)
        if 'triplewise' in C.LOSS:
            loss += self.triplewise_loss(S)
        return loss
