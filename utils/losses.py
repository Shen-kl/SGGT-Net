import torch
import torch.distributions as tdist
import torch.nn as nn
from fsspec import Callback
from typing import Callable
import math
import torch.nn.functional as F


def mask_reduce_losses(losses, mask):
    # Compute mean over vehicles, sum over time, masked
    # losses: (N,T)
    # mask: (N,T), \in {0,1}

    n_pred = torch.sum(mask, dim=0)  # (T,)
    t_losses = torch.sum(losses * mask, dim=0) / n_pred  # (T,)
    return torch.sum(t_losses)


class NLLMDNLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, sigma, pi, x, mask, is_tril=False):
        """
        N = batch size
        T = number of time steps
        k = number of states
        m = number of mixtures

        mu.shape (N, T, m, k)
        sigma.shape (N, T, m, k, k)
        pi.shape (N, m)
        x.shape (N, T, k)
        mask.shape (N, T)
        """

        pi_expanded = pi.unsqueeze(1).expand(-1, mu.shape[1], -1)
        mix = tdist.Categorical(logits=pi_expanded)  # Broadcast this over time dim.
        L = sigma if is_tril else torch.linalg.cholesky(sigma)
        mvn = tdist.MultivariateNormal(mu, scale_tril=L)
        gmm = tdist.MixtureSameFamily(mix, mvn)
        comp_loss = gmm.log_prob(x).neg()  # (N, T)
        return mask_reduce_losses(comp_loss, mask)


class EWTALoss(nn.Module):
    """Evolving Winner Takes All loss."""

    def __init__(self):
        super().__init__()
        self.huber = nn.HuberLoss(reduction='none')

    def forward(self, mu, x, mask, w=1):
        """
        N = batch size
        T = number of time steps
        k = number of states
        m = number of mixtures
        w = max number of winners

        mu.shape (N, T, m, k)
        x.shape (N, T, k)
        mask.shape (N, T)
        """
        x = x.unsqueeze(2).expand(-1, -1, mu.size(2), -1)  # (N, T, m, k)
        reg_loss = self.huber(mu, x).sum(dim=-1)  # (N, T, m)
        masked_loss = reg_loss * mask[..., None]  # (N, T, m)
        masked_time = masked_loss.sum(1)  # (N, m)
        vals, _ = torch.topk(masked_time, k=w, dim=-1, largest=False)
        loss = vals.mean()
        return loss


class ModeDist(nn.Module):
    """
    Calculates the summed pairwise distance between mixtures.
    """

    def __init__(self):
        super().__init__()
        self.pwd = nn.PairwiseDistance()

    def forward(self, pred_state, all_preds, mask):
        """
        N = batch size
        T = number of time steps
        k = number of states
        m = number of mixtures

        pred_state.shape (N, T k)
        all_preds.shape (N, T, m, k)
        mask.shape (N, T)
        """
        pwd_loss = self.pwd(pred_state.unsqueeze(2), all_preds)  # (N, T, m)
        return mask_reduce_losses(pwd_loss.sum(-1), mask)

@torch.no_grad()
def caculate_evaluation(graph_data, all_states_prediction, x_update, centroids_x, config, w):
    '''
    计算评价指标 包括位置与速度的RMSE
    :param graph_data:
    :param all_states_prediction:
    :param centroids_per_node_x:
    :param config:
    :return: loss
    '''
    mse = nn.MSELoss(reduction='none')
    # 质心偏移位置
    centroids_per_node_x = centroids_x[graph_data.batch].unsqueeze(1).unsqueeze(1)
    centroids_per_node_x_update = centroids_x[graph_data.batch].unsqueeze(1)
    # 反归一化
    all_states_prediction_renorm = all_states_prediction.clone()
    x_update_renorm = x_update.clone()
    # [x,y,vx,vy] -> [x.vx.y.vy]
    all_states_prediction_renorm = all_states_prediction_renorm[..., [0, 2, 1, 3]]
    all_states_prediction_renorm[:, :, :, [0, 2]] = all_states_prediction_renorm[:, :, :,[0, 2]] * config['S_POS'] + centroids_per_node_x
    all_states_prediction_renorm[:, :, :, [1, 3]] = all_states_prediction_renorm[:, :, :, [1, 3]] * config['S_VEL']

    x_update_renorm = x_update_renorm[..., [0, 2, 1, 3]]
    x_update_renorm[:, :, [0, 2]] = x_update_renorm[:, :, [0, 2]] * config['S_POS'] + centroids_per_node_x_update
    x_update_renorm[:, :, [1, 3]] = x_update_renorm[:, :, [1, 3]] * config['S_VEL']


    # 计算预测偏差
    real_mask = graph_data.real_mask[..., 0].to(torch.float32)
    target = graph_data.y
    target = target.unsqueeze(2).expand(-1, -1, all_states_prediction_renorm.size(2), -1)  # (N, T, m, k)
    mse_loss_prediction = mse(all_states_prediction_renorm, target)  # (N, T, m)
    location_mixture_rmse = mse_loss_prediction[:, :, :, [0, 2]].sum(dim=-1)
    velocity_mixture_rmse = mse_loss_prediction[:, :, :, [1, 3]].sum(dim=-1)

    masked_location_mixture_rmse = location_mixture_rmse * real_mask[..., None]
    masked_velocity_mixture_rmse = velocity_mixture_rmse * real_mask[..., None]

    masked_location_mixture_rmse_time = masked_location_mixture_rmse.sum(1)  # (N, m)
    location_rmse_prediction, _ = torch.topk(masked_location_mixture_rmse_time, k=w, dim=-1, largest=False)
    location_rmse_prediction = torch.sqrt(torch.sum(location_rmse_prediction) / torch.sum(real_mask)).item()

    masked_velocity_mixture_rmse_time = masked_velocity_mixture_rmse.sum(1)  # (N, m)
    velocity_rmse_prediction, _ = torch.topk(masked_velocity_mixture_rmse_time, k=w, dim=-1, largest=False)
    velocity_rmse_prediction = torch.sqrt(torch.sum(velocity_rmse_prediction) / torch.sum(real_mask)).item()

    # 计算更新偏差
    real_mask = graph_data.real_mask[..., 0].to(torch.float32)
    target = graph_data.y[:,0,:].unsqueeze(1)
    mse_loss_update = mse(x_update_renorm, target)  # (N, T, m)
    location_rmse = mse_loss_update[:, :, [0, 2]].sum(-1)
    velocity_rmse = mse_loss_update[:, :, [1, 3]].sum(-1)

    masked_location_rmse = location_rmse * real_mask[:, 0, None]
    masked_velocity_rmse = velocity_rmse * real_mask[:, 0, None]

    location_rmse_update = torch.sqrt(torch.sum(masked_location_rmse) / torch.sum(real_mask)).item()

    velocity_rmse_update = torch.sqrt(torch.sum(masked_velocity_rmse) / torch.sum(real_mask)).item()

    x_update_renorm = x_update_renorm.detach()

    # 计算量测偏差
    mse_loss_measurement = mse(graph_data.measurement_future, target[:,:,[0,2]]).sum(-1)  # (N, T, m)
    masked_location_rmse = mse_loss_measurement * real_mask[:, 0, None]
    location_rmse_measurement = torch.sqrt(torch.sum(masked_location_rmse) / torch.sum(real_mask)).item()

    return (location_rmse_prediction, velocity_rmse_prediction, location_rmse_update,velocity_rmse_update,
            x_update_renorm,location_rmse_measurement)


class LossCompute_NLL(nn.Module):
    # 极大似然
    def __init__(self):
        super(LossCompute_NLL, self).__init__()

    def forward(self, output, outputSigam, labels):
        # labels (batch, frames_num, tg_num_max, feature_size)
        # output (batch, frames_num, tg_num_max, feature_size)
        # outputSigam (batch, frames_num, tg_num_max, feature_size)
        # loss = torch.sum(torch.sum(torch.sum(torch.sum(0.5 * torch.exp(-outputSigam) * pow((output - labels), 2)
        #                                                + 0.5 * outputSigam + 1, dim=0), dim=0), dim=0), dim=0)
        # return loss/output.shape[0]/output.shape[1]/output.shape[2]
        loss = 0.5 * torch.exp(-outputSigam) * pow((output - labels), 2) + 0.5 * outputSigam + 1
        loss = loss.mean()
        return loss


def gaussian_nll(residual, cov, eps=1e-5):
    """
    residual: (B, d)
    cov:      (B, d, d)  must be SPD
    return: scalar NLL
    """
    B, _, d = residual.shape
    I = torch.eye(d, device=residual.device, dtype=residual.dtype).unsqueeze(0)

    cov = cov + eps * I
    #
    # # Cholesky factorization: cov = L L^T
    L = torch.linalg.cholesky(cov)  # (B, d, d)

    # Solve cov^{-1} residual
    r = residual.permute(0,2,1)  # (B, d, 1)
    sol = torch.cholesky_solve(r, L)  # (B, d, 1)

    quad = torch.bmm(r.transpose(1, 2), sol).squeeze(-1).squeeze(-1)  # (B,)

    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)  # (B,)

    nll = 0.5 * (quad + logdet + d * math.log(2 * math.pi))
    return nll.mean()


def loss_measurement_nll(z, x_pred, S):
    """
    z:      (B, dz)
    x_pred: (B, dx)
    S:      (B, dz, dz)
    H:      (dz, dx)
    """
    B = z.shape[0]
    residual = z - x_pred

    return gaussian_nll(residual, S)

def loss_P_calibration(x_gt, x_pred, P, detach_mean=True):
    """
    x_gt:   (B, dx)
    x_pred: (B, dx)
    P:      (B, dx, dx)
    """
    if detach_mean:
        x_pred = x_pred.detach()

    residual = x_gt - x_pred
    return gaussian_nll(residual, P)


def loss_R_calibration(z, x_gt, R):
    """
    z:    (B, dz)
    x_gt: (B, dx)
    R:    (B, dz, dz)
    H:    (dz, dx)
    """
    B = z.shape[0]
    residual = z - x_gt

    return gaussian_nll(residual, R)


def trace_regularizer(M):
    """
    M: (B, d, d)
    """
    return torch.mean(torch.diagonal(M, dim1=-2, dim2=-1).sum(-1))


def focal_loss(prob, targets, mask, alpha=0.25, gamma=2.0):
    """
    prob: (E,)
    targets: (E,)
    """

    bce_loss = F.binary_cross_entropy(
        prob, targets, reduction='none'
    )

    pt = targets * prob + (1 - targets) * (1 - prob)

    focal_weight = (1 - pt) ** gamma

    if alpha is not None:
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
        focal_weight = alpha_t * focal_weight

    loss = focal_weight * bce_loss * mask

    return loss.mean()


def edge_f1_score(probs, targets, threshold=0.5):
    """
    logits: (N)
    targets: (N)
    """
    preds = (probs > threshold).float()
    # 计算 TP, FP, FN
    TP = (preds * targets).sum()
    FP = (preds * (1 - targets)).sum()
    FN = ((1 - preds) * targets).sum()

    eps = 1e-8

    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)

    return f1.item(), precision.item(), recall.item()