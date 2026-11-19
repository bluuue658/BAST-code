import torch
import torch.nn as nn
import torch.nn.functional as F
# ================================================================
# Module 1: Ensemble Pseudo-Label
# ================================================================
def ensemble_pseudo_label(teacher_logit, student_logit=None,
                          alpha=0.7, temperature=0.1, nms_fn=None):
    with torch.no_grad():
        t_prob = F.softmax(teacher_logit, dim=1)
        if student_logit is None or alpha >= 1.0:
            masks = (t_prob >= 0.5).type(torch.int64)
            pseudo = masks[:, 1].contiguous()
            conf = t_prob[:, 1:2].contiguous()
            if nms_fn is not None:
                pseudo = nms_fn(pseudo)
            return pseudo.long(), conf, 1.0
        s_prob = F.softmax(student_logit, dim=1)
        ensemble_prob = alpha * t_prob + (1.0 - alpha) * s_prob
        masks = (ensemble_prob >= 0.5).type(torch.int64)
        pseudo = masks[:, 1].contiguous()
        conf = ensemble_prob[:, 1:2].contiguous()
        if nms_fn is not None:
            pseudo = nms_fn(pseudo)
        t_pred = (t_prob[:, 1] >= 0.5).long()
        s_pred = (s_prob[:, 1] >= 0.5).long()
        agree = float((t_pred == s_pred).float().mean().item())
    return pseudo.long(), conf, agree
# ================================================================
# Module 2: Boundary Weight Map
# ================================================================
def make_boundary_band(mask, radius=1):
 
    is_2d = (mask.dim() == 3)
    x = mask.float().unsqueeze(1)
    k = 2 * radius + 1
    pool_fn = F.max_pool2d if is_2d else F.max_pool3d
    dil = pool_fn(x, kernel_size=k, stride=1, padding=radius)
    ero = 1.0 - pool_fn(1.0 - x, kernel_size=k, stride=1, padding=radius)
    return (dil - ero).clamp(0, 1)
def boundary_weight_map(pseudo_label, lambda_boundary=1.0, radius=1,
                        iter_num=0, rampup=5000):
    ramp = min(float(iter_num) / max(float(rampup), 1.0), 1.0) if rampup > 0 else 1.0
    eff_lambda = lambda_boundary * ramp
    fg = (pseudo_label > 0).float()
    band = make_boundary_band(fg, radius=radius)
    weight = 1.0 + eff_lambda * band.squeeze(1)
    return weight, band
# ================================================================
# Module 3: Boundary Auxiliary Head
# ================================================================
class BoundaryAuxHead(nn.Module):
    def __init__(self, in_channels=2, mid_channels=16, is_2d=False):
        super().__init__()
        self.is_2d = is_2d
        Conv = nn.Conv2d if is_2d else nn.Conv3d
        BN = nn.BatchNorm2d if is_2d else nn.BatchNorm3d

        self.head = nn.Sequential(
            Conv(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            BN(mid_channels),
            nn.ReLU(inplace=True),
            Conv(mid_channels, 1, kernel_size=1, bias=True),
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.head:
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, seg_logit):
        return torch.sigmoid(self.head(seg_logit))
def generate_boundary_gt(label_map, radius=1):
    fg = (label_map > 0).float()
    return make_boundary_band(fg, radius=radius)
def boundary_aux_loss(boundary_pred, boundary_gt, valid_mask=None):
    if valid_mask is not None:
        pred = boundary_pred[valid_mask > 0]
        gt = boundary_gt[valid_mask > 0]
        if pred.numel() == 0:
            return torch.tensor(0.0, device=boundary_pred.device)
        return F.binary_cross_entropy(pred, gt)
    else:
        return F.binary_cross_entropy(boundary_pred, boundary_gt)
