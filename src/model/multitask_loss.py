import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_var_cls = nn.Parameter(torch.zeros(()))
        self.log_var_reg = nn.Parameter(torch.zeros(()))
        self.ema_alpha = 0.9

    def forward(self, loss_cls, loss_reg):
        # EMA动态调整
        with torch.no_grad():
            self.log_var_cls.data = self.ema_alpha * self.log_var_cls + (1 - self.ema_alpha) * loss_cls.detach()
            self.log_var_reg.data = self.ema_alpha * self.log_var_reg + (1 - self.ema_alpha) * loss_reg.detach()
        
        return 0.7 * loss_cls + 0.3 * loss_reg