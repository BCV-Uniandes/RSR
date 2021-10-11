import torch
import torch.nn as nn
import lpips
import torchvision
# from misc.kernel_loss import shave_a2b

class LPIPS(nn.Module):
    def __init__(self, net='alex', verbose=True, device='cpu', vgg19=False):
        super().__init__()
        if vgg19:
            self.lpips = VGGFeatureExtractor(device=device).to(device)
        else:
            self.lpips = lpips.LPIPS(net=net, verbose=verbose).to(device)
        # imagenet normalization for range [-1, 1]
        self.lpips.eval()

        for param in self.lpips.parameters():
            param.requires_grad = False        

    def perceptual_rec(self, x, y):
        loss_rgb = nn.L1Loss()(x, y)
        loss = loss_rgb + self(x, y)
        return loss

    @torch.no_grad()
    def forward(self, x, y):
        # normalization -1,+1
        # if x.size(-1) > y.size(-1):
        #     x = shave_a2b(x, y)
        # elif x.size(-1) < y.size(-1):
        #     y = shave_a2b(y, x)
        lpips_value = self.lpips(x, y, normalize=True)
        return lpips_value.mean()