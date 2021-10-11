import importlib
import torch
from collections import OrderedDict
import copy

from basicsr.models.srgan_model import SRGANModel

loss_module = importlib.import_module('basicsr.models.losses')


class RSRModel(SRGANModel):
    """RSRModel model for single image super-resolution."""
    def __init__(self, opt):
        super(RSRModel, self).__init__(opt)
        
        self.eps = self.opt['eps'] / 255.

    def optimize_parameters(self, current_iter, delta):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        for p in self.net_g.parameters():
            p.requires_grad = True

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq + delta)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0
                and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(
                    self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss (relativistic gan)
            real_d_pred = self.net_d(self.gt).detach()
            fake_g_pred = self.net_d(self.output)
            l_g_real = self.cri_gan(
                real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            l_g_fake = self.cri_gan(
                fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
            l_g_gan = (l_g_real + l_g_fake) / 2

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # gan loss (relativistic gan)

        # In order to avoid the error in distributed training:
        # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
        # the variables needed for gradient computation has been modified by
        # an inplace operation",
        # we separate the backwards for real and fake, and also detach the
        # tensor for calculating mean.

        # real
        fake_d_pred = self.net_d(self.output).detach()
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(
            real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(
            fake_d_pred - torch.mean(real_d_pred.detach()),
            False,
            is_disc=True) * 0.5
        l_d_fake.backward()
        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(loss_dict)


    def pgd_attack(self, current_iter):
        def clamp(X, lower_limit, upper_limit):
            return torch.max(torch.min(X, upper_limit), lower_limit)
        import visdom 
        import torchvision
        vis = visdom.Visdom(env='debug')
        alpha = 1./255.
        
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        for p in self.net_g.parameters():
            p.requires_grad = False
        
        delta = torch.zeros(self.lq.shape[0],self.lq.shape[1],int(self.lq.shape[2]//self.opt['noise_scale']),int(self.lq.shape[3]//self.opt['noise_scale'])).to(self.device)
        for i in range(self.lq.shape[1]):
            delta[:, i, :, :].uniform_(-self.eps, self.eps)
        delta = torch.nn.functional.interpolate(delta, size=(self.lq.shape[2],self.lq.shape[3]), mode='nearest', recompute_scale_factor =True)
        delta.requires_grad = True
        for reps in range(self.opt['iters_attack']):

            output = self.net_g(torch.clamp(self.lq + delta, 0., 1.))
            l_g_total = 0
            # perceptual loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(output, self.gt)
                l_g_total += l_g_pix
            # pixel loss
            if self.cri_perceptual:
                l_g_percep, _ = self.cri_perceptual(output, self.gt)
                l_g_total += l_g_percep
            if (current_iter % 5000 == 0):
                print("iteration {:.0f}, loss:{:.4f}".format(reps, l_g_total))

            l_g_total.backward()
            
            grad = delta.grad.detach()
            d = clamp(delta + alpha * torch.sign(grad), 
                      torch.tensor(-self.eps, device=self.device), 
                      torch.tensor(self.eps, device=self.device))
            d = clamp(d, 0 - self.lq, 1 - self.lq)
            delta.data = d
            delta.grad.zero_()
            
        return delta