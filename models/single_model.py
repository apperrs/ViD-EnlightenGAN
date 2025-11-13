import torch

from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
import random
from . import networks
import copy


def find_first_index_of_new_folder(sorted_paths):
    import os

    if not sorted_paths:
        return -1

    first_folder = os.path.dirname(sorted_paths[0])

    for i, path in enumerate(sorted_paths):
        current_folder = os.path.dirname(path)
        if current_folder != first_folder:
            return i

    return -1


def torch_sequential_cat(tensors, dim=0):
    if len(tensors) < 2:
        raise ValueError("At least two tensors are required for concatenation.")

    ndim = tensors[0].dim()
    for t in tensors[1:]:
        if t.dim() != ndim:
            raise ValueError(f"All input tensors must be of identical dimensionality. Detected mismatched dimensions: {ndim}D and {t.dim()}D.")

    result = tensors[0]
    for i in range(1, len(tensors)):
        for d in range(ndim):
            if d != dim and result.size(d) != tensors[i].size(d):
                raise ValueError(f"Shape mismatch in non-concat dimension {d}: {result.size(d)} vs {tensors[i].size(d)}")

        result = torch.cat([result, tensors[i]], dim=dim)

    return result


class SingleModel(BaseModel):
    def name(self):
        return 'SingleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)

        self.vgg_loss = networks.PerceptualLoss(opt)

        self.vgg_loss.cuda()
        self.vgg = networks.load_vgg16("./model", self.gpu_ids)
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        skip = True if opt.skip > 0 else False
        self.netG_A = networks.define_G(opt.input_nc * 2, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm,
                                        not opt.no_dropout, self.gpu_ids, skip=skip, opt=opt)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc * 2, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, False)
            self.netD_P = networks.define_D(opt.input_nc * 2, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_patchD, opt.norm, use_sigmoid, self.gpu_ids, True)
            self.netD_G = networks.define_D(2, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, False)
            self.netD_P_G = networks.define_D(2, opt.ndf,
                                              opt.which_model_netD,
                                              opt.n_layers_patchD, opt.norm, use_sigmoid, self.gpu_ids, True)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_P, 'D_P', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_B_pool = ImagePool(opt.pool_size)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            self.criterionCycle = torch.nn.L1Loss()

            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_G = torch.optim.Adam(self.netD_G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_P_G = torch.optim.Adam(self.netD_P_G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        if self.isTrain:
            networks.print_network(self.netD_A)
            if self.opt.patchD:
                networks.print_network(self.netD_P)
        if opt.isTrain:
            self.netG_A.train()
        else:
            self.netG_A.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        input = {k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
                 for k, v in input.items()}

        input_A = input['A'].clone()
        input_B = input['B'].clone() if input['B'] is not None else None

        A_paths = list(input['A_paths'])
        B_paths = list(input['B_paths']) if 'B_paths' in input else None

        split_index = find_first_index_of_new_folder(A_paths)
        if split_index in [1, 2]:
            A_paths = A_paths[split_index:]
            if B_paths: B_paths = B_paths[split_index:]
            input_A = input_A[split_index:]
            if input_B is not None: input_B = input_B[split_index:]
        elif split_index != -1:
            A_paths = A_paths[:split_index]
            if B_paths: B_paths = B_paths[:split_index]
            input_A = input_A[:split_index]
            if input_B is not None: input_B = input_B[:split_index]

        if len(input_A) >= 2:
            input_A_combined = torch_sequential_cat([input_A[:-1], input_A[1:]],
                                                    dim=1)
            input_B_combined = torch_sequential_cat([input_B[:-1], input_B[1:]],
                                                    dim=1) if input_B is not None else None
            A_paths = A_paths[1:]
            if B_paths: B_paths = B_paths[1:]
        else:
            raise RuntimeError(f"At least 2 frames of data are required, but currently there are only {len(input_A)} frames.")

        self.input_A.resize_(input_A_combined.shape).copy_(input_A_combined)
        if input_B_combined is not None:
            self.input_B.resize_(input_B_combined.shape).copy_(input_B_combined)
        self.input_img.resize_(input_A_combined.shape).copy_(input_A_combined)

        self.image_paths = A_paths
        self.image_pathsB = B_paths


    def set_predict_input(self, input):
        input = {k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
                 for k, v in input.items()}

        input_A = input['A'].clone()
        input_B = input['B'].clone() if input['B'] is not None else None

        A_paths = list(input['A_paths'])
        B_paths = list(input['B_paths']) if 'B_paths' in input else None

        if len(input_A) >= 2:
            input_A_combined = torch_sequential_cat([input_A[:-1], input_A[1:]],
                                                    dim=1)
            input_B_combined = torch_sequential_cat([input_B[:-1], input_B[1:]],
                                                    dim=1) if input_B is not None else None
            A_paths = A_paths[1:]
            if B_paths: B_paths = B_paths[1:]
        else:
            raise RuntimeError(f"At least 2 frames of data are required, but currently there are only {len(input_A)} frames.")

        self.input_A.resize_(input_A_combined.shape).copy_(input_A_combined)
        if input_B_combined is not None:
            self.input_B.resize_(input_B_combined.shape).copy_(input_B_combined)
        self.input_img.resize_(input_A_combined.shape).copy_(input_A_combined)
        self.image_paths = A_paths
        self.image_pathsB = B_paths

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise / 255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A)) / (torch.max(self.real_A) - torch.min(self.real_A))

        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A)
        else:
            self.fake_B = self.netG_A.forward(self.real_A)

        self.real_B = Variable(self.input_B, volatile=True)

    def predict(self):
        self.real_A = Variable(self.input_A, volatile=True)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise / 255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A)) / (torch.max(self.real_A) - torch.min(self.real_A))

        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A, _ = self.netG_A.forward(self.real_A)
        else:
            self.fake_B = self.netG_A.forward(self.real_A)

        real_A = util.tensor2im(self.real_A.data[:, 3:])
        fake_B = util.tensor2im(self.fake_B.data)

        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])

    def convert_video_to_grayscale_pairs(self, video_tensor):
        assert video_tensor.size(1) == 6, "The number of channels must be 6 (3 front and 3 rear)"

        prev_frames = video_tensor[:, :3, :, :]
        next_frames = video_tensor[:, 3:, :, :]

        prev_frames = (prev_frames + 1) / 2
        next_frames = (next_frames + 1) / 2

        grayscale_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=video_tensor.device).view(1, 3, 1, 1)

        prev_gray = (prev_frames * grayscale_weights).sum(dim=1, keepdim=True)
        next_gray = (next_frames * grayscale_weights).sum(dim=1, keepdim=True)

        grayscale_pairs = torch.cat([prev_gray, next_gray], dim=1)

        return grayscale_pairs

    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake, use_ragan):
        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake.detach())
        if use_ragan:
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) + self.criterionGAN(
                pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B
        fake_B = torch_sequential_cat([fake_B[:-1], fake_B[1:]], dim=1)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B[1:], fake_B, True)
        self.loss_D_A.backward()

    def backward_D_G(self):
        fake_B = self.fake_B
        fake_B = torch_sequential_cat([fake_B[:-1], fake_B[1:]], dim=1)
        self.loss_D_G = self.backward_D_basic(self.netD_G, self.convert_video_to_grayscale_pairs(self.real_B[1:]),
                                              self.convert_video_to_grayscale_pairs(fake_B), True)
        self.loss_D_G.backward()

    def backward_D_P(self):
        fake_patch = torch_sequential_cat([self.fake_patch[:-1], self.fake_patch[1:]],
                                          dim=1)
        loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch[1:], fake_patch, False)
        real_patch_1 = torch.stack(self.real_patch_1)
        fake_patch_1 = torch.stack(self.fake_patch_1)
        fake_patch_1 = torch_sequential_cat([fake_patch_1[:-1], fake_patch_1[1:]],
                                            dim=2)

        for i in range(fake_patch_1.shape[0]):
            loss_D_P += self.backward_D_basic(self.netD_P, real_patch_1[i], fake_patch_1[i], False)
        self.loss_D_P = loss_D_P / float(self.opt.patchD_3 + 1)

        self.loss_D_P.backward()

    def backward_D_P_G(self):
        fake_patch = torch_sequential_cat([self.fake_patch[:-1], self.fake_patch[1:]],
                                          dim=1)
        loss_D_P_G = self.backward_D_basic(self.netD_P_G, self.convert_video_to_grayscale_pairs(self.real_patch[1:]),
                                           self.convert_video_to_grayscale_pairs(fake_patch), False)
        real_patch_1 = torch.stack(self.real_patch_1)
        fake_patch_1 = torch.stack(self.fake_patch_1)
        fake_patch_1 = torch_sequential_cat([fake_patch_1[:-1], fake_patch_1[1:]],
                                            dim=2)

        for i in range(fake_patch_1.shape[0]):
            loss_D_P_G += self.backward_D_basic(self.netD_P_G, self.convert_video_to_grayscale_pairs(real_patch_1[i]),
                                                self.convert_video_to_grayscale_pairs(fake_patch_1[i]), False)
        self.loss_D_P_G = loss_D_P_G / float(self.opt.patchD_3 + 1)

        self.loss_D_P_G.backward()

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_img = Variable(self.input_img)

        self.fake_B, self.latent_real_A, self.gray = self.netG_A.forward(self.real_img)
        w = self.real_A.size(3)
        h = self.real_A.size(2)
        w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))

        import torch.nn.functional as F

        self.fake_patch = self.fake_B[:, :, h_offset:h_offset + self.opt.patchSize,
                          w_offset:w_offset + self.opt.patchSize]
        self.fake_patch = F.interpolate(
            self.fake_patch,
            size=(self.opt.patchSize, self.opt.patchSize),
            mode='bilinear'
        )
        self.real_patch = self.real_B[:, :, h_offset:h_offset + self.opt.patchSize,
                          w_offset:w_offset + self.opt.patchSize]
        self.input_patch = self.real_A[:, :, h_offset:h_offset + self.opt.patchSize,
                           w_offset:w_offset + self.opt.patchSize]

        self.fake_patch_1 = []
        self.real_patch_1 = []
        self.input_patch_1 = []
        w = self.real_A.size(3)
        h = self.real_A.size(2)

        for i in range(self.opt.patchD_3):
            scale = random.choice([1, 2, 3])
            current_size = self.opt.patchSize * scale

            w_offset_1 = random.randint(0, max(0, w - current_size - 1))
            h_offset_1 = random.randint(0, max(0, h - current_size - 1))

            fake_patch = self.fake_B[:, :,
                         h_offset_1:h_offset_1 + current_size,
                         w_offset_1:w_offset_1 + current_size]
            real_patch = self.real_B[:, :,
                         h_offset_1:h_offset_1 + current_size,
                         w_offset_1:w_offset_1 + current_size]
            input_patch = self.real_A[:, :,
                          h_offset_1:h_offset_1 + current_size,
                          w_offset_1:w_offset_1 + current_size]

            fake_patch = F.interpolate(fake_patch, size=(self.opt.patchSize, self.opt.patchSize), mode='bilinear')
            real_patch = F.interpolate(real_patch, size=(self.opt.patchSize, self.opt.patchSize), mode='bilinear')
            input_patch = F.interpolate(input_patch, size=(self.opt.patchSize, self.opt.patchSize), mode='bilinear')


            self.fake_patch_1.append(fake_patch)
            self.real_patch_1.append(real_patch)
            self.input_patch_1.append(input_patch)

    def backward_G(self):
        fake_B = torch_sequential_cat([self.fake_B[:-1], self.fake_B[1:]],
                                      dim=1)
        real_B = self.real_B[1:]
        pred_fake = self.netD_A.forward(fake_B)
        pred_real = self.netD_A.forward(real_B)

        self.loss_G_A = (self.criterionGAN(pred_real - torch.mean(pred_fake), False) +
                         self.criterionGAN(pred_fake - torch.mean(pred_real), True)) / 2

        pred_fake_G = self.netD_G.forward(self.convert_video_to_grayscale_pairs(fake_B))
        pred_real_G = self.netD_G.forward(self.convert_video_to_grayscale_pairs(real_B))
        self.loss_G_A += 1.0 * (self.criterionGAN(pred_real_G - torch.mean(pred_fake_G), False) +
                                self.criterionGAN(pred_fake_G - torch.mean(pred_real_G), True)) / 2

        loss_G_A = 0

        fake_patch = torch_sequential_cat([self.fake_patch[:-1], self.fake_patch[1:]],
                                          dim=1)
        pred_fake_patch = self.netD_P.forward(fake_patch)

        loss_G_A += self.criterionGAN(pred_fake_patch, True)

        fake_patch_1 = torch.stack(self.fake_patch_1)
        fake_patch_1 = torch_sequential_cat([fake_patch_1[:-1], fake_patch_1[1:]],
                                            dim=2)

        for i in range(fake_patch_1.shape[0]):
            pred_fake_patch_1 = self.netD_P.forward(fake_patch_1[i])
            loss_G_A += self.criterionGAN(pred_fake_patch_1, True)

        self.loss_G_A += loss_G_A / float(self.opt.patchD_3 + 1)

        loss_G_P_A = 0
        fake_patch = torch_sequential_cat([self.fake_patch[:-1], self.fake_patch[1:]], dim=1)
        pred_fake_patch = self.netD_P_G.forward(self.convert_video_to_grayscale_pairs(fake_patch))
        loss_G_P_A += self.criterionGAN(pred_fake_patch, True)
        fake_patch_1 = torch.stack(self.fake_patch_1)
        fake_patch_1 = torch_sequential_cat([fake_patch_1[:-1], fake_patch_1[1:]], dim=2)
        for i in range(fake_patch_1.shape[0]):
            pred_fake_patch_1 = self.netD_P_G.forward(self.convert_video_to_grayscale_pairs(fake_patch_1[i]))
            loss_G_P_A += self.criterionGAN(pred_fake_patch_1, True)
        self.loss_G_A += loss_G_P_A / float(self.opt.patchD_3 + 1)

        self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg,
                                                         self.fake_B[:, :, :, :],
                                                         self.real_A[:, 3:, :,
                                                         :]) * self.opt.vgg if self.opt.vgg > 0 else 0
        loss_vgg_patch = self.vgg_loss.compute_vgg_loss(self.vgg,
                                                        self.fake_patch[:, :, :, :],
                                                        self.input_patch[:, 3:, :, :]) * self.opt.vgg
        for i in range(self.opt.patchD_3):
            loss_vgg_patch += self.vgg_loss.compute_vgg_loss(self.vgg,
                                                             self.fake_patch_1[i][:, :, :, :],
                                                             self.input_patch_1[i][:, 3:, :, :]) * self.opt.vgg
        self.loss_vgg_b += loss_vgg_patch / float(self.opt.patchD_3 + 1)
        self.loss_G = self.loss_G_A + self.loss_vgg_b
        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.optimizer_D_A.zero_grad()
        self.backward_D_A()

        self.optimizer_D_G.zero_grad()
        self.backward_D_G()

        self.optimizer_D_P_G.zero_grad()
        self.backward_D_P_G()

        self.optimizer_D_P.zero_grad()
        self.backward_D_P()
        self.optimizer_D_A.step()
        self.optimizer_D_P.step()
        self.optimizer_D_G.step()
        self.optimizer_D_P_G.step()

    def get_current_errors(self, epoch):
        D_A = self.loss_D_A.item()
        D_P = self.loss_D_P.item() if self.opt.patchD else 0
        G_A = self.loss_G_A.item()
        D_G = self.loss_D_G.item()
        D_P_G = self.loss_D_P_G.item()

        vgg = self.loss_vgg_b.item() / self.opt.vgg if self.opt.vgg > 0 else 0
        return OrderedDict([('D_A', D_A), ('G_A', G_A), ("vgg", vgg), ("D_P", D_P), ("D_G", D_G), ("D_P_G", D_P_G)])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data[:, 3:])
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data[:, 3:])

        fake_patch = util.tensor2im(self.fake_patch.data)
        real_patch = util.tensor2im(self.real_patch.data[:, 3:, :, :])

        input_patch = util.tensor2im(self.input_patch.data[:, 3:, :, :])

        return OrderedDict([('real_A', real_A), ('fake_B', fake_B),  ('real_B', real_B),
                            ('real_patch', real_patch), ('fake_patch', fake_patch), ('input_patch', input_patch)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netD_P, 'D_P', label, self.gpu_ids)

    def update_learning_rate(self):
        if self.opt.new_lr:
            lr = self.old_lr / 2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        if self.opt.patchD:
            for param_group in self.optimizer_D_P.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr