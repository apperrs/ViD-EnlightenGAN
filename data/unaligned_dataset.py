import torch
from torch import nn
import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset#, store_dataset
import random
from PIL import Image
import PIL
from pdb import set_trace as st


'''def sort_files_by_number(file_paths):
    import re
    def get_number(file_path):
        filename = os.path.basename(file_path)
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0

    return sorted(file_paths, key=get_number)


def store_dataset(dir):
    def is_image_file(filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    images = []
    all_path = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                all_path.append(path)

    all_path = sort_files_by_number(all_path)

    for path in all_path:
        img = Image.open(path).convert('RGB')
        images.append(img)

    return images, all_path

class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        # self.A_paths = make_dataset(self.dir_A)
        # self.B_paths = make_dataset(self.dir_B)
        self.A_imgs, self.A_paths = store_dataset(self.dir_A)
        self.B_imgs, self.B_paths = store_dataset(self.dir_B)

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')
        #A = A.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        A = self.transform(A)

        B_path = self.B_paths[index]
        B = Image.open(B_path).convert('RGB')
        #B = B.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        B = self.transform(B)


        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        A_img, B_img = A, B
        if self.opt.resize_or_crop == 'no':
            r, g, b = A_img[0] + 1, A_img[1] + 1, A_img[2] + 1
            A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
            A_gray = torch.unsqueeze(A_gray, 0)
            input_img = A_img
            # A_gray = (1./A_gray)/255.
        else:
            w = A_img.size(2)
            h = A_img.size(1)

            # A_gray = (1./A_gray)/255.
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx)
                B_img = B_img.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
                B_img = B_img.index_select(1, idx)
            if self.opt.vary == 1 and (not self.opt.no_flip) and random.random() < 0.5:
                times = random.randint(self.opt.low_times, self.opt.high_times) / 100.
                input_img = (A_img + 1) / 2. / times
                input_img = input_img * 2 - 1
            else:
                input_img = A_img
            if self.opt.lighten:
                B_img = (B_img + 1) / 2.
                B_img = (B_img - torch.min(B_img)) / (torch.max(B_img) - torch.min(B_img))
                B_img = B_img * 2. - 1
            r, g, b = input_img[0] + 1, input_img[1] + 1, input_img[2] + 1
            A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
            A_gray = torch.unsqueeze(A_gray, 0)

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path, 'input_img': A, 'A_gray': A_gray}

    def __len__(self):
        return max(len(self.A_paths), len(self.B_paths))

    def name(self):
        return 'AlignedDataset'''


def pad_tensor(input):
    
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

            padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
            input = padding(input).data
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.shape[2], input.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]


def sort_files_by_number(file_paths):
    import os
    import re

    def get_sort_key(file_path):
        dirname = os.path.dirname(file_path)
        filename = os.path.basename(file_path)

        numbers = re.findall(r'\d+', filename)
        file_num = int(numbers[0]) if numbers else 0

        # Return a tuple (directory, number) for sorting
        return (dirname, file_num)

    return sorted(file_paths, key=get_sort_key)


def store_dataset(dir):
    def is_image_file(filename):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    images = []
    all_path = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                all_path.append(path)

    all_path = sort_files_by_number(all_path)

    for path in all_path:
        img = Image.open(path).convert('RGB')
        images.append(img)

    return images, all_path


def get_transform(opt, x, y, h, w, scale_radio):
    transform_list = []
    #transform_list.append(transforms.RandomCrop(opt.fineSize))
    if opt.isTrain:
        transform_list += [transforms.Resize((int(scale_radio * h), int(scale_radio* w))),
                           transforms.Lambda(lambda img: img.crop((x, y, x + opt.fineSize, y + opt.fineSize))),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    else:
        transform_list += [transforms.Resize((int(scale_radio * h), int(scale_radio * w))),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        # self.A_paths = make_dataset(self.dir_A)
        # self.B_paths = make_dataset(self.dir_B)
        self.A_imgs, self.A_paths = store_dataset(self.dir_A)
        self.B_imgs, self.B_paths = store_dataset(self.dir_B)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        self.transform = get_transform

        self.random_y = random.randint(0, self.opt.fineSize - 1)
        self.random_x = random.randint(0, self.opt.fineSize - 1)
        self.scale_radio = 1

    def __getitem__(self, index):
        # A_path = self.A_paths[index % self.A_size]
        # B_path = self.B_paths[index % self.Bcombined_to_flow_size]

        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        A_img = self.A_imgs[index % self.A_size]
        B_img = self.B_imgs[index % self.B_size]
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        # A_size = A_img.size
        # B_size = B_img.size
        # A_size = A_size = (A_size[0]//16*16, A_size[1]//16*16)
        # B_size = B_size = (B_size[0]//16*16, B_size[1]//16*16)
        # A_img = A_img.resize(A_size, Image.BICUBIC)
        # B_img = B_img.resize(B_size, Image.BICUBIC)
        # A_gray = A_img.convert('LA')
        # A_gray = 255.0-A_gray

        w, h = A_img.size

        A_img = self.transform(self.opt, self.random_x, self.random_y, h, w, self.scale_radio)(A_img)
        B_img = self.transform(self.opt, self.random_x, self.random_y, h, w, self.scale_radio)(B_img)
        
        if self.opt.resize_or_crop == 'no':
            r,g,b = A_img[0]+1, A_img[1]+1, A_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
            input_img = A_img
            # A_gray = (1./A_gray)/255.
        else:
            w = A_img.size(2)
            h = A_img.size(1)
            
            # A_gray = (1./A_gray)/255.
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx)
                B_img = B_img.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
                B_img = B_img.index_select(1, idx)
            if self.opt.vary == 1 and (not self.opt.no_flip) and random.random() < 0.5:
                times = random.randint(self.opt.low_times,self.opt.high_times)/100.
                input_img = (A_img+1)/2./times
                input_img = input_img*2-1
            else:
                input_img = A_img
            if self.opt.lighten:
                B_img = (B_img + 1)/2.
                B_img = (B_img - torch.min(B_img))/(torch.max(B_img) - torch.min(B_img))
                B_img = B_img*2. -1
            r,g,b = input_img[0]+1, input_img[1]+1, input_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
        return {'A': A_img, 'B': B_img, 'A_gray': A_gray, 'input_img': input_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'

    def set_random(self):
        self.scale_radio = random.uniform(0.36, 1)

        w, h = self.A_imgs[0].size
        w, h = int(w * self.scale_radio), int(h * self.scale_radio)

        self.random_y = random.randint(0, h - self.opt.fineSize - 1)
        self.random_x = random.randint(0, w - self.opt.fineSize - 1)

