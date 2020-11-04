from torch.utils.data.dataset import Dataset
from .tool import Tokenizer, preprocess_image
import os
import h5py
import numpy as np
from torch import from_numpy
from torch.autograd import Variable
from .tool import subsequent_mask
from torch import cuda

class HtrDataset(Dataset):
    def __init__(self, source='iam', pt='train', augment=False):

        #pt = ['train', 'valid', 'test']
        self.tokenizer = Tokenizer()
        self.dataset = dict()
        self.augment = augment
        source = os.path.join(f"{source}.hdf5")
        with h5py.File(source, "r") as f:
            self.dataset['img'] = np.array(f[pt]['dt'])
            self.dataset['lbl'] = np.array([x.decode() for x in f[pt]['gt']])

            if pt == 'train':
              randomize = np.arange(len(self.dataset['img']))
              np.random.seed(42)
              np.random.shuffle(randomize)
              self.dataset['img'] = self.dataset['img'][randomize]
              self.dataset['lbl'] = self.dataset['lbl'][randomize]


    def __len__(self):
        return len(self.dataset['lbl'])
    
    def __getitem__(self, index):
        '''
        line: (image path, label string)
        '''            
        img, string = self.dataset['img'][index], self.dataset['lbl'][index]
        img = preprocess_image(img, self.augment)
        img= np.transpose(img, (2, 0, 1))
        img = from_numpy(img).float()
        label = self.tokenizer.encode(string)
        if cuda.is_available():
            img=img.cuda()
            label=label.cuda()
        label_y = label[1:]
        label = label[:-1]

        return img, label_y, label

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, imgs, trg_y, trg, pad=0):
        self.src_mask = Variable(from_numpy(np.ones([imgs.size(0), 1, 560], dtype=np.bool)))
        if cuda.is_available():
            imgs=imgs.cuda()
            trg=trg.cuda()
            trg_y=trg_y.cuda()
            self.src_mask = self.src_mask.cuda()
        
        self.src = Variable(imgs, requires_grad=False)
        if trg is not None:
            self.trg = Variable(trg, requires_grad=False)
            self.trg_y = Variable(trg_y, requires_grad=False)
            self.trg_mask = self.make_std_mask(self.trg, pad)
            if cuda.is_available():
                self.trg_mask= self.trg_mask.cuda()
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return Variable(tgt_mask, requires_grad=False)