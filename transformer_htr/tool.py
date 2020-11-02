import unicodedata
import editdistance
import numpy as np
from torch import from_numpy
from skimage.transform import resize

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return from_numpy(subsequent_mask) == 0

def get_metrics(predicts, ground_truth, norm_accentuation=False, norm_punctuation=False):
    """Calculate Character Error Rate (CER), Word Error Rate (WER)"""

    if len(predicts) == 0 or len(ground_truth) == 0:
        return (1, 1)

    cer, wer, ser = [], [], []

    for (pd, gt) in zip(predicts, ground_truth):
        pd, gt = pd.lower(), gt.lower()

        if norm_accentuation:
            pd = unicodedata.normalize("NFKD", pd).encode("ASCII", "ignore").decode("ASCII")
            gt = unicodedata.normalize("NFKD", gt).encode("ASCII", "ignore").decode("ASCII")

        if norm_punctuation:
            pd = pd.translate(str.maketrans("", "", string.punctuation))
            gt = gt.translate(str.maketrans("", "", string.punctuation))

        pd_cer, gt_cer = list(pd), list(gt)
        dist = editdistance.eval(pd_cer, gt_cer)
        cer.append(dist / (max(len(pd_cer), len(gt_cer))))

        pd_wer, gt_wer = pd.split(), gt.split()
        dist = editdistance.eval(pd_wer, gt_wer)
        wer.append(dist / (max(len(pd_wer), len(gt_wer))))

    metrics = [cer, wer]
    metrics = np.mean(metrics, axis=1)

    return metrics

class Tokenizer():
    def __init__(self, max_len=140):
        self.max_len = max_len
        vocab =  "'"+' !"#$%&()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}'
        self.PAD = '<p>'
        self.START = '<s>'
        self.END = '</s>'
        self.char2token = {self.PAD:0, self.START:1, self.END:2}
        self.token2char = {v:k for k,v in self.char2token.items()}
        offset = len(self.char2token.keys())
        for i, c in enumerate(vocab):
            self.char2token[c] = i+offset
            self.token2char[i+offset] = c
    
    def encode(self, string):
        N = self.max_len
        tokens = np.zeros(N)
        tokens[0]=self.char2token[self.START]
        l = len(string)
        for n in range(1, N-1):
            if n-1 < l:
                try:
                    token = self.char2token[string[n-1]]
                except KeyError:
                    raise ValueError("Illegal character encountered:\n\t"+str(string[n-1])+"\n\tOnly use characters with ASCII hexcodes between 0x20 and 0x7E (inclusive)")
                else:
                    tokens[n] = token
            else:
                tokens[n] = self.char2token[self.PAD]
        tokens[N-1] = self.char2token[self.END]
        return from_numpy(tokens).long()

    def decode(self, tokens):
        string = ''
        if isinstance(tokens, int):
          tokens = [tokens]
        for token in tokens:
            token = token.item()
            try:
                char = self.token2char[token]
            except KeyError:
                raise ValueError("Unknown token code encountered:\n\t"+str(token))
            else:
                string += char
        return string

def resize_image(img):
    HEIGHT = 64
    WIDTH = 2227
    w, h = img.shape    
    nw, nh = int(w * HEIGHT/h), HEIGHT
    if nw < 10 : nw = 10
    img = resize(img, (nw, nh))
    a1 = int((WIDTH-nw)/2)
    a2= WIDTH-nw-a1
    pad1 = np.zeros((a1, HEIGHT), dtype=np.uint8)
    pad2 = np.zeros((a2, HEIGHT), dtype=np.uint8)
    img = np.concatenate((pad1, img, pad2), axis=0)
    img = np.stack((img,)*3, axis=-1)
    return img
