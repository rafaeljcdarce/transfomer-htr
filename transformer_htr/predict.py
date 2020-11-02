from .tool import subsequent_mask
from torch import from_numpy, ones, max, cat
def greedy_decode(model, src, max_len=140, start_symbol=1):
    model.eval()
    tokenizer=Tokenizer()
    src_mask = Variable(from_numpy(np.ones([src.size(0), 1, 560], dtype=np.bool)).cuda())
    memory = model.encode(src, src_mask)
    ys = ones(1, 1).fill_(start_symbol).long().cuda()
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .long().cuda()))
        prob = model.generator(out[:, -1])
        _, next_word = max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = cat([ys, 
                        ones(1, 1).long().cuda().fill_(next_word)], dim=1)
        if tokenizer.decode(next_word.item()) == '</s>':
            break
    out = ["".join(tokenizer.decode(y)) for y in ys.cpu().numpy()]
    return out
# print(greedy_decode(X))
# d = IamDataset()
# dataloader = torch.utils.data.DataLoader(d, batch_size=2, shuffle=False, num_workers=0)
# for epoch in range(1):
#   for batch_i, (imgs, labels_y, labels) in enumerate(dataloader):
#       print(greedy_decode(model, imgs[0].unsqueeze(0)))
#       break