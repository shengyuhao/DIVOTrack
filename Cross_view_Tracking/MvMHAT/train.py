import os
import config as C
os.environ["CUDA_VISIBLE_DEVICES"] = C.TRAIN_GPUS
import torch
import torchvision.models as models
from loss import CycleS
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from loader import Loader
from torch.cuda.amp import autocast as autocast
import pdb


def train(epoch):
    model.train()
    epoch_loss = 0
    for step_i, data in tqdm(enumerate(dataset_train), total=len(dataset_train)):
        optimizer.zero_grad()
        feature_ls = []
        with autocast():
            for view_i in range(len(data)):
                for frame_i in range(C.FRAMES):
                    img, box, lbl, scn = data[view_i][frame_i]
                    feature = model(img.squeeze(0).cuda())
                    feature_ls.append(feature)
            step_loss = cycle_loss(feature_ls)
            epoch_loss += step_loss.item()
            if epoch >= 0:
                step_loss.backward()
                optimizer.step()
    return epoch_loss / (step_i + 1)

if __name__ == '__main__':
    datasets = []
    for dataset in C.TRAIN_DATASET:
        datasets.append(Loader(views=C.VIEWS, frames=C.FRAMES, mode='train', dataset=dataset))
    datasets = ConcatDataset(datasets)
    dataset_train = DataLoader(datasets, num_workers=0, pin_memory=True, shuffle=C.LOADER_SHUFFLE)

    if C.NETWORK == 'resnet':
        model = models.resnet50(pretrained=False)
    model = model.cuda()


    if C.RE_ID:
        checkpoint_path = C.TRAIN_RESUME
        ckp = torch.load(checkpoint_path)['model']
    else:
        if C.NETWORK == 'resnet':
            checkpoint_path = './models/pretrained.pth'
        ckp = torch.load(checkpoint_path)

    model.load_state_dict(ckp)
    cycle_loss = CycleS()

    optimizer = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


    max_epoch = C.MAX_EPOCH
    max_loss = 1e8

    print('model: ' + C.EX_ID + ' '+
          'loss: ' + " ".join(C.LOSS) + ' '+
          'lr: ' + str(C.LEARNING_RATE) + ' '+
          'network: ' + C.NETWORK)

    for epoch_i in range(C.MAX_EPOCH):
        print("Epoch {}".format(epoch_i+1))
        epoch_loss = train(epoch_i)
        print(epoch_loss)
        if epoch_loss < max_loss:
            max_loss = epoch_loss
            print('save model')
            torch.save(
                {
                    'epoch': epoch_i,
                    'loss': epoch_loss,
                    'optimizer': optimizer.state_dict(),
                    'model': model.state_dict()
                },
                C.MODEL_SAVE_NAME
            )
        # print(epoch_loss)


