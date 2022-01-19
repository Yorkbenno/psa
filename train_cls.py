
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils
import argparse
import importlib
import torch.nn.functional as F
import dataset
from tqdm import tqdm

def validate(model, data_loader):
    print('\nvalidating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack[1]
            label = pack[2].cuda(non_blocking=True)

            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss': loss.item()})

    model.train()

    print('loss:', val_loss_meter.pop('loss'))

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epoches", default=15, type=int)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="resnet_cls", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    args = parser.parse_args()

    model = getattr(importlib.import_module(args.network), 'Net')()

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))
    scale = (0.7, 1)
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.831, 0.725, 0.858], std=[0.133, 0.173, 0.099])
        ])

    TrainDataset = dataset.OriginPatchesDataset(data_path_name=f'../WSSS4LUAD/Dataset_crag/1.training/img', transform=train_transform, num_class=2)
    print("train Dataset", len(TrainDataset))
    TrainDatasampler = torch.utils.data.RandomSampler(TrainDataset)
    train_data_loader = DataLoader(TrainDataset, batch_size=args.batch_size, num_workers=4, sampler=TrainDatasampler, drop_last=True)

    max_step = (len(TrainDataset) // args.batch_size) * args.max_epoches

    # val_dataset = voc12.data.VOC12ClsDataset(args.val_list, voc12_root=args.voc12_root,
    #                                          transform=transforms.Compose([
    #                     np.asarray,
    #                     model.normalize,
    #                     imutils.CenterCrop(500),
    #                     imutils.HWC_to_CHW,
    #                     torch.from_numpy
    #                 ]))
    # val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size,
    #                              shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        assert args.network == "network.resnet38_cls"
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    elif args.weights[-11:] == '.caffemodel':
        assert args.network == "network.vgg16_cls"
        import network.vgg16d
        weights_dict = network.vgg16d.convert_caffe_to_torch(args.weights)
    else:
        print("just load")
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss')

    timer = pyutils.Timer("Session started: ")

    for ep in range(args.max_epoches):

        for iter, pack in tqdm(enumerate(train_data_loader)):

            img = pack[0]
            # print(img.shape)
            label = pack[1].cuda(non_blocking=True)
            # print(label)

            x = model(img)
            # print(x, label)
            loss = F.multilabel_soft_margin_loss(x, label)

            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1)%50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'Loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

        # else:
        #     validate(model, val_data_loader)
        #     timer.reset_stage()

    torch.save(model.module.state_dict(), args.session_name + '.pth')
