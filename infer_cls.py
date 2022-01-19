
from unittest.mock import patch
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import voc12.data
import scipy.misc
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils
import argparse
from PIL import Image
import torch.nn.functional as F
import os.path
from torchvision import transforms
import dataset
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--infer_list", default="voc12/val.txt", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--voc12_root", required=False, type=str)
    parser.add_argument("--low_alpha", default=4, type=int)
    parser.add_argument("--high_alpha", default=32, type=int)
    parser.add_argument("--out_cam", default=None, type=str)
    parser.add_argument("--out_la_crf", default=None, type=str)
    parser.add_argument("--out_ha_crf", default=None, type=str)
    parser.add_argument("--out_cam_pred", default=None, type=str)

    args = parser.parse_args()

    num_class = 2
    model = getattr(importlib.import_module(args.network), 'Net')()
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    infer_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.831, 0.725, 0.858], std=[0.133, 0.173, 0.099])
        ])

    infer_dataset = dataset.OnlineDataset(data_path_name=f'../WSSS4LUAD/Dataset_crag/1.training/origin_ims', transform=infer_transform, patch_size=112, stride=56, scales=(1, 0.5, 1.5, 2.0))

    # infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
    #                                                scales=(1, 0.5, 1.5, 2.0),
    #                                                inter_transform=torchvision.transforms.Compose(
    #                                                    [np.asarray,
    #                                                     model.normalize,
    #                                                     imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers)

    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))

    for iter, (img_name, scaled_img_list, scaled_position_list, scales) in enumerate(infer_data_loader):
        img_name = img_name[0]

        img_path = os.path.join('../WSSS4LUAD/Dataset_crag/1.training/origin_ims', img_name)
        # img_path = voc12.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        w, h, _ = orig_img.shape
        ensemble_cam = np.zeros((num_class, w, h))
        side_length = 112
        for scale in scales:
            w_ = int(w*scale)
            h_ = int(h*scale)
            interpolatex = side_length
            interpolatey = side_length
            if w_ < side_length:
                interpolatex = w_
            if h_ < side_length:
                interpolatey = h_

            with torch.no_grad():
                for token in range(len(scales)):
                    cam_list = []
                    position_list = []
                    for ims, positions in zip(scaled_img_list[token], scaled_position_list[token]):
                        cam_scores = model.forward_cam(ims.cuda())
                        cam_scores = F.interpolate(cam_scores, (interpolatex, interpolatey), mode='bilinear', align_corners=False).detach().cpu().numpy()
                        cam_list.append(cam_scores)
                        position_list.append(positions)
                    cam_list = np.concatenate(cam_list)
                    # position_list = np.concatenate(position_list)
                    # print(position_list)
                    sum_cam = np.zeros((num_class, w_, h_))
                    sum_counter = np.zeros_like(sum_cam)
                
                    for k in range(cam_list.shape[0]):
                        y, x = position_list[k][0], position_list[k][1]
                        crop = cam_list[k]
                        try:
                            sum_cam[:, y:y+side_length, x:x+side_length] += crop
                        except:
                            _, a, b = sum_cam[:, y:y+side_length, x:x+side_length].shape
                            crop = crop[:, :a, :b]
                            sum_cam[:, y:y+side_length, x:x+side_length] += crop
                        sum_counter[:, y:y+side_length, x:x+side_length] += 1
                    sum_counter[sum_counter < 1] = 1
                    norm_cam = sum_cam / sum_counter
                    norm_cam = F.interpolate(torch.unsqueeze(torch.tensor(norm_cam),0), (w, h), mode='bilinear', align_corners=False).detach().cpu().numpy()[0]

                    # use the image-level label to eliminate impossible pixel classes
                    ensemble_cam += norm_cam                
        
            result_label = ensemble_cam.argmax(axis=0)

        # def _work(i, img):
        #     with torch.no_grad():
        #         with torch.cuda.device(i%n_gpus):
        #             cam = model_replicas[i%n_gpus].forward_cam(img.cuda())
        #             cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
        #             cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
        #             if i % 2 == 1:
        #                 cam = np.flip(cam, axis=-1)
        #             return cam

        # thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
        #                                     batch_size=12, prefetch_size=0, processes=args.num_workers)

        # cam_list = thread_pool.pop_results()

        # sum_cam = np.sum(cam_list, axis=0)
        # norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)
        norm_cam = ensemble_cam / (np.max(ensemble_cam, (1, 2), keepdims=True) + 1e-5)

        cam_dict = {}
        for i in range(num_class):
            cam_dict[i] = norm_cam[i]

        img_name = img_name.split(".")[0]
        print(img_name)
        if args.out_cam is not None:
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        if args.out_cam_pred is not None:
            # bg_score = [np.ones_like(norm_cam[0])*0.2]
            pred = np.argmax(norm_cam, 0)
            np.save(os.path.join(args.out_cam_pred, img_name + '.npy'), pred.astype(np.uint8))

        def _crf_with_alpha(cam_dict, alpha):
            v = np.array(list(cam_dict.values()))
            bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
            bgcam_score = np.concatenate((bg_score, v), axis=0)
            crf_score = imutils.crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

            n_crf_al = dict()

            n_crf_al[0] = crf_score[0]
            for i, key in enumerate(cam_dict.keys()):
                n_crf_al[key+1] = crf_score[i+1]

            return n_crf_al

        if args.out_la_crf is not None:
            crf_la = _crf_with_alpha(cam_dict, args.low_alpha)
            np.save(os.path.join(args.out_la_crf, img_name + '.npy'), crf_la)

        if args.out_ha_crf is not None:
            crf_ha = _crf_with_alpha(cam_dict, args.high_alpha)
            np.save(os.path.join(args.out_ha_crf, img_name + '.npy'), crf_ha)

        print(iter)

