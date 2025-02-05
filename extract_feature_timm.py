#!/usr/bin/env python
import os
import argparse
import torch
import torch.nn as nn
from list_dataset import ImageFilelist
import numpy as np
import pickle
from tqdm import tqdm
# import mmcv
import torchvision as tv
from torch.cuda.amp import autocast
import timm

from attack import attack_pgd_restart, ctx_noparamgrad
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import foolbox
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import  L2DeepFoolAttack, LinfBasicIterativeAttack, FGSM, L2CarliniWagnerAttack, LinfPGD, LinfDeepFoolAttack


def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('--data_root', default='/home/DATA/ITWM/lorenzp/cifar10', help='Path to data')
    # parser.add_argument('--out_file', default='/home/lorenzp/workspace/competence_estimation/features/cifar10/resnet18_test.npy', help='Path to output file')
    parser.add_argument('--out_file', default='/home/lorenzp/workspace/competence_estimation/features/cifar10/benign/resnet18_train.npy', help='Path to output file')
    parser.add_argument('--model', default='resnet18', help='Path to config')
    parser.add_argument('--datatype', default='spatial', choices=['benign', 'phase', '|phase|', 'magnitude'], help='')
    # parser.add_argument('--checkpoint', default='checkpoints/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth', help='Path to checkpoint')
    parser.add_argument('--preprocess', default='', choices=['MFS', 'PFS'], help='apply FFT?')
    parser.add_argument('--checkpoint', default='', help='Path to checkpoint')
    parser.add_argument('--img_list', default=None, help='Path to image list')
    parser.add_argument('--batch', type=int, default=256, help='Path to data')
    parser.add_argument('--workers', type=int, default=4, help='Path to data')
    parser.add_argument('--attack', default=None, choices=[None, 'pgd', 'fgsm', 'l2df', 'linfdf', 'linfpgd'], help='')
    parser.add_argument('--ε', type=float, default=8./255)
    parser.add_argument('--fc_save_path', default=None, help='Path to save fc')
    # parser.add_argument('--fc_save_path', default="/home/lorenzp/workspace/competence_estimation/features/cifar10/", help='Path to save fc')

    return parser.parse_args()


def create_dir(path):
    is_existing = os.path.exists(path)
    if not is_existing:
        os.makedirs(path)
        print("The new directory is created!", path)

    mean, std = normalization
    images[:,0,:,:] = (images[:,0,:,:] - mean[0]) / std[0]
    images[:,1,:,:] = (images[:,1,:,:] - mean[1]) / std[1]
    images[:,2,:,:] = (images[:,2,:,:] - mean[2]) / std[2]
    
    return images


def calculate_fourier_spectrum(im, typ='MFS'):
    # im = im.float()
    im = im.cpu()
    im = im.data.numpy() # transform to numpy
    fft = np.fft.fft2(im)
    if typ == 'MFS':
        fourier_spectrum = np.abs(fft)
    elif typ == 'PFS':
        fourier_spectrum = np.abs(np.angle(fft))
    # if  (args.net == 'cif100' or args.net == 'cif100vgg') and (args.attack=='cw' or args.attack=='df'):
    #     fourier_spectrum *= 1/np.max(fourier_spectrum)

    return torch.from_numpy(fourier_spectrum).float().cuda()


def load_model_timm(args):

    # https://huggingface.co/edadaltocg/resnet18_cifar10
    model = timm.create_model("resnet18", num_classes=10, pretrained=False)
    
    if args.datatype == 'spatial':
        # override model
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()  # type: ignore
        # model.fc = nn.Linear(512,  10)

        model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
            map_location="cpu", 
            file_name="resnet18_cifar10.pth",
            )
        )
    elif args.datatype == 'phase':
        checkpt = torch.load('/home/lorenzp/wide-resnet.pytorch/checkpoint_wrn/cifar10/resnet18_timm_phase_2023-10-14_15:42:13.pt')
        model.load_state_dict(checkpt)
    elif args.datatype == '|phase|':
        checkpt = torch.load('/home/lorenzp/wide-resnet.pytorch/checkpoint_wrn/cifar10/resnet18_timm_phase_2023-10-14_17:21:35.pt')
        model.load_state_dict(checkpt)
    elif args.datatype == 'magnitude':
        checkpt = torch.load('/home/lorenzp/wide-resnet.pytorch/checkpoint_wrn/cifar10/resnet18_timm_magnitude_2023-10-14_16:26:21.pt')
        model.load_state_dict(checkpt)

    model.eval()
    model.cuda()
    # cudnn.benchmark = True
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])

    return model 


def main():
    args = parse_args()
    
    print(args)

    torch.backends.cudnn.benchmark = True

    if args.fc_save_path is not None:
        model =  load_model_timm(args)
       
        create_dir(os.path.dirname(args.fc_save_path))
        # mmcv.mkdir_or_exist(os.path.dirname(args.fc_save_path))
        if args.model in ['repvgg_b3']:
            w = model.head.fc.weight.cpu().detach().numpy()
            b = model.head.fc.bias.cpu().detach().numpy()
        elif args.model in ['swin_base_patch4_window7_224', 'deit_base_patch16_224']:
            w = model.head.weight.cpu().detach().numpy()
            b = model.head.bias.cpu().detach().numpy()
        else:
            w = model.fc.weight.cpu().detach().numpy()
            b = model.fc.bias.cpu().detach().numpy()
        
        W_path = os.path.join(args.fc_save_path, args.model + '_W.npy')
        with open(W_path, 'wb') as f:
            # pickle.dump([w, b], f)
            np.save(f, w)
            
        b_path = os.path.join(args.fc_save_path, args.model + '_b.npy')
        with open(b_path, 'wb') as f:
            # pickle.dump([w, b], f)
            np.save(f, b)
            
        print("Save W: ", W_path)
        print("Save b: ", b_path)
        return

    model =  load_model_timm(args)


    nodes, _ = get_graph_node_names(model)

    if args.attack == 'pgd':
        def test_attacker(x, y):
            with ctx_noparamgrad(model):
                adv_delta = attack_pgd_restart(
                    model=model,
                    X=x,
                    y=y,
                    eps=args.ε,
                    alpha=args.ε / 4,
                    attack_iters=40,
                    n_restarts=10,
                    rs=True,
                    verbose=True,
                    linf_proj=True,
                    l2_proj=False,
                    l2_grad_update=False,
                    cuda=torch.cuda.is_available()
                )
            return x + adv_delta
    elif args.attack == 'fgsm':
        attack = FGSM()
    elif args.attack == 'l2df':
        args.ε = None
        attack = L2DeepFoolAttack()
    elif args.attack == 'linfdf':
        args.ε = None
        attack = LinfDeepFoolAttack()
    elif args.attack == 'linfpgd':
        attack = LinfPGD()



    transform = tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if args.img_list is not None:
        dataset = ImageFilelist(args.data_root, args.img_list, transform)
    else:
        if 'cifar10' in args.data_root:
            if args.datatype == 'spatial':
                mean = (0.4914, 0.4822, 0.4465)
                std = (0.2023, 0.1994, 0.2010)
            elif args.datatype == 'phase':
                mean = (1.5699, 1.5683, 1.5669)
                std = (0.9143, 0.9142, 0.9143)
            elif args.datatype == '|phase|':
                mean = (0.0045, 0.0043, 0.0043)
                std = (1.8167, 1.8153, 1.8141)
            elif args.datatype == 'magnitude':
                mean = (3.2836, 3.2311, 3.1789)
                std = (17.2909, 16.9801, 16.2524)      
                
            print("mean", mean, ", std", std)
            
            
            if args.attack is None:
                transform = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(mean, std),
                ])
            else:
                transform = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                ])
            normalization = [mean, std]

            train=True
            if 'test' in args.out_file:
                train=False
            dataset = tv.datasets.CIFAR10(root=args.data_root, train=train, download=True, transform=transform)
        else:
            dataset = tv.datasets.ImageFolder(args.data_root, transform)

            
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
        )

    feature_extractor = create_feature_extractor(model, return_nodes={'global_pool': 'features', 'fc': 'logits'})

    if not args.attack == None and not args.attack == 'pgd':
        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    
    features = []
    logits = []
    labels = []

    # with torch.no_grad():
    with autocast():
        for x, y in tqdm(dataloader):

            x = x.cuda()
            y = y.cuda()

            if not args.attack is None:
                if args.attack == 'pgd':
                    x = test_attacker(x,y)
                else: 
                    raw_x, x, success = attack(fmodel, x, criterion=foolbox.criteria.Misclassification(y), epsilons=args.ε)
                
            if len(args.preprocess) > 0:
                x = calculate_fourier_spectrum(x)

            x = normalize_images(x, normalization)

            feature = feature_extractor(x)
            features.append(feature['features'].cpu().detach().numpy())
            logits.append(feature['logits'].cpu().detach().numpy())

            if not args.attack is None:
                y_adv = model(x)
                y_adv = torch.argmax(y_adv, axis=1)

            labels.append(y.cpu().detach().numpy())

    features = np.concatenate(features, axis=0)
    logits   = np.concatenate(logits, axis=0)
    labels   = np.concatenate(labels, axis=0)

    create_dir(os.path.dirname(args.out_file))
    dirname = os.path.dirname(args.out_file)
    basename = os.path.basename(args.out_file)
    preprocess = ''
    if not args.attack is None:
        if len(args.preprocess) > 0:
            preprocess = args.preprocess + '_'
        basename =  args.attack + '_' + preprocess  + basename
    else: 
        if len(args.preprocess) > 0:
            preprocess = args.preprocess + '_'
        basename = preprocess + basename

    out = os.path.join(dirname, "features_" +  basename)
    print("save as: ", out)
    with open(out, 'wb') as f:
        np.save(f, features)
    
    with open(os.path.join(dirname, "logits_" + basename), 'wb') as f:
        np.save(f, logits)

    with open(os.path.join(dirname, "labels_" + basename), 'wb') as f:
        np.save(f, labels)


if __name__ == '__main__':
    main()
