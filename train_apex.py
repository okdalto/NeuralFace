# OMP_NUM_THREADS=4 python -m torch.distributed.run --nproc_per_node=4 train_apex.py --save-ckpt --data_root "/data/dataset_seonghyeon/out" --batch_size 8

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torch.nn.functional as F
from torch.utils import data
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter

import torchvision

from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *

import lpips

from model import Generator
from face_dataset import FaceDataset

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

def parse():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--data_root', type=str, default="/data/dataset_seonghyeon/video")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--save-ckpt', action='store_true')
    args = parser.parse_args()
    return args

def get_embedder():
    idt_encoder = torchvision.models.resnext50_32x4d(num_classes=512)
    pose_encoder = torchvision.models.mobilenet_v3_large(num_classes=512)
    return idt_encoder, pose_encoder

def get_generator() -> Generator:
    generator_args = argparse.Namespace(
        latent_dim = 512,
        n_mlp = 8,
        g_size = 256,
        channel_multiplier = 2,
        start_iter = 0,
        # ckpt = "./550000.pt"
        ckpt = "./550000_g_4ch.pt"
        
    )
    print("load model:", generator_args.ckpt)
    generator = Generator(
        generator_args.g_size, generator_args.latent_dim, generator_args.n_mlp, channel_multiplier=generator_args.channel_multiplier
    )
    ckpt = torch.load(generator_args.ckpt, map_location=lambda storage, loc: storage)
    # generator.load_state_dict(ckpt["g"], strict=False)
    generator.load_state_dict(ckpt, strict=False)
    return generator

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

def main():
    args = parse()

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = "cuda:" + os.environ['LOCAL_RANK']
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        dist.barrier()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    generator = get_generator().cuda()
    idt_encoder, pose_encoder = get_embedder()
    optimizer = torch.optim.Adam(
        list(idt_encoder.parameters()) + 
        list(pose_encoder.parameters()) + 
        list(generator.parameters()), 
        lr=1e-4
    )

    if args.save_ckpt and dist.get_rank() == 0:
        import time
        now = time.localtime()
        times = map(str, [now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec])
        ckpt_dir = "./checkpoint/checkpoints_{0}".format("_".join(times))
        print("making directory in : {0}".format(ckpt_dir))
        os.makedirs(ckpt_dir, exist_ok=True)
        writer = SummaryWriter(ckpt_dir)


    dataset = FaceDataset(args.data_root)
    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        generator = DDP(generator)
        idt_encoder = DDP(idt_encoder.cuda())
        pose_encoder = DDP(pose_encoder.cuda())

    # define loss function (criterion) and optimizer
    loss_fn = nn.MSELoss().cuda()
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    loss_fn_dice = DiceBCELoss()

    dataloader = sample_data(dataloader)
    rank = dist.get_rank()
    for i in range(300000+1):
        idt_imgs, pose_original_img, pose_transformed_img, pose_mask = next(dataloader)
        # idx_to_str = str(i).zfill(6)
        # torchvision.utils.save_image(idt_imgs[0], "{0}/idt_imgs_{1}_{2}.png".format(ckpt_dir, dist.get_rank(), idx_to_str), normalize=True, range=(-1,1))
        # torchvision.utils.save_image(pose_original_img, "{0}/pose_imgs_{1}_{2}.png".format(ckpt_dir, dist.get_rank(), idx_to_str), normalize=True, range=(-1,1))
        # torchvision.utils.save_image(pose_transformed_img, "{0}/pose_transformed_imgs_{1}_{2}.png".format(ckpt_dir, dist.get_rank(), idx_to_str), normalize=True, range=(-1,1))
        # torchvision.utils.save_image(pose_mask, "{0}/mask_imgs_{1}_{2}.png".format(ckpt_dir, dist.get_rank(), idx_to_str), normalize=True, range=(-1,1))

        pose_mask = pose_mask[:, 0:1, :, :].cuda()

        di = []
        for b in range(args.batch_size):
            di.append(idt_encoder(idt_imgs[b].cuda()))
        di = torch.stack(di, dim=0)
        di = di.mean(dim=1, keepdim=False)              # b, 512
        dp = pose_encoder(pose_transformed_img.cuda())  # b, 512

        img_fake, _ = generator([di + dp]) # bs, 4, 512, 512

        mask_fake = (img_fake[:, 3:4, :, :] + 1.0) * 0.5  # bs, 512, 512, 
        mask_fake = torch.clamp(mask_fake, min=0.0, max=1.0) # range: 0 ~ 1

        img_fake_maksed = img_fake[:, :3, :, :] * mask_fake
        pose_original_maksed = pose_original_img.cuda() * pose_mask

        loss_MSE = loss_fn(img_fake_maksed, pose_original_maksed)
        loss_LPIPS = loss_fn_alex(img_fake_maksed, pose_original_maksed).mean()
        dice_loss = loss_fn_dice(mask_fake, pose_mask)
        loss = loss_MSE + loss_LPIPS + dice_loss

        if rank == 0:
            writer.add_scalar("Loss_MSE/train", loss_MSE.item(), i)
            writer.add_scalar("Loss_LPIPS/train", loss_LPIPS.item(), i)
            writer.add_scalar("Loss_all/train", loss.item(), i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            if rank == 0:
                idx_to_str = str(i).zfill(6)
                print("rank : ", rank, i, loss.item(), "MSE: ", loss_MSE.item(), "LPIPS: ", loss_LPIPS.item(), "dice: ", dice_loss.item())
                if args.save_ckpt:
                    torchvision.utils.save_image(img_fake[:, :3, :, :], "{0}/img_fake_{1}.jpg".format(ckpt_dir, idx_to_str), normalize=True, range=(-1,1))
                    torchvision.utils.save_image(pose_original_img, "{0}/GT_pose_{1}.jpg".format(ckpt_dir, idx_to_str), normalize=True, range=(-1,1))
                    torchvision.utils.save_image(img_fake_maksed[:, :3, :, :], "{0}/img_fake_maksed_{1}.jpg".format(ckpt_dir, idx_to_str), normalize=True, range=(-1,1))
        if i % 10000 == 0 and i != 0:
            if rank == 0:
                idx_to_str = str(i).zfill(6)
                if args.save_ckpt:
                    torch.save(idt_encoder.state_dict(),   "{0}/idt_encoder_{1}.pth".format(ckpt_dir, idx_to_str))
                    torch.save(pose_encoder.state_dict(),  "{0}/pose_encoder_{1}.pth".format(ckpt_dir, idx_to_str))
                    torch.save(generator.state_dict(),     "{0}/generator_{1}.pth".format(ckpt_dir, idx_to_str))
    if rank == 0:
        writer.flush()
        writer.close()

if __name__ == '__main__':
    main()