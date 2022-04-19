import torch
import torchvision

from face_dataset import FaceDataset
from torch.utils.data import DataLoader

from model import Generator

import cv2
import numpy as np
import argparse

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class Trainer():
    def __init__(self) -> None:
        self.device = "cuda:0"
        self.batch_size = 4
        self.dataloader = self.get_dataloader(data_root = "/data/dataset_seonghyeon/video")
        self.idt_encoder, self.pose_encoder = self.get_embedder()
        self.generator = self.get_generator()
        self.g_mean = self.generator.mean_latent(512)
        self.optimizer = torch.optim.Adam(
            list(self.idt_encoder.parameters()) + 
            list(self.pose_encoder.parameters()) + 
            list(self.generator.parameters()), 
            lr=1e-3)
        self.criterion = torch.nn.MSELoss()
        

    def get_dataloader(self, data_root) -> DataLoader:
        dataloader = DataLoader(FaceDataset(data_root), batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True)
        return dataloader

    def get_embedder(self):
        idt_encoder = torchvision.models.resnext50_32x4d(num_classes=512).to(self.device)
        pose_encoder = torchvision.models.mobilenet_v3_large(num_classes=512).to(self.device)
        # idt_encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
        # pose_encoder = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        return idt_encoder, pose_encoder

    def get_generator(self) -> Generator:
        generator_args = argparse.Namespace(
            latent_dim = 512,
            n_mlp = 8,
            g_size = 256,
            channel_multiplier = 2,
            start_iter = 0,
            ckpt = "./550000.pt"
        )
        print("load model:", generator_args.ckpt)
        generator = Generator(
            generator_args.g_size, generator_args.latent_dim, generator_args.n_mlp, channel_multiplier=generator_args.channel_multiplier
        ).to(self.device)
        ckpt = torch.load(generator_args.ckpt, map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt["g"], strict=False)
        requires_grad(generator, True)
        return generator
    
    def train(self):
        for i in range(300000+1):
            idt_imgs, pose_original_img, pose_transformed_img = next(iter(self.dataloader))
            self.optimizer.zero_grad()
            
            di = []
            for b in range(self.batch_size):
                di.append(self.idt_encoder(idt_imgs[b].to(self.device)))
            di = torch.stack(di, dim=0)

            # b, 512
            di = di.mean(dim=1, keepdim=False)
            # b, 512
            dp = self.pose_encoder(pose_transformed_img.to(self.device))

            alpha = self.get_alpha(i, 0, 10000)
            # b * 3 * 256 * 256

            image, _ = self.generator([self.g_mean + (di+dp) * alpha])
            loss = self.criterion(pose_original_img.to(self.device), image)

            if i % 1000 == 0:
                print(i, loss.item(), alpha)
                # if self.show_image(i, (image + 1.0) * 0.5):
                # self.show_image(i, (image + 1.0) * 0.5)
                torchvision.utils.save_image(image, "./checkpoint/training_{0}.png".format(str(i).zfill(6)), normalize=True, range=(-1,1))
            if i % 10000 == 0 and i != 0:
                torch.save(self.idt_encoder.state_dict(),  "./checkpoint/idt_encoder_{0}.pth".format(str(i).zfill(6)))
                torch.save(self.pose_encoder.state_dict(),  "./checkpoint/pose_encoder_{0}.pth".format(str(i).zfill(6)))
                torch.save(self.generator.state_dict(),  "./checkpoint/generator_{0}.pth".format(str(i).zfill(6)))

            self.generator.zero_grad()
            loss.backward()
            self.optimizer.step()


    def get_alpha(self, i, start, end):
        diff = end - start
        return min((max(start, i) - start)/diff, 1.0)

    def show_image(self, idx, img):
        img_np = self.torch2numpy(img)
        img_np = img_np.transpose(1,2,0)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./checkpoint/{0}.jpg".format(str(idx).zfill(7)), img_np*255)
        # cv2.imshow("process", img_np)
        # return cv2.waitKey(1) & 0xFF == 27

    def torch2numpy(self, img):
        return img[0].detach().cpu().numpy()




trainer = Trainer()
trainer.train()