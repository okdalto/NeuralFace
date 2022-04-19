import torch
import torchvision
from model import Generator
import cv2
import argparse
import os

def torch2numpy(img):
    return img[0].detach().cpu().numpy()

sample_dir = "sample_3"
idt_dir = "./test_data/idt_victor"

def show_video(idx, img):
    global sample_dir
    
    img_np = torch2numpy(img)
    img_np = img_np.transpose(1,2,0)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_np = (img_np + 1.0) * 0.5
    os.makedirs("./checkpoint/{}".format(sample_dir), exist_ok=True)
    cv2.imwrite("./checkpoint/{0}/{1}.jpg".format(sample_dir, str(idx).zfill(7)), img_np*255)
    # cv2.imshow("process", img_np)
    return cv2.waitKey(1) & 0xFF == 27

def show_image(img):
    img_np = torch2numpy(img)
    img_np = img_np.transpose(1,2,0)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    cv2.imshow("process", (img_np + 1.0) * 0.5 )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def get_generator() -> Generator:
    generator_args = argparse.Namespace(
        latent_dim = 512,
        n_mlp = 8,
        g_size = 256,
        channel_multiplier = 2,
        start_iter = 0,
        # ckpt = "./550000.pt"
        
    )
    print("load G model")
    generator = Generator(
        generator_args.g_size, generator_args.latent_dim, generator_args.n_mlp, channel_multiplier=generator_args.channel_multiplier
    )
    return generator

from collections import OrderedDict
def preprocess_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    return new_state_dict


device = "cuda:0"

idt_encoder = torchvision.models.resnext50_32x4d(num_classes=512).to(device)
pose_encoder = torchvision.models.mobilenet_v3_large(num_classes=512).to(device)
generator = get_generator().to(device)
g_mean = generator.mean_latent(512)

checkpoint_dir = "checkpoints_2022_4_5_15_44_14"


idt_model = torch.load("./checkpoint/{}/idt_encoder_300000.pth".format(checkpoint_dir),)
idt_encoder.load_state_dict(preprocess_dict(idt_model))
pose_model = torch.load("./checkpoint/{}/pose_encoder_300000.pth".format(checkpoint_dir))
pose_encoder.load_state_dict(preprocess_dict(pose_model))
generator_model = torch.load("./checkpoint/{}/generator_300000.pth".format(checkpoint_dir))
generator.load_state_dict(preprocess_dict(generator_model))

from PIL import Image
import torchvision.transforms as transforms
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

import glob
idt_img_list = glob.glob("{}/idt/*.png".format(idt_dir))
pose_img_list = glob.glob("{}/pose/*.jpg".format(idt_dir))
pose_img_list.sort()

idt_img_list = torch.stack([transform(Image.open(img_path)) for img_path in idt_img_list], dim=0).to(device)
pose_img_list = [transform(Image.open(img_path)).to(device) for img_path in pose_img_list]


with torch.no_grad():
    idt_feature = idt_encoder(idt_img_list)
    idt_feature = idt_feature.mean(dim=0, keepdim=True)

    for i, pose_img in enumerate(pose_img_list):
        pose_feature = pose_encoder(pose_img.unsqueeze(0))
        print(idt_feature.shape)
        print(pose_feature.shape)
        image, _ = generator(
            [idt_feature + pose_feature]
        )
        if show_video(i, image):
            break
    

