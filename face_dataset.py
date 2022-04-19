from secrets import choice
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import random
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, root_dir):
        self.dir_list = glob.glob( os.path.join(root_dir, "*") )
        self.data_len = len(self.dir_list)
        # print(self.dir_list)



        self.transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])
        self.pose_transform = transforms.Compose([
                                transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3),
                                transforms.GaussianBlur(kernel_size=(5), sigma=(0.0001, 5)),
                                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.mask_transform = transforms.Compose([
                                transforms.ToTensor(),
                            ])

        # dir_name_list = [os.path.basename(sub_dir) for sub_dir in dir_list]
        self.images = {}
        for dir_path in self.dir_list:
            img_dir = os.path.join(dir_path, "img")
            msk_dir = os.path.join(dir_path, "mask")
            self.images[dir_path] = []
            images = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
            masks = glob.glob(os.path.join(msk_dir, "*.jpg")) + glob.glob(os.path.join(msk_dir, "*.png"))
            for image, mask in zip(images, masks):
                self.images[dir_path].append({"img":image, "mask":mask})
        # self.rand = random.Random()
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        idx = random.randint(0, self.data_len-1)
        candidates = self.images[self.dir_list[idx]]
        if len(candidates) <= 9:
            print(self.dir_list[idx])
            print(candidates[0]["img"])
            print("data too short!!")
        
        random.shuffle(candidates)
        # self.rand.shuffle(candidates)
        idts = torch.stack([self.transform(Image.open(candidate["img"])) for candidate in candidates[:8]], dim=0)
        pose_original = Image.open(candidates[8]["img"])
        pose_transformed = self.pose_transform(pose_original.copy())
        pose_mask = self.mask_transform(Image.open(candidates[8]["mask"]))
        return idts, self.transform(pose_original), pose_transformed, pose_mask




# from __future__ import print_function, division
# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# import glob
# from PIL import Image
# from skimage import io, transform
# import numpy as np
# import cv2


# # import matplotlib.pyplot as plt
# # from torchvision import transforms, utils
# # import random
# # import sys



# class FaceImgDataset(Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self, root_dir, warping=True, warp_scale=0.12):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """

#         # torch.manual_seed(0)

#         self.img_file_list = [glob.glob(root_dir+e, recursive=True) for e in ["/**/img/*.png", "/**/img/*.jpg"]]
#         self.img_file_list = sum(self.img_file_list, [])
#         self.img_file_list.sort()

#         self.mask_file_list = [glob.glob(root_dir+e, recursive=True) for e in ["/**/mask/*.png", "/**/mask/*.jpg"]]
#         self.mask_file_list = sum(self.mask_file_list, [])
#         self.mask_file_list.sort()

#         # self.img_file_list = [glob.glob(root_dir+e, recursive=True) for e in ["/**/img/*.png", "/**/img/*.jpg"]]
#         # self.img_file_list = sum(self.img_file_list, [])

#         self.index = len(self.img_file_list)

#         self.warping = warping
#         self.warp_scale = warp_scale

#         self.transform = transforms.Compose([
#                                 transforms.ToPILImage(),
#                                 transforms.Resize((128,128)),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                          ])

#         self.transform_mask = transforms.Compose([
#                                 transforms.ToPILImage(),
#                                 transforms.Resize((128,128)),
#                                 transforms.ToTensor(),
#                          ])



#         print(root_dir, self.index)

#     def __len__(self):
#         return self.index

#     def get_data_length(self):
#         return self.data_length

#     def __getitem__(self, idx):
#         image_file_path = self.img_file_list[idx]
#         mask_file_path = self.mask_file_list[idx]

#         mask = cv2.imread(mask_file_path)
#         # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

#         img = cv2.imread(image_file_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, dsize=(128,128), interpolation=cv2.INTER_LANCZOS4)
#         mask = cv2.resize( mask, dsize=(128,128), interpolation=cv2.INTER_LANCZOS4)

#         if self.warping:        
#             warped_img, target_img, mask = random_warp(img, mask, self.warp_scale)
#         else:
#             warped_img, target_img, mask = img, img.copy(), mask

#         if self.transform:
#             warped_img = self.transform(warped_img)
#             target_img = self.transform(target_img)
#             mask = self.transform_mask(mask)

#         return warped_img, target_img, mask


# def random_warp(image, mask, warp_scale=0.12):
#     # assert image.shape == (256, 256, 3)
#     # range_ = np.zeros(7)
#     w = 128
#     cell_size = [ w // (2**i) for i in range(1,4) ] [ np.random.randint(3) ]
#     # cell_size = 32
#     cell_count = w // cell_size + 1
#     # cell_count = 5
#     range_ = np.linspace( 0, w, cell_count)
#     # range_ = np.linspace(0, 256, 5)

#     mapx = np.broadcast_to(range_, (cell_count, cell_count)).copy()
#     mapy = mapx.T

#     # mapx[1:-1,1:-1] = mapx[1:-1,1:-1] + np.random.normal(size=(cell_count-2, cell_count-2), scale=15)
#     # mapy[1:-1,1:-1] = mapy[1:-1,1:-1] + np.random.normal(size=(cell_count-2, cell_count-2), scale=15)

#     mapx = mapx + np.random.normal(size=(cell_count, cell_count)) * (cell_size*warp_scale)
#     mapy = mapy + np.random.normal(size=(cell_count, cell_count)) * (cell_size*warp_scale)

#     half_cell_size = cell_size // 2
#     # interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
#     # interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')
#     # interp_mapx = cv2.resize(mapx, (128+8*4, 128+8*4) )[80-64:80+64,80-64:80+64].astype('float32')
#     # interp_mapy = cv2.resize(mapy, (128+8*4, 128+8*4) )[80-64:80+64,80-64:80+64].astype('float32')
#     interp_mapx = cv2.resize(mapx, (w+cell_size,)*2 )[half_cell_size:-half_cell_size,half_cell_size:-half_cell_size].astype(np.float32)
#     interp_mapy = cv2.resize(mapy, (w+cell_size,)*2 )[half_cell_size:-half_cell_size,half_cell_size:-half_cell_size].astype(np.float32)

#     warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

#     src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
#     dst_points = np.mgrid[0:w+1:cell_size, 0:w+1:cell_size].T.reshape(-1, 2)
    
#     mat = umeyama(src_points, dst_points, True)[0:2]

#     target_image = cv2.warpAffine(image, mat, (w, w))
#     mask = cv2.warpAffine(mask, mat, (w, w))

#     return warped_image, target_image, mask



# def umeyama(src, dst, estimate_scale):
#     """Estimate N-D similarity transformation with or without scaling.
#     Parameters
#     ----------
#     src : (M, N) array
#         Source coordinates.
#     dst : (M, N) array
#         Destination coordinates.
#     estimate_scale : bool
#         Whether to estimate scaling factor.
#     Returns
#     -------
#     T : (N + 1, N + 1)
#         The homogeneous similarity transformation matrix. The matrix contains
#         NaN values only if the problem is not well-conditioned.
#     References
#     ----------
#     .. [1] "Least-squares estimation of transformation parameters between two
#             point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
#     """

#     num = src.shape[0]
#     dim = src.shape[1]

#     # Compute mean of src and dst.
#     src_mean = src.mean(axis=0)
#     dst_mean = dst.mean(axis=0)

#     # Subtract mean from src and dst.
#     src_demean = src - src_mean
#     dst_demean = dst - dst_mean

#     # Eq. (38).
#     A = dst_demean.T @ src_demean / num

#     # Eq. (39).
#     d = np.ones((dim,), dtype=np.double)
#     if np.linalg.det(A) < 0:
#         d[dim - 1] = -1

#     T = np.eye(dim + 1, dtype=np.double)

#     U, S, V = np.linalg.svd(A)

#     # Eq. (40) and (43).
#     rank = np.linalg.matrix_rank(A)
#     if rank == 0:
#         return np.nan * T
#     elif rank == dim - 1:
#         if np.linalg.det(U) * np.linalg.det(V) > 0:
#             T[:dim, :dim] = U @ V
#         else:
#             s = d[dim - 1]
#             d[dim - 1] = -1
#             T[:dim, :dim] = U @ np.diag(d) @ V
#             d[dim - 1] = s
#     else:
#         T[:dim, :dim] = U @ np.diag(d) @ V

#     if estimate_scale:
#         # Eq. (41) and (42).
#         scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
#     else:
#         scale = 1.0

#     T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
#     T[:dim, :dim] *= scale

#     return T
