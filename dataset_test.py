from face_dataset import FaceDataset
from torch.utils.data import DataLoader
import cv2

def get_dataloader(data_root) -> DataLoader:
    dataloader = DataLoader(FaceDataset(data_root), batch_size=1, shuffle=True, num_workers=4, drop_last=True)
    return dataloader

def torch2numpy(img):
    return img[0].detach().cpu().numpy()

def show_video(idx, img):
    img_np = torch2numpy(img)
    img_np = img_np.transpose(1,2,0)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("./checkpoint/{0}.jpg".format(str(idx).zfill(7)), img_np*255)
    cv2.imshow("process", img_np)
    return cv2.waitKey(1) & 0xFF == 27

def show_image(img):
    img_np = torch2numpy(img)
    img_np = img_np.transpose(1,2,0)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    cv2.imshow("process", (img_np + 1.0) * 0.5 )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


dataloader = get_dataloader(data_root = "../dataset/video/")

for i in range(1000):
    idt_imgs, pose_original_img, pose_transformed_img = next(iter(dataloader))
    if show_video(0, idt_imgs[0]):
        break