import lpips
import torch
import numpy as np
import cv2

def main():
    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    im0 = cv2.imread("outputs/MED_loc/crops/00087.jpg_9.jpg")
    im1 = cv2.imread("outputs/MED_loc/crops/00097.jpg_0.jpg")
    im2 = cv2.imread("outputs/MED_loc/crops/00102.jpg_4.jpg")
    im0 = cv2.resize(im0, (128, 128)).astype(np.float32)
    im1 = cv2.resize(im1, (128, 128)).astype(np.float32)
    im2 = cv2.resize(im2, (128, 128)).astype(np.float32)

    im0 = 2*(im0-im0.min())/(im0.max()-im0.min())-1
    im1 = 2*(im1-im1.min())/(im1.max()-im1.min())-1
    im2 = 2*(im2-im2.min())/(im2.max()-im2.min())-1

    tm0 = torch.from_numpy(im0.transpose(2, 0, 1)[None])
    tm1 = torch.from_numpy(im1.transpose(2, 0, 1)[None])
    tm2 = torch.from_numpy(im2.transpose(2, 0, 1)[None])
    d0 = loss_fn_alex(tm0, tm1)
    print("d0 = ", d0)
    d1 = loss_fn_alex(tm0, tm2)
    print("d1= ", d1)
    d2 = loss_fn_alex(tm1, tm2)
    print("d2 = ", d2)


if __name__ == "__main__":
    main()