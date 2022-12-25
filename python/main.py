import os
import glob
import numpy as np
import cv2
from tqdm import tqdm

import trans

def main():
    h = 2000
    w = 2700

    # imgs_transed = np.zeros((5, h, w, 3), dtype=np.uint8)
    # imgs_transed[0] = cv2.rectangle(imgs_transed[0], (0*w//9, h//2-100), (1*w//9, h//2+100), (255, 0, 0), thickness=-1)
    # imgs_transed[1] = cv2.rectangle(imgs_transed[1], (2*w//9, h//2-100), (3*w//9, h//2+100), (0, 255, 0), thickness=-1)
    # imgs_transed[2] = cv2.rectangle(imgs_transed[2], (4*w//9, h//2-100), (5*w//9, h//2+100), (0, 0, 255), thickness=-1)
    # imgs_transed[3] = cv2.rectangle(imgs_transed[3], (6*w//9, h//2-100), (7*w//9, h//2+100), (255, 0, 255), thickness=-1)
    # imgs_transed[4] = cv2.rectangle(imgs_transed[4], (8*w//9, h//2-100), (9*w//9, h//2+100), (0, 255, 255), thickness=-1)

    # 画像読み込み[id, w, h, c]
    imgs = []
    path_imgs = sorted(glob.glob(os.path.join(".", "input", "*.png")))
    for path_img in path_imgs:
        print(path_img)
        img = cv2.imread(path_img)
        imgs.append(img)
    imgs = np.array(imgs)

    # 射影変換
    imgs_transed = []
    # 変換前座標（ペイントソフトより取得）
    uvs = np.array([[[329, 252], [846, 245], [328, 651], [847, 661]],
                    [[310, 250], [851, 247], [309, 654], [852, 658]],
                    [[299, 248], [850, 247], [298, 654], [851, 655]],
                    [[298, 248], [840, 249], [296, 657], [840, 654]],
                    [[302, 245], [820, 251], [303, 660], [820, 651]]])
    # 変換後座標
    st = np.array([[0, 0], [w, 0], [0, h], [w, h]])
    print('射影変換')
    for (i, (img, uv)) in tqdm(enumerate(zip(imgs, uvs))):
        imgs_transed.append(trans.transform(img, uv, st))
        cv2.imwrite(os.path.join('.', 'output', 'transed', f'{i}.png'), imgs_transed[-1])
    imgs_transed = np.array(imgs_transed)

    dot_w = 4
    img_stripe = np.zeros((h, w, 3))
    print('生成')
    for i in tqdm(range(w//dot_w)):
        if i % 5 == 0:
            img_stripe[:, i*dot_w:(i+1)*dot_w, :] = imgs_transed[0][:, i*dot_w:(i+1)*dot_w, :]
        elif i % 5 == 1:
            img_stripe[:, i*dot_w:(i+1)*dot_w, :] = imgs_transed[1][:, i*dot_w:(i+1)*dot_w, :]
        elif i % 5 == 2:
            img_stripe[:, i*dot_w:(i+1)*dot_w, :] = imgs_transed[2][:, i*dot_w:(i+1)*dot_w, :]
        elif i % 5 == 3:
            img_stripe[:, i*dot_w:(i+1)*dot_w, :] = imgs_transed[3][:, i*dot_w:(i+1)*dot_w, :]
        elif i % 5 == 4:
            img_stripe[:, i*dot_w:(i+1)*dot_w, :] = imgs_transed[4][:, i*dot_w:(i+1)*dot_w, :]
    cv2.imwrite(os.path.join('.', 'output', 'result.png'), img_stripe)


if __name__ == '__main__':
    main()
