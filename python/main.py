import os
import glob
import numpy as np
import cv2
from tqdm import tqdm

import trans

def main():
    h = 2000
    w = 2700

    # imgs = np.zeros((4, h, w, 3), dtype=np.uint8)
    # imgs[0] = cv2.rectangle(imgs[0], (w//2-100, 0*h//7), (w//2+100, 1*h//7), (255, 255, 255), thickness=-1)
    # imgs[1] = cv2.rectangle(imgs[1], (w//2-100, 2*h//7), (w//2+100, 3*h//7), (255, 0, 0), thickness=-1)
    # imgs[2] = cv2.rectangle(imgs[2], (w//2-100, 4*h//7), (w//2+100, 5*h//7), (0, 255, 0), thickness=-1)
    # imgs[3] = cv2.rectangle(imgs[3], (w//2-100, 6*h//7), (w//2+100, 7*h//7), (0, 0, 255), thickness=-1)

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
    # 変換前座標
    uvs = np.array([[[378, 320], [1017, 284], [378, 771], [1017, 821]],
                    [[360, 320], [1017, 299], [360, 783], [1017, 801]],
                    [[341, 305], [1010, 305], [341, 794], [1010, 794]],
                    [[332, 296], [ 992, 314], [332, 807], [ 992, 780]],
                    [[333, 287], [ 956, 323], [333, 821], [ 956, 765]]])
    # 変換後座標
    st = np.array([[0, 0], [w, 0], [0, h], [w, h]])
    print('射影変換')
    for (i, (img, uv)) in tqdm(enumerate(zip(imgs, uvs))):
        imgs_transed.append(trans.transform(img, uv, st))
        cv2.imwrite(os.path.join('.', 'output', 'transed', f'{i}.png'), imgs_transed[-1])
    imgs_transed = np.array(imgs_transed)

    haba = 4
    neo_image = np.zeros((h, w, 3))
    print('生成')
    for i in tqdm(range(w//haba)):
        if i % 5 == 0:
            neo_image[:, i*haba:(i+1)*haba, :] = imgs_transed[0][:, i*haba:(i+1)*haba, :]
        elif i % 5 == 1:
            neo_image[:, i*haba:(i+1)*haba, :] = imgs_transed[1][:, i*haba:(i+1)*haba, :]
        elif i % 5 == 2:
            neo_image[:, i*haba:(i+1)*haba, :] = imgs_transed[2][:, i*haba:(i+1)*haba, :]
        elif i % 5 == 3:
            neo_image[:, i*haba:(i+1)*haba, :] = imgs_transed[3][:, i*haba:(i+1)*haba, :]
        elif i % 5 == 4:
            neo_image[:, i*haba:(i+1)*haba, :] = imgs_transed[4][:, i*haba:(i+1)*haba, :]
    cv2.imwrite(os.path.join('.', 'output', 'result.png'), neo_image)


if __name__ == '__main__':
    main()
