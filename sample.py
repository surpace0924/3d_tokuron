import cv2
import numpy as np
import glob
import os
h = 2000
w = 2700

imgs = np.zeros((4, h, w, 3), dtype=np.uint8)
imgs[0] = cv2.rectangle(imgs[0], (w//2-100, 0*h//7), (w//2+100, 1*h//7), (255, 255, 255), thickness=-1)
imgs[1] = cv2.rectangle(imgs[1], (w//2-100, 2*h//7), (w//2+100, 3*h//7), (255, 0, 0), thickness=-1)
imgs[2] = cv2.rectangle(imgs[2], (w//2-100, 4*h//7), (w//2+100, 5*h//7), (0, 255, 0), thickness=-1)
imgs[3] = cv2.rectangle(imgs[3], (w//2-100, 6*h//7), (w//2+100, 7*h//7), (0, 0, 255), thickness=-1)

path_imgs = sorted(glob.glob(os.path.join(".", "input", "*.png")))
imgs = []
for path_img in path_imgs:
    print(path_img)
    img = cv2.imread(path_img)
    img = cv2.resize(img, dsize=(w, h))
    imgs.append(img)
imgs = np.array(imgs)


haba = 4
neo_image = np.zeros((h, w, 3))
for i in range(w//haba):
    if i % 5 == 0:
        neo_image[:, i*haba:(i+1)*haba, :] = imgs[0][:, i*haba:(i+1)*haba, :]
    elif i % 5 == 1:
        neo_image[:, i*haba:(i+1)*haba, :] = imgs[1][:, i*haba:(i+1)*haba, :]
    elif i % 5 == 2:
        neo_image[:, i*haba:(i+1)*haba, :] = imgs[2][:, i*haba:(i+1)*haba, :]
    elif i % 5 == 3:
        neo_image[:, i*haba:(i+1)*haba, :] = imgs[3][:, i*haba:(i+1)*haba, :]
    elif i % 5 == 4:
        neo_image[:, i*haba:(i+1)*haba, :] = imgs[4][:, i*haba:(i+1)*haba, :]
    # elif i % 5 == 4:
    #     neo_image[:, i*haba:(i+1)*haba, :] = image_5th[:, i*haba:(i+1)*haba, :]
cv2.imwrite("./sample" + "_haba-{0}_".format(haba) + ".png", neo_image)
