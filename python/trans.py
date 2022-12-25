import os
import cv2
import numpy as np


# def main():
#     # 画像を読み込んでチャネル分割
#     img = cv2.imread(os.path.join('.', 'input', '4.png'))

#     # 変換前座標（画像に映るA4用紙の4隅の座標）
#     # uv = np.array([[378, 320], [1017, 284], [378, 771], [1017, 821]])
#     # uv = np.array([[360, 320], [1017, 299], [360, 783], [1017, 801]])
#     # uv = np.array([[341, 305], [1010, 305], [341, 794], [1010, 794]])
#     # uv = np.array([[332, 296], [992, 314], [332, 807], [992, 780]])
#     uv = np.array([[333, 287], [956, 323], [333, 821], [956, 765]])
    
#     # 変換後座標
#     len_s = 2700
#     len_t = 2000
#     st = np.array([[0, 0], [len_s, 0], [0, len_t], [len_s, len_t]])


#     # 画像の書き出し
#     cv2.imwrite(os.path.join('.', '4.png'), img_transed)

def transform(img, uv, st):
    # 射影変換のパラメータ（h11, h12, h13, h21, h22, h23, h31, h32）の計算
    h = calParam(st, uv)

    # 各チャンネルごとに射影変換を実行
    img_transed = []
    for img_channel in cv2.split(img):
        img_transed.append(transformByParam(img_channel, st, h))
    img_transed = np.array(img_transed, dtype=np.uint8).transpose(1, 2, 0)

    return img_transed


# 射影変換のパラメータ（h11, h12, h13, h21, h22, h23, h31, h32）の計算
def calParam(st, uv):
    # 方程式の係数行列
    A = []
    for i in range(len(st)):
        s, t, u, v = st[i][0], st[i][1], uv[i][0], uv[i][1]
        A.append([s, t, 1, 0, 0, 0, -u*s, -u*t])
        A.append([0, 0, 0, s, t, 1, -v*s, -v*t])
    A = np.array(A)
    b = np.array([uv[i][j] for i in range(len(uv)) for j in range(2)])
    
    # 最小2乗解の計算
    h = np.linalg.solve(A, b)
    return h


# 射影変換の実行
def transformByParam(img, st, h):
    len_s, len_t = st[3][0], st[3][1]
    img_transed = np.zeros((len_t, len_s))
    for t in range(len_t):
        for s in range(len_s):
            # 変換後の座標stに対応する座標uvを求める
            u = (h[0]*s + h[1]*t + h[2])/(h[6]*s + h[7]*t + 1)
            v = (h[3]*s + h[4]*t + h[5])/(h[6]*s + h[7]*t + 1)

            # バイリニア補間をする
            ui, vi = int(u), int(v)
            p, q = u-ui, v-vi
            img_transed[t, s] = (1-q)*((1-p)*img[vi, ui] + p*img[vi, ui+1]) + q*((1-p)*img[vi+1, ui] + p*img[vi+1, ui+1])

    return img_transed.astype(np.uint8)
