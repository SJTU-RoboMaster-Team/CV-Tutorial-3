import numpy as np
import cv2

# 相机内参
K = np.array([
    [1114.1804893712708, 0.0, 1074.2415297217708, ],
    [0.0, 1113.4568392254073, 608.6477877664104, ],
    [0.0, 0.0, 1.0, ],
])
# 相机内参的逆
K_inv = np.linalg.inv(K)


def calc_pose_and_homography(c0, d0, c1, d1):
    # 特征点检测和匹配
    detector = cv2.SIFT_create()
    kp0, des0 = detector.detectAndCompute(c0, None)
    kp1, des1 = detector.detectAndCompute(c1, None)
    bf = cv2.BFMatcher_create()
    matches = bf.knnMatch(des0, des1, k=2)
    good = [m for m, n in matches if m.distance < 0.5 * n.distance]
    # 提取匹配到的点和它们的深度
    p_i0 = np.array([kp0[m.queryIdx].pt for m in good])  # 第一张图片的特征点像素坐标
    d_0 = np.array([d0[int(p[1]), int(p[0])] for p in p_i0])  # 第一张图片特征点对应的深度
    p_i1 = np.array([kp1[m.trainIdx].pt for m in good])  # 第二张图片的特征点像素坐标
    d_1 = np.array([d1[int(p[1]), int(p[0])] for p in p_i1])  # 第二张图片特征点对应的深度
    # 使用对极约束剔除误匹配
    E, status = cv2.findEssentialMat(p_i0, p_i1, K, cv2.RANSAC, 0.999, 1.0)
    p_i0 = np.array([p_i0[i] for i in range(len(status)) if status[i]])
    d_0 = np.array([d_0[i] for i in range(len(status)) if status[i]])
    p_i1 = np.array([p_i1[i] for i in range(len(status)) if status[i]])
    d_1 = np.array([d_1[i] for i in range(len(status)) if status[i]])

    # 使用PnP计算相对位姿
    p_i0_ = np.concatenate([p_i0, np.ones([p_i0.shape[0], 1])], axis=1)  # 第一张图片的特征点像素坐标(齐次坐标)
    p_c0 = d_0 * (K_inv @ p_i0_.T)  # 第一张图片的特征点对应的三维坐标(在第一张图片的相机坐标系下)
    p_c0 = p_c0.T
    rst, rvecs, tvecs, inliers = cv2.solvePnPRansac(p_c0, p_i1, K, None)  # 解PnP
    rmats, _ = cv2.Rodrigues(rvecs)  # 旋转向量转旋转矩阵
    # 相对位姿使用4x4的齐次变换矩阵表示
    T = np.eye(4)
    T[:3, :3] = rmats
    T[:3, 3] = tvecs.T

    # 计算投影变换的单应性矩阵
    H, status = cv2.findHomography(p_i0, p_i1, cv2.RANSAC, 5.0)
    return T, H


# 使用投影变换合成全景图
def perspective_method(color, depth, H0N):
    height, width, channel = color[0].shape

    # 平移变换矩阵(投影变换后可能有些像素区域在图像外侧，故平移图像，使得所有像素区域都在图像内部)
    transform = np.array([[1, 0, width],
                          [0, 1, height],
                          [0, 0, 1], ], dtype=np.float64)
    # 平移第一张图片，使得生成的图片的长宽变为原来的3倍，并且平移后的图像位于新图片的中央
    c0 = cv2.warpPerspective(color[0], transform, (width * 3, height * 3))
    # 从第二张图片开始，通过计算好的单应性矩阵，将后面的图片全部变换到第一张图片的视角，并且平移图片
    c0N = [cv2.warpPerspective(c, transform @ H0x, (width * 3, height * 3)) for c, H0x in zip(color[1:], H0N)]
    # 准备合并图片
    homo = c0
    for c0x in c0N:
        # 仅将新图片中的非零像素拷贝到合并后的图片中
        homo = np.where(c0x != 0, c0x, homo)
    # 图像显示和保存
    cv2.imwrite("homo.jpg", homo)
    homo = cv2.resize(homo, (-1, -1), fx=0.3, fy=0.3)
    cv2.imshow("homo", homo)
    cv2.waitKey(0)


# 使用重投影法合成全景图
def reproject_method(color, depth, T0N):
    height, width, channel = color[0].shape

    # 用于保存重投影结果(长宽变为原图的3倍，避免有的像素超出图片范围)
    reproject = np.zeros([height * 3, width * 3, channel], dtype=np.uint8)

    # 从第二张图片开始，遍历每张图片，依次进行重投影(第一张图片不需要重投影，直接复制到对应位置即可，因为所有图片都是投影到第一张图片的视角)
    for c, d, T0X in zip(color[1:], depth[1:], T0N):
        # 生成所有像素点的x坐标。例：如果图像大小为3x3，则生成的结果为[0,1,2,0,1,2,0,1,2]
        pXu = np.arange(width).reshape(1, width).repeat(height, axis=0).reshape(width * height)
        # 生成所有像素点的y坐标。例：如果图像大小为3x3，则生成的结果为[0,0,0,1,1,1,2,2,2]
        pXv = np.arange(height).reshape(height, 1).repeat(width, axis=0).reshape(width * height)
        # 生成所有像素点的深度。顺序为从上到下，从左到右。
        pXd = d[pXv, pXu]
        # 生成所有像素点的其次像素坐标
        pX_i_ = np.stack([pXu, pXv, np.ones_like(pXu)], axis=1).T
        # 计算所有像素点的三维坐标(在自己的相机坐标系下)
        pX_c = pXd * (K_inv @ pX_i_)
        # 三维坐标转齐次坐标
        pX_c_ = np.concatenate([pX_c, np.ones([1, pX_c.shape[1]])], axis=0)
        # 通过其次变换矩阵计算所有像素在第一张图片的相机坐标系下的三维坐标(此时结果为其次坐标系)
        p0_c_ = T0X @ pX_c_
        # 齐次坐标转费齐次坐标
        p0_c = p0_c_[:3]
        # 计算重投影后的像素坐标
        p0_i_ = K @ (p0_c / p0_c[2])
        # 像素坐标四舍五入
        p0_i = np.round(p0_i_[:2].T).astype(np.int)
        p0u = p0_i[:, 0]
        p0v = p0_i[:, 1]
        # 横坐标平移，并剔除超出图片外的像素
        distu = p0u + width
        distu = np.where(distu < 0, 0, distu)
        distu = np.where(distu >= width * 3, width * 3 - 1, distu)
        # 纵坐标平移，并剔除超出图片外的像素
        distv = p0v + height
        distv = np.where(distv < 0, 0, distv)
        distv = np.where(distv >= height * 3, height * 3 - 1, distv)
        # 重投影的点赋值到目标图像中
        reproject[distv, distu] = c[pXv, pXu]
    # 复制第一张图像
    reproject[height:height * 2, width:width * 2] = color[0]
    # 保存和显示图片
    cv2.imwrite("reproject.jpg", reproject)
    reproject = cv2.resize(reproject, (-1, -1), fx=0.3, fy=0.3)
    cv2.imshow("reproject", reproject)
    cv2.waitKey(0)


if __name__ == "__main__":
    # 读取所有图片和深度图
    color = [cv2.imread(f"stereo-data/{i}_orig.jpg") for i in range(3)]
    depth = [np.load(f"stereo-data/{i}_dpt.npy") for i in range(3)]

    # 从第二张图开始，计算每张图和第一张图之间的单应矩阵和齐次变换矩阵
    T0N = []
    H0N = []
    for c, d in zip(color[1:], depth[1:]):
        T, H = calc_pose_and_homography(c, d, color[0], depth[0])
        T0N.append(T)
        H0N.append(H)

    # 使用投影变换合成全景图
    perspective_method(color, depth, H0N)
    # 使用重投影法合成全景图
    reproject_method(color, depth, T0N)
    cv2.destroyAllWindows()
