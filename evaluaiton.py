import os
import time
import shutil
import argparse
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import convolve
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
from skimage import io
from skimage import feature
from skimage import img_as_ubyte
from skimage.filters import sobel
from multiprocessing import Pool

def conv_tri(image, radius):
    kernel_size = 2 * radius + 1
    triangle_kernel = np.convolve([1]*kernel_size, [1]*kernel_size)
    triangle_kernel = triangle_kernel / triangle_kernel.sum()
    triangle_kernel = triangle_kernel.reshape(-1, 1)
    image_filtered = convolve(image, triangle_kernel, mode='reflect')
    image_filtered = convolve(image_filtered, triangle_kernel.T, mode='reflect')
    return image_filtered

def gradient_2(im):
    im_filtered = gaussian_filter(im, sigma=1)
    gx, gy = np.gradient(im_filtered)
    return gx, gy

def non_max_suppression(E, O, radius=1):
    """
    自定义非极大值抑制函数
    E: 边缘强度
    O: 梯度方向
    radius: 邻域半径
    """
    Z = np.zeros(E.shape, dtype=np.float32)
    angle = O * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, E.shape[0]-1):
        for j in range(1, E.shape[1]-1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = E[i, j+1]
                    r = E[i, j-1]
                # angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = E[i+1, j-1]
                    r = E[i-1, j+1]
                # angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = E[i+1, j]
                    r = E[i-1, j]
                # angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = E[i-1, j-1]
                    r = E[i+1, j+1]

                if (E[i,j] >= q) and (E[i,j] >= r):
                    Z[i,j] = E[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass

    return Z

def process_image(args):
    mat_file, mat_dir, nms_dir = args
    # 读取 MAT 文件
    mat = loadmat(os.path.join(mat_dir, mat_file))
    # 假设变量名未知，取第一个变量
    var_name = list(mat.keys())[-1]
    x = mat[var_name]

    # 进行卷积和平滑
    E = conv_tri(x.astype(np.float32), 1)
    Ox, Oy = np.gradient(conv_tri(E, 4))
    Oxx, _ = np.gradient(Ox)
    Oxy, Oyy = np.gradient(Oy)
    O = np.mod(np.arctan2(Oyy * np.sign(-Oxy), Oxx + 1e-5), np.pi)

    # 非极大值抑制
    E_nms = non_max_suppression(E, O)
    # 保存结果
    nms_name = mat_file[:-4] + '.png'
    io.imsave(os.path.join(nms_dir, nms_name), img_as_ubyte(E_nms / E_nms.max()))
    return nms_name

def main():
    parser = argparse.ArgumentParser(description='Edge Detection Evaluation')
    parser.add_argument('--data_dir', type=str, default='map_folder/table5_pidinet', help='Data directory')
    parser.add_argument('--ablation', action='store_true', help='Ablation study flag')
    parser.add_argument('--suffix', type=str, default='_epoch_019', help='Suffix for directories')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads to use')
    args = parser.parse_args()

    data_dir = args.data_dir
    ablation = args.ablation
    suffix = args.suffix
    num_threads = args.num_threads

    print(f'Data dir: {data_dir}')

    start_time = time.time()
    # Section 1: NMS process
    print('NMS process...')
    mat_dir = os.path.join(data_dir, 'mats' + suffix)
    nms_dir = os.path.join(data_dir, 'nms' + suffix)
    os.makedirs(nms_dir, exist_ok=True)

    mat_files = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]

    # 使用多进程处理
    pool = Pool(num_threads)
    args_list = [(mat_file, mat_dir, nms_dir) for mat_file in mat_files]
    results = pool.map(process_image, args_list)
    pool.close()
    pool.join()

    # Section 2: Evaluate the edges
    print('Evaluate the edges...')
    if ablation:
        gt_dir = 'data/groundTruth/val'
    else:
        gt_dir = 'data/groundTruth/test'
    res_dir = nms_dir

    # 评估结果
    # 此处需要实现评估函数，计算精度、召回率等指标
    # 可以使用现有的评估工具，或者自行实现

    # 示例：计算每个图像的精度和召回率
    from sklearn.metrics import precision_recall_curve
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.png')]
    precisions = []
    recalls = []

    for gt_file in gt_files:
        gt = io.imread(os.path.join(gt_dir, gt_file), as_gray=True)
        gt = (gt > 128).astype(np.uint8).flatten()

        res_file = gt_file
        res = io.imread(os.path.join(res_dir, res_file), as_gray=True)
        res = (res > 0).astype(np.uint8).flatten()

        precision, recall, _ = precision_recall_curve(gt, res)
        precisions.append(precision[1])
        recalls.append(recall[1])

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    f1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-8)

    print(f'Average Precision: {avg_precision}')
    print(f'Average Recall: {avg_recall}')
    print(f'F1 Score: {f1_score}')

    # 删除临时文件夹
    shutil.rmtree(mat_dir)
    shutil.rmtree(nms_dir)

    elapsed_time = time.time() - start_time
    print(f'Total time: {elapsed_time:.2f} seconds')

if __name__ == '__main__':
    main()

