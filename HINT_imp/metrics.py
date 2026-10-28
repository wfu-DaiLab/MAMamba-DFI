import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import inception_v3
from scipy import linalg
import lpips
from PIL import Image  # 必须使用 PIL 以匹配 Evaluation.py

from glob import glob
from ntpath import basename
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.color import rgb2gray


# =========================================================
#  FID 计算辅助函数
# =========================================================
def get_activations(images, model, batch_size=32, dims=2048, device='cuda'):
    model.eval()

    if isinstance(images, list):
        images = torch.cat(images, dim=0)

    num_images = images.shape[0]
    pred_arr = np.empty((num_images, dims))

    for i in range(0, num_images, batch_size):
        batch = images[i:i + batch_size].to(device)

        # 严格匹配 Evaluation.py 的预处理:
        # 1. Resize 到 299x299 (bilinear)
        if batch.shape[2] != 299 or batch.shape[3] != 299:
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)

        # 2. 归一化到 [-1, 1]
        batch = batch * 2 - 1

        with torch.no_grad():
            pred = model(batch)

        if pred.dim() > 2:
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = F.adaptive_avg_pool2d(pred, 1)

        pred = pred.reshape(batch.size(0), -1)
        pred_arr[i:i + batch_size] = pred.cpu().numpy()

    return pred_arr


def calculate_activation_statistics(images, model, device='cuda'):
    act = get_activations(images, model, device=device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_fid(original_images_list, reconstructed_images_list, device='cuda'):
    print("Loading InceptionV3 model for FID...")

    # --- 关键修正 ---
    # 无论使用新版还是旧版 API，都必须显式设置 transform_input=False
    # Evaluation.py 显式设置了这个参数。
    # 如果不设置，InceptionV3 默认会进行内部归一化，导致 FID 计算错误。

    try:
        from torchvision.models import Inception_V3_Weights
        weights = Inception_V3_Weights.DEFAULT
        # 修正：显式加入 transform_input=False
        inception_model = inception_v3(weights=weights, transform_input=False).to(device)
    except (ImportError, AttributeError):
        inception_model = inception_v3(pretrained=True, transform_input=False).to(device)

    inception_model.fc = nn.Identity()
    inception_model.eval()

    mu1, sigma1 = calculate_activation_statistics(original_images_list, inception_model, device)
    mu2, sigma2 = calculate_activation_statistics(reconstructed_images_list, inception_model, device)

    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value


# =========================================================
#  Main Script
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path',
                        default='/home/li/wu/data_set/inpainting/MuralDH/Mural_seg/test/images256',
                        type=str)
    parser.add_argument('--output-path',
                        default='/home/li/wu/sec_work/HINT_imp/test_results/MuralDH/0.5-0.6/iter_120080/images',
                        type=str)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args


def compare_mae(img_true, img_test):
    # img_true, img_test 均为 numpy array, 范围 [0, 1]
    absolute_error = np.abs(img_true - img_test)
    mae_percent = np.mean(absolute_error) * 100
    return mae_percent


def calculate_l1(img_true, img_test):
    if len(img_true.shape) == 2:
        diff = img_true - img_test
    else:
        diff = np.mean(img_true, 2) - np.mean(img_test, 2)
    return np.mean(np.abs(diff) / 255)


if __name__ == "__main__":
    args = parse_args()
    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))

    path_true = args.data_path
    path_pred = args.output_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # LPIPS
    lpips_model = lpips.LPIPS(net='alex').to(device)

    psnr = []
    ssim = []
    mae = []
    l1 = []
    lpips_scores = []
    names = []

    gt_images_tensor_list = []
    pred_images_tensor_list = []

    index = 1
    files = list(glob(path_true + '/*.jpg')) + list(glob(path_true + '/*.png'))
    files = sorted(files)

    print(f"Found {len(files)} images. Starting evaluation...")

    # 标准 ToTensor
    to_tensor = transforms.ToTensor()

    for fn in files:
        name = basename(str(fn))
        names.append(name)

        # 1. 使用 PIL 读取
        img_gt_pil = Image.open(str(fn)).convert('RGB')
        img_pred_pil = Image.open(path_pred + '/' + name).convert('RGB')

        # 2. 尺寸对齐 (与 Evaluation.py 逻辑一致)
        if img_gt_pil.size != img_pred_pil.size:
            img_pred_pil = img_pred_pil.resize(img_gt_pil.size, Image.BICUBIC)

        # 3. 转 Tensor (用于 FID/LPIPS)
        img_gt_tensor = to_tensor(img_gt_pil).unsqueeze(0).to(device)
        img_pred_tensor = to_tensor(img_pred_pil).unsqueeze(0).to(device)

        gt_images_tensor_list.append(img_gt_tensor.cpu())
        pred_images_tensor_list.append(img_pred_tensor.cpu())

        # 4. 转 Numpy (用于 PSNR/SSIM/MAE/L1) - 范围 [0, 1]
        img_gt_rgb = img_gt_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_pred_rgb = img_pred_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # 计算 LPIPS
        with torch.no_grad():
            lpips_val = lpips_model(img_gt_tensor * 2 - 1, img_pred_tensor * 2 - 1)
            lpips_scores.append(lpips_val.item())

        # 计算 MAE
        mae.append(compare_mae(img_gt_rgb, img_pred_rgb))

        # 灰度化
        img_gt_gray = rgb2gray(img_gt_rgb)
        img_pred_gray = rgb2gray(img_pred_rgb)

        if args.debug != 0:
            plt.subplot('121')
            plt.imshow(img_gt_gray, cmap='gray')
            plt.title('Groud truth')
            plt.subplot('122')
            plt.imshow(img_pred_gray, cmap='gray')
            plt.title('Output')
            plt.show()

        # 计算 PSNR/SSIM/L1 (灰度)
        psnr.append(compare_psnr(img_gt_gray, img_pred_gray, data_range=1))
        ssim.append(compare_ssim(img_gt_gray, img_pred_gray, data_range=1, win_size=51))
        l1.append(calculate_l1(img_gt_gray, img_pred_gray))

        if np.mod(index, 10) == 0:
            print(
                str(index) + ' images processed',
                "PSNR: %.4f" % np.mean(psnr),
                "SSIM: %.4f" % np.mean(ssim),
                "MAE: %.4f" % np.mean(mae),
                "LPIPS: %.4f" % np.mean(lpips_scores)
            )
        index += 1

    # 计算 FID
    print("Calculating FID (this may take a while)...")
    fid_score = calculate_fid(gt_images_tensor_list, pred_images_tensor_list, device=device)

    # 保存结果
    np.savez(args.output_path + '/metrics.npz',
             psnr=psnr, ssim=ssim, mae=mae, l1=l1, lpips=lpips_scores, fid=fid_score, names=names)

    print("\n" + "=" * 50)
    print("Final Results:")
    print("PSNR: %.4f (Var: %.4f)" % (np.mean(psnr), np.var(psnr)))
    print("SSIM: %.4f (Var: %.4f)" % (np.mean(ssim), np.var(ssim)))
    print("MAE:  %.4f (Var: %.4f) %%" % (np.mean(mae), np.var(mae)))
    print("LPIPS: %.4f (Var: %.4f)" % (np.mean(lpips_scores), np.var(lpips_scores)))
    print("FID:   %.4f" % fid_score)
    print("L1:   %.4f (Var: %.4f)" % (np.mean(l1), np.var(l1)))
    print("=" * 50)