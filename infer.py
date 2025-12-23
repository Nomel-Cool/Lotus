import logging
import os
import argparse
from pathlib import Path
from PIL import Image
from contextlib import nullcontext

import numpy as np
import torch
import cv2
from tqdm.auto import tqdm
from diffusers.utils import check_min_version
from diffusers import AutoencoderKL

from pipeline import LotusGPipeline, LotusDPipeline
from utils.image_utils import colorize_depth_map
from utils.seed_all import seed_all

check_min_version('0.28.0.dev0')

def apply_gamma(image_np, gamma=0.9):
    """
    image_np: numpy array, float32, range [0, 255]
    gamma < 1 brightens dark regions
    """
    image_np = image_np / 255.0
    image_np = np.power(image_np, gamma)
    image_np = image_np * 255.0
    return image_np

def parse_args():
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Run Lotus..."
    )
    # model settings
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="pretrained model path from hugging face or local dir",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="sample",
        help="The used prediction_type. ",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=999,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="regression", # "generation"
        help="Whether to use the generation or regression pipeline."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="depth", # "normal"
    )
    parser.add_argument(
        "--disparity",
        action="store_true",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--depth_gamma",
        type=float,
        default=0.7,
        help="Gamma correction for depth output. <1 increases contrast."
    )

    # inference settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    
    # MODIFIED: 添加推理步数参数，默认 50 以获得高纹理
    parser.add_argument("--steps", type=int, default=50, help="Inference steps. Set to 1 for speed, 20-50 for high texture.")

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input directory."
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`. Default: `bilinear`",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="high relief, 3d sculpture, sharp facial features, deep depth",
        help="Prompt to guide the generative depth estimation."
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="flat, low relief, blurry, smooth surface",
        help="Prompt to guide the generative depth estimation."
    )

    args = parser.parse_args()

    return args

def preprocess_image_for_lotus(pil_image):
    """
    改进版预处理：平衡全局体积感与局部细节
    1. 减弱对暗部的暴力提亮，保留深度线索。
    2. 使用 '混合' 方式应用 CLAHE，而不是直接替换。
    """
    img = np.array(pil_image)
    
    # 转到 LAB 空间
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # === 1. 备份原始亮度 (用于恢复体积感) ===
    l_original = l_channel.copy()

    # === 2. CLAHE 增强 (针对细节) ===
    # 稍微降低 clipLimit (3.0 -> 2.0)，避免过度平坦化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l_channel)
    
    # === 3. 关键修改：高保真混合 (High-Fidelity Blend) ===
    # 我们不直接使用 l_clahe，而是把它和原始亮度混合。
    # 原始亮度提供“体积感”(头发是黑的)，CLAHE 提供“纹理”(发丝清晰)。
    # alpha=0.6 (原始权重), beta=0.4 (细节权重)
    # 你可以根据需要调整：想要体积感更强，就调高 alpha
    l_blend = cv2.addWeighted(l_original, 0.65, l_clahe, 0.35, 0)

    # === 4. 智能边缘锐化 (只针对边缘，不增加噪点) ===
    # 使用双边滤波 (Bilateral Filter) 做锐化掩膜，它能保边去噪
    blur = cv2.bilateralFilter(l_blend, 5, 75, 75)
    # 提取高频细节
    details = cv2.subtract(l_blend, blur)
    # 将细节叠加回去 (增强系数 2.0)
    l_final = cv2.addWeighted(l_blend, 1.0, details, 2.0, 0)
    
    # === 5. 可选：边缘暗角 (Vignette) ===
    # 如果主体在中间，稍微压暗四周可以强迫模型把四周推远
    rows, cols = l_final.shape
    kernel_x = cv2.getGaussianKernel(cols, cols/3) # 增大分母会减小暗角范围
    kernel_y = cv2.getGaussianKernel(rows, rows/3)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    
    # 稍微压暗一点点四周 (保持 90% 亮度，避免太黑)
    vignette_mask = mask * 0.2 + 0.8 
    l_final = (l_final * vignette_mask).astype(np.uint8)

    # 合并并输出
    merged_lab = cv2.merge([l_final, a_channel, b_channel])
    result_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(result_img)
    
def main():
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Run inference...")

    args = parse_args()

    # -------------------- Preparation --------------------
    # Random seed
    if args.seed is not None:
        seed_all(args.seed)

    # Output directories
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output dir = {args.output_dir}")

    output_dir_color = os.path.join(args.output_dir, f'{args.task_name}_vis')
    output_dir_npy = os.path.join(args.output_dir, f'{args.task_name}')
    
    # MODIFIED: 新增 16位 PNG 输出文件夹
    output_dir_16bit = os.path.join(args.output_dir, f'{args.task_name}_16bit')

    if not os.path.exists(output_dir_color): os.makedirs(output_dir_color)
    if not os.path.exists(output_dir_npy): os.makedirs(output_dir_npy)
    if not os.path.exists(output_dir_16bit): os.makedirs(output_dir_16bit)

    # half_precision
    if args.half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32
    
    # processing_res
    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"Device = {device}")

    # -------------------- Data --------------------
    root_dir = Path(args.input_dir)
    test_images = list(root_dir.rglob('*.png')) + list(root_dir.rglob('*.jpg'))
    test_images = sorted(test_images)
    print('==> There are', len(test_images), 'images for validation.')
    # -------------------- Model --------------------

    # === MODIFIED START: 强制加载高精度 VAE ===
    print("正在加载高精度 VAE (stabilityai/sd-vae-ft-mse)...")
    # 这个 VAE 对纹理的还原度比默认的 kl-f8 强很多
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", 
        torch_dtype=dtype
    ).to(device)
    # === MODIFIED END ===

    if args.mode == 'generation':
        pipeline = LotusGPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            torch_dtype=dtype,
        )
    elif args.mode == 'regression':
        pipeline = LotusDPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            torch_dtype=dtype,
        )
    else:
        raise ValueError(f'Invalid mode: {args.mode}')
    logging.info(f"Successfully loading pipeline from {args.pretrained_model_name_or_path}.")
    logging.info(f"processing_res = {processing_res or pipeline.default_processing_resolution}")
    logging.info(f"Inference steps = {args.steps}") # MODIFIED: 打印步数

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for i in tqdm(range(len(test_images))):
            if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
            else:
                autocast_ctx = torch.autocast(pipeline.device.type)
            with autocast_ctx:
                # Preprocess validation image
                test_image = Image.open(test_images[i]).convert('RGB')
                # 重要幻觉预处理利用修改
                test_image = preprocess_image_for_lotus(test_image) 
                test_image = np.array(test_image).astype(np.float32)
                test_image = torch.tensor(test_image).permute(2,0,1).unsqueeze(0)
                test_image = test_image / 127.5 - 1.0 
                test_image = test_image.to(device)

                task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(device)
                task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

                # Run
                # MODIFIED: 将 num_inference_steps 从硬编码的 1 改为 args.steps
                pred = pipeline(
                    rgb_in=test_image, 
                    prompt=args.prompt,
                    negative_prompt=args.neg_prompt,
                    num_inference_steps=args.steps, # <--- 关键修改：增加步数以获得纹理
                    generator=generator, 
                    # guidance_scale=0,
                    output_type='np',
                    timesteps=[args.timestep],
                    task_emb=task_emb,
                    processing_res=processing_res,
                    match_input_res=match_input_res,
                    resample_method=resample_method,
                    ).images[0]

                # Post-process the prediction
                save_file_name = os.path.basename(test_images[i])[:-4]
                
                if args.task_name == 'depth':
                    # ================= DEPTH 处理逻辑 =================

                    # 1. 取单通道（非常关键）
                    output_npy = pred[..., 0].astype(np.float32)

                    # 2. Robust percentile normalize（唯一一次归一化）
                    p_min, p_max = np.min(output_npy), np.max(output_npy)
                    depth_norm = (output_npy - p_min) / (p_max - p_min + 1e-6)

                    # 3. 极性处理
                    if not args.disparity:
                        depth_norm = 1.0 - depth_norm

                    # 5. Gamma（全局对比）
                    depth_norm = np.power(depth_norm, args.depth_gamma)

                    # 6. 转 16-bit
                    depth_uint16 = (depth_norm * 65535.0).astype(np.uint16)

                    # 7. 保存 16-bit depth
                    save_path_16bit = os.path.join(output_dir_16bit, f'{save_file_name}.png')
                    cv2.imwrite(save_path_16bit, depth_uint16)

                    # 8. 预览图 —— ❗用 depth_norm 而不是 output_npy
                    output_color = colorize_depth_map(depth_norm, reverse_color=args.disparity)

                else:
                    # ================= NORMAL 处理逻辑 (新增) =================
                    # output_npy 是 (H, W, 3) 的数据，范围 [0, 1]
                    output_npy = pred 
                    
                    # 1. 转换为 16-bit (每通道 16bit)
                    # 将 [0,1] 映射到 [0, 65535]
                    normal_16bit = (output_npy * 65535).astype(np.uint16)
                    
                    # 2. 颜色空间转换 (RGB -> BGR)
                    # OpenCV 默认使用 BGR 顺序保存图片，而 Lotus 输出的是 RGB
                    # 如果不转，保存出来的法线图颜色会是反的 (蓝色变红色)
                    normal_16bit_bgr = cv2.cvtColor(normal_16bit, cv2.COLOR_RGB2BGR)
                    
                    # 3. 保存 16-bit Normal
                    save_path_16bit = os.path.join(output_dir_16bit, f'{save_file_name}.png')
                    cv2.imwrite(save_path_16bit, normal_16bit_bgr)
                    
                    # 生成预览图
                    output_color = Image.fromarray((output_npy * 255).astype(np.uint8))

                # 保存通用的预览图和 npy 数据
                output_color.save(os.path.join(output_dir_color, f'{save_file_name}.png'))
                np.save(os.path.join(output_dir_npy, f'{save_file_name}.npy'), output_npy)

            torch.cuda.empty_cache()
            
    print('==> Inference is done. \n==> Results saved to:', args.output_dir)


if __name__ == '__main__':
    main()
