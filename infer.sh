export CUDA=0

export CHECKPOINT_DIR="jingheya/lotus-depth-g-v2-1-disparity"
export OUTPUT_DIR="output/Depth_G_Infer_Texture"
export TASK_NAME="depth"
export MODE="generation"

# export CHECKPOINT_DIR="jingheya/lotus-normal-g-v1-1"
# export TASK_NAME="normal"
# export MODE="generation"
# export OUTPUT_DIR="output/Normal_Texture"

# 把你的图片放在这里
export TEST_IMAGES="assets/in-the-wild_example"

# 关键修改：
# 1. 添加 --steps 50 (如果太慢可以改30，想要极致纹理试 50)
# 2. --disparity 确保开启
CUDA_VISIBLE_DEVICES=$CUDA python infer.py \
        --pretrained_model_name_or_path=$CHECKPOINT_DIR \
        --prediction_type="sample" \
        --seed=41 \
        --input_dir=$TEST_IMAGES \
        --task_name=$TASK_NAME \
        --mode=$MODE \
        --output_dir=$OUTPUT_DIR \
        --disparity \
        --steps=4 \
        --resample_method="bicubic" \
        --depth_gamma 1.2 \
        --processing_res=0 \
        --prompt "3d facial relief, sharp nose, deep depth map, highly detailed sculpture" \
        --neg_prompt "flat, low relief, blurry, smooth surface" \
        --half_precision
