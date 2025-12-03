#!/bin/bash
#SBATCH --job-name=TrainMoreGpu      # 作业名称

#SBATCH --partition=mixed              # 分区
#SBATCH --nodes=1                      # 使用一个节点
#SBATCH --ntasks-per-node=1            # 在该节点上运行一个任务
#SBATCH --cpus-per-task=40             # 为任务分配的 CPU 核心数
#SBATCH --gres=gpu:4                   # 请求一个 GPU
#SBATCH --time=0                       # 作业最大运行时间
#SBATCH --output=模型失语修复.out # 标准输出文件名
#SBATCH --error=模型失语修复.err  # 标准错误文件名


# --- 设置环境 ---
PROJECT_ROOT="/gpool/home/wanghongyang/WangHY/WYY/tree_enhanced_dl"

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "Error: Project root $PROJECT_ROOT not found."
    exit 1
fi

# 激活 conda 环境（如果需要）
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate SaLoRA

# 设置 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 切换到项目目录
cd "$PROJECT_ROOT"

# 打印节点信息（用于调试）
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

echo "Starting training at $(date)"

# 使用 srun 运行（SLURM 会自动设置分布式环境变量）
python train_moregpu.py \
    --rs 32 \
    --ds 32 \
    --bs 16 \
    --rank 16 \
    --lr 1e-4 \
    --port 29500


echo "SaLoRA train_moregpu.py finished at $(date)!"
