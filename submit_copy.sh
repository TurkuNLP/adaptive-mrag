#!/bin/bash
#SBATCH --account=project_2002820
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:v100:2
#SBATCH --output=MLDR-stage1-out.txt
#SBATCH --error=MLDR-stage1-err.txt

module load pytorch
source venv-mrag/bin/activate

srun python3 mteb-task.py

deactivate