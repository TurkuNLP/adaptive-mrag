#!/bin/bash
#SBATCH --account=project_
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=output.txt
#SBATCH --error=error.txt

module load pytorch
source venv-mrag/bin/activate

srun python3 create-pkl.py 

deactivate