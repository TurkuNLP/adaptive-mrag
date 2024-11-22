#!/bin/bash
#SBATCH --account=project_2000539
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=output.txt
#SBATCH --error=error.txt

source venv-mrag/bin/activate
module load pytorch

srun python3 create-pkl.py 

deactivate