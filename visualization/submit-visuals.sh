#!/bin/bash
#SBATCH --account=project_2002820
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=visualize-output.txt
#SBATCH --error=visualize-error.txt

source venv-mrag/bin/activate
module load pytorch

srun python3 visualize.py

deactivate