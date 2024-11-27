#!/bin/bash
#SBATCH --account=project_2000539
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:v100:6,nvme:100
#SBATCH --output=output.txt
#SBATCH --error=error.txt

source venv-mrag/bin/activate
module load pytorch

UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip /scratch/project_2000539/maryam/adaptive-mrag/news.zip -d $LOCAL_SCRATCH

while true; do
  sleep 3600  # Wait for an hour
  rsync -av $LOCAL_SCRATCH/*.pkl /scratch/project_2000539/maryam/adaptive-mrag/output
done &
srun python3 create-pkl.py

rsync -av $LOCAL_SCRATCH/*.pkl /scratch/project_2000539/maryam/adaptive-mrag/output

deactivate