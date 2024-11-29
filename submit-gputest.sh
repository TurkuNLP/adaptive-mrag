#!/bin/bash
#SBATCH --account=project_2000539
#SBATCH --partition=gputest
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=output.txt
#SBATCH --error=error.txt

source venv-mrag/bin/activate
module load pytorch

unzip /scratch/project_2000539/maryam/adaptive-mrag/news.zip -d $LOCAL_SCRATCH

while true; do
  sleep 60  
  rsync -av $LOCAL_SCRATCH/test /scratch/project_2000539/maryam/adaptive-mrag/output
done &
srun python3 create-pkl.py

rsync -av $LOCAL_SCRATCH/test /scratch/project_2000539/maryam/adaptive-mrag/output

deactivate