#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --account=blanca-clearlab1
#SBATCH --partition=blanca-clearlab1
#SBATCH --qos=blanca-clearlab1
#SBATCH --gres=gpu:1
#SBATCH --output=job_logs/squad-clusters-%j.out
#SBATCH --time=12:00:00

# 1. Changing to project directory
cd /projects/tejo9855/Projects/ComputationalModelsOfDiscourseRAG

# 2. Loading Modules
module load anaconda
module load cuda/12.1.1

# 3. Loading conda environment
conda activate teagan-conda-env-curc

# 4. Generate coreference clusters for all 412 SQuAD articles
echo "========================================"
echo "Generating SQuAD coreference clusters"
echo "========================================"
python -u squad_testing/scripts/generate_squad_clusters.py
