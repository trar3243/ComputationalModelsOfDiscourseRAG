#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --account=blanca-clearlab1
#SBATCH --partition=blanca-clearlab1
#SBATCH --qos=blanca-clearlab1
#SBATCH --gres=gpu:1
#SBATCH --output=job_logs/squad-metrics-%j.out
#SBATCH --time=24:00:00

# Toggle retrieval mode: "filtered" or "global"
RETRIEVAL="global"

STRATEGIES=(
    "baseline_258_tok"
    "most_recent_low_acc_258t_w128_wtd"
    "most_recent_258t_w254"
    "nonlinear_258t_w254"
)

# 1. Changing to project directory
cd /projects/tejo9855/Projects/ComputationalModelsOfDiscourseRAG

# 2. Loading Modules
module load anaconda
module load cuda/12.1.1

# 3. Loading conda environment
conda activate teagan-conda-env-curc

# 4. Compute metrics for each chunking strategy
for STRATEGY in "${STRATEGIES[@]}"; do
    echo "========================================"
    echo "Computing metrics: $STRATEGY | retrieval: $RETRIEVAL"
    echo "========================================"
    python -u squad_testing/scripts/compute_squad_metrics.py \
        --strategy $STRATEGY \
        --retrieval $RETRIEVAL
done
