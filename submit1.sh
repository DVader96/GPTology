#!/bin/bash
#SBATCH --time=2:10:00
#SBATCH --mem=30GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH -o './logs/%A.out'
#SBATCH -e './logs/%A.err'

if [[ "$HOSTNAME" == *"tiger"* ]]
then
    echo "It's tiger"
    module load anaconda
    source activate torch-env
elif [[ "$HOSTNAME" == *"della"* ]]
then
    echo "It's Della"
    module load anaconda3/2021.11
    source activate torch_env
else
    module load anacondapy
    source activate srm
fi

echo 'Requester:' $USER
echo 'Node:' $HOSTNAME
echo 'Start time:' `date`
echo "$@"
if [[ -v SLURM_ARRAY_TASK_ID ]]
then
    python "$@" --electrodes $SLURM_ARRAY_TASK_ID
else
    python "$@"
fi
echo 'End time:' `date`
