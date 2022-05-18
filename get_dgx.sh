#!/bin/bash
module load slurm cuda/11.3 gcc/9.3

if [[ $1 == "-i" ]]; then
    srun --time=3-00:00:00 -A eecs -p dgx2 --gres=gpu:1 --pty bash
elif [[ $1 == "-b" ]]; then
    if [[ -n $2 ]]; then
        sbatch --time=3-00:00:00 -A eecs -p dgx2 --gres=gpu:1 --pty bash python $2
    else
        echo "please specify the python filename you want to execute"
    fi
else
    echo "please specify whether to run batch (-b) or interactive session (-i)"
fi
