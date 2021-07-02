#!/bin/bash


# Comments starting with #OAR are used by the resource manager if using "oarsub -S"
#
# Note : quoting style of parameters matters, follow the example
#OAR -l /nodes=1/gpunum=1, walltime=00:30:00
#OAR -p gpu='YES' and host='nefgpu33.inria.fr'
#
# The job is submitted to the default queue
#OAR -q default


# pick first argument (101 or 50) in VAR1
VAR1=$1

# Place here your submission script body
conda activate semseg;
cd /data/chorale/share/cartizzu/6_SEGMENTATION/semseg/;
python3 eval_multipro.py --gpus 0 --cfg config/r50_upp_rot_e100_nef_30.yaml;
