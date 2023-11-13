#!/bin/bash
#SBATCH -o /home/jaehyunkang/slurm_log/genvis_mevis/%j.log

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo $CUDA_VISIBLE_DEVICES
gpustat

echo "num_nodes: ${num_nodes} | current node: ${node_idx}"
echo "CLIP_LEN: ${CLIP_LEN} | CLIP_NUM: ${CLIP_NUM} | num_gpus: $num_gpus | DIST_URL: ${DIST_URL}"

num_gpus=2
dataset="-B /datasets/ytvis_2021:/datasets/ytvis_2021 -B /datasets/ytvis_2019:/datasets/ytvis_2019 -B /home/jaehyunkang/datasets/COCO_new:/datasets/coco -B /datasets/OVIS:/datasets/ovis"
docker=" /home/jaehyunkang/simages/rvos_miran.simg "
exec_file=" python /home/jaehyunkang/GenVIS/train_net_genvis.py "
machine_cfg=" --num-gpus ${num_gpus} "
config_file=" --config-file /home/jaehyunkang/GenVIS/configs/genvis/mevis/genvis_R50_bs8_semi_online.yaml "
out_dir=" OUTPUT_DIR /home/jaehyunkang/GenVIS/output/mevis "
dataloader_cpu=" DATALOADER.NUM_WORKERS 1"

ml purge
ml load singularity
singularity exec --nv $dataset $mnt $docker $exec_file --resume $config_file $machine_cfg $out_dir $dataloader_cpu TEST.EVAL_PERIOD 0
