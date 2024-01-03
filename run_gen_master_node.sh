#!/bin/bash
#SBATCH -o /home/jaehyunkang/slurm_log/lmpm_mevis/%j.log

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo $CUDA_VISIBLE_DEVICES
gpustat

MASTER_NODE=$SLURM_NODELIST # sbatch output environment 사용
DIST_URL="tcp://$MASTER_NODE:$MASTER_PORT"
echo "num_nodes: ${num_nodes} | current node: ${node_idx}"
echo "CLIP_LEN: ${CLIP_LEN} | CLIP_NUM: ${CLIP_NUM} | num_gpus: $num_gpus | DIST_URL: ${DIST_URL}"

dataset="-B /datasets/ytvis_2021:/datasets/ytvis_2021 -B /datasets/ytvis_2019:/datasets/ytvis_2019 -B /home/jaehyunkang/datasets/COCO_new:/datasets/coco -B /datasets/OVIS:/datasets/ovis"
docker=" /home/jaehyunkang/simages/rvos_miran.simg "
exec_file=" python /home/jaehyunkang/meviplus/train_net_lmpm.py "
machine_cfg=" --num-gpus ${num_gpus} --num-machines ${num_nodes} --machine-rank ${node_idx}"
config_file=" --config-file /home/jaehyunkang/meviplus/configs/lmpm_SWIN_bs8.yaml "
out_dir=" OUTPUT_DIR /home/jaehyunkang/meviplus/output/reproduce "
dataloader_cpu=" DATALOADER.NUM_WORKERS 1"
dist_url="--dist-url ${DIST_URL}" 

ml purge
ml load singularity
singularity exec --nv $dataset $mnt $docker $exec_file --resume $config_file $machine_cfg $dist_url $out_dir $dataloader_cpu
