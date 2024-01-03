#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
echo [SCRIPT_DIR] $SCRIPT_DIR

export num_gpus=2
export num_nodes=4
export MASTER_PORT=12359

DURATION='2-0:0:0'
slurm_setting="-p base -q base_qos"
slurm_setting2="-p big -q big_qos"

curr_date="$(date +'%m/%d-%H:%M:%S')"
JNAME="lmpm-g${num_gpus}n${num_nodes}-${curr_date}" 
echo "[MASTER] JNAME: ${JNAME} | DURATION: ${DURATION} | num_gpus: ${num_gpus}"

### Run master node
export DETECTRON2_DATASETS=/datasets
export node_idx=0
sbatch --gres=gpu:$num_gpus --cpus-per-task=8 $slurm_setting -J $JNAME --time=$DURATION ./${SCRIPT_DIR}/run_gen_master_node.sh
sleep 10

### get master node address
master_node_jobID=$(squeue --name=${JNAME})
master_node_jobID=$(echo $master_node_jobID | awk '{print $9}')
echo "[NON-MASTER] $SCRIPT_DIR | MASTER-JID: ${master_node_jobID} | num_gpus: ${num_gpus} | JNAME: ${JNAME} | DURATION: ${DURATION}"

line=$(scontrol show job $master_node_jobID | grep '  NodeList=' | awk '{print $1}')
node_list=${line:9}
echo "[node_list] ${node_list}"

while [[ ${node_list} == '(null)' ]]
do
  sleep 180
  line=$(scontrol show job $master_node_jobID | grep '  NodeList=' | awk '{print $1}')
  node_list=${line:9}
  echo "[master_node_jid] ${master_node_jobID} [node_list] ${node_list}"
done

echo 'Run with master node: ' $node_list
export MASTER_NODE=$node_list
echo 'MASTER_NODE' $MASTER_NODE

### Run all non-master nodes
for i in $(seq 1 $(($num_nodes-1)))
do
  echo "node $i"
  export node_idx=$i
  sbatch --gres=gpu:$num_gpus --cpus-per-task=8 $slurm_setting2 -J $JNAME --time=$DURATION ./${SCRIPT_DIR}/run_gen_slave_node.sh
done

