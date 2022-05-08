#!/bin/bash

# fetches the directory where the shell file resides
file_path=$(realpath "$0")
dir_path=$(dirname "$file_path")


#setting up the defaults
LANG='hi,mr,te,ta,en,gu,bn,kn'
GPUS=1
MODEL_DIR=$dir_path   #optional
PYTHON=$(which python)  #change required
SCRATCH_DIR=/tmp/XAlign
mkdir -p $SCRATCH_DIR
CHECKPOINT_PATH=$SCRATCH_DIR/checkpoint   #change required


BATCH_SIZE=2
TEST_BATCH_SIZE=4
EPOCHS=30
LR=1e-3

# seq length related configuration
SRC_MAX_SEQ_LENGTH=250
TGT_MAX_SEQ_LENGTH=200
#transformer model to use
MODEL_NAME='google/mt5-small'
PRETRAINED=1

ONLINE_SYNC=0  #control w&b online syncronization, 0 means inactive

DATASET_DIR=$SCRATCH_DIR/datasets

printf "\n\n"
#dynamically set above default values through shell arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus=*)
      GPUS="${1#*=}"
      ;;
    --checkpoint_path=*)
      CHECKPOINT_PATH="${1#*=}"/checkpoint
      ;;
    --model_dir=*)
      MODEL_DIR="${1#*=}"
      ;;
    --python=*)
      PYTHON="${1#*=}"
      ;;
    --batch_size=*)
      BATCH_SIZE="${1#*=}"
      TEST_BATCH_SIZE=$BATCH_SIZE
      ;;
    --test_batch_size=*)
      TEST_BATCH_SIZE="${1#*=}"
      ;;
    --epochs=*)
      EPOCHS="${1#*=}"
      ;;
    --lr=*)
      LR="${1#*=}"
      ;;
    --src_max_seq_len=*)
      SRC_MAX_SEQ_LENGTH="${1#*=}"
      ;;
    --tgt_max_seq_len=*)
      TGT_MAX_SEQ_LENGTH="${1#*=}"
      ;;
    --model_name=*)
      MODEL_NAME="${1#*=}"
      ;;
    --pretrained=*)
      PRETRAINED="${1#*=}"
      ;;
    --online=*)
      ONLINE_SYNC="${1#*=}"
      ;;
    --lang=*)
      LANG="${1#*=}"
      ;;
    --dataset_dir=*)
      DATASET_DIR="${1#*=}"
      ;;  
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument. please check argument $1 *\n"
      printf "***************************\n"
      exit 1
  esac
  shift
done

# api key weight & Biases (uncomment when using online logger and replace <value> with API)
# export WANDB_API_KEY=<your_key>  


#########################################################
#print argument captures in shell script
echo "<< ----------- Experiment configurations -------------"
echo "GPUS : $GPUS"
echo "CHECKPOINT_PATH : $CHECKPOINT_PATH"
echo "MODEL_DIR : $MODEL_DIR"
echo "PYTHON : $PYTHON"
echo "BATCH_SIZE : $BATCH_SIZE"
echo "TEST_BATCH_SIZE : $TEST_BATCH_SIZE"
echo "EPOCHS : $EPOCHS"
echo "LR : $LR"
echo "SRC_MAX_SEQ_LENGTH : $SRC_MAX_SEQ_LENGTH"
echo "TGT_MAX_SEQ_LENGTH : $TGT_MAX_SEQ_LENGTH"
echo "MODEL_NAME : $MODEL_NAME"
echo "PRETRAINED : $PRETRAINED"
echo "ONLINE_SYNC : $ONLINE_SYNC"
echo "DATASET_DIR : $DATASET_DIR"
echo "LANGUAGE : $LANG"
echo "--------------------------------------------------- >>"
printf "\n"

# execute training
$PYTHON $MODEL_DIR/main.py --dataset_path $DATASET_DIR --epochs $EPOCHS --gpus $GPUS --batch_size $BATCH_SIZE --eval_batch_size $TEST_BATCH_SIZE --src_max_seq_len $SRC_MAX_SEQ_LENGTH --tgt_max_seq_len $TGT_MAX_SEQ_LENGTH --checkpoint_path $CHECKPOINT_PATH --learning_rate $LR --model_name $MODEL_NAME --online_mode $ONLINE_SYNC --use_pretrained $PRETRAINED --lang $LANG --verbose