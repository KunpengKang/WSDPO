#!/bin/bash


# Initialize our own variables
input=""
num_splits=""
model_folder=""
model_family="" # determines inference template
sample_strategy="" # determine how to sample models

# Parse the command line arguments
while getopts ":i:d:n:m:g:s:f:t:" opt; do
  case $opt in
    i) input="$OPTARG"
    ;;
    n) num_splits="$OPTARG"
    ;;
    m) model_folder="$OPTARG"
    ;;
    g) IFS=',' read -r -a gpus <<< "$OPTARG" 
    ;;
    s) train_step="$OPTARG"
    ;;
    f) model_family="$OPTARG"
    ;;
    t) sample_strategy="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# Now you can use these variables in your scripts
# For example:
echo "Input file: $input"
echo "Number of splits: $num_splits"
echo "Model directory: $model_folder"
echo "Checkpoint step: $train_step"
echo "Model Family: $model_family"
echo "Sample Strategy: $sample_strategy"
if [ -n "$gpus" ] 
then
  echo "Spicified Devices: ${gpus[@]}" 
else
  gpus=$(seq 0 $(($num_splits-1)))
  echo "Default Devices: ${gpus[@]}"
fi
if [ -n "$train_step" ]
then
  echo "Checkpoint step: $train_step"
  saving_name=${model_folder}-checkpoint-${train_step}
else
  echo "No checkpoint step specified, use final meta file."
  saving_name=${model_folder}
fi

# cd .. # go back to the root directory
pwd

# split_data.sh
python split_data.py --input ../data/${input}.json --output_dir ../data/splits --num_splits $num_splits

# run_inference.sh
index=0
for i in ${gpus[@]}
do
  
  if [ -n "$train_step" ] # specified which checkpoint to infer
  then
    CUDA_VISIBLE_DEVICES=$i python -u DPO_Sample.py --dataset_path ../data/splits/${input}_part${index}.json  --sampling_output ../result/${saving_name}_sampling_${sample_strategy}_${input}_part${index}.json --llm_path ../${model_folder}/checkpoint-${train_step} --model_family ${model_family} --sample_strategy ${sample_strategy} &
  else # use final model by default
    CUDA_VISIBLE_DEVICES=$i python -u DPO_Sample.py --dataset_path ../data/splits/${input}_part${index}.json  --sampling_output ../result/${saving_name}_sampling_${sample_strategy}_${input}_part${index}.json --llm_path ../${model_folder} --model_family ${model_family} --sample_strategy ${sample_strategy} &
  fi

  index=$((index+1)) 
done
wait

# merge_outputs.sh
python merge_outputs.py --input_dir ../result --output ../result/${saving_name}_sampling_${sample_strategy}_${input}.json --prefix ${saving_name}_sampling_${sample_strategy}_${input} --num_splits $num_splits --type sample

# clear intermediate files
rm -rf ../data/splits
