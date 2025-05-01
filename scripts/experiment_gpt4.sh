dataset_name=$1
model=$2 #or "gpt4-o"
if [ -z "$dataset_name" ]; then
  echo "Usage: $0 <dataset_name> <model_name>"
  exit 1
fi

# if model is not either gpt4-o or gpt4-o-mini, then error out
if [[ $model != "gpt4-o" && $model != "gpt4-o-mini" ]]; then
  echo "Usage: $0 <dataset_name> <model_name>, where model_name is either gpt4-o or gpt4-o-mini, you gave $2"
  exit 1
fi

# These commands must be run consecutively after the previous one completes running in OpenAI Batch generation. 
# Run the same command again to check if it is done running. 

python main.py grounding_experiment --dataset_name $dataset_name --model $model --stage identification
#python main.py grounding_experiment --dataset_name $dataset_name --model $model --stage full_information
#python main.py grounding_experiment --dataset_name $dataset_name --model $model --stage image_reference
#python main.py grounding_experiment --dataset_name $dataset_name --model $model --stage trivial
#python main.py grounding_experiment --dataset_name $dataset_name --model $model --stage evaluation