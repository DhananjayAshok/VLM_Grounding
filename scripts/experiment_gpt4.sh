dataset_name=$1
model="gpt4-0-mini" #or "gpt4-0"
if [ -z "$dataset_name" ]; then
  echo "Usage: $0 <dataset_name>"
  exit 1
fi

# These commands must be run consecutively after the previous one completes running in OpenAI Batch generation. 
# Run the same command again to check if it is done running. 

python main.py grounding_experiment --dataset_name $dataset_name --model $model --stage identification
#python main.py grounding_experiment --dataset_name $dataset_name --model $model --stage full_information
#python main.py grounding_experiment --dataset_name $dataset_name --model $model --stage image_reference
#python main.py grounding_experiment --dataset_name $dataset_name --model $model --stage trivial
#python main.py grounding_experiment --dataset_name $dataset_name --model $model --stage evaluation