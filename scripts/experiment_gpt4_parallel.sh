# This script is meant to run every dataset in parallel to make gpt4 generation easier to run. 
# Adjust the scripts/experiment_gpt4.sh script to run the gpt4 generation for each dataset

datasets=(
    "mnist"
    "cifar100"
    "food101"
    "landmarks"
)

models=("gpt-4o-mini" "gpt-4o")

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Calling GPT4 Training script for $dataset with $model"
        # Run the gpt4 generation script in the background
        bash scripts/experiment_gpt4.sh "$dataset" "$model"
    done
done