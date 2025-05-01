vlms=(
  "llava-v1.6-vicuna-7b-hf"
  "llava-v1.6-vicuna-13b-hf"
  "llava-v1.6-mistral-7b-hf"
  "instructblip-vicuna-7b"
  "instructblip-vicuna-13b"
)

datasets=(
    "mnist"
    "cifar100"
    "cifar100_mcq"
    "food101"
    "food101_mcq"
    "landmarks"
    "landmarks_mcq"
)

echo "RUNNING SCRIPT WITH DATASETS ${datasets[@]} AND VLMS ${vlms[@]}"
for dataset in "${datasets[@]}"; do
    for vlm in "${vlms[@]}"; do
        echo "XXXXX Running experiment for $dataset with $vlm XXXXX"
        if [[ $vlm == "llava-v1.6-vicuna-7b-hf" ]]; then
            python main.py grounding_experiment --dataset_name $dataset --model $vlm --variant hidden_state_vocab_projection --stage all
            python main.py visualize_vocab_projection --dataset $dataset
        else
            python main.py grounding_experiment --dataset_name $dataset --model $vlm --stage all
        fi
    done
done