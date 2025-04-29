vlm=$1
if [ -z "$vlm" ]; then
  echo "Usage: $0 <vlm_name>"
  exit 1
fi

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

# gpt-4o-mini and gpt-4o need to be done separately, stage by stage, due to the way batching works

for dataset_name in "${datasets[@]}"; do
  echo "Running experiment for $dataset_name with $vlm"
    if [[ $vlm == "llava-v1.6-vicuna-7b-hf" ]]; then
        echo "Running hidden state tracking and vocabulary projection too"
        python main.py grounding_experiment --dataset_name $dataset_name --model $vlm --variant hidden_state_vocab_projection
    else
        python main.py grounding_experiment --dataset_name $dataset_name --model $vlm
    fi
done