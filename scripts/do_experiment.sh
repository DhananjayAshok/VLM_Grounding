dataset=$1
vlm=$2
# vlm options are: llava-v1.6-vicuna-7b-hf, llava-v1.6-vicuna-13b-hf, llava-v1.6-mistral-7b-hf, instructblip-vicuna-7b, instructblip-vicuna-13b, gpt-4o-mini, gpt-4o
if [ -z "$dataset" ]; then
  echo "Usage: $0 <dataset> <vlm>"
  exit 1
fi
if [ -z "$vlm" ]; then
  echo "Usage: $0 <dataset> <vlm>"
  exit 1
fi

python main.py grounding_experiment --dataset_name $dataset --model $vlm --stage all