dataset=$1
vlm=$2
if [ -z "$dataset" ]; then
  echo "Usage: $0 <dataset> <vlm>"
  exit 1
fi
if [ -z "$vlm" ]; then
  echo "Usage: $0 <dataset> <vlm>"
  exit 1
fi

python main.py grounding_experiment --dataset_name $dataset --model $vlm --stage all