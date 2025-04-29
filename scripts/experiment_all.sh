vlms=(
  "llava-v1.6-vicuna-7b-hf"
  "llava-v1.6-vicuna-13b-hf"
  "llava-v1.6-mistral-7b-hf"
  "instructblip-vicuna-7b"
  "instructblip-vicuna-13b"
)

for vlm in "${vlms[@]}"; do
    bash scripts/experiment_all_datasets.sh $vlm
done