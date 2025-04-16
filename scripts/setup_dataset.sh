dataset_name=$1
echo "Setting up dataset: $dataset_name. If you have not set up the DataCreator properly, this will fail."
CUDA_VISIBLE_DEVICES=0 python main.py validate_classes --dataset_names $dataset_name

echo "Completed class validation. Now generating QA pairs, validating and deduplicating"
CUDA_VISIBLE_DEVICES=0 python main.py full_qa_pipeline --dataset_name $dataset_name

echo "Completed QA set up. Now setting up the dataset"
CUDA_VISIBLE_DEVICES=0 python main.py setup_data --dataset_names $dataset_name
