dataset_name=$1
echo "Setting up dataset: $dataset_name. If you have not set up the DataCreator properly, this will fail."
python main.py validate_classes --dataset_names $dataset_name

echo "Completed class validation. Now generating QA pairs, validating and deduplicating"
python main.py full_qa_pipeline --dataset_name $dataset_name

echo "Completed QA set up. Now setting up the dataset"
python main.py setup_dataset --dataset_names $dataset_name
