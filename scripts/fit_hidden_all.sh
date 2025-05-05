datasets=("mnist" "cifar100" "food101" "landmarks ""okvqa")

for dataset in "${datasets[@]}"; do
    python main.py fit_hidden_states --datasets "$dataset"
done

python main.py fit_hidden_states --datasets cifar100 --datasets landmarks --datasets food101 --datasets okvqa