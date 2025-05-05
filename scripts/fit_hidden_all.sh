datasets=("mnist" "cifar100" "food101" "landmarks ""okvqa")

for dataset in "${datasets[@]}"; do
    echo "XXX Fitting on $dataset XXX"
    python main.py fit_hidden_states --datasets "$dataset"
done

echo "Fitting OOD"
python main.py fit_hidden_states --datasets cifar100 --datasets landmarks --datasets food101 --datasets okvqa