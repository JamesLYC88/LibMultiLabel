data_list=(ecthr_a ecthr_b scotus eurlex ledgar unfair_tos)
algo_list=(1vsrest thresholding cost_sensitive)

linear_algo_list=(1vsrest thresholding cost_sensitive)
data=$1
algo=$2

if [[ ! " ${data_list[*]} " =~ " ${data} " ]]; then
    echo "Invalid argument! Data ${data} is not in (${data_list[*]})."
    exit
fi

if [[ ! " ${algo_list[*]} " =~ " ${algo} " ]]; then
    echo "Invalid argument! Algorithm ${algo} is not in (${algo_list[*]})."
    exit
fi

if [[ " ${linear_algo_list[*]} " =~ " ${algo} " ]]; then
    multilabel_unlabeled_data_list=(ecthr_a ecthr_b unfair_tos)
    multilabel_labeled_data_list=(eurlex)
    multiclass_labeled_data_list=(scotus ledgar)
    if [[ " ${multilabel_unlabeled_data_list[*]} " =~ " ${data} " ]]; then
        python3 main.py --config config/${data}/l2svm.yml --linear_technique ${algo} --zero
    elif [[ " ${multilabel_labeled_data_list[*]} " =~ " ${data} " ]]; then
        python3 main.py --config config/${data}/l2svm.yml --linear_technique ${algo}
    elif [[ " ${multiclass_labeled_data_list[*]} " =~ " ${data} " ]]; then
        python3 main.py --config config/${data}/l2svm.yml --linear_technique ${algo} --multi_class
    else
        echo "Should never reach here..."
        exit
    fi
fi
