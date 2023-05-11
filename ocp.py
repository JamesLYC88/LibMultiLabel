# parse_data.py

config_list = [
    'ecthr_a',
    'ecthr_b',
    'scotus',
    'eurlex',
    'ledgar',
    'unfair_tos',
]

config2task = {
    'ecthr_a': 'multi_label',
    'ecthr_b': 'multi_label',
    'scotus': 'multi_class',
    'eurlex': 'multi_label',
    'ledgar': 'multi_class',
    'unfair_tos': 'multi_label',
}

config2hier = {
    'ecthr_a': True,
    'ecthr_b': True,
    'scotus': True,
    'eurlex': False,
    'ledgar': False,
    'unfair_tos': False,
}

split_list = [
    'train',
    'validation',
    'test',
]

split2name = {
    'train': 'train.txt',
    'validation': 'valid.txt',
    'test': 'test.txt',
}

replace_dict = {
    'ecthr_a': 'ECtHR (A)',
    'ecthr_b': 'ECtHR (B)',
    'scotus': 'SCOTUS',
    'eurlex': 'EUR-LEX',
    'ledgar': 'LEDGAR',
    'unfair_tos': 'UNFAIR-ToS'
}

# result.py / time.py

model_list = [
#    'l2svm_1vsrest',
#    'l2svm_thresholding',
#    'l2svm_cost_sensitive',
#    'l2svm_cost_sensitive_micro',
    'bert_default',
    'bert_tuned',
    'bert_reproduce',
#    'bert_lwan_rand',
]

metric_list = [
    'Micro-F1',
    'Macro-F1',
]

model2method = {
    'l2svm_1vsrest': 'one-vs-rest',
    'l2svm_thresholding': 'thresholding',
    'l2svm_cost_sensitive': 'cost-sensitive',
    'l2svm_cost_sensitive_micro': 'cost-sensitive-micro',
    'bert_default': 'BERT (default\_mean)',
    'bert_tuned': 'BERT (tuned\_mean)',
    'bert_reproduce': 'BERT (reproduce\_mean)',
    'bert_lwan_rand': 'BERT-LWAN',
}

# seed.py

seed_list = [
    1,
    2,
    3,
    4,
    5,
]

# parameter.py

param_algo_list = [
    'tune',
    'best',
]

model2sparams = {
    'BERT': ['max_seq_length', 'learning_rate', 'dropout'],
}

param_algo2suffix = {
    'tune': '_tune.yml',
    'best': '_tuned.yml',
}
