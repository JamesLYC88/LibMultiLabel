import glob
import json
import yaml
import argparse

# python3 parse_best_params.py -td runs/Wiki10-31K_bigru_lwan_tune/ -rp temp/bigru_example.yml -op example_config/Wiki10-31K/bigru_lwan.yml -vm RP@15

dataset2metrics = {
    'EUR-Lex': ['P@1', 'P@3', 'P@5', 'RP@5', 'nDCG@5', 'Loss'],
    'EUR-Lex-57k': ['P@1', 'P@3', 'P@5', 'RP@5', 'nDCG@5', 'Loss'],
    'EUR-Lex-57k-1V': ['P@1', 'P@3', 'P@5', 'RP@5', 'nDCG@5', 'Loss'],
    'EUR-Lex-57k-3V': ['P@1', 'P@3', 'P@5', 'RP@5', 'nDCG@5', 'Loss'],
    'EUR-Lex-57k-20V': ['P@1', 'P@3', 'P@5', 'RP@5', 'nDCG@5', 'Loss'],
    'MIMIC': ['Macro-F1', 'Micro-F1', 'RP@15', 'nDCG@15', 'Loss'],
    'MIMIC-1V': ['Macro-F1', 'Micro-F1', 'RP@15', 'nDCG@15', 'Loss'],
    'MIMIC-10V': ['Macro-F1', 'Micro-F1', 'RP@15', 'nDCG@15', 'Loss'],
    'MIMIC-20V': ['Macro-F1', 'Micro-F1', 'RP@15', 'nDCG@15', 'Loss'],
    'Wiki10-31K': ['P@1', 'P@3', 'P@5', 'RP@15', 'nDCG@15', 'Loss'],
    'AmazonCat-13K': ['P@1', 'P@3', 'P@5', 'RP@5', 'nDCG@5', 'Loss'],
    'Amazon-670K': ['P@1', 'P@3', 'P@5', 'RP@5', 'nDCG@5', 'Loss'],
}

def get_args():
    parser = argparse.ArgumentParser()
    # path / directory
    parser.add_argument('-td', '--tune_dir', type=str, required=True,
                        help='The directory to the tuning directory')
    parser.add_argument('-rp', '--refer_path', type=str, required=True,
                        help='Path to the referenced template')
    parser.add_argument('-op', '--output_path', type=str, default='params.yml',
                        help='Path to output the best params (default: %(default)s)')
    # eval
    parser.add_argument('-vm', '--val_metric', type=str, required=True,
                        help='The metric to select the best run')
    parser.add_argument('-l', '--last', action='store_true',
                        help='Whether to select the best run by the last checkpoint')
    args = parser.parse_args()
    return args

def main():
    # args
    args = get_args()

    # best by experiment state
    exp_state = glob.glob(f'{args.tune_dir}/experiment_state*')
    assert(len(exp_state) == 1)
    exp_state = exp_state[0]
    with open(exp_state) as f:
        logs = json.load(f)
    best_val = -1
    for checkpoint in logs['checkpoints']:
        log = json.loads(checkpoint)
        log_val = log['metric_analysis'][f'val_{args.val_metric}']
        # choice: ['max', 'min', 'avg', 'last', 'last-5-avg', 'last-10-avg']
        if args.last:
            val = log_val['last']
        else:
            val = log_val['max']
        if val > best_val:
            best_val = val
            best_config = log['config']
            best_dir = log['logdir'].split('/')[-1]

    # best by best_trail
    with open(f'{args.tune_dir}/best_trial/params.yml') as f:
        best_params = yaml.safe_load(f)

    # compare
    assert(sorted(best_config) == sorted(best_params))
    same_result = True
    for k in sorted(best_config):
        # silent is true in best_config but becomes false in best_params
        if k != 'silent' and best_config[k] != best_params[k]:
            same_result = False
    assert(same_result == True)
    # print(f'compare = {same_result}')
    print(best_dir)
    print(best_val)
    params = best_params if same_result else best_config

    # monitor metrics
    dataset = args.tune_dir.rstrip('/').split('/')[-1].split('_')[0]
    assert(dataset in dataset2metrics)
    params['monitor_metrics'] = dataset2metrics[dataset]

    # extend
    for param, value in params.items():
        if type(value) == dict:
            rec = value
    for k, v in rec.items():
        params[k] = v

    # write
    fw = open(args.output_path, 'w')
    with open(args.refer_path) as f:
        for line in f:
            msg = None
            if line.startswith(('\n', '#', 'network_config')):
                msg = line
            else:
                param = line.lstrip().split(':')[0]
                value = str(params[param])
                if value in ['True', 'False']:
                    value = value.lower()
                elif value == 'None':
                    value = 'null'
                if param == 'data_dir':
                    value = '/'.join(value.split('/')[-2:])
                msg = f'{line.rstrip()} {value}\n'
            fw.write(msg)

if __name__ == '__main__':
    main()
