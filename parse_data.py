import argparse
import os
import re
from datasets import load_dataset

from ocp import config_list, config2task, config2hier, split_list, split2name


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', type=str, default='data')
    parser.add_argument('-f', '--format', type=str, choices=['linear', 'nn', 'case'], required=True)
    args = parser.parse_args()
    return args


def get_texts(config, dataset, case=False):
    if 'ecthr' in config:
        if not case:
            texts = [' '.join(text) for text in dataset['text']]
        else:
            texts = [' [CASE] '.join(text) for text in dataset['text']]
        return [' '.join(text.split()) for text in texts]
    elif config == 'scotus' and case:
        texts = [' [CASE] '.join(re.split('\n{2,}', text)) for text in dataset['text']]
        # Huggingface tokenizer ignores newline and tab,
        # so it's okay to replace them with a space here.
        for i in range(len(texts)):
            texts[i] = texts[i].replace('\n', ' ')
            texts[i] = texts[i].replace('\r', ' ')
            texts[i] = texts[i].replace('\t', ' ')
        return texts
    elif config == 'case_hold':
        return [contexts[0] + ' [SEP] '.join(holdings)
                for contexts, holdings in zip(dataset['contexts'], dataset['endings'])]
    else:
        return [' '.join(text.split()) for text in dataset['text']]


def get_labels(config, dataset, task):
    if task == 'multi_class':
        return list(map(str, dataset['label']))
    else:
        if config == 'eurlex':
            return [' '.join(map(str, [l for l in label if l < 100])) for label in dataset['labels']]
        else:
            return [' '.join(map(str, label)) for label in dataset['labels']]


def save_data(data_path, data):
    with open(data_path, 'w') as f:
        for text, label in zip(data['text'], data['labels']):
            assert '\n' not in label+text
            assert '\r' not in label+text
            assert '\t' not in label+text
            formatted_instance = '\t'.join([label, text])
            f.write(f'{formatted_instance}\n')


def main():
    # args
    args = get_args()
    data_path = f'{args.data_path}_{args.format}'
    os.makedirs(data_path, exist_ok=True)

    # parse
    for config in config_list:
        if args.format == 'case' and not config2hier[config]:
            continue
        config_path = os.path.join(data_path, config)
        os.makedirs(config_path, exist_ok=True)
        processed_data = {}
        for split in split_list:
            dataset = load_dataset('lex_glue', config, split=split)
            texts = get_texts(config, dataset, case=args.format == 'case')
            labels = get_labels(config, dataset, config2task[config])
            assert len(texts) == len(labels)
            print(f'{config} ({split}): num_instance = {len(texts)}')
            processed_data[split] = {'text': texts, 'labels': labels}
        # format
        if args.format == 'linear':
            # train
            train_path = os.path.join(config_path, split2name['train'])
            train_data = {
                'text': processed_data['train']['text'] + processed_data['validation']['text'],
                'labels': processed_data['train']['labels'] + processed_data['validation']['labels']
            }
            save_data(train_path, train_data)
            # test
            test_path = os.path.join(config_path, split2name['test'])
            test_data = processed_data['test']
            save_data(test_path, test_data)
        elif args.format == 'nn' or args.format == 'case':
            # train/validation/test
            for split in processed_data:
                split_path = os.path.join(config_path, split2name[split])
                save_data(split_path, processed_data[split])


if __name__ == '__main__':
    main()
