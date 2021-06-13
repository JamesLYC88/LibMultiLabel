import argparse
import glob
import itertools
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.utilities.parsing import AttributeDict
from ray import tune

from libmultilabel import data_utils
from libmultilabel.callbacks import EarlyStoppingWithCheckpoint
from libmultilabel.model import Model
from libmultilabel.utils import dump_log, init_device, set_seed

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')


class Trainable(tune.Trainable):
    def setup(self, config, data):
        self.config = AttributeDict(config)
        self.datasets = data['datasets']
        self.word_dict = data['word_dict']
        self.classes = data['classes']

    def step(self):
        self.config.run_name = '{}_{}_{}_{}'.format(
            self.config.data_name,
            Path(
                self.config.config).stem if self.config.config else self.config.model_name,
            datetime.now().strftime('%Y%m%d%H%M%S'),
            self.trial_id
        )
        logging.info(f'Run name: {self.config.run_name}')

        checkpoint_dir = os.path.join(
            self.config.result_dir, self.config.run_name)
        best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.ckpt')
        last_checkpoint_path = os.path.join(
            checkpoint_dir, 'lastest_model.ckpt')
        early_stop_callback = EarlyStoppingWithCheckpoint(
            best_checkpoint_path=best_checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            monitor=self.config.val_metric,
            patience=self.config.patience,
            verbose=not self.config.silent,
            mode='max'
        )

        trainer = pl.Trainer(checkpoint_callback=False, logger=False,
                             num_sanity_val_steps=0,
                             gpus=0 if config.cpu else 1,
                             progress_bar_refresh_rate=0 if self.config.silent else 1,
                             max_epochs=self.config.epochs,
                             callbacks=[early_stop_callback])

        model = Model(self.config, self.word_dict, self.classes)

        train_loader = data_utils.get_dataset_loader(
            model.config, self.datasets['train'], model.word_dict, model.classes,
            shuffle=model.config.shuffle, train=True)
        val_loader = data_utils.get_dataset_loader(
            model.config, self.datasets['val'], model.word_dict, model.classes, train=False)

        trainer.fit(model, train_loader, val_loader)

        test_val_results = dict()

        # run and dump test result
        if 'test' in self.datasets:
            test_loader = data_utils.get_dataset_loader(
                model.config, self.datasets['val'], model.word_dict, model.classes, train=False)
            test_metric_dict = trainer.test(test_dataloaders=test_loader,
                                            ckpt_path=best_checkpoint_path)[0]
            dump_log(config=self.config, metrics=test_metric_dict, split='test')
            for k, v in test_metric_dict.items():
                test_val_results[f'test_{k}'] = v

        # return best val result
        val_metric_dict = trainer.test(test_dataloaders=val_loader,
                                       ckpt_path=best_checkpoint_path)[0]
        for k, v in val_metric_dict.items():
            test_val_results[f'val_{k}'] = v

        # remove model_best.pt and model_last.pt
        for model_path in glob.glob(os.path.join(self.config.result_dir, self.config.run_name, '*.pt')):
            logging.info(f'Removing {model_path} ...')
            os.remove(model_path)
        return test_val_results


def init_model_config(config_path):
    with open(config_path) as fp:
        args = yaml.load(fp, Loader=yaml.SafeLoader)

    # create directories that hold the shared data
    os.makedirs(args['result_dir'], exist_ok=True)
    if args['embed_cache_dir']:
        os.makedirs(args['embed_cache_dir'], exist_ok=True)

    # set relative path to absolute path (_path, _file, _dir)
    for k, v in args.items():
        if isinstance(v, str) and os.path.exists(v):
            args[k] = os.path.abspath(v)

    model_config = ArgDict(args)
    set_seed(seed=model_config.seed)
    model_config.device = init_device(model_config.cpu)
    return model_config


def init_search_params_spaces(model_config):
    """Initialize the sample space defined in ray tune.
    See the random distributions API listed here: https://docs.ray.io/en/master/tune/api_docs/search_space.html#random-distributions-api
    """
    search_spaces = ['choice', 'grid_search', 'uniform', 'quniform', 'loguniform',
                     'qloguniform', 'randn', 'qrandn', 'randint', 'qrandint']
    for key, value in model_config.items():
        if isinstance(value, list) and len(value) >= 2 and value[0] in search_spaces:
            model_config[key] = getattr(tune, value[0])(*value[1:])
    return model_config


def init_search_algorithm(search_alg, metric=None, mode=None):
    """Specify a search algorithm and you must pip install it first.
    See more details here: https://docs.ray.io/en/master/tune/api_docs/suggestion.html
    """
    if search_alg == 'optuna':
        assert metric and mode, "Metric and mode cannot be None for optuna."
        from ray.tune.suggest.optuna import OptunaSearch
        return OptunaSearch(metric=metric, mode=mode)
    elif search_alg == 'bayesopt':
        assert metric and mode, "Metric and mode cannot be None for bayesian optimization."
        from ray.tune.suggest.bayesopt import BayesOptSearch
        return BayesOptSearch(metric=metric, mode=mode)
    logging.info(f'{search_alg} search is found, run BasicVariantGenerator().')


def load_static_data(config):
    datasets = data_utils.load_datasets(config)
    return {
        "datasets": datasets,
        "word_dict": data_utils.load_or_build_text_dict(config, datasets['train']),
        "classes": data_utils.load_or_build_label(config, datasets)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', help='Path to configuration file (default: %(default)s). Please specify a config with all arguments in LibMultiLabel/main.py::get_config.')
    parser.add_argument('--cpu_count', type=int, default=4,
                        help='Number of CPU per trial (default: %(default)s)')
    parser.add_argument('--gpu_count', type=int, default=1,
                        help='Number of GPU per trial (default: %(default)s)')
    parser.add_argument('--local_dir', default=os.getcwd(),
                        help='Directory to save training results of tune (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of running trials. If the search space is `grid_search`, the same grid will be repeated `num_samples` times. (default: %(default)s)')
    parser.add_argument('--mode', default='max', choices=[
                        'min', 'max'], help='Determines whether objective is minimizing or maximizing the metric attribute. (default: %(default)s)')
    parser.add_argument('--search_alg', default=None, choices=[
                        'basic_variant', 'bayesopt', 'optuna'], help='Search algorithms (default: %(default)s)')
    args = parser.parse_args()

    """Other args in the model config are viewed as resolved values that are ignored from tune.
    https://github.com/ray-project/ray/blob/34d3d9294c50aea4005b7367404f6a5d9e0c2698/python/ray/tune/suggest/variant_generator.py#L333
    """
    model_config = init_model_config(args.config)
    model_config = init_search_params_spaces(model_config)
    search_alg = args.search_alg if args.search_alg else model_config.search_alg
    data = load_static_data(model_config)

    """Run tune analysis.
    If no search algorithm is specified, the default search algorighm is BasicVariantGenerator.
    https://docs.ray.io/en/master/tune/api_docs/suggestion.html#tune-basicvariant
    """
    all_monitor_metrics = [f'{split}_{metric}' for split, metric in itertools.product(
        ['val', 'test'], model_config.monitor_metrics)]
    tune.run(
        tune.with_parameters(Trainable, data=data),
        # run one step "libmultilabel.model.train"
        stop={"training_iteration": 1},
        search_alg=init_search_algorithm(
            search_alg, metric=model_config.val_metric, mode=args.mode),
        local_dir=args.local_dir,
        metric=f'val_{model_config.val_metric}',
        mode=args.mode,
        num_samples=args.num_samples,
        resources_per_trial={
            'cpu': args.cpu_count, 'gpu': args.gpu_count},
        progress_reporter=tune.CLIReporter(metric_columns=all_monitor_metrics),
        config=model_config)


# calculate wall time.
wall_time_start = time.time()
main()
logging.info(f"\nWall time: {time.time()-wall_time_start}")
