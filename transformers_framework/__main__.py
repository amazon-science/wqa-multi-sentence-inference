import os
from argparse import ArgumentParser

import datasets
import pytorch_lightning as pl
import transformers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_warn
from transformers_lightning.callbacks import RichProgressBar, TransformersModelCheckpointCallback
from transformers_lightning.defaults import DefaultConfig
from transformers_lightning.utils import get_classes_from_module

from transformers_framework import models
from transformers_framework.datamodules import TransformersDataModule
from transformers_framework.utilities import ExtendedNamespace, write_dict_to_disk


def main(hyperparameters):

    # too much complains of the tokenizers
    transformers.logging.set_verbosity_error()

    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    datasets.config.IN_MEMORY_MAX_SIZE = 1024 * 1024 * 1024 * hyperparameters.datasets_max_in_memory  # in GB

    # set the random seed
    seed_everything(seed=hyperparameters.seed, workers=True)

    # instantiate PL model
    pl_model_class = all_models[hyperparameters.model]
    model = pl_model_class(hyperparameters)

    # default tensorboard logger
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(hyperparameters.output_dir, hyperparameters.tensorboard_dir),
        name=hyperparameters.name,
    )
    loggers = [tb_logger]

    # save pre-trained models to
    save_transformers_callback = TransformersModelCheckpointCallback(hyperparameters)

    # and log learning rate
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    # and normal checkpoints with
    checkpoints_dir = os.path.join(hyperparameters.output_dir, hyperparameters.checkpoints_dir, hyperparameters.name)
    checkpoint_callback_args = dict(verbose=True, dirpath=checkpoints_dir)

    if hyperparameters.monitor is not None:
        checkpoint_callback_args = dict(
            **checkpoint_callback_args,
            monitor=hyperparameters.monitor,
            save_last=True,
            mode=hyperparameters.monitor_direction,
            save_top_k=1,
        )
    checkpoint_callback = ModelCheckpoint(**checkpoint_callback_args)

    # rich progress bar
    rich_progress_bar = RichProgressBar(leave=True)

    # modelsummary callback
    model_summary = RichModelSummary(max_depth=2)

    # all callbacks
    callbacks = [
        save_transformers_callback,
        lr_monitor_callback,
        checkpoint_callback,
        rich_progress_bar,
        model_summary,
    ]

    # early stopping if defined
    if hyperparameters.early_stopping:
        if hyperparameters.monitor is None:
            raise ValueError("cannot use early_stopping without a monitored variable")

        early_stopping_callback = EarlyStopping(
            monitor=hyperparameters.monitor,
            patience=hyperparameters.patience,
            verbose=True,
            mode=hyperparameters.monitor_direction,
        )
        callbacks.append(early_stopping_callback)

    # disable find unused parameters to improve performance
    kwargs = dict()
    if hyperparameters.strategy in ("dp", "ddp2"):
        rank_zero_warn("This repo is not designed to work with DataParallel. Use strategy `ddp` instead.")

    if hyperparameters.strategy == "ddp":
        kwargs['strategy'] = DDPPlugin(find_unused_parameters=False)

    # instantiate PL trainer
    trainer = pl.Trainer.from_argparse_args(
        hyperparameters,
        default_root_dir=hyperparameters.output_dir,
        logger=loggers,
        callbacks=callbacks,
        profiler='simple',
        **kwargs,
    )

    # DataModules
    datamodule = TransformersDataModule(hyperparameters, trainer, tokenizer=model.tokenizer)

    # Train!
    if datamodule.do_train():
        trainer.fit(model, datamodule=datamodule)

    # Test!
    if datamodule.do_test():
        if datamodule.do_train() and hyperparameters.monitor is not None:
            rank_zero_warn(
                f"Going to test on best ckpt chosen over "
                f"{hyperparameters.monitor}: {checkpoint_callback.best_model_path}"
            )
            trainer.test(datamodule=datamodule, ckpt_path='best')
        else:
            rank_zero_warn("Going to test on last or pretrained ckpt")
            trainer.test(model, datamodule=datamodule)

    if datamodule.do_predict():
        assert hasattr(model, "predict_step") and hasattr(model, "predict_epoch_end"), (
            "To do predictions, the model must implement both `predict_step` and `predict_epoch_end`"
        )

        if trainer._accelerator_connector.is_distributed:
            rank_zero_warn("Predicting on more than 1 GPU may give results in different order, use keys to sort them.")

        predictions = trainer.predict(model, datamodule=datamodule, return_predictions=True)
        predictions = model.predict_epoch_end(predictions)

        basepath = os.path.join(hyperparameters.output_dir, hyperparameters.predictions_dir, hyperparameters.name)
        write_dict_to_disk(predictions, basepath, trainer=trainer)


if __name__ == '__main__':

    # Read config for defaults and eventually override with hyperparameters from command line
    parser = ArgumentParser(add_help=False)

    # model classname
    all_models = get_classes_from_module(models, parent=pl.LightningModule)
    parser.add_argument('--model', type=str, required=True, choices=all_models.keys())

    # experiment name, used both for checkpointing, pre_trained_names, logging and tensorboard
    parser.add_argument('--name', type=str, required=True, help='Name of the model')

    # various options
    parser.add_argument('--seed', type=int, default=1337, help='Set the random seed')
    parser.add_argument('--monitor', type=str, help='Value to monitor for best checkpoint', default=None)
    parser.add_argument(
        '--monitor_direction', type=str, help='Monitor value direction for best', default='max', choices=['min', 'max']
    )
    parser.add_argument('--early_stopping', action="store_true", help="Use early stopping")
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        required=False,
        help="Number of non-improving validations to wait before early stopping"
    )
    parser.add_argument(
        '--find_unused_parameters',
        action="store_true",
        help="Whether to check for unused params at each iteration"
    )
    parser.add_argument(
        '--datasets_max_in_memory', type=int, default=0, help="Datasets max in memory cache (in GB)"
    )

    # I/O folders
    parser.add_argument(
        '--predictions_dir', type=str, default="predictions", required=False, help="Predictions folder"
    )

    DefaultConfig.add_defaults_args(parser)

    # retrieving model with temporary parsered arguments
    tmp_params, extra = parser.parse_known_args()

    # get pl_model_class in advance to know which params it needs
    all_models[tmp_params.model].add_model_specific_args(parser)
    TransformersDataModule.add_datamodule_specific_args(parser)

    # add callback / logger specific parameters
    TransformersModelCheckpointCallback.add_callback_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --devices --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    # get NameSpace of paramters
    hyperparameters = parser.parse_args()
    hyperparameters = ExtendedNamespace.from_namespace(hyperparameters)
    main(hyperparameters)
