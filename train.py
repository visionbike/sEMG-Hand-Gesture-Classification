from copy import deepcopy
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, NeptuneLogger
from config_parser import *
from data import *
from agent import *

if __name__ == '__main__':
    # load arguments
    parser = YamlConfigParser(description='Configuration for training')
    args = parser.parse()

    # set seed
    pl.seed_everything(args.ExpConfig.seed)

    # load data module
    data_kwargs = deepcopy(args.DataConfig)
    data_ = get_data(**data_kwargs)
    data_.setup(stage=args.ExpConfig.phase)
    data_train = data_.train_dataloader()
    data_val = data_.val_dataloader()
    data_test = data_.test_dataloader()

    # setup model
    network_kwargs = deepcopy(args.NetworkConfig)
    model = Trainer(network_kwargs=network_kwargs,
                    optimizer_kwargs=args.OptimConfig,
                    criterion_kwargs=args.LossConfig,
                    scheduler_kwargs=args.SchedulerConfig,
                    metric_kwargs=args.MetricConfig)

    # model summary callback
    model_summary_callback = ModelSummary()
    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ExpConfig.exp_dir + '/checkpoints',
        filename='net_{epoch:02d}',
        monitor='loss/val',
        save_top_k=args.ExpConfig.epochs,
        mode='min'
    )
    # learning rate callback
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

    # setup loggers
    tb_logger = TensorBoardLogger(save_dir=args.ExpConfig.exp_dir + '/tb_logs/')
    neptune_logger = NeptuneLogger(
        project='asrlab/semg-gesture-classification',
        api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NGFlZTgyMy1jMjQ2LTQ0YTEtYTk2ZC0zNTg4MzJmMmU0Y2YifQ==',
        log_model_checkpoints=False,
        tags=[
            args.DataConfig.name,
            args.NetworkConfig.name,
            args.NetworkConfig.attention.get('name'),
        ]
    )
    neptune_logger.log_hyperparams(args)

    # prepare training
    trainer = pl.Trainer(accelerator='gpu' if args.ExpConfig.use_cuda else 'cpu',
                         auto_select_gpus=True,
                         callbacks=[checkpoint_callback, model_summary_callback, lr_monitor_callback],
                         logger=[tb_logger, neptune_logger],
                         max_epochs=args.ExpConfig.epochs,
                         benchmark=True,
                         deterministic=False)

    # train model
    trainer.fit(model, train_dataloaders=data_train, val_dataloaders=data_val)

    # save best model checkpoint after training
    shutil.copyfile(checkpoint_callback.best_model_path,
                    args.ExpConfig.exp_dir + '/net_best.ckpt')

    # test
    trainer.test(ckpt_path=args.ExpConfig.exp_dir + '/net_best.ckpt', dataloaders=data_test)