from copy import deepcopy
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.neptune import NeptuneLogger
from config_parser import *
from data import *
from agent import *

if __name__ == '__main__':
    # load arguments
    parser = YamlConfigParser(description='Configuration for training')
    args = parser.parse()

    # set seed
    pl.seed_everything(args.ExpConfig.seed, workers=True)

    # load data module
    data_kwargs = deepcopy(args.DataConfig)
    data_ = get_data(seed=args.ExpConfig.seed, **data_kwargs)
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

    # setup callbacks
    callbacks = []
    # model summary callback
    # callbacks += [ModelSummary(max_depth=-1)]
    # checkpoint callback
    callbacks += [ModelCheckpoint(
        dirpath=args.ExpConfig.exp_dir + '/checkpoints',
        filename='net_{epoch:02d}',
        monitor='val/loss',
        mode='min',
        save_last=True,
        save_on_train_epoch_end=False
    )]
    # learning rate callback
    if args.ExpConfig.logging:
        callbacks += [LearningRateMonitor(logging_interval='epoch')]

    # setup loggers
    if args.ExpConfig.logging:
        # tb_logger = TensorBoardLogger(save_dir=args.ExpConfig.exp_dir + '/tb_logs/')
        neptune_logger = NeptuneLogger(
            project='asrlab106/sEMG-Gesture-Classification',
            api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NGFlZTgyMy1jMjQ2LTQ0YTEtYTk2ZC0zNTg4MzJmMmU0Y2YifQ==',
            log_model_checkpoints=False,
            tags=[
                args.DataConfig.name,
                args.NetworkConfig.name,
                args.NetworkConfig.attention.get('name'),
            ],
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=False,
            name=args.ExpConfig.name
        )
        neptune_logger.log_hyperparams(args)

    # prepare training
    trainer = pl.Trainer(accelerator='gpu' if args.ExpConfig.use_cuda else 'cpu',
                         devices=args.ExpConfig.gpu_ids,
                         auto_select_gpus=True,
                         callbacks=callbacks,
                         logger=(neptune_logger if args.ExpConfig.logging else args.ExpConfig.logging),
                         max_epochs=args.ExpConfig.epochs,
                         log_every_n_steps=1,
                         benchmark=False,
                         deterministic=False,
                         num_sanity_val_steps=0)

    # train model
    trainer.fit(model, train_dataloaders=data_train, val_dataloaders=data_val)

    # save best model checkpoint after training
    shutil.copyfile(callbacks[0].best_model_path, args.ExpConfig.exp_dir + '/net_best.ckpt')
    shutil.copyfile(callbacks[0].last_model_path, args.ExpConfig.exp_dir + '/net_last.ckpt')

    # test
    # trainer.test(ckpt_path=args.ExpConfig.exp_dir + '/net_best.ckpt', dataloaders=data_test)
    # trainer.validate(model, dataloaders=data_val)
    # trainer.validate(model, dataloaders=data_test, ckpt_path='best')
    # trainer.validate(model, dataloaders=data_test, ckpt_path='last')
    trainer.test(model, ckpt_path='best', dataloaders=data_test)
    # trainer.test(model, ckpt_path='last', dataloaders=data_test)
