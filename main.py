from base.constructor import (
    load_metadata,
    check_wandb,
    prepare_run
)
from data_loader.data_generator import DataGenerator
from models.model_builder import ModelBuilder
from trainers.trainer_maker import TrainerMaker


@load_metadata
@check_wandb
@prepare_run
def run(config):
    # Load data
    data = DataGenerator(config)

    # Build model
    model = ModelBuilder(config).model

    # Make Trainer
    trainer = TrainerMaker(config, model, data).trainer

    # Run
    if config.mode == 'train':
        trainer.train()
    else:
        if config.analysis_method:
            trainer.analysis()
        else:
            trainer.test()


if __name__ == '__main__':
    run()
