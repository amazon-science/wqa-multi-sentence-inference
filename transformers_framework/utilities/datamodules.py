from pytorch_lightning.trainer.states import TrainerFn


STAGES_TO_NAMES = {
    TrainerFn.FITTING: 'train',
    TrainerFn.VALIDATING: 'valid',
    TrainerFn.TESTING: 'test',
    TrainerFn.PREDICTING: 'predict',
}
