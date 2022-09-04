from ensemble_models import run_ensemble_models
from generate_submission import run_generate_submission
from prepare_dataset import run_prepare_dataset
from train_model import run_train_model

if __name__ == '__main__':
    run_prepare_dataset()
    run_train_model()
    run_generate_submission()
    run_ensemble_models()