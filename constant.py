import os
from datetime import datetime

from configuration import set_configuration
set_configuration()

cwd = os.getcwd()

cleaned_train_file_path = os.path.join(
    cwd,
    'cleaned_dataset',
    'train.csv'
)
cleaned_test_file_path = os.path.join(
    cwd,
    'cleaned_dataset',
    'test.csv'
)
scaled_testing_file_path = os.path.join(
    cwd,
    'extra',
    'test.npy'
)
timestamp = f'{datetime.now().strftime("%Y-%m-%d_%H-%M")}'
application_log_file_path = os.path.join(
    cwd,
    'application_log',
    f'logs_{timestamp}.log'
)
train_file_path = os.path.join(
    cwd,
    'dataset',
    'train.csv'
)
test_file_path = os.path.join(
    cwd,
    'dataset',
    'test.csv'
)
sample_submission_file_path = os.path.join(
    cwd,
    'dataset',
    'SampleSubmission.csv'
)
model_configuration_file_path = os.path.join(
    cwd,
    'model_configuration.json'
)
best_model_file_path = os.path.join(
    cwd,
    'best_model',
    'model.joblib'
)
trained_model_dir = os.path.join(
    cwd,
    'trained_model'
)
submission_dir = os.path.join(
    cwd,
    'submission'
)
score_file_path = os.path.join(
    cwd,
    'extra',
    'score.json'
)
scaler_object_file_path = os.path.join(
    cwd,
    'extra',
    'scalor.joblib'
)

target = 'pm2_5'
base_score = 13.0
n_splits = 10
random_state = 2022