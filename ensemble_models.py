import os

import numpy as np
import pandas as pd

import utility
import constant
from logger import logging
logger = logging.getLogger(__name__)

def run_ensemble_models() -> None:
    logger.info(f'{10*"="} Started generating ensemble submission file. {10*"="}')

    logger.info('Reading the trained model score.')
    score_data = utility.read_json_file(
        file_path=constant.score_file_path
    )

    all_model_mean_score = []
    model_name = []
    for s in score_data['score']:
        all_model_mean_score.append(s['mean_score'])
        model_name.append(s['model_name'])

    logger.info('Computing weights.')
    weights = utility.compute_weight(
        score=all_model_mean_score
    )

    logger.info(f'All model score - {all_model_mean_score}')
    logger.info(f'All model name - {model_name}')
    logger.info(f'Weight for all model - {weights}')

    logger.info('Loading scaled testing data.')
    X_test = utility.load_numpy_array(
        file_path=constant.scaled_testing_file_path
    )

    y_test = np.zeros(X_test.shape[0])

    logger.info('Generating ensemble submission file.')
    submission_file_names = os.listdir(constant.submission_dir)
    for i in range(len(submission_file_names)):
        mn = model_name[i]
        logger.info(f'Model name - {mn}')
        logger.info('Reading the csv file.')
        csv_file_path = os.path.join(
            constant.submission_dir,
            submission_file_names[i]
        )
        df = pd.read_csv(
            filepath_or_buffer=csv_file_path
        )
        logger.info('Computing prediction.')
        predictions = df[constant.target] * weights[i]
        y_test += predictions

    logger.info('Writing ensembled output to the submission file.')
    logger.info('Reading sample submission file.')
    submission = pd.read_csv(
        filepath_or_buffer=constant.sample_submission_file_path
    )
    submission[constant.target] = y_test
    submission.to_csv(
        path_or_buf=os.path.join(
            constant.submission_dir,
            'Ensembled.csv'
        ),
        index=False
    )

    logger.info(f'{10*"="} Finished generating ensemble submission file. {10*"="}')
