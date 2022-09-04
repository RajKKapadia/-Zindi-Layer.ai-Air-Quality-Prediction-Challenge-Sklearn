import os

import pandas as pd

import constant
import utility
from logger import logging
logger = logging.getLogger(__name__)

def run_generate_submission() -> None:
    logger.info(f'{10*"="} Started generating submission file. {10*"="}')

    logger.info('Loading scaled testing data.')
    X_test = utility.load_numpy_array(
        file_path=constant.scaled_testing_file_path
    )

    logger.info('Reading sample submission file.')
    submission = pd.read_csv(
        filepath_or_buffer=constant.sample_submission_file_path
    )

    logger.info('Loading all the trained models.')
    models = os.listdir(
        path=constant.trained_model_dir
    )

    logger.info('Started creating submission file.')
    for model in models:
        model_name = model.split('.')[0]
        logger.info(f'Loading model - {model_name}')
        model_object = utility.load_object(
            file_path=os.path.join(
                constant.trained_model_dir,
                model
            )
        )
        logger.info('Predicting...')
        y_test = model_object.predict(X_test)
        this_submission = submission.copy()
        this_submission[constant.target] = y_test
        logger.info('Writing submission to file.')
        this_submission.to_csv(
            path_or_buf=os.path.join(
                constant.submission_dir,
                f'{model_name}.csv'
            ),
            index=False
        )

    logger.info(f'{10*"="} Finished generating submission file. {10*"="}')
