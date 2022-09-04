import pandas as pd

import constant
import utility
from logger import logging
logger = logging.getLogger(__name__)

def run_prepare_dataset() -> None:
    logger.info(f'{10*"="} Started preparing data. {10*"="}')

    train_df = pd.read_csv(
        filepath_or_buffer=constant.train_file_path
    )
    test_df = pd.read_csv(
        filepath_or_buffer=constant.test_file_path
    )

    train_data = utility.create_date_features(
        df=train_df,
        drop_features=['ID', 'device']
    )
    test_data = utility.create_date_features(
        df=test_df,
        drop_features=['ID', 'device']
    )

    train_data = utility.clean_dataframe(
        df=train_data,
        target='pm2_5'
    )
    test_data = utility.clean_dataframe(df=test_data)

    logger.info(f'Training data shape - {train_data.shape}')
    logger.info(f'Testing data shape - {test_data.shape}')

    columns_not_in_testing_data = []
    for c in train_data.columns:
        if c not in test_data.columns and c != constant.target:
            columns_not_in_testing_data.append(c)

    if len(columns_not_in_testing_data) > 0:
        logger.info('Columns not in testing data.')
        logger.info(columns_not_in_testing_data)
        logger.info('Dropping them from training data.')

        train_data.drop(
            labels=columns_not_in_testing_data,
            axis=1,
            inplace=True
        )
    else:
        logger.info('Training and testing data is okay.')

    logger.info('Writing cleaned data.')
    train_data.to_csv(
        path_or_buf=constant.cleaned_train_file_path
    )
    test_data.to_csv(
        path_or_buf=constant.cleaned_test_file_path
    )

    logger.info(f'{10*"="} Finished preparing data. {10*"="}')
