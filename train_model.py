import os
from collections import namedtuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import constant
import utility
from logger import logging
logger = logging.getLogger(__name__)

def run_train_model() -> None:
    logger.info(f'{10*"="} Started model training. {10*"="}')

    logger.info('Reading cleaned files.')
    train_df = pd.read_csv(
        filepath_or_buffer=constant.cleaned_train_file_path
    )
    test_df = pd.read_csv(
        filepath_or_buffer=constant.cleaned_test_file_path
    )

    X = train_df.drop(
        labels=[constant.target],
        axis=1
    ).values
    y = train_df[constant.target].values
    logger.info(f'Training data shape - {X.shape, y.shape}')

    X_test = test_df.values
    X_test.shape
    logger.info(f'Tesing data shape - {X_test.shape}')

    logger.info('Scaling the data.')
    ss = StandardScaler()
    ss.fit(X=X)
    X_scaled = ss.transform(X=X)
    X_test_scaled = ss.transform(X=X_test)

    logger.info('Saving Scaling Transformer.')
    utility.save_object(
        object=ss,
        file_path=constant.scaler_object_file_path
    )

    logger.info('Saving the preprocessed testing data.')
    utility.save_numpy_array(
        array=X_test_scaled,
        file_path=constant.scaled_testing_file_path
    )

    logger.info('Creating list of models.')
    ModelInformation = namedtuple(
        'ModelInformation',
        [
            'model_name',
            'model_object',
            'model_parameters',
            'model_file_name'
        ]
    )
    models = []

    model_data = utility.read_json_file(file_path=constant.model_configuration_file_path)

    for data in model_data['models']:
        module_name = data['module']
        class_name = data['class']
        parameters = data['parameters']
        model_object = utility.get_class_reference(
            module_name=module_name,
            class_name=class_name
        )
        if len(parameters) == 0:
            models.append(
                ModelInformation(
                    model_name=data['name'],
                    model_object=model_object(),
                    model_parameters=parameters,
                    model_file_name=data['model_file_name']
                )
            )
        else:
            updated_model_object = utility.update_property_of_class(
                instance_reference=model_object,
                parameters=parameters
            )
            models.append(
                ModelInformation(
                    model_name=data['name'],
                    model_object=updated_model_object(),
                    model_parameters=parameters,
                    model_file_name=data['model_file_name']
                )
            )

    base_score = constant.base_score
    best_model_information = None
    model_score = {
        'score': []
    }

    for model in models:
        logger.info(f'Training {model.model_name}')
        kfold = KFold(
            n_splits=constant.n_splits,
            random_state=constant.random_state,
            shuffle=True
        )
        ''' KFold Cross Validation
        '''
        model_object = model.model_object
        score = []
        _fold = 1
        for train_index, valid_index in kfold.split(X=X_scaled, y=y):
            # get the training and validation set od data
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

            # train the model
            model_object.fit(X_train, y_train)

            y_predicted = model_object.predict(X_valid)

            this_score = mean_absolute_error(y_valid, y_predicted)

            logger.info(f'Fold - {_fold}, Score - {this_score}')
            score.append(this_score)
            _fold += 1

        model_file_path = os.path.join(
            constant.trained_model_dir,
            model.model_file_name
        )
        utility.save_object(
            object=model_object,
            file_path=model_file_path
        )
        logger.info(f'CV score - {score}')
        mean_score = np.mean(score)
        logger.info(f'Mean score - {mean_score}')
        if mean_score < base_score:
            logger.info(f'Found a best model - {model.model_name}')
            logger.info(f'Replacing the best score {base_score} to {mean_score}.')
            base_score = mean_score
            best_model_information = ModelInformation(
                model_name=model.model_name,
                model_object=model_object,
                model_parameters=model.model_parameters,
                model_file_name=model.model_file_name
            )

        model_score['score'].append(
            {
                'model_name': model.model_name,
                'cv_Score': f'{score}',
                'mean_score': mean_score
            }
        )

    logger.info('Writing model score file.')
    utility.write_json_file(
        dictionary=model_score,
        file_path=constant.score_file_path
    )

    if best_model_information == None:
        logger.info('No best model found. Change the base score in "constant.py".')
    else:
        logger.info('Found a best model.')
        utility.save_object(
            object=best_model_information.model_object,
            file_path=constant.best_model_file_path
        )

    logger.info(f'{10*"="} Finished model training. {10*"="}')
