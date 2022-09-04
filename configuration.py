import os

def set_configuration() -> None:
    ''' Set up the configuration
    '''
    cwd = os.getcwd()

    # Create the required folders
    cleaned_dataset_dir = os.path.join(
        cwd,
        'cleaned_dataset'
    )
    os.makedirs(
        cleaned_dataset_dir,
        exist_ok=True
    )
    best_model_dir = os.path.join(
        cwd,
        'best_model'
    )
    os.makedirs(
        best_model_dir,
        exist_ok=True
    )
    application_log_dir = os.path.join(
        cwd,
        'application_log'
    )
    os.makedirs(
        application_log_dir,
        exist_ok=True
    )
    trained_model_dir = os.path.join(
        cwd,
        'trained_model'
    )
    os.makedirs(
        trained_model_dir,
        exist_ok=True
    )
    submission_dir = os.path.join(
        cwd,
        'submission'
    )
    os.makedirs(
        submission_dir,
        exist_ok=True
    )
    extra_dir = os.path.join(
        cwd,
        'extra'
    )
    os.makedirs(
        extra_dir,
        exist_ok=True
    )
