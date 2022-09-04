import logging

import constant

logging.basicConfig(
    filename=constant.application_log_file_path,
    filemode='w',
    format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)