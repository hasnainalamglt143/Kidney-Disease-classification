import logging
import os
import sys

logging_srt='[%(asctime)s] : %(module)s: %(levelname)s: %(message)s'

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

file_path=os.path.join(log_dir,'running_logs.log')

logging.basicConfig(level=logging.INFO,
                    format=logging_srt,
                    handlers=[logging.FileHandler(filename=file_path),
                              logging.StreamHandler(sys.stdout)])


logger = logging.getLogger("cnnClassifierLogger")