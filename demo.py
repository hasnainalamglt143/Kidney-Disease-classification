from cnn_classifier.utils import common
from pathlib import Path

print(common.read_yaml(Path("./dvc.yaml")).start.key1)