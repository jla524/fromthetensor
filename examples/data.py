import random
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from string import ascii_letters
import requests
import numpy as np
from unidecode import unidecode

random.seed(0)
DATA_DIR = Path("..").resolve() / "data"


def download_names(names_dir, labels_dir):
    if not labels_dir.is_dir():
        names_dir.mkdir(parents=True, exist_ok=True)
        response = requests.get("https://download.pytorch.org/tutorial/data.zip", timeout=5)
        assert response.status_code == 200
        zip_file = BytesIO(response.content)
        with ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(names_dir)


def name_to_array(name, char_to_idx):
    array = np.zeros((len(name), 1, len(char_to_idx)), dtype=np.float32)
    for i, char in enumerate(name):
        array[i][0][char_to_idx[char]] = 1
    return array


def fetch_names(input_dtype, output_dtype):
    names_dir = DATA_DIR / "names"
    labels_dir = names_dir / "data" / "names"
    download_names(names_dir, labels_dir)
    char_to_idx = {letter: i for i, letter in enumerate(ascii_letters + " .,:;-'")}
    lang_to_label = {file_path.stem: i for i, file_path in enumerate(labels_dir.iterdir())}
    dataset = []
    for file_path in labels_dir.iterdir():
        with file_path.open("r") as file:
            for name in [unidecode(line.rstrip()) for line in file]:
                if all(letter in char_to_idx for letter in name):
                    x = input_dtype(name_to_array(name, char_to_idx))
                    y = output_dtype([lang_to_label[file_path.stem]])
                    dataset.append((x, y))
    random.shuffle(dataset)
    train_data = dataset[:round(len(dataset)*.9)]
    test_data = dataset[round(len(dataset)*.9):]
    return train_data, test_data, char_to_idx, lang_to_label
