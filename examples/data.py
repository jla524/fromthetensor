from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from string import ascii_letters
import requests
import numpy as np
from unidecode import unidecode
from sklearn.model_selection import train_test_split

DATA_DIR = Path("..").resolve() / "data"


def name_to_array(name, char_to_idx):
    array = np.zeros((len(name), 1, len(char_to_idx)), dtype=np.float32)
    for i, char in enumerate(name):
        array[i][0][char_to_idx[char]] = 1
    return array


def download_names(names_dir, labels_dir):
    if not labels_dir.is_dir():
        names_dir.mkdir(parents=True, exist_ok=True)
        response = requests.get("https://download.pytorch.org/tutorial/data.zip", timeout=5)
        assert response.status_code == 200
        zip_file = BytesIO(response.content)
        with ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(labels_dir)


def fetch_names(dtype):
    names_dir = DATA_DIR / "names"
    labels_dir = names_dir / "data" / "names"
    download_names(names_dir, labels_dir)
    char_to_idx = {letter: i for i, letter in enumerate(ascii_letters + " .,:;-'")}
    lang_to_label = {file_path.stem: i for i, file_path in enumerate(labels_dir.iterdir())}
    input_names, target_langs = [], []
    for file_path in labels_dir.iterdir():
        with file_path.open("r") as file:
            for name in [unidecode(line.rstrip()) for line in file]:
                if all(letter in char_to_idx for letter in name):
                    input_names.append(name_to_array(name, char_to_idx))
                    target_langs.append(lang_to_label[file_path.stem])
    train_idx, test_idx = train_test_split(
        range(len(target_langs)), test_size=0.1, random_state=1337, shuffle=True, stratify=target_langs
    )
    train_dataset = [(dtype(input_names[i]), np.array(target_langs[i])) for i in train_idx]
    test_dataset = [(dtype(input_names[i]), np.array(target_langs[i])) for i in test_idx]
    return train_dataset, test_dataset, char_to_idx, lang_to_label
