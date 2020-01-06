import numpy as np
import h5py
import pandas
from functools import reduce
from urllib.request import urlretrieve
from tqdm import tqdm
import tarfile


SVHN_TRAIN_DIR = './data/train/'
SVHN_TEST_DIR = './data/train/'

SVHN_TRAIN_CSV = './data/train/digitStruct.csv'
SVHN_TEST_CSV = './data/train/digitStruct.csv'


# TAR files and dataset information from http://ufldl.stanford.edu/housenumbers/

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_tar(url, local_dir_path, local_file_name=None):
    local_file_name = local_file_name or url.split('/')[-1]
    local_file_path = '/'.join([local_dir_path, local_file_name])

    with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        path, message = urlretrieve(url, local_file_path, reporthook=t.update_to)

    return local_file_name, path, message


def extract_tar(file_path):
    f = tarfile.open(file_path)
    dir_comps = file_path.split('/')
    dir_path = '/'.join(dir_comps[:-1])
    f.extractall(path=dir_path)


def download_and_extract_tar(url, local_dir_path, local_file_name=None):
    name, path, _ = download_tar(url, local_dir_path, local_file_name)
    extract_tar(path)


def read_matlab_digit_struct(file_path, target_csv_path, max_n=None):
    rows = []

    with h5py.File(file_path, 'r') as h:
        d = h['digitStruct']

        boxes = d['bbox']
        names = d['name']

        max_n = max_n or len(boxes)
        max_n = min(max_n, len(boxes))

        assert len(boxes) == len(names)

        for i in tqdm(range(max_n)):
            name_ref = names[i][0]
            box_ref = boxes[i][0]

            name_raw = h[name_ref]
            box_raw = h[box_ref]

            name = ''.join([chr(x) for x in name_raw])

            row = {'name': name}

            for key in box_raw:
                key_raw = box_raw[key]
                np_values = []

                if key_raw.shape == (1, 1):
                    # Single box in this frame, so don't try to iterate, just decode
                    value = key_raw[0][0]
                    np_values.append(value)
                else:
                    # Multiple boxes
                    for key_i in range(len(key_raw)):
                        key_ref = key_raw[key_i][0]
                        key_val = h[key_ref]
                        value = np.array(key_val, dtype=key_val.dtype)
                        np_values.append(value)

                np_values = np.array(np_values).flatten()

                row[key] = np_values

            rows.append(pandas.DataFrame(row))

    # After reading file, flatten data frames into one
    def combine_rows(df1, df2): return df1.append(df2)

    df = reduce(combine_rows, rows)

    # Reset the index to make indexes actually unique again
    df = df.reset_index()
    df = df.drop('index', 1)

    # Convert labels of 10 to 0, since they actually represent the digit 0
    df['label'][df['label'] == 10] = 0

    # Save data frame to file so we don't have to run this every time
    # with open(csv_file_path, 'w') as csv_file:
    df.to_csv(target_csv_path)


if __name__ == '__main__':
    # 1. Full Numbers
    train = "http://ufldl.stanford.edu/housenumbers/train.tar.gz"
    test = "http://ufldl.stanford.edu/housenumbers/test.tar.gz"
    extra = "http://ufldl.stanford.edu/housenumbers/extra.tar.gz"

    number_urls = [train, test]

    # 2. Cropped 32x32 numbers
    train_32 = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    test_32 = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    extra_32 = "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"

    cropped_urls = [train_32, test_32]

    # for url in number_urls:
    #     download_and_extract_tar(url, './data')

    # Local file paths for restructuring the matlab bounding box files to csv files
    train_file_path = './data/train/digitStruct.mat'

    test_file_path = './data/train/digitStruct.mat'

    read_matlab_digit_struct(train_file_path, SVHN_TRAIN_CSV, max_n=50)
    read_matlab_digit_struct(test_file_path, SVHN_TEST_CSV, max_n=50)
