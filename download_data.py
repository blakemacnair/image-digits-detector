from urllib.request import urlretrieve
from tqdm import tqdm
import tarfile


# From http://ufldl.stanford.edu/housenumbers/

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


if __name__ == '__main__':
    for url in number_urls:
        download_and_extract_tar(url, './data')
