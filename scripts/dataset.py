import os
from six.moves import urllib
import tarfile
import sys

SPEECH_COMMANDS_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

def maybe_download_and_extract(data_url, dest_directory):
    if os.path.exists(dest_directory):
        return
    os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            percent = float(count * block_size) / float(total_size) * 100.0
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, percent))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print ''
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    os.remove(filepath)

def get_speech_commands_dataset(directory):
    maybe_download_and_extract(SPEECH_COMMANDS_URL, directory)
    dirs = filter(lambda x: os.path.isdir(os.path.join(directory, x)), os.listdir(directory))
    dataset = {}
    for d in dirs:
        d_full = os.path.join(directory, d)
        files = filter(lambda x: x.endswith('.wav'), os.listdir(d_full))
        dataset[d] = map(lambda x: os.path.join(d_full, x), files)
    return dataset
