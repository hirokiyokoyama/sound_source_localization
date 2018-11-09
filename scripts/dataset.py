import os
from six.moves import urllib
import tarfile
import sys
import yaml
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan

SPEECH_COMMANDS_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

def maybe_download_and_extract(data_url, dest_directory):
    if os.path.exists(dest_directory):
        return
    os.makedirs(dest_directory)
    print 'Dataset does not exist.'
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            percent = float(count * block_size) / float(total_size) * 100.0
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, percent))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print ''
    print 'Extracting...'
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

def get_recorded_dataset(directories):
    if isinstance(directories, str):
        directories = [directories]
    dataset = []
    suffixes = {'sound': '.wav',
                'meta': '.txt',
                'mic_pose': '.msg',
                'speaker_pose': '.msg',
                'self_pose': '.msg',
                'other_pose': '.msg',
                'self_scan': '.msg',
                'other_scan': '.msg'}
    for directory in directories:
        d_full = os.path.abspath(directory)
        _dataset = {}
        for f in os.listdir(d_full):
            f_full = os.path.join(d_full, f)
            for prefix, suffix in suffixes.items():
                if f.startswith(prefix+'_') and f.endswith(suffix):
                    id = f[len(prefix)+1:-len(suffix)]
                    if id in _dataset:
                        entry = _dataset[id]
                    else:
                        entry = {}
                        _dataset[id] = entry
                    if prefix == 'sound':
                        entry['recorded_sound_file'] = f_full
                    elif prefix == 'meta':
                        with open(f_full, 'r') as _f:
                            entry.update(yaml.load(_f.read()))
                    elif prefix.endswith('pose'):
                        ps = PoseStamped()
                        with open(f_full, 'rb') as _f:
                            ps.deserialize(_f.read())
                        entry[prefix] = ps
                    elif prefix.endswith('scan'):
                        ls = LaserScan()
                        with open(f_full, 'rb') as _f:
                            ls.deserialize(_f.read())
                        entry[prefix] = ls
                        
        print 'Found {} entries in directory "{}"'.format(len(_dataset), d_full)
        for k, v in _dataset.iteritems():
            v['id'] = os.path.basename(directory)+'/'+k
            dataset.append(v)
    return dataset
