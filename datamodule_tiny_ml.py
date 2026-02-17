# --
# datamodule tiny ml

import sys
import re
import torch
import yaml
import numpy as np
import soundfile
import gzip
import functools

from pathlib import Path


class DatamoduleTinyMl(torch.utils.data.Dataset):
  """
  datamodule tiny ml
  """

  def __init__(self, cfg={}, **kwargs):

    # super constructor
    super().__init__()

    # arguments
    self.cfg = cfg
    self.kwargs = kwargs

    # init config
    self.cfg_init()

    # required package paths
    [sys.path.append(p) for p in self.cfg['add_python_paths'] if p not in sys.path]

    # members
    self.fp_getitem = None
    self.features = None
    self.targets = None
    self.sample_ids = None
    self.length = None
    self.label_dict = None
    self.classes = None
    self.intermediate_info = {}
    self.cache_info = {}
    self.feature_constants = None
    self.do_windows_fn = None

    # assertions
    assert Path(self.cfg['dataset']['root_path']).is_dir(), "***No dataset root path exists: [{}]".format(self.cfg['dataset']['root_path'])

    # paths
    self.dataset_path = Path(self.cfg['dataset']['root_path'])
    self.intermediate_path = Path(self.cfg['intermediate']['root_path']) / self.cfg['intermediate']['intermediate_id']
    self.cached_path = Path(self.cfg['caching']['root_path']) / self.cfg['caching']['cache_id']

    # assertions
    assert self.dataset_path.is_dir(), "***Wrong data folder selected? [{}]".format(self.dataset_path)

    # path creation
    [p.mkdir(parents=True) for p in [self.intermediate_path, self.cached_path] if not p.is_dir()]

    # setup get item function
    self.fp_getitem = self.getitem_to_torch if self.cfg['to_torch'] else self.getitem_numpy

    # intermediate dataset
    self.create_intermediate_dataset()

    # caching
    self.caching()

    # load cache
    if self.cfg['load_cache']['load_on_init']: self.load_cache()


  def __len__(self):
    return self.length


  def __getitem__(self, idx):
    return self.fp_getitem(idx)


  def getitem_to_torch(self, idx):
    return torch.from_numpy(self.features[idx]), torch.from_numpy(np.squeeze(self.targets[idx])), torch.from_numpy(np.squeeze(self.sample_ids[idx]))


  def getitem_numpy(self, idx):
    return self.features[idx], np.squeeze(self.targets[idx]), np.squeeze(self.sample_ids[idx])


  def cfg_init(self, **cfg_overwrites):
    """
    config init
    """

    # default config
    cfg_default = {
      'add_python_paths': [],
      'target_audio_slice_duration_ms': 3000,
      'target_sample_rate': 24000,
      'feature_extraction': {
        'window_len': 4096,
        'window_stride': 512,
        'window_scaling_bits': 12,
        'mel_n_channels': 40,
        'mel_low_hz': 125,
        'mel_high_hz': 7500,
        'mel_post_scaling_bits': 6,
        },
      'dataset': {
        'root_path': '/path/to/the/downloaded/dataset',
        'file_ext': '.wav',
        },
      'intermediate': {
        'root_path': './output/01_intermediate',
        'intermediate_id': 'intermediate0',
        'redo': False,
        'filter_files': {'is_used': False, 're_contains': '.*'},
        'file_naming': {'target_file_ext': '.npz', 'method': 'keeping_parent_folder'},
        'compress': True,
        },
      'caching': {
        'root_path': './output/02_features',
        'cache_id': 'cache0',
        'redo': False,
        'filter_files': {'is_used': False, 're_contains': '.*'},
        'file_naming': {'target_file_ext': '.npz', 'method': 'parent_folder_to_filename'},
        'compress': True,
        },
      'load_cache': {
        'load_on_init': True,
        'filter_files': {'is_used': False, 're_contains': '.*'},
        },

      # folder structure
      'train_folder': 'train',
      'validation_folder': 'validation',
      'test_folder': 'test',

      # other flags
      'redo': False,
      'verbose': False,
      'to_torch': True,
      }

    # config update
    self.cfg = {**cfg_default, **cfg_overwrites, **self.cfg, **self.kwargs}


  def create_intermediate_dataset(self):
    """
    create an intermediate dataset, overwrite this
    """

    # already processed return empty list
    if any([f for f in self.intermediate_path.glob('**/*') if f.is_file()]) and not self.cfg['intermediate']['redo'] and not self.cfg['redo']: return []

    # info
    self.cfg['verbose']: print("Create intermediate to: ", self.intermediate_path)

    # clean directory
    [(print("remove file: ", f) if self.cfg['verbose'] else None, f.unlink()) for f in self.intermediate_path.glob('**/*') if f.is_file()]
    [(print("remove folder: ", f) if self.cfg['verbose'] else None, f.rmdir()) for f in sorted(self.intermediate_path.glob('**/*'))[::-1] if f.is_dir()]

    # get files
    files = sorted(list(self.dataset_path.glob('**/*' + self.cfg['dataset']['file_ext'])))

    # filter files
    files = self.filter_files_with_config(files, self.cfg['intermediate']['filter_files'])
    
    # intermediate info
    self.intermediate_info.update({'intermediate_file_to_dataset_file': {}})

    # process files
    for file in files:

      # out file
      out_file_path = self.file_naming_by_config(self.cfg['intermediate']['file_naming'], file, target_path=self.intermediate_path, file_root_dir=self.dataset_path)
      self.intermediate_info['intermediate_file_to_dataset_file'].update({str(out_file_path): str(file)})

      # read audio
      audio_array, sample_rate = soundfile.read(file)

      # assert len and fs
      assert int(len(audio_array) / sample_rate * 1000) == self.cfg['target_audio_slice_duration_ms'], "Audio file: [{}] has ms: [{}] and should be: [{}]".format(file, int(len(audio_array) / sample_rate * 1000), self.cfg['target_audio_slice_duration_ms'])
      assert sample_rate == self.cfg['target_sample_rate'], "Audio file: [{}] has sample rate: [{}] and should be: [{}]".format(file, sample_rate, self.cfg['target_sample_rate'])

      # to int16 conversion for serialization
      audio_array_int16 = (audio_array * np.iinfo(np.int16).max).astype(np.int16)

      # save function (compress?)
      f_save = np.savez_compressed if self.cfg['intermediate']['compress'] else np.savez

      # save file
      f_save(file=out_file_path, x=audio_array_int16, fs=sample_rate)


  def caching(self):
    """
    cache some data
    """

    # already cached (simple statement)
    if any([f for f in self.cached_path.glob('**/*') if f.is_file()]) and not self.cfg['caching']['redo'] and not self.cfg['redo']: return

    # info
    self.cfg['verbose']: print("Caching features to: ", self.cached_path)

    # clean directory
    [(print("cache remove: ", f) if self.cfg['verbose'] else None, f.unlink()) for f in self.cached_path.glob('**/*') if f.is_file()]
    [(print("remove folder: ", f) if self.cfg['verbose'] else None, f.rmdir()) for f in sorted(self.cached_path.glob('**/*'))[::-1] if f.is_dir()]

    # get files
    files = sorted(list(self.intermediate_path.glob('**/*' + self.cfg['intermediate']['file_naming']['target_file_ext'])))

    # filter files
    files = self.filter_files_with_config(files, self.cfg['caching']['filter_files'])

    # get label dict
    self.label_dict = self.at_caching_extract_label_dict_from_files(files)

    # cache info
    self.cache_info = {'label_dict': {**self.label_dict}, 'feature_sizes': [], 'target_sizes': [], 'files': {'dataset': [], 'intermediate': [], 'cached': []}, 'cfg_caching': self.cfg['caching']}

    # cache info
    self.at_caching_add_something_before_file_processing()

    # go through each file
    for file in files:

      # target
      y = self.at_caching_extract_target_from_file(file)

      # features
      x = self.at_caching_extract_features_from_file(file)

      # out file name
      out_file_path = self.file_naming_by_config(self.cfg['caching']['file_naming'], file, self.cached_path, file_root_dir=self.intermediate_path)

      # cache info update
      self.cache_info['feature_sizes'].append(list(x.shape))
      self.cache_info['target_sizes'].append(list(y.shape))
      self.cache_info['files']['cached'].append(str(out_file_path))
      self.cache_info['files']['intermediate'].append(str(file))
      self.cache_info['files']['dataset'].append(self.intermediate_info['intermediate_file_to_dataset_file'][str(file)])

      # info
      if self.cfg['verbose']: print("cached file saved to: ", out_file_path)

      # save function (compress?)
      f_save = np.savez_compressed if self.cfg['caching']['compress'] else np.savez

      # save file
      f_save(file=out_file_path, x=x, y=y)

    # add something after file processing
    self.at_caching_add_something_after_file_processing()

    # write yaml file
    yaml.dump(self.cache_info, open(self.cached_path / 'cache_info.yaml', 'w'))


  def load_cache(self, cache_id=None, additional_file_filter_cfg=None):
    """
    load data from cache
    """

    # target cached path
    target_cached_path = self.cached_path

    # target cache id 
    if not cache_id is None:

      # change cached path
      target_cached_path = self.cached_path.parent / cache_id

      # check if path exists
      if not target_cached_path.exists(): raise ValueError('cach_id: {} does not exist in path: {}!'.format(cache_id, self.cached_path.parent))

    # info
    print("{} load cache with id: [{}] ...".format(self.__class__.__name__, target_cached_path.name))

    # cache info
    self.cache_info = yaml.safe_load(open(str(target_cached_path / 'cache_info.yaml')))

    # get label dict
    self.label_dict = self.cache_info['label_dict']
    self.classes = list(self.label_dict.keys())

    # cached files
    cached_files_filtered = self.filter_files_with_config(sorted(list(target_cached_path.glob('**/*.npz'))), self.cfg['load_cache']['filter_files'])

    # additional filtering
    cached_files_filtered = self.filter_files_with_config(cached_files_filtered, additional_file_filter_cfg)

    # allocate memory
    self.at_load_cache_allocate_memory_before_adding_data(len(cached_files_filtered))

    # load each file and add to array
    for i, cached_file in enumerate(cached_files_filtered):

      # data 
      data = np.load(cached_file)

      # stack data
      self.features[i] = self.at_load_cache_process_features(data['x'])
      self.targets[i] = data['y']
      self.sample_ids[i] = [idx for idx, name in enumerate(self.cache_info['files']['cached']) if re.search(str(cached_file), name)]

    # length
    assert(len(self.features) == len(self.targets))

    # dataset length
    self.length = len(self.targets)

    # success
    if self.cfg['verbose']: print("Datamodule successfully loaded!")


  def load_train_dataset(self):
    """
    train datraset loading, overwrite
    """
    print("No train dataset available")
    self.load_cache(additional_file_filter_cfg={'is_used': False, 're_contains': 'train'})


  def load_validation_dataset(self):
    """
    train datraset loading, overwrite
    """
    print("No validation dataset available")
    self.load_cache(additional_file_filter_cfg={'is_used': False, 're_contains': 'validation'})


  def load_test_dataset(self):
    """
    train datraset loading, overwrite
    """
    print("No test dataset available")
    self.load_cache(additional_file_filter_cfg={'is_used': False, 're_contains': 'test'})


  def file_naming_by_config(self, cfg_file_naming, input_file, target_path, file_root_dir, file_name_addon='', overwrite_file_ext=None):
    """
    outfile naming
    """

    # base name
    base_name = input_file.stem + file_name_addon + (cfg_file_naming['target_file_ext'] if overwrite_file_ext is None else overwrite_file_ext)

    # simply name after file and ignore folders
    if cfg_file_naming['method'] == 'just_filename': return target_path / base_name

    # root substraction
    re_root_substraction = re.sub(r'\./', '', str(Path(file_root_dir))) + '/'

    # substract root path
    file_path_substracted_root = Path(re.sub(re_root_substraction, '', str(input_file)))

    # keep parent folder
    if cfg_file_naming['method'] == 'keeping_parent_folder':

      # target path
      target_path /= file_path_substracted_root.parent

      # create folder if it does not exist
      if not target_path.is_dir(): target_path.mkdir(parents=True)

      # with folder structure
      return target_path / base_name

    # no other option
    # use parent folder in file naming
    if cfg_file_naming['method'] == 'parent_folder_to_filename': pass

    # parent paths
    parent_path_names = re.sub(r'\.', '', str(file_path_substracted_root.parent))
    parent_path_names = re.sub(r'/', '.', parent_path_names)
    parent_path_names += '.' if len(parent_path_names)  else ''

    return target_path / (parent_path_names + base_name)


  def filter_files_with_config(self, files, cfg_filter_files={'is_used': False, 're_contains': '.*'}):
    """
    filter files with config {filter_files: {is_used, re_contains}}
    """

    # returns
    if cfg_filter_files is None: return files
    if not cfg_filter_files['is_used']: return files

    # assert string in re
    assert isinstance(cfg_filter_files['re_contains'], str)

    # filter files
    filtered_files = [f for f in files if re.search(cfg_filter_files['re_contains'], str(f))]

    # assert that there are still some files left
    assert len(filtered_files), 'No files left after filtering!!! Change in config.yaml -> filter_files.re_contains'

    # filter files
    return filtered_files


  def at_caching_extract_label_dict_from_files(self, files):
    """
    extract label dict from files
    """

    # labels
    labels = np.unique([self.at_caching_extract_label_from_file(f) for f in files])

    # label dict
    cache_label_dict = {str(l): i for i, l in enumerate(labels)}

    return cache_label_dict


  def at_caching_extract_label_from_file(self, file):
    """
    extract label from file (overwrite this)
    """
    return file.parent.stem


  def at_caching_add_something_before_file_processing(self):
    """
    add something before file processing
    """

    # packages for feature extraction
    from biodcase_tiny.feature_extraction.feature_extraction import process_window, make_constants
    from numpy.lib.stride_tricks import sliding_window_view

    # add more cache info
    self.cache_info.update({'x_len': [], 'fs': [], 'feature_sizes_origin': []})

    # feature constants
    self.feature_constants = make_constants(
      win_samples=self.cfg['feature_extraction']['window_len'],
      sample_rate=self.cfg['target_sample_rate'],
      window_scaling_bits=self.cfg['feature_extraction']['window_scaling_bits'],
      mel_n_channels=self.cfg['feature_extraction']['mel_n_channels'],
      mel_low_hz=self.cfg['feature_extraction']['mel_low_hz'],
      mel_high_hz=self.cfg['feature_extraction']['mel_high_hz'],
      mel_post_scaling_bits=self.cfg['feature_extraction']['mel_post_scaling_bits'],
    )

    # window function
    apply_windowed = lambda data, window_len, window_stride, fn: np.array([fn(row) for row in sliding_window_view(data, window_len)[::window_stride]])

    # this partial stuff is just a way to set all config parameters, so we have a function that only takes data as input
    self.do_windows_fn = functools.partial(
      apply_windowed,
      window_len=self.cfg['feature_extraction']['window_len'],
      window_stride=self.cfg['feature_extraction']['window_stride'],
      fn=functools.partial(
        process_window,
        hanning=self.feature_constants.hanning_window,
        mel_constants=self.feature_constants.mel_constants,
        fft_twiddle=self.feature_constants.fft_twiddle,
        window_scaling_bits=self.feature_constants.window_scaling_bits,
        mel_post_scaling_bits=self.feature_constants.mel_post_scaling_bits
      ),
    )


  def at_caching_add_something_after_file_processing(self):
    """
    add something after file processing (overwrite this)
    """
    pass
    #self.cache_info.update({'feature_params': {**self.feature_extraction.get_feature_kwargs(), **{k: v.tolist() for k, v in self.feature_extraction.get_pre_computes().items() if k in ['mel_frequencies']}, **self.feature_extraction.get_cfg()}})


  def at_caching_extract_target_from_file(self, file):
    """
    extract target from file (overwrite this)
    """
    return np.array([self.label_dict[file.parent.stem]])


  def at_caching_extract_features_from_file(self, file):
    """
    extract features from file
    """

    # load file
    data = np.load(str(file))
    x, fs = data['x'], data['fs']

    # add cache info
    self.cache_info['x_len'].append(len(x))
    self.cache_info['fs'].append(int(fs))

    # compute features
    features = self.do_windows_fn(x).astype(np.float32)

    # origin sizes
    self.cache_info['feature_sizes_origin'].append(list(features.shape))

    # flatten
    features = features.flatten()

    return features


  def at_load_cache_allocate_memory_before_adding_data(self, num_cached_files):
    """
    memory allocation (overwrite this)
    """

    # allocate memory space
    self.features = np.empty(shape=(num_cached_files,) + tuple(self.cache_info.get('feature_sizes')[0]), dtype=np.uint8)
    self.targets = np.empty(shape=(num_cached_files, 1), dtype=np.uint8)
    self.sample_ids = np.empty(shape=(num_cached_files, 1), dtype=np.uint32)


  def at_load_cache_process_features(self, x):
    """
    postprocess cached data, e.g. format mismatches (overwrite this)
    """
    return x


  def info(self):
    """
    info
    """
    print("\n--\n{} info: ".format(self.__class__.__name__))
    print("label dict: ", self.get_label_dict())
    print("--\n")


  def get_label_dict(self): return self.label_dict
  def get_cache_info(self): return self.cache_info
  def get_cache_info_from_cached_folder(self): return yaml.safe_load(open(str(self.cached_path / 'cache_info.yaml')))
  def get_targets(self): return np.squeeze(self.targets)
  def get_file_names_by_single_sid(self, sid): return [self.cache_info['files']['dataset'][sid], self.cache_info['files']['intermediate'][sid], self.cache_info['files']['cached'][sid]]
  def get_file_name_id_by_single_sid(self, sid): return Path(self.get_file_names_by_single_sid(sid)[-1]).stem



if __name__ == '__main__':
  """
  datamodule tiny ml
  """

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # datamodule
  datamodule = DatamoduleTinyMl(cfg['datamodule'], redo=False)
  datamodule.info()

  # train dataset
  datamodule.load_train_dataset()
  #datamodule.load_validation_dataset()
  #datamodule.load_test_dataset()

  # loader
  dataloader = torch.utils.data.DataLoader(datamodule, **cfg['dataloader_kwargs'])

  # test loader
  x, y, sid = next(iter(dataloader))

  print(x)
  print(x.shape)
  print(x.dtype)
  print(y)
  print(sid)