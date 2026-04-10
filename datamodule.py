# --
# datamodule tiny ml

import sys
import re
import yaml
import numpy as np
import soundfile
import gzip
import functools

from pathlib import Path
from feature_handler import FeatureHandler


class DatamoduleTinyMl():
  """
  datamodule tiny ml
  """

  def __init__(self, cfg={}, **kwargs):

    # arguments
    self.cfg = cfg
    self.kwargs = kwargs

    # init config
    self.cfg_init()

    # required package paths
    [sys.path.append(p) for p in self.cfg['add_python_paths'] if p not in sys.path]

    # members
    self.features = None
    self.targets = None
    self.sample_ids = None
    self.length = None
    self.label_dict = None
    self.classes = None
    self.intermediate_info = {}
    self.cache_info = {}
    self.load_info = {}
    self.feature_handler = None

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

    # intermediate dataset
    self.create_intermediate_dataset()

    # caching
    self.caching()

    # load set on init
    if self.cfg['load_set_on_init'] is not None:

      # assertions
      assert self.cfg['load_set_on_init'] in ['train', 'validation', 'test'], "load_set_on_init must be a string of one of: {}!".format(['train', 'validation', 'test'])

      # load corresponding set
      if self.cfg['load_set_on_init'] == 'train': self.load_train_dataset()
      elif self.cfg['load_set_on_init'] == 'validation': self.load_validation_dataset()
      elif self.cfg['load_set_on_init'] == 'test': self.load_test_dataset()
      return

    # load cache
    if self.cfg['load_cache']['load_on_init']: self.load_cache()


  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    return self.features[idx], self.targets[idx], self.sample_ids[idx]


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
      'feature_handler_add_kwargs': {
        'target_sample_rate': 24000,
        'transpose_features_extracted': True,
        'normalize_features': True,
        'to_float': True,
        'to_torch': True,
        'add_channel_dimension': True,
        'add_batch_dimension': False,
        'channel_dimension_at_end': False,
        },
      'dataset': {
        'root_path': '/path/to/the/downloaded/dataset',
        'file_ext': '.wav',
        },
      'intermediate': {
        'root_path': './output/01_intermediate',
        'intermediate_id': 'intermediate0',
        'filter_files': {'is_used': False, 're_contains': '.*'},
        'file_naming': {'target_file_ext': '.npz', 'method': 'keeping_parent_folder'},
        'compress': True,
        },
      'caching': {
        'root_path': './output/02_features',
        'cache_id': 'cache0',
        'filter_files': {'is_used': False, 're_contains': '.*'},
        'file_naming': {'target_file_ext': '.npz', 'method': 'keeping_parent_folder'},
        'compress': True,
        },
      'load_cache': {
        'load_on_init': False,
        'filter_files': {'is_used': False, 're_contains': '.*'},
        },

      # folder structure
      'train_folder': 'Train',
      'validation_folder': 'Validation',
      'test_folder': 'Validation',

      # load set
      'load_set_on_init': None,

      # other flags
      'redo_all': False,
      'redo_intermediate': False,
      'redo_cache': False,
      'verbose': False,
      }

    # config update
    self.cfg = {**cfg_default, **cfg_overwrites, **self.cfg, **self.kwargs}


  def create_intermediate_dataset(self):
    """
    create an intermediate dataset, overwrite this
    """

    # already processed return empty list
    if any([f for f in self.intermediate_path.glob('**/*') if f.is_file()]) and not self.cfg['redo_intermediate'] and not self.cfg['redo_all']: 

      # intermediate info - needed for further processing
      self.intermediate_info = yaml.safe_load(open(str(self.intermediate_path / 'intermediate_info.yaml')))
      return

    # info
    print("Datamodule - Create intermediate to: ", self.intermediate_path)

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

    # save intermediate info
    yaml.dump(self.intermediate_info, open(self.intermediate_path / 'intermediate_info.yaml', 'w'))


  def caching(self):
    """
    cache some data
    """

    # already cached (simple statement)
    if any([f for f in self.cached_path.glob('**/*') if f.is_file()]) and not self.cfg['redo_cache'] and not self.cfg['redo_all']: return

    # info
    print("Datamodule - Caching features to: ", self.cached_path)

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
    self.cache_info = {'label_dict': {**self.label_dict}, 'feature_size': None, 'target_size': None, 'files': {'dataset': [], 'intermediate': [], 'cached': []}}

    # cache info
    self.at_caching_add_something_before_file_processing()

    # go through each file
    for sid, file in enumerate(files):

      # target
      y = self.at_caching_extract_target_from_file(file)

      # features
      x = self.at_caching_extract_features_from_file(file)

      # out file name
      out_file_path = self.file_naming_by_config(self.cfg['caching']['file_naming'], file, self.cached_path, file_root_dir=self.intermediate_path)

      # cache info update
      if self.cache_info['feature_size'] is None: self.cache_info['feature_size'] = list(x.shape)
      if self.cache_info['target_size'] is None: self.cache_info['target_size'] = list(y.shape)

      # assertions
      assert self.cache_info['feature_size'] == list(x.shape)
      assert self.cache_info['target_size'] == list(y.shape)

      # add files
      self.cache_info['files']['cached'].append(str(out_file_path))
      self.cache_info['files']['intermediate'].append(str(file))
      self.cache_info['files']['dataset'].append(self.intermediate_info['intermediate_file_to_dataset_file'][str(file)])

      # info
      if self.cfg['verbose']: print("cached file saved to: ", out_file_path)

      # save function (compress?)
      f_save = np.savez_compressed if self.cfg['caching']['compress'] else np.savez

      # save file
      f_save(file=out_file_path, x=x, y=y, sid=np.array(sid).astype(np.uint32))

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
    print("{} load cache with id: [{}] and filter: [{}] ...".format(self.__class__.__name__, target_cached_path.name, additional_file_filter_cfg['re_contains']))

    # cache info
    self.cache_info = yaml.safe_load(open(str(target_cached_path / 'cache_info.yaml')))

    # get label dict
    self.label_dict = self.cache_info['label_dict']
    self.classes = list(self.label_dict.keys())

    # cached files
    cached_files_filtered = self.filter_files_with_config(sorted(list(target_cached_path.glob('**/*.npz'))), self.cfg['load_cache']['filter_files'])

    # additional filtering
    cached_files_filtered = self.filter_files_with_config(cached_files_filtered, additional_file_filter_cfg)

    # load info
    self.load_info = {'label_dict': {**self.label_dict}, 'feature_shape_at_load': None}

    # allocate memory
    self.at_load_cache_allocate_memory_before_adding_data(len(cached_files_filtered))

    # load each file and add to array
    for i, cached_file in enumerate(cached_files_filtered):

      # data 
      data = np.load(cached_file)

      # stack data
      self.features[i] = self.at_load_cache_process_features(data['x'])
      self.targets[i] = data['y']
      self.sample_ids[i] = data['sid']

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
    self.check_if_folder_exists_in_root_path_non_recursive(self.cfg['train_folder'])
    self.load_cache(additional_file_filter_cfg={'is_used': True, 're_contains': self.cfg['train_folder']})


  def load_validation_dataset(self):
    """
    train datraset loading, overwrite
    """
    self.check_if_folder_exists_in_root_path_non_recursive(self.cfg['validation_folder'])
    self.load_cache(additional_file_filter_cfg={'is_used': True, 're_contains': self.cfg['validation_folder']})


  def load_test_dataset(self):
    """
    train datraset loading, overwrite
    """
    self.check_if_folder_exists_in_root_path_non_recursive(self.cfg['test_folder'])
    self.load_cache(additional_file_filter_cfg={'is_used': True, 're_contains': self.cfg['test_folder']})


  def check_if_folder_exists_in_root_path_non_recursive(self, folder_name):
    """
    check folder existance in root path
    """

    # root directories
    root_dirs = [path_dir.name for path_dir in sorted(self.dataset_path.iterdir()) if path_dir.is_dir()]

    # assert existance
    assert folder_name in root_dirs, "Your root path should include a folder with name: '{}' (change folder in config.yaml), actual folders: {}".format(folder_name, root_dirs)


  def file_naming_by_config(self, cfg_file_naming, input_file, target_path, file_root_dir, file_name_addon='', overwrite_file_ext=None):
    """
    outfile naming
    """

    # base name
    base_name = input_file.stem + file_name_addon + (cfg_file_naming['target_file_ext'] if overwrite_file_ext is None else overwrite_file_ext)

    # simply name after file and ignore folders
    if cfg_file_naming['method'] == 'just_filename': return target_path / base_name

    # root subtraction
    re_root_subtraction = re.sub(r'\./', '', str(Path(file_root_dir))) + '/'

    # subtract root path
    file_path_subtracted_root = Path(re.sub(re_root_subtraction, '', str(input_file)))

    # keep parent folder
    if cfg_file_naming['method'] == 'keeping_parent_folder':

      # target path
      target_path /= file_path_subtracted_root.parent

      # create folder if it does not exist
      if not target_path.is_dir(): target_path.mkdir(parents=True)

      # with folder structure
      return target_path / base_name

    # no other option
    # use parent folder in file naming
    if cfg_file_naming['method'] == 'parent_folder_to_filename': pass

    # parent paths
    parent_path_names = re.sub(r'\.', '', str(file_path_subtracted_root.parent))
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

    # add more cache info
    self.cache_info.update({'x_len': None, 'fs': None, 'feature_size_origin': None})

    # feature handler init
    self.feature_handler = FeatureHandler(**{**self.cfg['feature_extraction'], **self.cfg['feature_handler_add_kwargs']})


  def at_caching_add_something_after_file_processing(self):
    """
    add something after file processing (overwrite this)
    """
    self.cache_info.update({
      'feature_extraction': self.cfg['feature_extraction'],
      'sample_rate': self.cfg['target_sample_rate'],
      })


  def at_caching_extract_target_from_file(self, file):
    """
    extract target from file (overwrite this)
    """
    return np.array(self.label_dict[file.parent.stem]).astype(np.uint8)


  def at_caching_extract_features_from_file(self, file):
    """
    extract features from file
    """

    # load file
    data = np.load(str(file))
    x, fs = data['x'], data['fs']

    # extract features
    features = self.feature_handler.extract(x)

    # cache info update
    if self.cache_info['x_len'] is None: self.cache_info['x_len'] = len(x)
    if self.cache_info['fs'] is None: self.cache_info['fs'] = int(fs)
    if self.cache_info['feature_size_origin'] is None: self.cache_info['feature_size_origin'] = list(features.shape)

    # assertions
    assert self.cache_info['x_len'] == len(x)
    assert self.cache_info['fs'] == fs
    assert self.cache_info['feature_size_origin'] == list(features.shape)

    # flatten
    features = features.flatten()

    return features


  def at_load_cache_allocate_memory_before_adding_data(self, num_cached_files):
    """
    memory allocation (overwrite this)
    """

    # target feature shape
    target_feature_shape = tuple(self.cache_info['feature_size_origin'])

    # add feature shape at load
    self.load_info.update({'feature_shape_at_load': target_feature_shape})

    # allocate memory space
    self.features = np.empty(shape=(num_cached_files,) + target_feature_shape, dtype=np.float32)
    self.targets = np.empty(shape=(num_cached_files), dtype=np.uint8)
    self.sample_ids = np.empty(shape=(num_cached_files), dtype=np.uint32)


  def at_load_cache_process_features(self, x):
    """
    postprocess cached data, flattened array to original shape
    """

    # reshape to original shape
    x = x.reshape(self.cache_info['feature_size_origin'])

    return x


  def info(self):
    """
    info
    """
    print("\n--\n{} info: ".format(self.__class__.__name__))
    print("label dict: ", self.get_label_dict())
    if self.load_info.get('feature_shape_at_load') is not None: print("feature size at load: ", self.get_feature_shape_at_load())
    print("--\n")


  def play_sound_by_single_sid(self, sid):
    """
    play sound by single sid
    """
    import sounddevice
    sounddevice.play(*self.get_raw_waveform_and_fs_by_single_sid(sid))


  def get_feature_shape_at_load(self):
    """
    feature shape at load
    """
    assert self.load_info.get('feature_shape_at_load') is not None, "Please load data before!"
    return self.load_info['feature_shape_at_load']


  def get_raw_waveform_file_by_single_sid(self, sid):
    """
    waveform file of origin
    """
    return self.get_file_names_by_single_sid(sid)[0]


  def get_raw_waveform_and_fs_by_single_sid(self, sid):
    """
    waveform data of origin file
    """
    return soundfile.read(self.get_raw_waveform_file_by_single_sid(sid))


  def get_cache_info_spec_fs(self):
    """
    get spec sampling rate (fs)
    """
    return self.cfg['target_sample_rate'] / self.cfg['feature_extraction']['window_stride']


  def get_label_dict(self): return self.label_dict
  def get_target_to_label_dict(self): return {v: k for k, v in self.label_dict.items()}
  def get_cache_info(self): return self.cache_info
  def get_cache_info_from_cached_folder(self): return yaml.safe_load(open(str(self.cached_path / 'cache_info.yaml')))
  def get_targets(self): return np.squeeze(self.targets)
  def get_file_names_by_single_sid(self, sid): return [self.cache_info['files']['dataset'][sid], self.cache_info['files']['intermediate'][sid], self.cache_info['files']['cached'][sid]]
  def get_file_name_id_by_single_sid(self, sid): return Path(self.get_file_names_by_single_sid(sid)[-1]).stem



if __name__ == '__main__':
  """
  datamodule tiny ml
  """

  import torch
  from plots import plot_waveform_and_features

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # datamodule
  datamodule = DatamoduleTinyMl(cfg['datamodule'], redo_all=False, redo_cache=False, load_set_on_init='train')
  datamodule.info()

  # loader TODO
  dataloader = torch.utils.data.DataLoader(datamodule, **{'batch_size': 4, 'shuffle': True})

  # get label dict
  target_to_label_dict = datamodule.get_target_to_label_dict()

  # show some examples
  for data_batch in dataloader:

    # batch 
    for data in zip(*data_batch):

      # extract components
      x, y, sid = data

      # extract waveform and features
      waveform, fs = datamodule.get_raw_waveform_and_fs_by_single_sid(sid)
      features = x[0]
      file_name = datamodule.get_file_name_id_by_single_sid(sid)

      # show plot flag
      show_plot_flag = True

      # play sound
      if show_plot_flag: datamodule.play_sound_by_single_sid(sid)

      # some prints
      print("file_name: ", file_name)
      print("feature shape: ", x.shape)
      print("feature type: ", x.dtype)
      print("sample ids: ", sid)
      print("targets: ", y)
      print("label: ", target_to_label_dict[int(y)])

      # plot waveform and features
      plot_waveform_and_features(waveform, features, show_plot_flag=show_plot_flag, title=file_name, fs=fs, spec_fs=datamodule.get_cache_info_spec_fs())

    break
