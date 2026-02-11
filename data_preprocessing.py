from functools import partial
from pathlib import Path
from typing import Iterable

import yaml
import librosa
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask
from tqdm.dask import TqdmCallback
import pyarrow as pa
import soundfile

from config import load_config, Config
from paths import CLIPS_DIR, PREPROC_PRQ_PATH

_PREPROC_DASK_BATCH_SIZE = 1000

SPLIT_FOLDER_TO_SPLIT = {"train": "train", "val": "validation", "test": "test"}


def extract_loudest_slice(audio_array, sample_rate, audio_slice_duration_ms):
    """
    Find the max of the audio, then return a slice of duration audio_slice_duration_ms centred on the max.
    Also deal with boundary conditions.

    :param audio_slice_duration_ms:
    :return: slice with duration audio_slice_duration_ms
    """

    slice_n_samples = int(audio_slice_duration_ms / 1000 * sample_rate)
    audio_n_samples = audio_array.shape[0]
    left_edge = slice_n_samples // 2
    right_edge = slice_n_samples - left_edge

    max_index = np.argmax(audio_array)
    start_index = max(max_index - left_edge, 0)
    end_index = min(max_index + right_edge, audio_n_samples)

    # Handle edge cases where the slice would go beyond bounds
    if end_index - start_index < slice_n_samples:
        if start_index == 0:  # Cut at left edge
            end_index = slice_n_samples
        else:  # Cut at right edge
            start_index = audio_n_samples - slice_n_samples
            end_index = audio_n_samples
    return audio_array[start_index:end_index]


def run_preprocessing(config: Config, clips_dir=CLIPS_DIR, preproc_prq_path=PREPROC_PRQ_PATH):
    """
    run preprocessing - check sample rate, create classes, etc
    """

    # batch function
    def do_batch(batch: Iterable[Path], **kwargs):

        # data
        slice_data = []

        # get class dict
        actual_class_dict = kwargs.get('class_dict')
        assert actual_class_dict is not None

        # for each sample
        for path in batch:

            # todo: resample all to specific fs
            # load audio
            #audio_array, sample_rate = librosa.load(path, sr=config.data_preprocessing.sample_rate, mono=True)
            audio_array, sample_rate = librosa.load(path, sr=None, mono=True)

            # info
            #print("path: ", Path(path).name), print("fs: ", sample_rate), print("class: ", actual_class_dict[path.parent.name]), print("split: ", SPLIT_FOLDER_TO_SPLIT.get(path.parents[1].name, None)), print("soundfile: ", soundfile.info(path))

            audio_array_int16 = (audio_array * np.iinfo(np.int16).max).astype(np.int16)
            audio_slice = extract_loudest_slice(audio_array_int16, sample_rate, config.data_preprocessing.audio_slice_duration_ms)
            slice_data.extend([{
                "data": audio_slice,
                "path": str(path),
                "label": actual_class_dict[path.parent.name],
                "split": SPLIT_FOLDER_TO_SPLIT.get(path.parents[1].name, None),
                "sample_rate": sample_rate
            }])
        return pd.DataFrame(slice_data)

    # all clips
    clips = sorted(list(clips_dir.rglob("*.wav")))

    # create class dict
    class_dict = {k: i for i, k in enumerate(sorted(np.unique([str(Path(clip.parent.name)) for clip in clips]).tolist()))}

    # save class dict
    yaml.dump({'class_dict': class_dict}, open(preproc_prq_path.parent / 'class_dict.yaml', 'w'))

    batches = []
    total_batches = (len(clips) + _PREPROC_DASK_BATCH_SIZE - 1) // _PREPROC_DASK_BATCH_SIZE
    for batch_idx in range(total_batches):
        start_idx = batch_idx * _PREPROC_DASK_BATCH_SIZE
        end_idx = min(start_idx + _PREPROC_DASK_BATCH_SIZE, len(clips))
        batches.append(clips[start_idx:end_idx])

    # process clips in batches
    with TqdmCallback(desc="Preprocessing clips in batches"):

        # config
        dask.config.set({"dataframe.convert-string": False})

        # batch processing
        all_data: dd.DataFrame = dd.from_map(
            do_batch, 
            batches, 
            meta=pd.DataFrame({
                "data": pd.Series([], dtype=object),
                "path": pd.Series(dtype="string"),
                "split": pd.Series(dtype="string"),
                "label": pd.Series(dtype="int32"),
                "sample_rate": pd.Series(dtype="int32"),
                }),
            class_dict=class_dict
        )

        # make sure array is serialized correctly (data)
        all_data.to_parquet(preproc_prq_path, write_index=False, schema={'data': pa.list_(pa.int16())})


if __name__ == "__main__":
    """
    run preprocessing
    """

    # load config
    config = load_config()

    # run preprocessing
    run_preprocessing(config)
