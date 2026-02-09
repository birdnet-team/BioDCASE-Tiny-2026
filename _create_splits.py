from pathlib import Path
from typing import Callable, Iterable, Any

import librosa
import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm.dask import TqdmCallback
import dask.dataframe as ddf
import dask


from constants import (
    POS_ORIGINAL_PATH, NEG_ORIGINAL_PATH,
    POS_PARQUET_PATH, NEG_PARQUET_PATH,
    TRAIN_SPLIT_PATH, NEG_TEST_SPLIT_PATH, POS_TEST_SPLIT_PATH
)


POS_MD_SCHEMA = {
    "clip_n": pa.int32(),
    "date": pa.string(),
    "loc": pa.string(),
    "distance": pa.string(),
    "recorder_pos": pa.int32(),
    "audiomoth_n": pa.int32(),
    "test_n": pa.string(),
    "yh_id": pa.string(),
    "phrase_type": pa.string(),
    "song_pos_n": pa.string()
}

NEG_MD_SCHEMA = {
    "clip_n": pa.int32(),
    "neg_mark": pa.string(),
    "neg_n": pa.int32(),
    "sound_file": pa.string()
}


TEST_INDIVIDUALS = {'9', '10'}
NEG_SOUND_FILES_TRAIN_RATIO = 0.8


def _positive_extract_metadata(pos_fname_stems: list[str]):
    md = {k: [] for k in POS_MD_SCHEMA}
    for pos_fname_stem in pos_fname_stems:
        _, clip_n, date, loc, *rest = pos_fname_stem.split("_")
        if "m" in rest[0]:
            distance, recorder_pos, audiomoth_n, test_n, _, yh_id, phrase_type, *song_pos_n = rest
            song_pos_n = "_".join(song_pos_n)
        else:
            *distance, recorder_pos, audiomoth_n, test_n, _, yh_id, phrase_type, song_pos_n = rest
            distance = ".".join(distance)
        md["clip_n"].append(int(clip_n))
        md["date"].append(date)
        md["loc"].append(loc)
        md["distance"].append(".".join(distance))
        md["recorder_pos"].append(int(recorder_pos))
        md["audiomoth_n"].append(int(audiomoth_n))
        md["test_n"].append(test_n)
        md["yh_id"].append(yh_id)
        md["phrase_type"].append(phrase_type)
        md["song_pos_n"].append(song_pos_n)
    return md


def _negative_extract_metadata(neg_fname_stems: list[str]):
    md = {k: [] for k in NEG_MD_SCHEMA}
    for neg_fname_stem in neg_fname_stems:
        parts = neg_fname_stem.split("_")
        _, clip_n, neg_mark, *rest = neg_fname_stem.split('_')
        if neg_mark == "pink":  # pink_noise,
            neg_mark = f"{neg_mark}_{rest[0]}"
            neg_n, *sound_file = rest[1:]
        else:
            neg_n, *sound_file = rest
        md["clip_n"].append(int(clip_n))
        md["neg_mark"].append(neg_mark)
        md["neg_n"].append(int(neg_n))
        md["sound_file"].append("_".join(sound_file))
    return md


def audio_folder_to_pyarrow(
        folder_path: Path,
        output_path: Path,
        metadata_extraction_fn: Callable[[list[str]], dict],
        md_schema: dict[str, Any],
        batch_size: int = 1000
):
    audio_files = [f for f in folder_path.glob("*.wav")]

    samples, expected_sr = librosa.load(audio_files[0], sr=None)
    expected_n_samples = len(samples)

    # process files in batches (avoid popping RAM)
    def do_batch(batch: Iterable[Path]):
        audio_data = []
        file_names = []

        # Process each file in the batch
        for file_path in batch:
            x, sr = librosa.load(str(file_path), sr=None)
            assert sr == expected_sr, f"wrong sampling rate detected at {file_path}"
            assert len(x) == expected_n_samples, f"wrong clip length detected at {file_path}"
            audio_data.append(x)
            file_names.append(file_path.name)
        metadata = metadata_extraction_fn([p.stem for p in batch])

        df = pd.DataFrame({
            'file_name': file_names,
            'data': audio_data,
            'sample_rate': sr,
            **metadata
        })
        return df

    batches = []
    total_batches = (len(audio_files) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(audio_files))
        batches.append(audio_files[start_idx:end_idx])

    with TqdmCallback(desc="Processing batches"):
        all_data = ddf.from_map(do_batch, batches)
        all_data.to_parquet(
            output_path,
            write_index=False,
            schema=pa.schema([
                pa.field('file_name', pa.string()),
                pa.field('data', pa.list_(pa.float32())),
                pa.field('sample_rate', pa.int32()),
            ] + [
                pa.field(name, type) for name, type in md_schema.items()
            ])
        )


if __name__ == '__main__':
    dask.config.set({"dataframe.convert-string": False})
    print("Converting audio folders to parquet")
    audio_folder_to_pyarrow(POS_ORIGINAL_PATH, POS_PARQUET_PATH, _positive_extract_metadata, POS_MD_SCHEMA)
    audio_folder_to_pyarrow(NEG_ORIGINAL_PATH, NEG_PARQUET_PATH, _negative_extract_metadata, NEG_MD_SCHEMA)

    print("Splitting data into train and test sets")
    print("Train...")
    # we don't have lots of data, so let's not do folds. Let's split the data in train / test based on
    # recorded individual (positives) and recording name (negative)
    # split 80 / 20?
    pos_data = ddf.read_parquet(POS_PARQUET_PATH)
    neg_data = ddf.read_parquet(NEG_PARQUET_PATH)

    # save the training data, save them as a single dataset, as metadata doesn't concern the participants
    pos_train = pos_data[~pos_data["yh_id"].isin(TEST_INDIVIDUALS)]
    pos_train["yh"] = True
    pos_train = pos_train[["data", "sample_rate", "yh"]]

    neg_sound_f_df = pd.read_parquet(NEG_PARQUET_PATH, columns=["sound_file"])
    neg_sound_f_ratio = neg_sound_f_df['sound_file'].value_counts(normalize=True)
    neg_sound_f_ratio_items = list(neg_sound_f_ratio.to_dict().items())
    np.random.seed(42)
    np.random.shuffle(neg_sound_f_ratio_items)
    cutoff = np.searchsorted(np.cumsum([x[1] for x in neg_sound_f_ratio_items]), NEG_SOUND_FILES_TRAIN_RATIO)
    neg_sound_f_train = set(
        x[0] for x in neg_sound_f_ratio_items[:cutoff]  # approx 80% of neg data
    )


    neg_train = neg_data[neg_data['sound_file'].isin(neg_sound_f_train)]
    neg_train["yh"] = False
    neg_train = neg_train[["data", "sample_rate", "yh"]]

    all_train = ddf.concat([pos_data, neg_train], interleave_partitions=True)
    all_train.to_parquet(
        TRAIN_SPLIT_PATH,
        write_index=False,
        schema=pa.schema([
            pa.field("data", pa.list_(pa.float32())),
            pa.field("sample_rate", pa.int32()),
            pa.field("yh", pa.bool_()),
        ])
    )

    print("Test...")
    # save the test data. Split it in positive and negative examples,
    # as the positive one have interesting metadata (e.g. distance) we might want to check against predictions
    pos_test = pos_data[pos_data["yh_id"].isin(TEST_INDIVIDUALS)]
    pos_test["yh"] = True
    pos_test.compute().to_parquet(
        POS_TEST_SPLIT_PATH,
        schema=pa.schema([
            pa.field('file_name', pa.string()),
            pa.field('data', pa.list_(pa.float32())),
            pa.field('sample_rate', pa.int32()),
        ] + [
            pa.field(name, type) for name, type in POS_MD_SCHEMA.items()
        ]))

    neg_test = neg_data[~neg_data['sound_file'].isin(neg_sound_f_train)]
    neg_test["yh"] = False
    neg_test.compute().to_parquet(
        NEG_TEST_SPLIT_PATH,
        schema=pa.schema([
            pa.field('file_name', pa.string()),
            pa.field('data', pa.list_(pa.float32())),
            pa.field('sample_rate', pa.int32())
        ] + [
            pa.field(name, type) for name, type in NEG_MD_SCHEMA.items()
        ]))
    print("Done")




