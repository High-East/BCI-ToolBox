from typing import List, Union

import numpy as np
from mne.filter import resample
from torch.utils.data import Dataset
from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.preprocess import Preprocessor
from braindecode.datautil.preprocess import preprocess
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.windowers import create_windows_from_events

from utils.utils import print_off, print_on, AttrDict


class BNCI20142a(Dataset):
    def __init__(
            self,
            subject: Union[int, List] = None,
            preproces_params: Union[dict, AttrDict] = None,
            phase: str = 'train',
            get_ch_names: bool = False
    ):
        if preproces_params is None:
            preproces_params = dict(
                band=[[0, 40]],
                start_time=-0.5
            )

        x_bundle, y_bundle = [], []
        for (low_hz, high_hz) in preproces_params['band']:
            x_list = []
            y_list = []

            if isinstance(subject, int):
                subject = [subject]

            print_off()

            # Load data from MOABBDataset
            dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=subject)

            # Preprocess data
            factor_new = 1e-3
            init_block_size = 1000

            preprocessors = [
                # Keep only EEG sensors
                Preprocessor(fn='pick_types', eeg=True, meg=False, stim=False, apply_on_array=True),
                # Convert from volt to microvolt
                Preprocessor(fn=lambda x: x * 1e+06, apply_on_array=True),
                # Apply bandpass filtering
                Preprocessor(fn='filter', l_freq=low_hz, h_freq=high_hz, apply_on_array=True),
                # Apply exponential moving standardization
                Preprocessor(fn=exponential_moving_standardize, factor_new=factor_new,
                             init_block_size=init_block_size, apply_on_array=True)
            ]
            preprocess(dataset, preprocessors)

            # Check sampling frequency
            sfreq = dataset.datasets[0].raw.info['sfreq']
            if not all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets]):
                raise ValueError("Not match sampling rate.")

            # Divide data by trial
            trial_start_offset_samples = int(preproces_params['start_time'] * sfreq)

            windows_dataset = create_windows_from_events(
                dataset,
                trial_start_offset_samples=trial_start_offset_samples,
                trial_stop_offset_samples=0,
                preload=True
            )
            print_on()

            # Make session-to-session data (subject dependent)
            if preproces_params['is_session']:
                if phase == 'train':
                    for trial in windows_dataset.split('session')['session_T']:
                        x_list.append(trial[0])
                        y_list.append(trial[1])
                else:
                    for trial in windows_dataset.split('session')['session_E']:
                        x_list.append(trial[0])
                        y_list.append(trial[1])

            # Return numpy array
            x_list = np.array(x_list)
            y_list = np.array(y_list)

            # Cut time points
            if preproces_params['end_time'] is not None:
                len_time = preproces_params['end_time'] - preproces_params['start_time']
                x_list = x_list[..., : int(len_time * sfreq)]

            # Resampling
            if preproces_params['resampling'] is not None:
                x_list = resample(np.array(x_list, dtype=np.float64), preproces_params['resampling'] / sfreq)

            x_bundle.append(x_list)
            y_bundle.append(y_list)

        self.x = np.stack(x_bundle, axis=1)
        self.y = np.array(y_bundle[0])

        if get_ch_names:
            self.ch_names = windows_dataset.datasets[0].windows.info.ch_names

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = [self.x[idx], self.y[idx]]
        return sample


class BNCI20142b(Dataset):
    def __init__(
            self,
            subject: Union[int, List] = None,
            preproces_params: Union[dict, AttrDict] = None,
            phase: str = 'train',
            get_ch_names: bool = False
    ):
        if preproces_params is None:
            preproces_params = AttrDict(
                band=[[0, 40]],
                start_time=-0.5
            )

        x_bundle, y_bundle = [], []
        for (low_hz, high_hz) in preproces_params['band']:
            x_list = []
            y_list = []

            if isinstance(subject, int):
                subject = [subject]

            print_off()

            # Load data from MOABBDataset
            dataset = MOABBDataset(dataset_name="BNCI2014004", subject_ids=subject)

            # Preprocess data
            factor_new = 1e-3
            init_block_size = 1000

            preprocessors = [
                # Keep only EEG sensors
                Preprocessor(fn='pick_types', eeg=True, meg=False, stim=False, apply_on_array=True),
                # Convert from volt to microvolt
                Preprocessor(fn=lambda x: x * 1e+06, apply_on_array=True),
                # Apply bandpass filtering
                Preprocessor(fn='filter', l_freq=low_hz, h_freq=high_hz, apply_on_array=True),
                # Apply exponential moving standardization
                Preprocessor(fn=exponential_moving_standardize, factor_new=factor_new,
                             init_block_size=init_block_size, apply_on_array=True)
            ]
            preprocess(dataset, preprocessors)

            # Check sampling frequency
            sfreq = dataset.datasets[0].raw.info['sfreq']
            if not all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets]):
                raise ValueError("Not match sampling rate.")

            # Divide data by trial
            trial_start_offset_samples = int(preproces_params['start_time'] * sfreq)

            windows_dataset = create_windows_from_events(
                dataset,
                trial_start_offset_samples=trial_start_offset_samples,
                trial_stop_offset_samples=0,
                preload=True
            )
            print_on()

            # Make session-to-session data (subject dependent)
            if preproces_params['is_session']:
                if phase == 'train':
                    for session in ['session_0', 'session_1', 'session_2']:
                        for trial in windows_dataset.split('session')[session]:
                            x_list.append(trial[0])
                            y_list.append(trial[1])
                else:
                    for session in ['session_3', 'session_4']:
                        for trial in windows_dataset.split('session')[session]:
                            x_list.append(trial[0])
                            y_list.append(trial[1])

            # Return numpy array
            x_list = np.array(x_list)
            y_list = np.array(y_list)

            # Cut time points
            if preproces_params['end_time'] is not None:
                len_time = preproces_params['end_time'] - preproces_params['start_time']
                x_list = x_list[..., : int(len_time * sfreq)]

            # Resampling
            if preproces_params['resampling'] is not None:
                x_list = resample(np.array(x_list, dtype=np.float64), preproces_params['resampling'] / sfreq)

            x_bundle.append(x_list)
            y_bundle.append(y_list)

        self.x = np.stack(x_bundle, axis=1)
        self.y = np.array(y_bundle[0])

        if get_ch_names:
            self.ch_names = windows_dataset.datasets[0].windows.info.ch_names

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = [self.x[idx], self.y[idx]]
        return sample
