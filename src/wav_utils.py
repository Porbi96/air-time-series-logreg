import os
from typing import List

import skfda
from scipy.io import wavfile

def create_wav_matrix_raw(directory: str) -> List[List[int]]:
    if not os.path.isdir(directory):
        raise NotADirectoryError('Directory not found')

    data_matrix = []
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            _, data = wavfile.read(os.path.join(directory, filename))
            data_matrix.append(data)
    return data_matrix


def create_wav_fdatagrid(directory):
    if not os.path.isdir(directory):
        raise NotADirectoryError('Directory not found')

    samplerate = None
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            samplerate, _ = wavfile.read(os.path.join(directory, filename))
            break

    if samplerate:
        grid_points = [x for x in range(0, samplerate)]
        data_matrix = create_wav_matrix_raw(directory)
    else:
        raise FileNotFoundError('There are no WAV files inside given directory.')

    return skfda.FDataGrid(
        data_matrix=data_matrix,
        grid_points=grid_points,
    )