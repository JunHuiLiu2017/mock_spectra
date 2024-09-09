import random
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from laspec import mrs
from scipy.optimize import least_squares
from laspec.normalization import normalize_spectrum_general
import time
import joblib
from joblib import dump, load
'''
To obtain the delta chi2 from binary spectra subtract single spectra.
'''

G = 1.327458213e11  # Msun^-1 (km/s)^2
pi = np.pi




if __name__ == "__main__":
    mock_binary_data_path = './20220708_mock_binary_spectra_for_criteria.dump'
    mock_binary_date = load(mock_binary_data_path)
    mock_flux_binarys = mock_binary_date['mock_flux_binary']
    mock_flux_err_binarys = mock_binary_date['mock_flux_err_binary']
    mock_flux_norm_Star1s = mock_binary_date['mock_flux_norm_Star1']
    mock_flux_norm_Star2s = mock_binary_date['mock_flux_norm_Star2']

    print(len(mock_binary_date['params']))
    for _i in range(63):
        chi2 = np.log10(np.sqrt(np.mean(((mock_flux_binarys[_i] - mock_flux_norm_Star1s[_i]) / mock_flux_err_binarys[_i]) ** 2)))
        print(chi2)
    # b = {'params': params, 'mock_flux_binary': mock_flux_binary, 'mock_flux_err_binary': mock_flux_err_binary,
    #      'mock_flux_norm_Star1': mock_flux_norm_Star1, 'mock_flux_norm_Star2': mock_flux_norm_Star2}
    # dump(b, '20220708_mock_binary_spectra_for_criteria.dump')
