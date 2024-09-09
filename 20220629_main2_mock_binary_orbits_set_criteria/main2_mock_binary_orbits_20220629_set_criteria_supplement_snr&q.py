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
To set the random snr based on the LAMOST snr distribution.
After the generation of the whole sample, I exchange the snr 100 to random snr values.
'''

G = 1.327458213e11  # Msun^-1 (km/s)^2
pi = np.pi

if __name__ == "__main__":
    whole_parameter_path = '20220714_rdm_snr_rdm_q_mh-1_am0.4_mock_binary_whole_parameters_criteria.csv'
    whole_parameter = pd.read_csv(whole_parameter_path)

    expdistri = np.random.exponential(scale=1 / (1.47 * 10 ** -2), size=500000)
    expdistri_30 = expdistri[expdistri > 30]
    # expdistri_30 = np.zeros(30000) + 100

    random_index = random.sample(range(0, len(expdistri_30)), 10000)

    '''
    Function 1
    '''
    # Only add snr, the predecessor file likes "20220712_snr_100_random_q_mock_binary_whole_parameters_criteria.csv"
    for _i in range(len(whole_parameter['snr'])):
        whole_parameter['snr'][_i] = round(expdistri_30[_i], 2)

    '''
    Function 2
    '''
    # Add snr, m2, teff1, teff2, rv1_obs and rv2_obs,
    # and the predecessor file likes "20220712_snr_100_random_q_mock_binary_whole_parameters_criteria.csv"

    # Load mist
    # workdir = '/Users/liujunhui/PycharmProjects/LamostBinary/'
    # mist_model = joblib.load(workdir + 'mist_eep_202_454_teff_3500_8000_logt_6.83_10.26_mh_-2.1_0.7_gaia_2mass.dump')
    # sp_mist, colname_input, colname_output, acc, sp_val = mist_model
    #
    # for _i in range(int(len(whole_parameter['snr']) / 3)):
    #     q = round((np.random.uniform(0.4, 1.0, 1)[0]), 2)
    #     for _j in [0, 1, 2]:
    #         index = _i * 3 + _j
    #         mact1 = whole_parameter['mact1'][index]
    #         logage = whole_parameter['logage'][index]
    #         mh = -1.0
    #         alpha_m = 0.4
    #         whole_parameter['mh'][index] = mh
    #         whole_parameter['alpha_m'][index] = alpha_m
    #         teff1, logg1, R1, *other_output1 = sp_mist.predict([mact1, logage, mh])
    #         teff2, logg2, R2, *other_output2 = sp_mist.predict([mact1 * q, logage, mh])
    #
    #         whole_parameter['mact2'][index] = round(mact1 * q, 2)
    #         whole_parameter['teff1'][index] = round(teff1, 2)
    #         whole_parameter['teff2'][index] = round(teff2, 2)
    #         whole_parameter['logg1'][index] = round(logg1, 2)
    #         whole_parameter['logg2'][index] = round(logg2, 2)
    #         whole_parameter['R1'][index] = round(R1, 2)
    #         whole_parameter['R2'][index] = round(R2, 2)
    #
    #         whole_parameter['q'][index] = q
    #         whole_parameter['snr'][index] = round(expdistri_30[_i], 2)
    #         whole_parameter['gamma'][index] = 0
    #         if index % 3 == 0:
    #             whole_parameter['rv1_obs'][index] = 0
    #             whole_parameter['rv2_obs'][index] = 0
    #         elif index % 3 == 1:
    #             whole_parameter['rv1_obs'][index] = round(100 - (100 / (1 + q)), 2)
    #             whole_parameter['rv2_obs'][index] = round(-(100 / (1 + q)), 2)
    #         elif index % 3 == 2:
    #             whole_parameter['rv1_obs'][index] = round(200 - (200 / (1 + q)), 2)
    #             whole_parameter['rv2_obs'][index] = round(-(200 / (1 + q)), 2)

    whole_parameter.to_csv('./20220714_rdm_snr_rdm_q_mh-1_am0.4_mock_binary_whole_parameters_criteria.csv', index=False)
