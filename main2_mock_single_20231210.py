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


"""
Teff: 3500--13700
logg: 0.13--5.0
M/H: -2--0.5
A/M: -0.5--0.7

logage: 6.6--12.4
rv1: -22-24
rv2: -515--515
m1: 0.13--2.0
m2: 0.13--2.0
"""

'''
teff: 3500--10000
logg: 0--5
m/h: -2.5--0.5
a/m: -1--1

(all["ASPCAPFLAG"]==0)&
(all["SNR"]>30)&
(all["combined_snrg"]>30)&
(all["TEFF"]>3000)&
(all["TEFF"]<8000)&
(all["LOGG"]>-1)&
(all["M_H"]>-2.0)

！！！！！！！！！！！2022.04.08 the logg be set to higher than 3.
'''
G = 1.327458213e11  # Msun^-1 (km/s)^2
pi = np.pi


def read_mock_paras(mock_paras, snr, index):
    corres_paras = mock_paras[index]

    logage = corres_paras['logage']
    mh = corres_paras['feh']
    teff1 = corres_paras['teff1']
    logg1 = corres_paras['logg1']
    mact1 = corres_paras['mact1']
    R1 = corres_paras['R1']
    alpha_m = corres_paras['afe']

    return {'phi': [teff1,  logg1, mh, alpha_m, R1, snr, mact1, logage]}


def model_mock_single(phi, r, wave):
    '''
    'phi' [Teff, logg, mh, alpha_m, R1, snr]
    Args:
        phi:
        sp:
        sp_mist:
        regli:
        spec_num:
        wave:

    Returns:

    '''

    flux_single_model = r.predict_spectrum([phi[0], phi[1], phi[2], phi[3]], wave_new = wave)
    scale = np.maximum(flux_single_model / phi[5],
                       np.zeros(len(flux_single_model)))  # Set snr based on whether the flux_obs is bigger than 0. phi['phi'][10] = corresponding snr

    flux_single_obs = np.random.normal(loc=flux_single_model, scale=scale)

    flux_single_norm, flux_single_cont = normalize_spectrum_general(wave, flux_single_obs)
    flux_single_norm_err = scale / flux_single_cont

    return flux_single_norm, flux_single_norm_err, flux_single_obs


if __name__ == "__main__":
    '''
    load models
    '''
    workdir = '/Users/liujunhui/PycharmProjects/LamostBinary/'
    sp = joblib.load(workdir + 'sp2021_12_28_23_48_41_Step_62577_1.dmp')
    mist_model = joblib.load(workdir + 'mist_eep_202_454_teff_3500_8000_logt_6.83_10.26_mh_-2.1_0.7_gaia_2mass.dump')
    sp_mist, colname_input, colname_output, acc, sp_val = mist_model
    # regli = joblib.load(workdir + 'sp2022_02_12_10_49_58_Step_26468_1regli_conti_in_log.dmp')
    r = joblib.load('./R1800_regli.dump')

    wave = np.arange(3950, 5750, 1.)
    sp.wave = wave

    '''
    load data
    '''
    mock_data_file = '/Users/liujunhui/Desktop/2021workMac/202111012totallynew/cnn_binary_mock_data_v22_tgm.fits'
    mock_paras = fits.getdata(mock_data_file)
    mock_paras = mock_paras[
        (mock_paras['teff1'] > 3500) & (mock_paras['teff1'] < 9000) & (mock_paras['logg1'] > 3.5)]
    expdistri = np.random.exponential(scale=1 / (1.47 * 10 ** -2), size=500000)
    expdistri_30 = expdistri[expdistri > 30]

    samplesize = 10000
    random_index = random.sample(range(0, len(mock_paras)), samplesize)

    Set_spec_num = 1
    params = []
    mock_flux_single = np.zeros((samplesize, Set_spec_num, len(wave)), dtype=float)
    mock_flux_err_single = np.zeros((samplesize, Set_spec_num, len(wave)), dtype=float)
    T = []
    for i in range(len(random_index)):
        one_mock_paras = read_mock_paras(mock_paras, expdistri_30[random_index[i]], random_index[i])

        ####################### add noise to flux ###################
        flux_single_norm, flux_single_norm_err, flux_single_obs = model_mock_single(
            one_mock_paras['phi'], r, wave)
        params.append(one_mock_paras['phi'])
        mock_flux_single[i] = flux_single_norm
        mock_flux_err_single[i] = flux_single_norm_err
        T.append(one_mock_paras['phi'][0])

        # print(one_mock_paras['phi'])
        # print(mock_flux_single[i])
        # print(mock_flux_err_single[i])
        print(i)
    plt.hist(T)
    plt.show()
    b = {'params': params, 'mock_flux_binary': mock_flux_single, 'mock_flux_err_binary': mock_flux_err_single}
    dump(b, './20231210_mock_binary_data_logg_35_Teff_3500_9000_1_epoche_SingleStar.dump')


