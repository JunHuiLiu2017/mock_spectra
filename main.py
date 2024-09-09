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


def read_mock_paras(mock_paras, snr, index):
    corres_paras = mock_paras[index]

    logage = corres_paras['logage']
    mh = corres_paras['feh']
    teff1 = corres_paras['teff1']
    logg1 = corres_paras['logg1']
    mact1 = corres_paras['mact1']
    R1 = corres_paras['R1']
    rv1 = corres_paras['rv1']
    alpha_m = corres_paras['afe']

    teff2 = corres_paras['teff2']
    logg2 = corres_paras['logg2']
    mact2 = corres_paras['mact2']
    R2 = corres_paras['R2']
    rv2 = corres_paras['rv2']
    q = corres_paras['q']

    return {'phi': [teff1, teff2, logg1, logg2, mh, alpha_m, R1, R2, rv1, rv2, snr, 2, mact1, mact2, logage, q]}


def model_mock_binary2(phi, sp, regli, wave):
    '''
    'phi': [teff1, teff2, logg1, logg2, mh, alpha_m, R1, R2, rv1, rv2, snr]

    Args:
        phi:
        sp:
        sp_mist:
        regli:
        spec_num:
        wave:

    Returns:

    '''

    # continuum interpolator (Regli)
    flux_cont1 = regli.predict((phi[0], phi[2], phi[4], phi[5]))
    flux_cont2 = regli.predict((phi[1], phi[3], phi[4], phi[5]))
    # plt.plot(wave, flux_cont1, 'r-')
    # plt.plot(wave, flux_cont2, 'b-')
    # plt.show()

    # flux_binary = flux_norm1 * flux_cont1 + flux_norm2 * flux_cont2
    # normalized flux

    flux_norm1 = sp.predict_one_spectrum_rv((phi[0], phi[2], phi[4], phi[5]), rv=phi[8])
    flux_norm2 = sp.predict_one_spectrum_rv((phi[1], phi[3], phi[4], phi[5]), rv=phi[9])
    # plt.plot(wave, flux_norm1, 'r-')
    # plt.plot(wave, flux_norm2, 'b-')
    # plt.show()

    flux_binary_obs = phi[6] ** 2 * flux_norm1 * 10 ** flux_cont1 + phi[7] ** 2 * flux_norm2 * 10 ** flux_cont2
    flux_binary_norm, flux_binary_cont = normalize_spectrum_general(wave, flux_binary_obs)
    return flux_binary_norm, flux_binary_obs, flux_norm1, flux_norm2


def model_mock_binary(phi, sp, regli, wave):
    '''
    'phi': [teff1, teff2, logg1, logg2, mh, alpha_m, R1, R2, rv1, rv2, snr]

    Args:
        phi:
        sp:
        sp_mist:
        regli:
        spec_num:
        wave:

    Returns:

    '''

    flux_cont1 = regli.predict((phi[0], phi[2], phi[4], phi[5]))
    flux_cont2 = regli.predict((phi[1], phi[3], phi[4], phi[5]))

    flux_norm1 = sp.predict_one_spectrum_rv((phi[0], phi[2], phi[4], phi[5]), rv=phi[8])
    flux_norm2 = sp.predict_one_spectrum_rv((phi[1], phi[3], phi[4], phi[5]), rv=phi[9])

    flux_binary_model = phi[6] ** 2 * flux_norm1 * 10 ** flux_cont1 + phi[7] ** 2 * flux_norm2 * 10 ** flux_cont2
    scale = np.maximum(flux_binary_model / phi[10],
                       np.zeros(len(flux_binary_model)))  # Set snr based on whether the flux_obs is bigger than 0. phi['phi'][10] = corresponding snr

    flux_binary_obs = np.random.normal(loc=flux_binary_model, scale=scale)

    flux_binary_norm, flux_binary_cont = normalize_spectrum_general(wave, flux_binary_obs)
    flux_binary_norm_err = scale / flux_binary_cont

    return flux_binary_norm, flux_binary_norm_err, flux_binary_obs, flux_norm1, flux_norm2





if __name__ == "__main__":
    '''
    load models
    '''
    workdir = '/Users/liujunhui/PycharmProjects/LamostBinary/'
    sp = joblib.load(workdir + 'sp2021_12_28_23_48_41_Step_62577_1.dmp')
    mist_model = joblib.load(workdir + 'mist_eep_202_454_teff_3500_8000_logt_6.83_10.26_mh_-2.1_0.7_gaia_2mass.dump')
    sp_mist, colname_input, colname_output, acc, sp_val = mist_model
    regli = joblib.load(workdir + 'sp2022_02_12_10_49_58_Step_26468_1regli_conti_in_log.dmp')

    wave = np.arange(3950, 5750, 1.)
    sp.wave = wave

    '''
    load data
    '''
    mock_data_file = '/Users/liujunhui/Desktop/2021workMac/202111012totallynew/cnn_binary_mock_data_v22_tgm.fits'
    mock_paras = fits.getdata(mock_data_file)
    mock_paras = mock_paras[
        (mock_paras['teff1'] > 4000) & (mock_paras['teff1'] < 7000) & (mock_paras['teff2'] < 7000) & (
                    mock_paras['teff2'] > 4000) & (mock_paras['logg1'] > 3) & (mock_paras['logg2'] > 3) & (
                mock_paras['teff2'] < mock_paras['teff1'])]
    expdistri = np.random.exponential(scale=1 / (1.47 * 10 ** -2), size=500000)
    expdistri_30 = expdistri[expdistri > 30]

    print(expdistri_30)

    samplesize = 10000
    random_index = random.sample(range(0, len(mock_paras)), samplesize)

    params = np.zeros((samplesize, 16), dtype=float)
    mock_flux_binary = np.zeros((samplesize, len(wave)), dtype=float)
    mock_flux_err_binary = np.zeros((samplesize, len(wave)), dtype=float)
    for i in range(len(random_index)):
        phi = read_mock_paras(mock_paras, expdistri_30[random_index[i]], random_index[i])
        ##### add noise to norm flux
        # flux_binary_norm, flux_binary_obs, flux_norm1, flux_norm2 = model_mock_binary(phi['phi'], sp, regli, wave)
        # scale = np.maximum(flux_binary_norm / phi['phi'][10],
        #                    np.zeros(len(flux_binary_norm)))  # Set snr based on whether the flux_obs is bigger than 0. phi['phi'][10] = corresponding snr
        # flux_obs_snr = np.random.normal(loc=flux_binary_norm, scale=scale)
        # params[i] = phi['phi']
        # mock_flux_binary[i] = flux_obs_snr
        # mock_flux_err_binary[i] = scale
        ##### add noise to norm flux

        ####################### add noise to flux ###################
        flux_binary_norm, flux_binary_norm_err, flux_binary_obs, flux_norm1, flux_norm2 = model_mock_binary(phi['phi'], sp, regli, wave)
        params[i] = phi['phi']
        mock_flux_binary[i] = flux_binary_norm
        mock_flux_err_binary[i] = flux_binary_norm_err
    b = {'params': params, 'mock_flux_binary': mock_flux_binary, 'mock_flux_err_binary': mock_flux_err_binary}
    dump(b, './mock_binary_data_logg_3_Teff_4000_7000.dump')



