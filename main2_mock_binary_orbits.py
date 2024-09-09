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


def read_mock_paras(mock_paras, snr, index, mu=5.03, sigma=2.28, low_p=-0.7, upper_p=1.2, epoches=2):
    corres_paras = mock_paras[index]

    logage = corres_paras['logage']
    mh = corres_paras['feh']
    teff1 = corres_paras['teff1']
    logg1 = corres_paras['logg1']
    mact1 = corres_paras['mact1']
    R1 = corres_paras['R1']
    # rv1 = corres_paras['rv1']
    alpha_m = corres_paras['afe']

    teff2 = corres_paras['teff2']
    logg2 = corres_paras['logg2']
    mact2 = corres_paras['mact2']
    R2 = corres_paras['R2']
    # rv2 = corres_paras['rv2']
    q = corres_paras['q']
    period = get_period(mu, sigma, low_p, upper_p)
    orbital_paras = getRvsBy_ramdom_Phases(q, 10 ** period * 84600, mact1, e=0, omega=0, phases_num=epoches)
    # rv1_list = orbital_paras[2]
    # rv2_list = orbital_paras[3]
    # gamma_list = orbital_paras[1]
    # phase_list = orbital_paras[-1]
    # P,                                                                                                      gamma.tolist(),        q_dyn, rv1_list, rv2_list, phases.tolist()
    return {'phi': [teff1, teff2, logg1, logg2, mh, alpha_m, R1, R2, snr, period, 2, mact1, mact2, logage, q, orbital_paras[1][0], orbital_paras[2][0], \
                    orbital_paras[3], orbital_paras[4], orbital_paras[5]]}


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

    spec_num = len(phi[-3])

    flux_norm1_array = np.zeros((spec_num, len(wave)), dtype=float)
    flux_norm2_array = np.zeros((spec_num, len(wave)), dtype=float)
    flux_obs_array = np.zeros((spec_num, len(wave)), dtype=float)
    flux_norm_array = np.zeros((spec_num, len(wave)), dtype=float)
    flux_norm_err_array = np.zeros((spec_num, len(wave)), dtype=float)

    for _i in range(spec_num):
        flux_norm1 = sp.predict_one_spectrum_rv((phi[0], phi[2], phi[4], phi[5]), rv=phi[-3][_i])
        flux_norm2 = sp.predict_one_spectrum_rv((phi[1], phi[3], phi[4], phi[5]), rv=phi[-2][_i])

        flux_binary_model = phi[6] ** 2 * flux_norm1 * 10 ** flux_cont1 + phi[7] ** 2 * flux_norm2 * 10 ** flux_cont2
        scale = np.maximum(flux_binary_model / phi[8],
                        np.zeros(
                            len(flux_binary_model)))  # Set snr based on whether the flux_obs is bigger than 0. phi['phi'][10] = corresponding snr

        flux_binary_obs = np.random.normal(loc=flux_binary_model, scale=scale)
        flux_binary_norm, flux_binary_cont = normalize_spectrum_general(wave, flux_binary_obs)
        flux_binary_norm_err = scale / flux_binary_cont

        flux_norm1_array[_i] = flux_norm1
        flux_norm2_array[_i] = flux_norm2
        flux_norm_array[_i] = flux_binary_norm
        flux_norm_err_array[_i] = flux_binary_norm_err
        flux_obs_array[_i] = flux_binary_obs
    return flux_norm_array, flux_norm_err_array, flux_obs_array, flux_norm1_array, flux_norm2_array


# In[getPhasesWithRandomT0]:
def getPhasesWithRandomT0(deltaTs, P):
    ts = np.insert(np.cumsum(deltaTs), 0, 0.) + np.uniform(0, P)  # Random T0
    return (ts % P) / P


# In[getE]:
def getE(e, phase):
    E = 0.
    ranE = [0, 2 * pi]
    while (True):
        E = sum(ranE) / 2.
        r0 = ranE[0] - e * np.sin(ranE[0]) - (2 * pi * phase)
        r1 = ranE[1] - e * np.sin(ranE[1]) - (2 * pi * phase)
        r = E - e * np.sin(E) - (2 * pi * phase)
        if (r0 < r and r < 0):
            ranE[0] = E
        if (0 < r and r < r1):
            ranE[1] = E
        if (abs(r) < 0.0001):
            break
    return E


# In[getTheta]:
def getTheta(e, phase):
    if (phase < 0):
        phase += 1
    if (e == 0.):
        E = 2 * pi * phase
    else:
        E = getE(e, phase)
    factor = (np.cos(E) - e) / (1 - e * np.cos(E))
    theta = np.arccos(factor)
    if (E >= pi):
        theta = - theta
    return theta


# In[getRv]:
def getRv(i, q, P, m1, e, omega, theta):
    part1 = 2. * pi * np.sin(i) / (P * np.sqrt(1 - e ** 2))
    a = ((G * m1 * (1 + q) * P ** 2) / (4 * pi ** 2)) ** (1. / 3.)
    a1 = a / (1. + 1. / q)
    part3 = np.cos(theta + omega)
    part4 = e * np.cos(omega)
    return part1 * a1 * (part3 + part4)


# In[getRvMaxAndMin]:
def getRvMaxAndMin(i, q, P, m1, e, omega):
    part1 = 2. * pi * np.sin(i) / (P * np.sqrt(1 - e ** 2))
    a = ((G * m1 * (1 + q) * P ** 2) / (4 * pi ** 2)) ** (1. / 3.)
    a1 = a / (1. + 1. / q)
    part4 = e * np.cos(omega)
    return part1 * a1 * (part4 + 1), part1 * a1 * (part4 - 1)


def getK1K2(i, q, P, m1, e):
    part1 = 2. * pi * np.sin(i) / (P * np.sqrt(1 - e ** 2))
    a = ((G * m1 * (1 + q) * P ** 2) / (4 * pi ** 2)) ** (1. / 3.)
    a1 = a / (1. + 1. / q)
    a2 = a - a1
    return abs(part1 * a1), abs(part1 * a2)


# In[getRvsByPhases]:
def getRvsByPhases(i, q, P, m1, e, omega, phases):
    rv1List = []
    part1 = 2.*pi*np.sin(i) / (P*np.sqrt(1-e**2))
    a = ( (G*m1*(1+q)*P**2)/(4*pi**2) )**(1./3.)
    a1 = a / (1.+1./q)
    part4 = e*np.cos(omega)
    factor1 = part1 * a1
    for ph in phases:
        theta = getTheta(e, ph)
        part3 = np.cos(theta + omega)
        rv1List.append(factor1 * (part3 + part4))
    rv1List = np.array(rv1List)
    rv2List = -np.array(rv1List)*m1/(m1*q)
    return rv1List, rv2List


def getRvsBy_ramdom_Phases(q, P, m1, e=0, omega=0, phases_num=10, i=np.array([pi/2])):
    phases = np.random.uniform(0.0, 1.0, phases_num)
    gamma = np.random.normal(0, 70, 1)
    rv1s = []
    rv2s = []
    for ph in phases:
        rv1, rv2 = getRvsByPhases(i, q, P, m1, e, omega, [ph])
        rv1s.append(rv1[0][0])
        rv2s.append(rv2[0][0])
    q_dyn = (-(np.array(rv1s)/np.array(rv2s)))
    rv1s_obs = rv1s + gamma
    rv2s_obs = gamma + (gamma - rv1s_obs) / q_dyn

    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
    return [P, gamma.tolist(), [q_dyn[-1]], rv1s_obs, rv2s_obs, phases.tolist()]
    # rv1s_obs是多次观测的主星速度，rv2s_obs是多次观测的次星速度。

def get_period(mu, sigma, low_p, upper_p):
    s = np.random.normal(mu, sigma, 10000)
    select_s = s[(s>low_p) & (s<upper_p)]
    return select_s[0]


if __name__ == "__main__":
    '''
    load models
    '''
    workdir = '/Users/liujunhui/PycharmProjects/LamostBinary/'
    sp = joblib.load(workdir + 'sp2022_05_29_13_09_44_Step_64013_1.dmp')
    mist_model = joblib.load(workdir + 'mist_eep_202_454_teff_3500_8000_logt_6.83_10.26_mh_-2.1_0.7_gaia_2mass.dump')
    sp_mist, colname_input, colname_output, acc, sp_val = mist_model
    regli = joblib.load(workdir + 'sp2022_05_29_11_07_00_Step_26468_1regli_conti_in_log.dmp')

    wave = np.arange(3950, 5750, 1.)
    sp.wave = wave

    '''
    load data
    '''
    mock_data_file = '/Users/liujunhui/Desktop/2021workMac/202111012totallynew/cnn_binary_mock_data_v22_tgm.fits'
    mock_paras = fits.getdata(mock_data_file)
    mock_paras = mock_paras[
        (mock_paras['teff1'] > 3800) & (mock_paras['teff1'] < 7800) & (mock_paras['teff2'] < 7800) & (
                mock_paras['teff2'] > 3800) & (mock_paras['logg1'] > 3) & (mock_paras['logg2'] > 3) & (
                mock_paras['teff2'] < mock_paras['teff1'])]
    expdistri = np.random.exponential(scale=1 / (1.47 * 10 ** -2), size=500000)
    expdistri_30 = expdistri[expdistri > 30]
    #  set the expdistri = [10000,...., 10000]
    # expdistri_10000 = np.zeros(500000)+10000  #   2022.06.29 set the snr to be 10000

    samplesize = 10000
    random_index = random.sample(range(0, len(mock_paras)), samplesize)

    set_spec_num = 2

    params = []
    mock_flux_binary = np.zeros((samplesize, set_spec_num, len(wave)), dtype=float)
    mock_flux_err_binary = np.zeros((samplesize, set_spec_num, len(wave)), dtype=float)
    mock_flux_norm_Star1 = np.zeros((samplesize, set_spec_num, len(wave)), dtype=float)
    mock_flux_norm_Star2 = np.zeros((samplesize, set_spec_num, len(wave)), dtype=float)

    for i in range(len(random_index)):
        one_mock_paras = read_mock_paras(mock_paras, expdistri_30[random_index[i]], random_index[i], epoches=set_spec_num)

        ####################### add noise to flux ###################
        flux_binary_norm, flux_binary_norm_err, flux_binary_obs, flux_norm1, flux_norm2 = model_mock_binary(one_mock_paras['phi'],
                                                                                                            sp, regli,
                                                                                                            wave)

        # plt.plot(wave, flux_binary_norm[0], '-')
        # plt.plot(wave, flux_norm1[0]+0.6, '-')
        # plt.plot(wave, flux_norm2[0]+0.3, '-')
        # plt.show()
        params.append(one_mock_paras['phi'])
        # print(one_mock_paras['phi'])
        mock_flux_binary[i] = flux_binary_norm
        mock_flux_err_binary[i] = flux_binary_norm_err
        mock_flux_norm_Star1[i] = flux_norm1
        mock_flux_norm_Star2[i] = flux_norm2
        # plt.plot(wave, flux_binary_norm[0])
        # plt.plot(wave, flux_binary_norm_err[0])
        # plt.show()

    b = {'params': params, 'mock_flux_binary': mock_flux_binary, 'mock_flux_err_binary': mock_flux_err_binary,
         'mock_flux_norm_Star1': mock_flux_norm_Star1, 'mock_flux_norm_Star2': mock_flux_norm_Star2}
    # dump(b, './20220814_mock_binary_elu_correct_rv_logg_3_Teff_3800_7800_2_epoches.dump')