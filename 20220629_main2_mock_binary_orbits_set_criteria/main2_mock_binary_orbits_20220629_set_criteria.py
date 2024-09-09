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
!!!!!!!!!!!!!!!!!2022.06.29 make the mock binary spectra to get the criteria.
'''

G = 1.327458213e11  # Msun^-1 (km/s)^2
pi = np.pi


def generate_pre_paras(mock_paras_database, database_filter, empty_dataframe, loc_index, mu=5.03, sigma=2.28, low_p=-0.7,
                       upper_p=1.2):
    '''
    To generate the pre parameters only hold mact1, logage, mh, alpha_m, q and period.
    Args:
        mock_paras:
        snr:
        index:
        mu:
        sigma:
        low_p:
        upper_p:
        epoches:

    Returns:

    '''

    # Step 2: load the database of parameters, and select one with random index in the range of range(0, len(mock_paras[filter]).
    samplesize = 1
    random_index = random.sample(range(0, len(mock_paras_database[database_filter])), samplesize)
    corres_paras = mock_paras_database[database_filter][random_index[0]]

    # Step 3: assemble the df. Note: 'round' to retain the specified number of the significant digits.
    empty_dataframe.loc[loc_index] = [round(corres_paras['mact1'], 1), corres_paras['logage'], corres_paras['feh'],
                                      corres_paras['afe'], round(corres_paras['q'], 1), get_period(mu, sigma, low_p, upper_p)]
    return empty_dataframe


def corresponding_criteria(mock_paras_database, mact1_lower, mact1_upper):
    criteria = (mock_paras_database['teff1'] > 4000) & (mock_paras_database['teff1'] < 7000) & (
                mock_paras_database['teff2'] < 7000) & (
                       mock_paras_database['teff2'] > 4000) & (mock_paras_database['logg1'] > 3) & (
                           mock_paras_database['logg2'] > 3) & (
                       mock_paras_database['teff2'] < mock_paras_database['teff1']) & (
                           mock_paras_database['q'] > 0.99) & (mock_paras_database['mact1'] > mact1_lower) & (
                       mock_paras_database['mact1'] < mact1_upper) & (mock_paras_database['logage'] < 10)
    return criteria


# def add_rv(dataframe):
#     mact1 = dataframe['mact1']
#     logage = dataframe['logage']
#     mh = dataframe['feh']
#     alpha_m = dataframe['afe']
#     q = dataframe['q']
#     period = dataframe['period']
#     orbital_paras = getRvsBy_delta_RV(q, 10 ** period * 84600, mact1, e=0, omega=0)

def generate_whole_mock_paras(pre_dataframe, empty_dataframe, index, sp_mist, snr=10000):
    corres_paras = pre_dataframe

    mact1 = corres_paras['mact1'][index]
    logage = corres_paras['logage'][index]
    mh = corres_paras['feh'][index]
    alpha_m = corres_paras['afe'][index]
    q = corres_paras['q'][index]
    period = corres_paras['period'][index]
    mact2 = mact1*q

    teff1, logg1, R1, *other_output = sp_mist.predict([mact1, logage, mh])
    teff2, logg2, R2, *other_output = sp_mist.predict([mact1*q, logage, mh])

    orbital_paras = getRvsBy_delta_RV(q, 10 ** period * 84600, mact1, e=0, omega=0)

    for _i in range(len(orbital_paras[3])):
        gamma = orbital_paras[1][0]
        rv1_obs = orbital_paras[3][_i]
        rv2_obs = orbital_paras[4][_i]
        single_whole_paras = [round(teff1, 2), round(teff2, 2), round(logg1, 2), round(logg2, 2), round(mh, 2),
                              round(alpha_m, 2), round(R1, 2), round(R2, 2), snr, round(period, 2), 1,
                              round(mact1, 2), round(mact2, 2), round(logage, 2), round(q, 2),
                              round(gamma, 2), round(rv1_obs, 2), round(rv2_obs, 2)]
        empty_dataframe.loc[len(empty_dataframe)] = single_whole_paras # Based on the length of empty_dataframe, to set the index of loc.
    return empty_dataframe


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

    spec_num = 1

    flux_norm1_array = np.zeros((spec_num, len(wave)), dtype=float)
    flux_norm2_array = np.zeros((spec_num, len(wave)), dtype=float)
    flux_obs_array = np.zeros((spec_num, len(wave)), dtype=float)
    flux_norm_array = np.zeros((spec_num, len(wave)), dtype=float)
    flux_norm_err_array = np.zeros((spec_num, len(wave)), dtype=float)

    for _i in range(spec_num):
        flux_norm1 = sp.predict_one_spectrum_rv((phi[0], phi[2], phi[4], phi[5]), rv=phi[-2])
        flux_norm2 = sp.predict_one_spectrum_rv((phi[1], phi[3], phi[4], phi[5]), rv=phi[-1])

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
    ts = np.insert(np.cumsum(deltaTs), 0, 0.) + ran.uniform(0, P)  # Random T0
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
    part1 = 2. * pi * np.sin(i) / (P * np.sqrt(1 - e ** 2))
    a = ((G * m1 * (1 + q) * P ** 2) / (4 * pi ** 2)) ** (1. / 3.)
    a1 = a / (1. + 1. / q)
    part4 = e * np.cos(omega)
    factor1 = part1 * a1
    for ph in phases:
        theta = getTheta(e, ph)
        part3 = np.cos(theta + omega)
        rv1List.append(factor1 * (part3 + part4))
    rv1List = np.array(rv1List)
    rv2List = -np.array(rv1List) * m1 / (m1 * q)
    return rv1List, rv2List


def getRvsBy_ramdom_Phases(q, P, m1, e=0, omega=0, phases_num=10, i=np.array([pi / 2])):
    phases = np.random.uniform(0.0, 1.0, phases_num)
    gamma = np.random.normal(0, 70, 1)
    rv1s = []
    rv2s = []
    for ph in phases:
        rv1, rv2 = getRvsByPhases(i, q, P, m1, e, omega, [ph])
        rv1s.append(rv1[0][0])
        rv2s.append(rv2[0][0])
    q_dyn = (-(np.array(rv1s) / np.array(rv2s)))
    rv1s_obs = rv1s + gamma
    rv2s_obs = gamma + (gamma - rv1s_obs) / q_dyn

    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
    return [P, gamma.tolist(), [q_dyn[-1]], rv1s_obs, rv2s_obs, phases.tolist()]


def getRvsBy_delta_RV(q, P, m1, e=0, omega=0, delta_rv_list=[0, 100, 200], i=np.array([pi / 2])):
    # get the max delta RV
    rv_max, rv_min = getRvMaxAndMin(i, q, P, m1, e, omega)
    max_delta_rv = np.abs(rv_max * q + rv_max)

    gamma = np.random.normal(0, 70, 1)
    rv1s = []
    for delta_rv in delta_rv_list:
        if delta_rv < max_delta_rv:
            rv1 = q * delta_rv / (q + 1)
            rv1s.append(rv1)
    rv1s_obs = rv1s + gamma
    rv2s_obs = gamma + (gamma - rv1s_obs) / q

    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
    return [P, gamma.tolist(), [q], rv1s_obs, rv2s_obs]


def get_period(mu, sigma, low_p, upper_p):
    s = np.random.normal(mu, sigma, 10000)
    select_s = s[(s > low_p) & (s < upper_p)]
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
    expdistri = np.random.exponential(scale=1 / (1.47 * 10 ** -2), size=500000)
    expdistri_30 = expdistri[expdistri > 30]
    #  set the expdistri = [10000,...., 10000]
    expdistri_10000 = np.zeros(500000) + 100  # 2022.06.29 set the snr to be 10000
    #
    # '''
    # Step 1: create the empty dataframe, generate the pre parameters. This step is used to search the stellar
    # parameters in "cnn_binary_mock_data_v22_tgm.fits" which is useful for objects only have mass ranges but do not
    # have other atmospheric parameters. So if you assign these parameters, this step is redundant.
    # '''
    # pre_df = pd.DataFrame(columns=['mact1', 'logage', 'feh', 'afe', 'q', 'period'])
    #
    # mact1_lower_range = np.arange(1.15, 0.55, -0.2)
    # mact1_upper_range = np.arange(1.25, 0.65, -0.2)
    #
    # for _i in range(len(mact1_lower_range)):
    #     filter = corresponding_criteria(mock_paras, mact1_lower_range[_i], mact1_upper_range[_i])
    #     Pre_DF = generate_pre_paras(mock_paras, filter, pre_df, _i)
    # Pre_DF.to_csv('20220708_mock_binary_pre_parameters_criteria.csv', index=False)

    '''
    Step 2 set the multi-q manually.  --> file: 20220707_mock_binary_pre_parameters_criteria_expended.csv
    '''

    '''
    Step 3 read multi-q pre parameter file. To generate the whole parameter file.
    '''
    # whole_df = pd.DataFrame(columns=['teff1', 'teff2', 'logg1', 'logg2', 'mh', 'alpha_m', 'R1', 'R2', 'snr', 'period',
    #                                  'spec_num', 'mact1', 'mact2', 'logage', 'q', 'gamma', 'rv1_obs', 'rv2_obs'])
    #
    # pre_dataframe_path = '20220710_mock_binary_pre_parameters_criteria_expended.csv'
    # pre_dataframe = pd.read_csv(pre_dataframe_path)
    # for _i in range(len(pre_dataframe)):
    #     Whole_DF = generate_whole_mock_paras(pre_dataframe, whole_df, _i, sp_mist, snr=100)
    # Whole_DF.to_csv('20220710_mock_binary_whole_parameters_criteria.csv', index=False)

    '''
    Step 4 generate spectra
    '''
    Whole_DF = pd.read_csv('./20220714_rdm_snr_rdm_q_mh-1_am0.4_mock_binary_whole_parameters_criteria.csv')
    samplesize = len(Whole_DF)
    set_spec_num = 1

    params = []
    mock_flux_binary = np.zeros((samplesize, set_spec_num, len(wave)), dtype=float)
    mock_flux_err_binary = np.zeros((samplesize, set_spec_num, len(wave)), dtype=float)
    mock_flux_norm_Star1 = np.zeros((samplesize, set_spec_num, len(wave)), dtype=float)
    mock_flux_norm_Star2 = np.zeros((samplesize, set_spec_num, len(wave)), dtype=float)

    for _i in range(samplesize):
        ####################### add noise to flux ###################
        flux_binary_norm, flux_binary_norm_err, flux_binary_obs, flux_norm1, flux_norm2 = model_mock_binary(Whole_DF.loc[_i],
                                                                                                            sp, regli,
                                                                                                            wave)

        # plt.plot(wave, flux_binary_norm[0], '-')
        # plt.plot(wave, flux_norm1[0]+0.6, '-')
        # plt.plot(wave, flux_norm2[0]+0.3, '-')
        # plt.show()
        params.append(Whole_DF.loc[_i])
        print(Whole_DF.loc[_i])
        mock_flux_binary[_i] = flux_binary_norm
        mock_flux_err_binary[_i] = flux_binary_norm_err
        mock_flux_norm_Star1[_i] = flux_norm1
        mock_flux_norm_Star2[_i] = flux_norm2
        # plt.plot(wave, flux_binary_norm[0])
        # plt.plot(wave, flux_binary_norm_err[0])
        # plt.show()

    b = {'params': params, 'mock_flux_binary': mock_flux_binary, 'mock_flux_err_binary': mock_flux_err_binary,
         'mock_flux_norm_Star1': mock_flux_norm_Star1, 'mock_flux_norm_Star2': mock_flux_norm_Star2}
    dump(b, '20220714_rdm_snr_rdm_q_mh-1_am0.4_mock_binary_spectra_for_criteria.dump')
