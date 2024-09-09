import numpy as np
import numpy.random as ran
import matplotlib.pyplot as plt
from joblib import load, dump
import pandas as pd
import time

G = 1.327458213e11 #Msun^-1 (km/s)^2
pi = np.pi


# In[getPhasesWithRandomT0]:
def getPhasesWithRandomT0(deltaTs, P):
    ts = np.insert(np.cumsum(deltaTs), 0, 0.) + ran.uniform(0, P) # Random T0
    return (ts%P)/P


# In[getE]:
def getE(e, phase):
    E = 0.
    ranE = [0, 2*pi]
    while (True):
        E = sum(ranE)/2.
        r0 = ranE[0] - e*np.sin(ranE[0]) - (2*pi*phase)
        r1 = ranE[1] - e*np.sin(ranE[1]) - (2*pi*phase)
        r = E - e*np.sin(E) - (2*pi*phase)
        if (r0<r and r<0):
            ranE[0] = E
        if (0<r and r<r1):
            ranE[1] = E
        if (abs(r)<0.0001):
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


def get_period(mu, sigma, low_p, upper_p):
    s = np.random.normal(mu, sigma, 10000)
    select_s = s[(s > low_p) & (s < upper_p)]
    return select_s[0]


if __name__ == '__main__':
    start = time.time()
    day = 86400  # s

    gamma = np.array([0])
    path = '/Users/liujunhui/Desktop/2021workMac/202111012totallynew/cnn_binary_mock_data_v22_tgm.csv'
    data = pd.read_csv(path)
    m1 = data['mact1']
    m1 = [0.731]
    q = data['q']
    q = [0.882]

    mu, sigma = 5.03, 2.28

    dv = []
    qs = []
    rv1 = []
    rv2 = []
    ps = []
    q_dyn = []
    for i in range(1):
        period = get_period(mu, sigma, -0.7, 1.2)
        period = 0.991
        A = getRvsBy_ramdom_Phases(q[i], period * day, m1[i], e=0, omega=0)
        rv111 = A[3]
        rv222 = A[4]
        plt.plot(A[-1], rv111, 'r.')
        plt.plot(A[-1], rv222, 'b.')

        q_dyn.append(A[2][0])
        rv1.append(rv111)
        rv2.append(rv222)
    print(A)
    plt.show()
    # print(min(q_dyn), max(q_dyn))
    # plt.hist(q_dyn, 10)
    # plt.xlabel('q_dyn')
    # plt.show()

    # print(len(dv))
    # plt.hist(dv, 40)
    # plt.xlabel('delta v')
    # plt.show()
    #
    # plt.hist(qs, 40)
    # plt.show()
    #
    # plt.hist(np.concatenate(rv1, 40))
    # plt.show()
    #
    # plt.hist(np.concatenate(rv2, 40))
    # plt.show()
    #
    # plt.hist(q_dyn, 40)
    # plt.show()
    end = time.time()
    print(end - start)

"""
i inclination
q mass ratio
P period
m1 mass
e ecc
omega
"""

