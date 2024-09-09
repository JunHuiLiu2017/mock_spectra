from juliacall import Main as jl
import matplotlib.pyplot as plt
import time
import numpy as np
from laspec.qconv import conv_spec_Gaussian
import joblib
from joblib import dump, load
from laspec.normalization import normalize_spectrum_spline
# jl.seval("using Pkg")
# jl.Pkg.add("Korg")
jl.seval("using Korg")
Korg = jl.Korg


def Stellar_spectra(wave_start=2950, wave_end=8750, wave_new=np.arange(3950, 5750, .01), add_RV=0,
                    R_lo=50000, Teff=5777, logg=4.0, MH=0.0, alphaM=0.0):
    lines = Korg.get_VALD_solar_linelist()  # lines for the Sun from 3000 to 9000 Å
    # this generates a model atmosphere for a dwarf with Teff = 5221 K and logg=4.32
    dwarf_atm = Korg.interpolate_marcs(Teff, logg)  # we could also provide abundance parameters for non-solar values

    # run this cell again to see how much faster it is after precompilation.
    t = time.process_time()
    sol = Korg.synthesize(dwarf_atm, lines, Korg.format_A_X(0), wave_start, wave_end)
    processtime = time.process_time() - t
    print(processtime)

    # Results: sol.wavelengths, sol.flux, sol.cntm
    # re-resolution
    wave_temp, flux_temp = conv_spec_Gaussian(sol.wavelengths, sol.flux, R_hi=500000, R_lo=R_lo,
                                              wave_new=wave_new)
    # Add RV
    flux_obs = np.interp(wave_temp, wave_temp * (1 + add_RV / 299792.458), flux_temp)

    return sol.wavelengths, sol.flux, sol.cntm, wave_temp, flux_temp, flux_obs


def binary_parameters(sp_mist, mist_mass1, mist_age, mist_q, mist_MH=0.0):
    """
    Using MIST model to convert absolute parameters into atmospheric paras.
    Args:
        sp_mist: The MIST model
        mist_mass1:
        mist_age:
        mist_q:
        mist_MH:

    Returns:
    """
    mist_mass2 = mist_mass1 * mist_q
    mist_Teff1, mist_logg1, mist_R1, *mist_other_outputs1 \
        = sp_mist.predict([mist_mass1, mist_age, mist_MH])
    mist_Teff2, mist_logg2, mist_R2, *mist_other_outputs2 \
        = sp_mist.predict([mist_mass2, mist_age, mist_MH])
    return mist_Teff1, mist_logg1, mist_R1, mist_Teff2, mist_logg2, mist_R2, # mist_other_outputs1, mist_other_outputs2


if __name__ == "__main__":
    '''
    load models
    '''
    workdir = '/Users/liujunhui/PycharmProjects/LamostBinary/'
    mist_model = joblib.load(workdir + 'mist_eep_202_454_teff_3500_8000_logt_6.83_10.26_mh_-2.1_0.7_gaia_2mass.dump')
    sp_mist, colname_input, colname_output, acc, sp_val = mist_model

    Mass_range = np.arange(1.0, 1.5, 0.05)
    q_range = np.arange(0.9, 1.1, 0.1)

    for _i in range(len(Mass_range)):
        for _j in range(len(q_range)):
            # 1. Generate atmospheric parameters based on mass, age and MH.
            Binary_paras = binary_parameters(sp_mist, Mass_range[_i], 9.65, q_range[_j], mist_MH=0.0)
            # Binary_paras outputs: mist_Teff1, mist_logg1, mist_R1, mist_Teff2, mist_logg2, mist_R2
            print(Binary_paras)

            # 2. Generate template spectra according to atmospheric parameters.
            wavelengths_pri, flux_pri, cntm_pri, wave_temp_pri, flux_temp_pri, flux_obs_pri = \
                Stellar_spectra(wave_start=3950, wave_end=5750, wave_new=np.arange(3950, 5750, .1), add_RV=50,
                                R_lo=50000, Teff=Binary_paras[0], logg=Binary_paras[1], MH=0.0, alphaM=0.0)

            wavelengths_sec, flux_sec, cntm_sec, wave_temp_sec, flux_temp_sec, flux_obs_sec = \
                Stellar_spectra(wave_start=3950, wave_end=5750, wave_new=np.arange(3950, 5750, .1), add_RV=-50,
                                R_lo=50000, Teff=Binary_paras[3], logg=Binary_paras[4], MH=0.0, alphaM=0.0)

            print(np.sum(wavelengths_sec - wavelengths_pri))

            # 3. Construct binary spectra.
            flux_obs_binary = flux_obs_pri * Binary_paras[2]**2 + flux_obs_sec * Binary_paras[5]**2

            # 4. Normalizing spectra of primary, secondary and binary stars.
            # flux_temp_norm_pri, flux_temp_cont_pri = normalize_spectrum_spline(wave_temp_pri, flux_obs_pri, niter=3)
            flux_obs_norm_pri, flux_obs_cont_pri = normalize_spectrum_spline(np.array(wave_temp_pri),
                                                                             np.array(flux_obs_pri), niter=3)

            # flux_temp_norm_sec, flux_temp_cont_sec = normalize_spectrum_spline(wave_temp_sec, flux_obs_sec, niter=3)
            flux_obs_norm_sec, flux_obs_cont_sec = normalize_spectrum_spline(np.array(wave_temp_sec),
                                                                             np.array(flux_obs_sec), niter=3)

            flux_obs_norm_binary, flux_obs_cont_binary = normalize_spectrum_spline(np.array(wave_temp_sec),
                                                                             np.array(flux_obs_binary), niter=3)

            plt.figure(figsize=(15, 6))
            plt.plot(np.array(wave_temp_pri), flux_obs_norm_pri, 'k.-', lw=3, label="Primary")
            plt.plot(np.array(wave_temp_sec), flux_obs_norm_sec, lw=3, label="Secondary")
            plt.plot(np.array(wave_temp_sec), flux_obs_norm_binary, lw=3, label="Binary")
            plt.xlim(5160, 5200)
            plt.xlabel("$\lambda$ [$\mathrm{\AA}$]")
            plt.ylabel("erg/s/cm$^2$/cm")
            plt.title("Mg I")
            plt.legend()
            plt.grid(True)
            plt.show()



    # lines = Korg.read_linelist("linelist.vald", format="vald")
    lines = Korg.get_VALD_solar_linelist()  # lines for the Sun from 3000 to 9000 Å
    # lines = Korg.get_APOGEE_DR17_linelist() # if you want to do something in the infrared (this is for 15,000 Å - 17,000 Å)
    # lines = Korg.get_GALAH_DR3_linelist()

    '''
    # this generates a model atmosphere for a dwarf with Teff = 5221 K and logg=4.32
    dwarf_atm = Korg.interpolate_marcs(5221, 4.32)  # we could also provide abundance parameters for non-solar values

    # run this cell again to see how much faster it is after precompilation.
    t = time.process_time()
    sol = Korg.synthesize(dwarf_atm, lines, Korg.format_A_X(0), 2950, 8750)
    time.process_time() - t

    sol.flux
    print(sol.wavelengths[-10000:-9990])
    print(sol.wavelengths[9990:10000])
    sol.cntm

    # plt.figure(figsize=(12, 4))
    # plt.plot(sol.wavelengths, np.array(sol.flux), "k-")
    # plt.xlabel("$\lambda$ [Å]")
    # plt.ylabel("rectified flux")
    # # plt.show()
    wave_uh = np.array(sol.wavelengths)
    flux_uh = np.array(sol.flux)

    wave_hi, flux_hi = conv_spec_Gaussian(wave_uh, flux_uh, R_hi=500000, R_lo=50000,
                                          wave_new=np.arange(3000, 10000, .01))
    wave_me, flux_me = conv_spec_Gaussian(wave_uh, flux_uh, R_hi=500000, R_lo=7500,
                                          wave_new=np.arange(3000, 10000, .01))
    wave_lo, flux_lo = conv_spec_Gaussian(wave_uh, flux_uh, R_hi=500000, R_lo=1800, wave_new=np.arange(3000, 10000, 1))

    plt.figure(figsize=(15, 6))
    plt.plot(wave_uh, flux_uh, lw=3, label="R=500000", c="gray")
    print(wave_uh[[wave_uh > 8350] and [wave_uh < 8580]])
    plt.plot(wave_hi, flux_hi, lw=3, label="R=50000")
    print("R=500000", wave_hi[[wave_hi > 8350] and [wave_hi < 8580]])
    plt.plot(wave_me, flux_me, lw=3, label="R=7500")
    print('R_lo=7500', wave_me[[wave_me > 8350] and [wave_me < 8580]])
    plt.plot(wave_lo, flux_lo, lw=3, label="R=1800")
    print('R_lo=1800', wave_lo[[wave_lo > 8350] and [wave_lo < 8580]])
    plt.xlim(8350, 8580)
    plt.xlabel("$\lambda$ [$\mathrm{\AA}$]")
    plt.ylabel("erg/s/cm$^2$/cm")
    plt.title("H$\\alpha$")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(15, 6))
    plt.plot(wave_uh, flux_uh, lw=3, label="R=500000", c="gray")
    plt.plot(wave_hi, flux_hi, lw=3, label="R=80000")
    plt.plot(wave_me, flux_me, lw=3, label="R=7500")
    plt.plot(wave_lo, flux_lo, lw=3, label="R=1800")
    plt.xlim(5160, 5290)
    plt.xlabel("$\lambda$ [$\mathrm{\AA}$]")
    plt.ylabel("erg/s/cm$^2$/cm")
    plt.title("Mg I")
    plt.legend()    # wavelengths, flux, cntm, wave_temp, flux_temp, flux_obs = \
    #     Stellar_spectra(wave_start=2950, wave_end=8750, wave_new=np.arange(3950, 5750, .01), add_RV=100,
    #                 R_lo=50000, Teff=5777, logg=4.0, MH=0.0, alphaM=0.0)
    # plt.figure(figsize=(15, 6))
    # plt.plot(wavelengths, flux, lw=3, label="R=50000", c="gray")
    # plt.plot(wave_temp, flux_obs, lw=3, label="R=50000")
    # plt.xlim(5160, 5290)
    # plt.xlabel("$\lambda$ [$\mathrm{\AA}$]")
    # plt.ylabel("erg/s/cm$^2$/cm")
    # plt.title("Mg I")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # 
    # flux_temp_norm, flux_temp_cont = normalize_spectrum_spline(wave_temp, flux_obs, niter=3)
    # # print(flux_temp_norm, flux_temp_cont)
    # flux_obs_norm, flux_obs_cont = normalize_spectrum_spline(np.array(wavelengths), np.array(flux), niter=3)
    # 
    # plt.figure(figsize=(15, 6))
    # plt.plot(wave_temp, flux_temp_norm, lw=3, label="R=50000", c="gray")
    # plt.plot(np.array(wavelengths), flux_obs_norm, lw=3, label="R=50000")
    # plt.xlim(5160, 5290)
    # plt.xlabel("$\lambda$ [$\mathrm{\AA}$]")
    # plt.ylabel("erg/s/cm$^2$/cm")
    # plt.title("Mg I")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    plt.grid(True)
    plt.show()
    '''
