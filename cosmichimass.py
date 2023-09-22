import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import ascii as asc
import matplotlib.patheffects as pe


def read_hidata(file, cosmo):

    if file == '21cm':
        rhoh1 = asc.read('LitRhoHI_21cm.lst')
    elif file == 'stack':
        rhoh1 = asc.read('LitRhoHI_stack.lst')
    elif file == 'dla':
        rhoh1 = asc.read('LitRhoHI_dla.lst')
    elif file == 'dsa2000':
        rhoh1 = asc.read('DSA2000.lst')
    else:
        raise ValueError
    z = rhoh1['z']
    zlo = rhoh1['zlow']
    zhi = rhoh1['zhigh']
    ref = rhoh1['ref']

    zlo = (cosmo.lookback_time(z) - cosmo.lookback_time(z - zlo)).value
    zhi = (cosmo.lookback_time(z + zhi) - cosmo.lookback_time(z)).value
    z = cosmo.lookback_time(z).value
    rho = rhoh1['rho'] * 1.3
    rholo = rhoh1['rholo'] * 1.3
    rhohi = rhoh1['rhohi'] * 1.3

    return z, zlo, zhi, rho, rholo, rhohi, ref


def plot_cosden(lowz=True, stack=True, dla=True, fit=True, dsa=True, region1=True, region2=True):
    cosmo = FlatLambdaCDM(70, 0.3)
    # figure properties
    _customplotpar_()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    color = ['maroon', 'navy', 'darkgreen', 'orange']
    # 21cm points
    if lowz:
        z, zlo, zhi, rho, rholo, rhohi, ref = read_hidata('21cm', cosmo)
        ax.errorbar(z, rho, xerr=np.array([zlo, zhi]), yerr=np.array([rholo, rhohi]),
                    ls='', color='maroon', alpha=1.0, elinewidth=2.0)
    if stack:
        if dsa is False:
            alpha = 0.8
        else:
            alpha = 0.2
        z, zlo, zhi, rho, rholo, rhohi, ref = read_hidata('stack', cosmo)
        ax.errorbar(z, rho, xerr=np.array([zlo, zhi]), yerr=np.array([rholo, rhohi]),
                    ls='', color='darkgreen', alpha=alpha, elinewidth=2.0)
    if dla:
        if dsa is False:
            alpha = 0.8
        else:
            alpha = 0.2
        z, zlo, zhi, rho, rholo, rhohi, ref = read_hidata('dla', cosmo)
        gd = ref == 1
        ax.errorbar(z[gd], rho[gd], xerr=np.array([zlo[gd], zhi[gd]]), yerr=np.array([rholo[gd], rhohi[gd]]),
                    ls='', color='navy', alpha=1.0, elinewidth=2.0)
        bd = ref == 0
        ax.errorbar(z[bd], rho[bd], xerr=np.array([zlo[bd], zhi[bd]]), yerr=np.array([rholo[bd], rhohi[bd]]),
                    ls='', color='navy', alpha=alpha, elinewidth=2.0)
    if dsa:
        z, zlo, zhi, rho, rholo, rhohi, ref = read_hidata('dsa2000', cosmo)
        ax.errorbar(z, rho / 1.16, xerr=np.array([zlo, zhi]), yerr=np.array([rholo, rhohi]),
                    ls='', color='maroon', alpha=1.0, elinewidth=2.0)
    if fit:
        z = np.arange(0, 5, 0.01)
        age = cosmo.lookback_time(z).value
        bestfit = 0.4524 * np.tanh(1.0 + z - 2.8368) + 1.0119
        linfit = 0.3 * z + 0.6
        ax.plot(age, bestfit, color='black', lw=2, ls='--', label='Walter et al. 2020')
        ax.fill_between(age, 10**(np.log10(bestfit) - 0.20), y2=10**(np.log10(bestfit) + 0.20), alpha=0.1,
                        color='black')
        ax.plot(age, linfit, color='black', lw=2, ls='-.', label='Peroux & Howk 2020')
    if region1:
        ax.fill_between([0, cosmo.lookback_time(0.5).value], [3, 3], y2=[0, 0], alpha=0.3, color='maroon')
        ax.text(1.3, 2.1, 'DSA-2000\n  direct', fontsize=14, color='white',
                path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
    if region2:
        ax.fill_between([cosmo.lookback_time(0.5).value, cosmo.lookback_time(1).value], [3, 3], y2=[0, 0],
                        alpha=0.2, color='darkgreen')
        ax.text(5.1, 2.1, 'DSA-2000\n stacking', fontsize=14, color='white',
                path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
    # axis labels and specifics
    ax.set_xlim(-0.2, 12.55)
    ax.set_ylim(0, 3)
    ax.legend(loc=2)
    ax.set_xlabel('Lookback time (Gyr)', fontsize=14)
    ax.set_ylabel('$\\rho_{\\rm HI}$ (10$^8$ M$_\\odot$ Mpc$^{-3}$)', fontsize=14)
    ax2 = ax.twiny()
    ax2.set_xlim(-0.2, 12.55)
    zlabel = cosmo.lookback_time([0, 1, 2, 3, 4]).value
    ax2.set_xticks(zlabel)
    ax2.set_xticklabels(['0', '1', '2', '3', '4'])
    ax2.set_xlabel('Redshift', fontsize=14)
    # save the figure
    if dsa:
        plt.savefig('../Fig/Omega_HI_DSA2000.pdf', dpi=300)
    else:
        plt.savefig('../Fig/Omega_HI.pdf', dpi=300)
    plt.close('all')


#for idx, file in enumerate(['21cm', 'stack', 'dla', 'dsa2000']):
#    z, zlo, zhi, rho, rholo, rhohi = read_hidata(file, cosmo)
#    ax.errorbar(z, rho, xerr=np.array([zlo, zhi]),
#                yerr=np.array([rholo, rhohi]),
#                ls='', color=color[idx], alpha=1.0, elinewidth=2.0)

#ax.plot(age, bestfit, color='black', lw=2, ls='--',
#        label='Walter et al. 2020')
#ax.fill_between(age, 10**(np.log10(bestfit) - 0.20),
#                y2=10**(np.log10(bestfit) + 0.20), alpha=0.1,
#                color='black')
#ax.plot(age, linfit, color='black', lw=2, ls='-.',
#        label='Peroux & Howk 2020')
#ax.plot([0, 10.0], [1.02, 1.02], color='black', lw=2, ls=':',
#        label='Braun 2012')
#
#ax.fill_between([0, cosmo.lookback_time(0.5).value], [3, 3], y2=[0, 0],
#                alpha=0.3, color='maroon')
#ax.fill_between([cosmo.lookback_time(0.5).value,
#                 cosmo.lookback_time(1).value], [3, 3], y2=[0, 0],
#                alpha=0.3, color='darkgreen')
#ax.text(2, 2.1, 'DSA-2000', fontsize=20, color='white',
#        path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])


def _customplotpar_():
    ################################
    # set custom figure parameters #
    mpl.rcParams.update(mpl.rcParamsDefault)
    font = {'family': 'DejaVu Sans',
            'weight': 'normal',
            'size': 10}
    mpl.rc('font', **font)
    mpl.rc('mathtext', fontset='stixsans')
    mpl.rc('axes', lw=2.5)
    mpl.rc('lines', lw=2.0)
    mpl.rc('xtick.major', size=4, width=1)
    mpl.rc('ytick.major', size=4, width=1)
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
