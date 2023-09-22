"""Code dealing with the mock catalog from Obreschkow."""
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as const
from scipy.stats import poisson
from scipy.optimize import curve_fit
from scipy.special import gamma
from matplotlib.patches import Ellipse
from abc import ABC
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox


def plot_massvsz(show_plot=False):
    """
    Plot the HI mass versus redshift for the mock catalog above the detection limit.
    """
    cosmo = FlatLambdaCDM(70, 0.3)
    # figure properties
    _customplotpar_()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # plot the data
    z = _selectdata_('zobs', cosmo=cosmo)
    print(np.where(z < 0.4)[0].shape)
    print(np.where(z > 0.4)[0].shape)
    mhi = _selectdata_('mhi', cosmo=cosmo)
    ax.plot(z, np.log10(mhi), ',', color='navy', alpha=0.2)
    # plot the limits
    zlimit = np.arange(0.0, 1.0, 0.01)
    flim = 0.0213 - 0.0063 * zlimit
    nhilimit = 2.356E5 / (1 + zlimit) * cosmo.luminosity_distance(zlimit).value**2 * flim
    ax.plot(zlimit, np.log10(nhilimit), color='navy')
    # Axis and label properties
    xlim = [0, 1.0]
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(8.2, 12)
    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel('log HI Mass [M$_\\odot$]', fontsize=14)
    ax2 = ax.twiny()
    ax2.set_xlim(cosmo.luminosity_distance(xlim[0]).value,
                 cosmo.luminosity_distance(xlim[1]).value)
    ax2.set_xlabel('Luminosity distance (Mpc)', fontsize=14)
    # save figure
    if show_plot:
        plt.show()
    else:
        plt.savefig('../Fig/fig_mhiz.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_wedge(show_plot=False):
    """
    Plot a polar 'sky wedge' of the DSA2000 coverage.
    """
    cosmo = FlatLambdaCDM(70, 0.3)
    ra = _selectdata_('ra', cosmo=cosmo)
    dec = _selectdata_('dec', cosmo=cosmo)
    zobs = _selectdata_('zobs', cosmo=cosmo)
    # now restict by RA slice (within 5 degrees of 03:00)
    mask = np.abs(90 - ra) < (5 / np.cos(dec * np.pi / 180))
    dec03 = np.pi / 2 - (dec[mask] * np.pi / 180)
    zobs03 = zobs[mask]
    # now restict by RA slice (within 5 degrees of 15:00)
    mask = np.abs(270 - ra) < (5 / np.cos(dec * np.pi / 180))
    dec15 = dec[mask] * np.pi / 180 - np.pi / 2
    zobs15 = zobs[mask]
    # figure properties
    _customplotpar_()
    mpl.rc('xtick.major', size=4, width=1)
    mpl.rcParams['xtick.direction'] = 'out'
    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(projection='polar')
    # plot the data
    print(len(dec03) + len(dec15))
    ax.plot(dec03, np.tanh(4 * zobs03), ',', alpha=0.5, color='navy')
    ax.plot(dec15, np.tanh(4 * zobs15), ',', alpha=0.5, color='navy')
    # axis properties and labels
    plt.rcParams['ytick.left'] = False
    ax.set_theta_zero_location("N")
    ax.set_thetamin(-120)
    ax.set_thetamax(120)
    ax.set_rmin(0.0)
    ax.set_rmax(0.999)
    ax.xaxis.set_tick_params(labelsize=5, size=-1)
    ax.set_xticks(np.arange(-120, 150, 30) * np.pi / 180)
    ax.yaxis.set_tick_params(rotation=-30, labelsize=5)
    ax.set_xticklabels(['-30$^{\\rm o}$', '0$^{\\rm o}$', '30$^{\\rm o}$',
                        '60$^{\\rm o}$', '90$^{\\rm o}$', '60$^{\\rm o}$',
                        '30$^{\\rm o}$', '0$^{\\rm o}$', '-30$^{\\rm o}$'])
    ax.set_yticks(np.tanh(4 * np.array([0, 0.1, 0.2, 0.3, 0.4, 1.0])))
    ax.set_yticklabels([' ', '0.1', '0.2', '0.3', '0.4', '1.0'])
    dc = cosmo.luminosity_distance([0.1, 0.2, 0.3, 0.4]).value
    dc = [round(i, -1) for i in dc]
    fig.text(0.505, 0.37, '0', fontsize=6)
    fig.text(0.38, 0.30, '{:3.0f}'.format(dc[0]), fontsize=6, rotation=30)
    fig.text(0.285, 0.245, '{:3.0f}'.format(dc[1]), fontsize=6, rotation=30)
    fig.text(0.23, 0.212, '{:2.0f}'.format(dc[2]), fontsize=6, rotation=30)
    fig.text(0.19, 0.19, '{:2.0f}'.format(dc[3]), fontsize=6, rotation=30)
    fig.text(0.60, 0.23, 'Redshift', rotation=-30, color='black', fontsize=10)
    fig.text(0.28, 0.20, 'Distance (Mpc)', rotation=30, color='black', fontsize=10)
    fig.text(0.5, 0.83, 'Declination', color='black', fontsize=10, ha='center')
    # save figure
    if show_plot:
        plt.show()
    else:
        plt.savefig('../Fig/fig_wedge.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_skypatch(show_plot=False):
    """
        Plot a patch of sky in the simulated data
    """
    cosmo = FlatLambdaCDM(70, 0.3)
    ra = _selectdata_('ra', cosmo=cosmo)
    dec = _selectdata_('dec', cosmo=cosmo)
    zobs = _selectdata_('zobs', cosmo=cosmo)
    rdisk = _selectdata_('rgas_disk_apparent', cosmo=cosmo)
    incl = _selectdata_('inclination', cosmo=cosmo)
    # now restict by RA and dec
    center = (13.35, 1.65)
    size = (0.1, 0.1)
    mask = (np.abs(center[0] - ra) < size[0]) & (np.abs(center[1] - dec) < size[1])
    ra = ra[mask]
    dec = dec[mask]
    zobs = zobs[mask]
    rdisk = rdisk[mask] / 3600
    incl = incl[mask]
    qh = np.sqrt(np.cos(incl * np.pi / 180) ** 2 + 0.01 * np.sin(incl * np.pi / 180) ** 2)
    # figure properties
    _customplotpar_()
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6))
    plt.subplots_adjust(left=0.12, right=1.00, top=0.98, bottom=0.08)
    # plot the data
    np.random.seed(1234)
    pa = np.random.rand(len(ra)) * 360
    ax.set_xlim(-size[0], size[0])
    ax.set_ylim(-size[1], size[1])
    color = plt.get_cmap('viridis')(np.tanh(4 * zobs))
    for idx in np.arange(len(ra)):
        ax.add_artist(Ellipse(xy=(ra[idx] - center[0], dec[idx] - center[1]), width=4 * rdisk[idx],
                              height=4 * rdisk[idx] * qh[idx], angle=pa[idx], color=color[idx]))
    # beam
    ax.add_artist(AnchoredEllipse(ax.transData, width=3.8/3600, height=3.8/3600, angle=0, loc='lower left',
                                  facecolor='black', borderpad=0.5))
    # color bar + labels
    ticks = np.tanh(4 * np.array([0., 0.1, 0.2, 0.3, 0.5, 1.0]))
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, ticks=ticks)
    cb.ax.set_yticklabels(['0.0', '0.1', '0.2', '0.3', '0.5', '1.0'])
    fig.text(0.02, 0.54, r'$\Delta$ Decl. ($^{\rm o}$)', fontsize=14, rotation=90, va='center')
    fig.text(0.52, 0.02, r'$\Delta$ R.A. ($^{\rm o}$)', fontsize=14, ha='center')
    fig.text(0.96, 0.54, r'Redshift', fontsize=14, rotation=90, va='center')
    # save figure
    if show_plot:
        plt.show()
    else:
        plt.savefig('../Fig/fig_skypatch.png', dpi=300)
        plt.close()


def plot_rdiskz(show_plot=True):
    """
        Plot the disk radius (twice the exponential disk length) vs redshift and compare this to the beam size
    """
    cosmo = FlatLambdaCDM(70, 0.3)
    zobs = _selectdata_('zobs', cosmo=cosmo)
    rdisk = _selectdata_('rgas_disk_apparent', cosmo=cosmo)
    incl = _selectdata_('inclination', cosmo=cosmo)
    qh = np.sqrt(np.cos(incl * np.pi / 180) ** 2 + 0.01 * np.sin(incl * np.pi / 180) ** 2)
    gal_size = np.pi * rdisk ** 2 * 4 * qh
    zobs2 = _selectdata_('zobs', cosmo=cosmo, correct_for_beam=False)
    rdisk2 = _selectdata_('rgas_disk_apparent', cosmo=cosmo, correct_for_beam=False)
    incl2 = _selectdata_('inclination', cosmo=cosmo, correct_for_beam=False)
    qh2 = np.sqrt(np.cos(incl2 * np.pi / 180) ** 2 + 0.01 * np.sin(incl2 * np.pi / 180) ** 2)
    gal_size2 = np.pi * rdisk2 ** 2 * 4 * qh2
    zobs_c = np.arange(0, 1, 0.01)
    beam_sig = 1.42 / 1.35 * 3.5 * (1 + zobs_c) / np.sqrt(8 * np.log(2))
    beam_size = beam_sig ** 2 * 2 * np.pi
    # nbeams = np.where(gal_size > beam_size, gal_size / beam_size, 1)
    # figure properties
    _customplotpar_()
    plt.figure(figsize=(6, 6))
    ax = plt.subplot()
    # plot the data
    ax.plot(zobs2, gal_size2, '.', color='gray', alpha=0.2, label='Undetected')
    ax.plot(zobs, gal_size, '.', color='navy', alpha=0.2, label='Detected')
    ax.plot(zobs_c, beam_size, '-', color='maroon', alpha=1.0, label='beam')
    # axis and labels
    ax.legend(loc='upper right')
    xlim = [0, 1.0]
    ax.plot(xlim, [0, 0], ls=':', color='black')
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(0, 1000)
    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel('Galaxy size (arcsec$^2$)', fontsize=14)
    ax2 = ax.twiny()
    ax2.set_xlim(cosmo.luminosity_distance(xlim[0]).value, cosmo.luminosity_distance(xlim[1]).value)
    ax2.set_xlabel('Luminosity distance (Mpc)', fontsize=14)
    # save the figure
    if show_plot:
        plt.show()
    else:
        plt.savefig('../Fig/fig_rdiskz.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_histz(show_plot=False):
    """
       make a histogram of the galaxies that will be detected (as a function of z)
    """
    cosmo = FlatLambdaCDM(70, 0.3)
    zobs = _selectdata_('zobs', cosmo=cosmo)
    # figure properties
    _customplotpar_()
    plt.figure(figsize=(6, 6))
    ax = plt.subplot()
    # plot the data
    ax.hist(zobs, bins=10, range=(0, 1), color='navy', log=True)
    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel('Number of Galaxies', fontsize=14)
    # save the figure
    if show_plot:
        plt.show()
    else:
        plt.savefig('../Fig/fig_histz.png', dpi=300, bbox_inches='tight')
        plt.close()


def _customplotpar_():
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


def _selectdata_(dat1, cosmo=None, correct_for_beam=True):
    if cosmo is None:
        cosmo = FlatLambdaCDM(70, 0.3)
    d1 = []
    for idx in np.arange(64):
        f = h5py.File('../DatFiles/mocksky_dsa_wide_1.' + str(idx) + '.hdf5')
    # mask by hi flux limit (assuming 150 km/s line, 10sigma detection and a flux limit that depends on the
    # redshift due to the varying depth at different frequencies (overlap in mosaics at high freq.)
    # the 10sigma sensitivity limit is a linear fit to the points z=0, 21.3 mJy km/s and z=1, 15 mJy km/s
    # see email by Fabian 3/17/2023
        if correct_for_beam:
            fhi = _fluxperbeam_(f)
        else:
            fhi = f['galaxies']['hiline_flux_int_vel'][:]
        mask = fhi > 0.0213 - 0.0063 * f['galaxies']['zobs'][:]
        if dat1 == 'mhi':
            zobs = f['galaxies']['zobs'][:][mask]
            d1.extend(2.356E5 / (1 + zobs) * cosmo.luminosity_distance(zobs).value ** 2 *
                      f['galaxies']['hiline_flux_int_vel'][:][mask])
        else:
            d1.extend(f['galaxies'][dat1][:][mask])
    return np.array(d1)


def _fluxperbeam_(f):
    """
        calculate the integrated flux taking into account the resolved nature
        THIS NEEDS A FIX TO CORRECTLY ACCOUNT FOR ADJACENT BEAMS
    """
    fhi = f['galaxies']['hiline_flux_int_vel'][:]
    zobs = f['galaxies']['zobs'][:]
    rdisk = f['galaxies']['rgas_disk_apparent'][:]
    incl = f['galaxies']['inclination'][:]
    qh = np.sqrt(np.cos(incl * np.pi / 180) ** 2 + 0.01 * np.sin(incl * np.pi / 180) ** 2)
    gal_size = np.pi * rdisk**2 * 3 * qh
    beam_sig = 1.42 / 1.35 * 3.5 * (1 + zobs) / np.sqrt(8 * np.log(2))
    beam_size = beam_sig**2 * 2 * np.pi
    nbeams = np.where(gal_size > beam_size, gal_size / beam_size, 1)
    return fhi / np.sqrt(nbeams)   # changed to sqrt(n) to partially account for integrated quantity which we care for.

#################################
#### BELOW THIS LINE THE CODE HAS NOT BEEN UPDATED FOR THE NEW SIMS, SO THESE NEXT PIECES OF CODE WILL LIKELY FAIL AND WILL NEED TO BE UPDATED.
#################################

def plot_himf():
    for field in ['deep', 'pulsar', 'wide']:
        zbins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
        if field == 'wide':
            zbins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]
        color = {'deep': 'maroon', 'pulsar': 'darkgreen', 'wide': 'navy'}
        _customplotpar_()
        fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex='all', sharey='all')
        plt.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.10, wspace=0.0, hspace=0.0)
        for idx, zbin in enumerate(zbins):
            ax = axs[idx // 3, np.mod(idx, 3)]
            _himf_(zbin, field=field, plot=True, ax=ax, fig=fig, color=color[field])
        if field == 'wide':
            axs[1, 1].axis('off')
        axs[1, 2].axis('off')
        fig.text(0.5, 0.05, r'log M$_{HI}$ [M$_\odot$]', fontsize=14, ha='center')
        fig.text(0.05, 0.5, r'log $\phi$(M$_{\rm HI}$) [Mpc$^{-3}$ dex$^{-1}$]', rotation=90, fontsize=14, va='center')
        plt.savefig('../Fig/himassfunction_' + field + '.png', dpi=300, bbox_inches='tight')
        plt.close('all')


def _himf_(zbin=(0.0, 0.1), field='pulsar', obslen='full', plot=False, cosmo=None, nsamples=100, scale=0.1,
           ax=None, fig=None, color='navy'):
    # define cosmological properties
    if cosmo is None:
        cosmo = FlatLambdaCDM(70, 0.3)
    omega = {'deep': 0.009138522, 'pulsar': 0.6092348, 'wide': 9.1385226}  # area in steradians
    rho_c = (3 * (70 * u.km / u.s / u.Mpc)**2 / (8 * np.pi * const.G)).to(u.Msun / u.Mpc**3).value
    # figure properties
    if plot:
        _customplotpar_()
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # mass limit for the redshift bin.
    logmhi_lim = np.log10(_get_mhilimit_(np.mean(zbin), field=field, obslen=obslen))
    # calculate the number of galaxies above the mass in the Mock data
    z, mass = _selectdata_(field, obslen, cosmo=cosmo)
    logmass_cat = np.log10(mass[np.log10(mass) > logmhi_lim])
    ngal_cat, binedge = np.histogram(logmass_cat, bins=np.arange(7, 12.2, 0.2))
    ngal_cat = ngal_cat[binedge[:-1] + 0.1 > logmhi_lim]
    ngal_cat = np.round(scale * ngal_cat)
    # calculate the expected/needed number of galaxies per bin from HIMF (from ALFALFA data)
    logmass_cal = np.arange(7.1, 12.1, 0.2)
    logmass_cal = logmass_cal[logmass_cal > logmhi_lim]
    volume = (omega[field] / (4 * np.pi) * (cosmo.comoving_volume(zbin[1]) -
                                            cosmo.comoving_volume(zbin[0])).value)
    logphi_cal = np.log10(4.7E-3 * np.log(10)) + (1 - 1.25) * (logmass_cal - 9.94) -\
        10**(logmass_cal - 9.94)*np.log10(np.exp(1))
    ngal_cal = 10**(logphi_cal + np.log10(volume * 0.2))
    # correction factor to apply
    c = ngal_cal / ngal_cat
    phistar, alpha, mstar, omega_hi = [], [], [], []
    for idx in np.arange(nsamples):
        # randomize the galaxies using Poisson stats
        ngal_ran = c * (1 * (poisson.rvs(ngal_cat) - ngal_cat) + ngal_cat)
        ngal_ran[np.isnan(ngal_ran)] = 0
        logphi_ran = np.log10(ngal_ran) - np.log10(volume * 0.2)
        # set the values of the high mass bins (those with less than 1 detection to the theoretical value)
        logphi_ran[ngal_ran < 1] = logphi_cal[ngal_ran < 1]
        dlogphi_ran = np.log10(ngal_cal + np.sqrt(ngal_cal + 1)) - np.log10(ngal_cal)
        if idx == nsamples - 1 and plot:
            ax.errorbar(logmass_cal, logphi_ran, yerr=dlogphi_ran, ls=' ', color='black')
            ax.plot(logmass_cal, logphi_ran, '.', color='black')
        # calculate the schechter fit
        p, _pcov = curve_fit(_schechter_func_, logmass_cal, logphi_ran, sigma=dlogphi_ran, p0=[10.5E-3, -3.25, 8.94])
        if plot:
            ax.plot(np.arange(7, 12, 0.1), _schechter_func_(np.arange(7, 12, 0.1), *p), '-', color=color, alpha=0.1)
        phistar.append(p[0])
        alpha.append(p[1])
        mstar.append(p[2])
        omega_hi.append(1 / rho_c * p[0] * 10 ** p[2] * gamma(p[1] + 2))
    if plot:
        p = (4.7E-3, -1.25, 9.94)
        ax.plot(np.arange(7, 12, 0.1), _schechter_func_(np.arange(7, 12, 0.1), *p), ls='--', color='navy')
        ax.text(0.93, 0.90, 'z = [{} - {}]'.format(zbin[0], zbin[1]), transform=ax.transAxes, ha='right')
        ax.text(0.08, 0.08, r'$\rho_{\rmHI}$ = ' +
                r'({:4.2f}  $\pm$ {:4.2f}) x 10$^7$ M$_\odot$'.format(np.mean(omega_hi) * rho_c * 1E-7,
                                                                      np.std(omega_hi) * rho_c * 1E-7),
                transform=ax.transAxes, ha='left')
        ax.text(0.08, 0.15, r'$\Omega_{\rmHI}$ = ' +
                r'({:4.3f}  $\pm$ {:4.3f})'.format(np.mean(omega_hi) * 1E3, np.std(omega_hi) * 1E3) + ' x 10$^{-3}$',
                transform=ax.transAxes, ha='left')
        ax.text(0.08, 0.22, r'$log M_*$ = ' + r'{:4.3f} $\pm$ {:4.3f}'.format(np.mean(mstar), np.std(mstar)),
                transform=ax.transAxes, ha='left')
        ax.text(0.08, 0.29, r'$\alpha$ = ' + r'{:4.3f} $\pm$ {:4.3f}'.format(np.mean(alpha), np.std(alpha)),
                transform=ax.transAxes, ha='left')
        ax.text(0.08, 0.36, r'$\phi_*$ = ' + r'{:4.2g} $\pm$ {:4.2g}'.format(np.mean(phistar), np.std(phistar)),
                transform=ax.transAxes, ha='left')
        ax.text(0.08, 0.43, r'Volume = ' + r'{:4.2g} Mpc$^3$'.format(volume), transform=ax.transAxes,
                ha='left')
        ax.set_xlim(7, 12)
        ax.set_ylim(-7, 0)
        # save figure
        if plot == 'show':
            plt.show()
        if plot == 'save':
            plt.savefig('../Fig/himassfunction_' + field + '_z{}-{}'.format(*zbin) + '.png', dpi=300,
                        bbox_inches='tight')
    himfdict = {'z': zbin, 'rhohi': np.mean(omega_hi) * rho_c, 'drhohi': np.std(omega_hi) * rho_c,
                'omegahi': np.mean(omega_hi), 'domegahi': np.std(omega_hi)}
    return himfdict


def plot_mhimstarvsz(show_plot=False):
    """
    Plot the HI/Stellar mass ratio as a function of redshift.
    """
    cosmo = FlatLambdaCDM(70, 0.3)
    # figure properties
    _customplotpar_()
    plt.figure(figsize=(6, 6))
    ax = plt.subplot()
    # plot the data
    zobs = _selectdata_('zobs', cosmo=cosmo)
    mhi = _selectdata_('mhi', cosmo=cosmo)
    mst = _selectdata_('mstars_disk', cosmo=cosmo) + _selectdata_('mstars_bulge', cosmo=cosmo)
    ax.plot(zobs, np.log10(mhi / mst), ',', color='navy', alpha=0.8)
    # axis and labels
    xlim = [0, 1.0]
    ax.plot(xlim, [0, 0], ls=':', color='black')
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(-2, 2)
    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel('log M$_{\\rm HI}$ / M$_\\star$', fontsize=14)
    ax2 = ax.twiny()
    ax2.set_xlim(cosmo.luminosity_distance(xlim[0]).value, cosmo.luminosity_distance(xlim[1]).value)
    ax2.set_xlabel('Luminosity distance (Mpc)', fontsize=14)
    # save the figure
    if show_plot:
        plt.show()
    else:
        plt.savefig('../Fig/fig_mhimstarz.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_zvswhiwidth():
    cosmo = FlatLambdaCDM(70, 0.3)
    # figure properties
    _customplotpar_()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # plot the data
    zobs = _selectdata_('zobs', cosmo=cosmo)
    hiw50 = _selectdata_('hiline_width_50', cosmo=cosmo)
    vvir = _selectdata_('vvir_subhalo', cosmo=cosmo)
    ax.plot(zobs, hiw50, '.', color='red')
    ax.plot(zobs, vvir, '.', color='blue')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, 3600)
    plt.show()


def _get_mhilimit_(z, field='deep', obslen='full', cosmo=None):
    if cosmo is None:
        cosmo = FlatLambdaCDM(70, 0.3)
    limit = {'deep': 9, 'pulsar': 18, 'wide': 45}
    scale = {'full': 1, '1hour': 2, '15min': 4}
    flim = ((limit[field] * u.mJy * u.km /
             u.s).to(u.Jy * u.km / u.s).value * scale[obslen])
    return 2.356E5 / (1 + z) * cosmo.luminosity_distance(z).value ** 2 * flim


def _schechter_func_(x, a, b, c):
    return np.log10(a * np.log(10)) + (1 + b) * (x - c) - 10**(x - c)*np.log10(np.exp(1))


class AnchoredEllipse(AnchoredOffsetbox, ABC):
    def __init__(self, transform, width, height, angle, loc,
                 pad=0.1, borderpad=0.1, prop=None, frameon=True,
                 facecolor='White', alpha=1.0, **kwargs):
        """
        Draw an ellipse the size in data coordinate of the give axes.
        pad, borderpad in fraction of the legend font size (or prop)
        """
        self._box = AuxTransformBox(transform)
        self.ellipse = (Ellipse((0, 0), width, height, angle,
                        facecolor=facecolor, alpha=alpha, **kwargs))
        self._box.add_artist(self.ellipse)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=self._box, prop=prop, frameon=frameon)
