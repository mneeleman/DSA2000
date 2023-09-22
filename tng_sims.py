"""Code to create mock HI observations from the TNG simulations."""
from martini.sources import TNGSource
from martini import DataCube, Martini
from martini.beams import GaussianBeam
from martini.noise import GaussianNoise
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import AdaptiveKernel, GaussianKernel
from martini.sph_kernels import CubicSplineKernel
import astropy.units as u
import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import h5py


def find_halo(vmax=[200, 220], sfr=[1, 10]):
    """
    Find subhalo with the given parameters.

    Parameters
    ----------
    vmax : TYPE, optional
        DESCRIPTION. The default is [200, 220].
    sfr : TYPE, optional
        DESCRIPTION. The default is [1, 10].

    Returns
    -------
    subhalo index number

    """
    basePath = ('/data/beegfs/astro-storage/groups/walter/neeleman' +
                '/TNG/TNG50-1/output')
    fields = ['SubhaloVmax', 'SubhaloSFRinMaxRad']
    subhalos = il.groupcat.loadSubhalos(basePath, 99, fields=fields)

    a = ((subhalos['SubhaloVmax'] > vmax[0]) * (subhalos['SubhaloVmax'] < vmax[1]) *
         (subhalos['SubhaloSFRinMaxRad'] > sfr[0]) * )
    b = [i for i, x in enumerate(a) if x]
    print(b)
    print('find_halo: Will return just the first instance of the halos that ' +
          'satisfy the requirements.')

    return subhalos, b[0]


def run_model(haloID=449659, distance=30*u.Mpc, inclination=60*u.deg,
              shape=[128, 64], pix_size=10*u.arcsec, channel_width=40*u.km/u.s,
              do_convolve=False, add_noise=False):
    """
    Create a model from the TNG50-1 simulation.

    Parameters
    ----------
    haloID : TYPE, optional
        DESCRIPTION. The default is 38350.
    distance : TYPE, optional
        DESCRIPTION. The default is 30*u.Mpc.
    inclination : TYPE, optional
        DESCRIPTION. The default is 60*u.deg.
    convolve : TYPE, optional
        DESCRIPTION. The default is False.
    noise : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    basePath = ('/data/beegfs/astro-storage/groups/walter/neeleman' +
                '/TNG/TNG50-1/output')

    # define the different components of the martinin model
    source = TNGSource(basePath, 99, haloID, distance=distance,
                       rotation={'L_coords': (inclination, 0*u.deg)},
                       ra=180*u.deg, dec=37.2333*u.deg)
    datacube = DataCube(n_px_x=shape[0], n_px_y=shape[0], n_channels=shape[1],
                        px_size=pix_size, channel_width=channel_width,
                        velocity_centre=source.vsys, ra=source.ra,
                        dec=source.dec)
    beam = GaussianBeam(bmaj=3.5*u.arcsec, bmin=3.5*u.arcsec, bpa=0*u.deg,
                        truncate=3)
    noise = GaussianNoise(rms=2.E-6*u.Jy/u.arcsec**2)
    spectral_model = GaussianSpectrum(sigma='thermal')
    sph_kernel = AdaptiveKernel((CubicSplineKernel(),
                                 GaussianKernel(truncate=6)))

    # run martini
    M = Martini(source=source, datacube=datacube, beam=beam, noise=noise,
                spectral_model=spectral_model, sph_kernel=sph_kernel)
    M.insert_source_in_cube(printfreq=20)
    if add_noise:
        M.add_noise()
    if do_convolve:
        M.convolve_beam()

    # write the results to disk
    name = '{}_{}_{}_{}_{}_{}_{}'.format(haloID, distance.value,
                                         inclination.value, shape[0], shape[1],
                                         pix_size.value, channel_width.value)
    M.write_fits('../Fits/' + name + '.fits', channels='frequency')
    M.write_beam_fits('../Fits/' + name + '_beam.fits', channels='frequency')
    M.write_hdf5('../Hdf5/' + name + '.hdf5', channels='frequency')


def create_quick_moments(name):
    """
    Create quick moments for the model.

    Parameters
    ----------
    name : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    try:
        f = h5py.File('../Hdf5/' + name + '.hdf5', 'r')
    except IOError:
        f = h5py.File('../TNGModel/Hdf5/' + name + '.hdf5', 'r')
    FluxCube = f['FluxCube'][()]
    vch = f['channel_mids'][()] / 1E3

    np.seterr(all='ignore')
    fig = plt.figure(figsize=(20, 5))
    sp1 = fig.add_subplot(1, 3, 1)
    sp2 = fig.add_subplot(1, 3, 2)
    sp3 = fig.add_subplot(1, 3, 3)
    rms = np.std(FluxCube[:16, :16])
    clip = np.where(FluxCube > 0 * rms, 1, 0)
    mom0 = np.sum(FluxCube, axis=-1)
    mask = np.where(mom0 > .00000, 1, np.nan)
    mom1 = np.sum(FluxCube * clip * vch, axis=-1) / mom0
    mom2 = np.sqrt(np.sum(FluxCube * clip *
                          np.power(vch - mom1[..., np.newaxis], 2),
                          axis=-1) / mom0)
    im1 = sp1.imshow(mom0.T, cmap='Greys', aspect=1.0, origin='lower')
    plt.colorbar(im1, ax=sp1, label='mom0 (Jy/pixel)')
    im2 = sp2.imshow((mom1*mask).T, cmap='RdBu', aspect=1.0, origin='lower')
    plt.colorbar(im2, ax=sp2, label='mom1 (km/s)')
    im3 = sp3.imshow((mom2*mask).T, cmap='magma', aspect=1.0, origin='lower')
    plt.colorbar(im3, ax=sp3, label='mom2 (km/s)')
    for sp in sp1, sp2, sp3:
        sp.set_xlabel('x (arcsec)')
        sp.set_ylabel('y (arcsec)')
    plt.subplots_adjust(wspace=.3)

    plt.savefig('../Fig/' + name + '.pdf', dpi=300)

