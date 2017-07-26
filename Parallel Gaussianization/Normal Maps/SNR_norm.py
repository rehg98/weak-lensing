from astropy.io import fits
import numpy as np
import scipy.ndimage
from scipy import fftpack
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
from PIL import Image

np.seterr(divide = 'ignore', invalid = 'ignore')

def power1D(image, num_bins):
    
    y, x = np.indices(image.shape)
    center = np.array([(x.max() - x.min()) / 2., (x.max() - x.min()) / 2.])
    
    if image.shape[0] % 2 == 0:
        center += 0.5
    
    radii = np.hypot(x - center[0], y - center[1])
    
    sorted_radii_indices = np.argsort(radii.flat)
    sorted_radii = radii.flat[sorted_radii_indices]
    sorted_pixels = image.flat[sorted_radii_indices]
    
    bins = np.logspace(0, np.log10(image.shape[0]/2.), num_bins + 1)
    
    bin_weights = np.histogram(sorted_radii, bins)[0]
    bin_edges = np.cumsum(bin_weights)
    pixel_sums = np.cumsum(sorted_pixels, dtype=float)
    bin_totals = pixel_sums[bin_edges[1:] - 1] - pixel_sums[bin_edges[:-1] - 1]
    radial_prof = bin_totals/bin_weights[1:]
    
    return bins[1:], radial_prof

def PowerSpectrum(psd2D, sizedeg = 12.25, size = 2048, bins = 50):
    
    ells, psd1D = power1D(psd2D, num_bins = 50)
    
    edge2center = lambda x: x[:-1]+0.5*(x[1:]-x[:-1])
    ells = edge2center(ells)
    
    ells *= 360. / np.sqrt(sizedeg)
    norm = ((2 * np.pi * np.sqrt(sizedeg) / 360.0) ** 2) / (size ** 2) ** 2
    powspec = ells * (ells + 1) / (2 * np.pi) * norm * psd1D
    
    last_nan = np.where(np.isnan(powspec))[0][-1]
    ells = ells[last_nan + 1:]
    powspec = powspec[last_nan + 1:]
    return ells, powspec

def SNR(ells, powerspecs, covar):
    #Calculate Signal-to-Noise ratio given a set of powerspectra 
    
    powermean = np.mean(powerspecs, axis = 0) 

    cut = np.argmax(ells >= 5000)
    powermeanmat = np.mat(powermean[:cut])    
    
    SNRsquare = powermeanmat * (covar.I * powermeanmat.T)
    
    return np.sqrt(SNRsquare), powermean

def corr_mat(covar):
    #Calculate the correlation matrix
    
    diag_sqrt = np.sqrt(np.diag(covar))
    X, Y = np.meshgrid(diag_sqrt, diag_sqrt)
    return covar / (X*Y)


def toPowspec(image_num):
    image = fits.open('/tigress/jialiu/CMBL_maps_46cosmo/Om0.296_Ol0.704_w-1.000_si0.786/WLconv_z1100.00_' + '{:04d}'.format(image_num) + 'r.fits')[0].data.astype(float)
    #1 Arcminute = 9.75; 2 Arc = 19.5; 5 Arc = 48.76; 10 Arc = 97.5
    image = scipy.ndimage.filters.gaussian_filter(image, 9.75)
    F = fftpack.fftshift(fftpack.fft2(image))
    psd2D = np.abs(F)**2
    ells, powspec = PowerSpectrum(psd2D, sizedeg = 12.25, size = 2048, bins = 50)

    return ells, powspec


image_range = np.arange(1, 1025)

from emcee.utils import MPIPool
pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

results = np.array(pool.map(toPowspec, image_range))
pool.close()

ells = results[0, 0]
powspecs = np.array([r[1] for r in results])


covar = np.mat(np.cov(powspecs, rowvar = 0))
print("\nCovariance Matrix: ")
print(covar)

fig1 = plt.figure(figsize=(6, 3.4))

ax = fig1.add_subplot(111)
ax.set_title('Covariance Matrix Heat Map')
plt.imshow(np.array(covar), cmap = 'hot')
ax.set_aspect('equal')

cax = fig1.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation = 'vertical')

fig1.savefig("covar.png")


correl = corr_mat(covar)
print("\nCorrelation Matrix: ")
print(correl)

fig2 = plt.figure(figsize=(6, 3.4))

ax = fig2.add_subplot(111)
ax.set_title('Correlation Matrix Heat Map')
plt.imshow(np.array(correl), cmap = 'hot')
ax.set_aspect('equal')

cax = fig2.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation = 'vertical')

fig2.savefig("corrmat.png")



s2r, powermean = SNR(ells, powspecs, covar)
print("\nSignal-to-Noise ratio: ")
print(s2r)


fig3 = plt.figure()
ax1 = fig3.add_subplot(111)

ax1.set_xscale("log", nonposx='clip')
ax1.set_yscale("log", nonposy='clip')

std_P = np.std(powspecs, axis = 0)
plt.errorbar(ells, powermean, std_P)

ax1.set_title("Mean Power Spectrum -- Normal Maps, Ungaussianized, 1 Arcminute Smoothing (7/18/17)")
ax1.set_ylabel(r'$\frac{\ell (\ell + 1) C_\ell}{2\pi}$', fontsize = 20)
ax1.set_xlabel(r'$\ell$', fontsize = 20)
ax1.set_xlim(1e2, 1e4)
fig3.savefig("powermean.png", bbox_inches = 'tight')


fig4 = plt.figure()
for p in powspecs:
    plt.loglog(ells, p)
plt.title("All Power Spectra -- Normal Maps, Ungaussianized, 1 Arcminute Smoothing (7/18/17)")
plt.ylabel(r'$\frac{\ell (\ell + 1) C_\ell}{2\pi}$', fontsize = 20)
plt.xlabel(r'$\ell$', fontsize = 20)
fig4.savefig("powerspecs.png", bbox_inches = 'tight')
