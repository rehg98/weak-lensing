from astropy.io import fits
import numpy as np
import scipy.ndimage
from scipy import fftpack
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import scipy.special as SSp

np.seterr(divide = 'ignore', invalid = 'ignore')

def gaussianizepdf(denf,avgrepeats=True, sigmagauss = None,assumelognormal=True):
    denshape = denf.shape
    denff = denf.flatten()
    o_f = np.argsort(denff)
    gaussf = 0.*denff.astype(np.float)
    lenny = len(gaussf)

    if (sigmagauss == None):
        if assumelognormal:
            sigmagauss = np.sqrt(np.log1p(np.var(denff)))
        else:
            sigmagauss = np.std(denff)
    #print(sigmagauss)

    step = 1./lenny

    gaussf[o_f] = np.sqrt(2.)*sigmagauss*SSp.erfinv(2.*np.arange(0.5*step,1,step)-1.)

    # average together repeated elements
    if (avgrepeats):
        cuts = np.searchsorted(denff[o_f],np.unique(denff[o_f]))
        #print(len(cuts),'cuts')
        for i in range(len(cuts)-1):
            gaussf[o_f[cuts[i]:cuts[i+1]]] = np.mean(gaussf[o_f[cuts[i]:cuts[i+1]]])
        # get the last one
        gaussf[o_f[cuts[-1]:]]=np.mean(gaussf[o_f[cuts[-1]:]])

    gaussf = gaussf.reshape(denshape)

    return gaussf

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

def SNR(powermean, covar):
    #Calculate Signal-to-Noise ratio given a set of powerspectra 

    powermeanmat = np.mat(powermean)    

    SNRsquare = powermeanmat * (covar.I * powermeanmat.T)
    
    return np.sqrt(SNRsquare)

def corr_mat(covar):
    #Calculate the correlation matrix
    
    diag_sqrt = np.sqrt(np.diag(covar))
    X, Y = np.meshgrid(diag_sqrt, diag_sqrt)
    return covar / (X*Y)


def toPowspec(image_num):
    #print(image_num)
    image = fits.open('/tigress/jialiu/CMBL_maps_46cosmo/Om0.296_Ol0.704_w-1.000_si0.786/WLconv_z1100.00_' + '{:04d}'.format(image_num) + 'r.fits')[0].data.astype(float)
    #1 Arcminute = 9.75; 2 Arc = 19.5; 5 Arc = 48.76; 10 Arc = 97.5
    image_1 = scipy.ndimage.filters.gaussian_filter(image, 9.75)
   # image_2 = scipy.ndimage.filters.gaussian_filter(image, 19.5)
   # image_5 = scipy.ndimage.filters.gaussian_filter(image, 48.76)
   # image_10 = scipy.ndimage.filters.gaussian_filter(image, 97.5)
   
    image_1 = gaussianizepdf(image_1)
   # image_2 = gaussianizepdf(image_2)
   # image_5 = gaussianizepdf(image_5)
   # image_10 = gaussianizepdf(image_10)

    F_1 = fftpack.fftshift(fftpack.fft2(image_1))
    psd2D_1 = np.abs(F_1)**2
   # F_2 = fftpack.fftshift(fftpack.fft2(image_2))
   # psd2D_2 = np.abs(F_2)**2
   # F_5 = fftpack.fftshift(fftpack.fft2(image_5))
   # psd2D_5 = np.abs(F_5)**2
   # F_10 = fftpack.fftshift(fftpack.fft2(image_10))
   # psd2D_10 = np.abs(F_10)**2

    ells, powspec_1 = PowerSpectrum(psd2D_1, sizedeg = 12.25, size = 2048, bins = 50)
   # ells, powspec_2 = PowerSpectrum(psd2D_2, sizedeg = 12.25, size = 2048, bins = 50)
   # ells, powspec_5 = PowerSpectrum(psd2D_5, sizedeg = 12.25, size = 2048, bins = 50)
   # ells, powspec_10 = PowerSpectrum(psd2D_10, sizedeg = 12.25, size = 2048, bins = 50)

    return ells, powspec_1#,  powspec_2, powspec_5, powspec_10


image_range = np.arange(1, 1025)

from emcee.utils import MPIPool
pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

results = np.array(pool.map(toPowspec, image_range))
pool.close()

ells = results[0, 0]

powspecs_1 = np.array([r[1] for r in results])
powermean_1 = np.mean(powspecs_1, axis = 0) 
#powspecs_2 = np.array([r[1] for r in results])
#powermean_2 = np.mean(powspecs_2, axis = 0) 
#powspecs_5 = np.array([r[2] for r in results])
#powermean_5 = np.mean(powspecs_5, axis = 0) 
#powspecs_10 = np.array([r[3] for r in results])
#powermean_10 = np.mean(powspecs_10, axis = 0) 

cut = np.argmax(ells >= 3000)

tpowspecs_1 = np.array([p[:cut] for p in powspecs_1])
tpowermean_1 = powermean_1[:cut]
#tpowspecs_2 = np.array([p[:cut] for p in powspecs_2])
#tpowermean_2 = powermean_2[:cut]
#tpowspecs_5 = np.array([p[:cut] for p in powspecs_5])
#tpowermean_5 = powermean_5[:cut]
#tpowspecs_10 = np.array([p[:cut] for p in powspecs_10])
#tpowermean_10 = powermean_10[:cut]

covar = np.mat(np.cov(powspecs_1, rowvar = 0))
covar_1 = np.mat(np.cov(tpowspecs_1, rowvar = 0))
#covar_2 = np.mat(np.cov(tpowspecs_2, rowvar = 0))
#covar_5 = np.mat(np.cov(tpowspecs_5, rowvar = 0))
#covar_10 = np.mat(np.cov(tpowspecs_10, rowvar = 0))

#print("\nCovariance Matrix: ")
#print(covar)

#fig1 = plt.figure(figsize=(6, 3.4))

#ax = fig1.add_subplot(111)
#ax.set_title('Covariance Matrix Heat Map')
#plt.imshow(np.array(covar), cmap = 'hot')
#ax.set_aspect('equal')

#cax = fig1.add_axes([0.12, 0.1, 0.78, 0.8])
#cax.get_xaxis().set_visible(False)
#cax.get_yaxis().set_visible(False)
#cax.patch.set_alpha(0)
#cax.set_frame_on(False)
#plt.colorbar(orientation = 'vertical')

#fig1.savefig("covar_gauss.png")


correl = corr_mat(covar)
#print("\nCorrelation Matrix: ")
#print(correl)

fig2 = plt.figure(figsize=(6, 3.4))

ax = fig2.add_subplot(111)
ax.set_title('Correlation Matrix Heat Map - Noiseless, Gaussianized')
plt.imshow(np.array(correl), cmap = 'hot', vmin = -0.07, vmax = 1.0)
ax.set_aspect('equal')

cax = fig2.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation = 'vertical')

fig2.savefig("corrmat_gauss.png")

s2r_1 = SNR(tpowermean_1, covar_1)
#s2r_2 = SNR(tpowermean_2, covar_2)
#s2r_5 = SNR(tpowermean_5, covar_5)
#s2r_10 = SNR(tpowermean_10, covar_10)
print("\nSignal-to-Noise ratio: ")
print(s2r_1)
#print(s2r_2)
#print(s2r_5)
#print(s2r_10)


#fig3 = plt.figure()
#ax1 = fig3.add_subplot(111)
#
#ax1.set_xscale("log", nonposx='clip')
#ax1.set_yscale("log", nonposy='clip')
#
#std_P = np.std(powspecs_1, axis = 0)
#plt.errorbar(ells, powermean_1, std_P)
#
#ax1.set_title("Mean Power Spectrum -- Normal Maps, Gaussianized, 1 Arcminute Smoothing (7/18/17)")
#ax1.set_ylabel(r'$\frac{\ell (\ell + 1) C_\ell}{2\pi}$', fontsize = 20)
#ax1.set_xlabel(r'$\ell$', fontsize = 20)
#ax1.set_xlim(1e2, 1e4)
#fig3.savefig("powermean_gauss.png", bbox_inches = 'tight')



#fig4 = plt.figure()
#for p in powspecs:
#    plt.loglog(ells, p)
#plt.title("All Power Spectra -- Normal Maps, Gaussianized, 1 Arcminute Smoothing (7/18/17)")
#plt.ylabel(r'$\frac{\ell (\ell + 1) C_\ell}{2\pi}$', fontsize = 20)
#plt.xlabel(r'$\ell$', fontsize = 20)
#fig4.savefig("powerspecs_gauss.png", bbox_inches = 'tight')
