from netCDF4 import Dataset
import numpy as np
from matplotlib import pyplot as plt
import glob
from scipy import sparse
plt.ion()

# Basins
ncbasins = Dataset('/home/andy/Dropbox/Papers/InProgress/DrainageArea_Discharge/GPM_TRMM/BasinsAtGauge_USA_raster_6arcmin_GPM.nc', 'r')
Xb,Yb = np.meshgrid(ncbasins.variables['lon'][:], ncbasins.variables['lat'][:])
basins = ncbasins.variables['basins'][:]
mean_precip_mmyr_PRISM = ncbasins.variables['PRISMprecip'][:]

# GPM files
GPMfiles = sorted(glob.glob('/home/andy/Dropbox/GPM_data_all_20181226/*.nc4'))[:30000][::100]

# Initialize with a set of data of the right size
#ncdata = Dataset('/home/awickert/Downloads/3B-HHR-E.MS.MRG.3IMERG.20140312-S000000-E002959.0000.V05B.HDF5.nc')
ncdata = Dataset(GPMfiles[0])
Y,X = np.meshgrid(ncdata.variables['lat'][:], ncdata.variables['lon'][:])
Y = np.flipud(Y.transpose())
X = X.transpose()

# Pad drainage basin map to match
# Storm map should always be >= basin map area or else there is an error
wpad = int( np.round ( ( np.min(Xb) - np.min(X) ) * 10 ) )
epad = int( np.round ( ( np.max(X) - np.max(Xb) ) * 10 ) )
spad = int( np.round ( ( np.min(Yb) - np.min(Y) ) * 10 ) )
npad = int( np.round ( ( np.max(Y) - np.max(Yb) ) * 10 ) )
basinspad = np.pad(basins, [(0,0), (npad,spad), (wpad,epad)], mode='constant', constant_values=np.nan)

# Grid area
cell_area_km2 = np.cos(Y * np.pi/180.) * np.pi * 12713.6 / 360. * np.pi * 12756 / 360. * 0.1**2

# Basin area
# Could update this to be array-based...
basin_ncells = []
basinsizes_km2 = []
for basin in basinspad:
    basin_ncells.append( np.sum(basin) )
    basinsizes_km2.append( np.sum( basin * cell_area_km2 ) )
basin_ncells = np.array(basin_ncells)
basinsizes_km2 = np.array(basinsizes_km2)


# Size distributions of basin coverage for each storm (and rainfall rate)
# =======================================================================

stormfract_all = []
precip_in_basin_all = []
precip_in_basin_mean_all = []

ii = 1
for GPMfile in GPMfiles[:5]:

    print(GPMfile.split('/')[-1])
    print('%06.2f' %(100 * ii / float(len(GPMfiles))) +'%')
    ii += 1

    # Precipitation
    ncdata = Dataset(GPMfile, 'r')
    precip = ncdata.variables['precipitationCal'][:]
    precip = np.flipud(precip.transpose())

    # Precip map
    basinprecip = basinspad * precip
    basinprecip_sparse = []
    for _slice in basinprecip:
        basinprecip_sparse.append(sparse.csr_matrix(_slice))
    precip_in_basin_all.append(basinprecip_sparse)
    
    # Precip total -- weight for area
    precip_in_basin_mean_all.append(np.sum( basinprecip * cell_area_km2,
                                            axis=(1,2)) / basinsizes_km2)

    # Storm fraction
    basinstorm_areas = np.sum( (basinprecip > 0) * cell_area_km2, axis=(1,2))
    stormfract = basinstorm_areas / basinsizes_km2
    stormfract_all.append(stormfract)

stormfract_all = np.array(stormfract_all)
precip_in_basin_all = np.array(precip_in_basin_all)
precip_in_basin_mean_all = np.array(precip_in_basin_mean_all)

# First-pass analysis
# ===================

stormfract_nan = stormfract_all.copy()
stormfract_nan[stormfract_nan == 0] = np.nan
mean_stormfract = np.nanmean(stormfract_nan, axis=0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.scatter( basinsizes_km2, mean_stormfract, c=np.log(mean_precip_mmyr_PRISM), marker='o', cmap='seismic_r' )

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.scatter( basinsizes_km2, np.nanmedian(stormfract_nan, axis=0), c=np.log(mean_precip_mmyr_PRISM), marker='o', cmap='seismic_r' )

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.scatter( basinsizes_km2, np.nanmax(stormfract_nan, axis=0), c=np.log(mean_precip_mmyr_PRISM), marker='o', cmap='seismic_r' )

def adjacent_average(val):
    return (val[:-1] + val[1:])/2.

plt.figure()
for i in range(10):
    hist = np.histogram(stormfract_nan[:,i][np.isfinite(stormfract_nan[:,i])], 20)
    plt.plot(adjacent_average(hist[1]), hist[0], linewidth=2)
    
subset = stormfract_nan[:, (24000 <= basinsizes_km2) * (basinsizes_km2 < 26000)]
subset_precip = mean_precip_mmyr_PRISM[(24000 <= basinsizes_km2) * (basinsizes_km2 < 26000)]
for i in range(subset.shape[-1]):
    hist = np.histogram(subset[:,i][np.isfinite(subset[:,i])], 20)
    plt.loglog(adjacent_average(hist[1]), hist[0], linewidth=2)

subset = stormfract_nan[:, (9000 <= basinsizes_km2) * (basinsizes_km2 < 11000)]
for i in range(subset.shape[-1]):
    hist = np.histogram(subset[:,i][np.isfinite(subset[:,i])], 20)
    plt.loglog(adjacent_average(hist[1]), hist[0], linewidth=2)

subset = stormfract_nan[:, (500 <= basinsizes_km2) * (basinsizes_km2 < 1000)]
for i in range(subset.shape[-1]):
    hist = np.histogram(subset[:,i][np.isfinite(subset[:,i])], 20)
    plt.loglog(adjacent_average(hist[1]), hist[0], 'ko', linewidth=2)


# Contour plot: (x) Drainage area, (y) mean annual precip, (c) mean storm fract
# Well, starting as scatter plot, and then contouring

# Grid data
#from matplotlib.mlab import griddata
from scipy.interpolate import griddata
from scipy.ndimage.filters import gaussian_filter

def grid(x, y, z, resX=50, resY=50):
    "Convert 3 column data to matplotlib grid"
    x = x[np.isfinite(z)]
    y = y[np.isfinite(z)]
    z = z[np.isfinite(z)]
    x = x[np.isfinite(y)]
    z = z[np.isfinite(y)]
    y = y[np.isfinite(y)]
    y = y[np.isfinite(x)]
    z = z[np.isfinite(x)]
    x = x[np.isfinite(x)]
    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata( (x, y), z, (X, Y), method='linear')
    return X[0,:], Y[:,0], Z

def loggrid(x, y, z, resX=50, resY=50, filter=True, sigma=0.7):
    "Convert 3 column data to matplotlib grid"
    x = x[np.isfinite(z)]
    y = y[np.isfinite(z)]
    z = z[np.isfinite(z)]
    x = x[np.isfinite(y)]
    z = z[np.isfinite(y)]
    y = y[np.isfinite(y)]
    y = y[np.isfinite(x)]
    z = z[np.isfinite(x)]
    x = x[np.isfinite(x)]
    xi = np.logspace(np.log10(min(x)), np.log10(max(x)), resX)
    yi = np.logspace(np.log10(min(y)), np.log10(max(y)), resY)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata( (x, y), z, (X, Y), method='linear')
    if filter:
        Z = gaussian_filter(Z, sigma)
    return X, Y, Z


"""
from scipy.stats import binned_statistic_2d

def grid(x, y, z)
ret = binned_statistic_2d(x, y, None, 'count', bins=[binx,biny], \
    expand_binnumbers=True)
"""


fig = plt.figure()
ax = fig.add_subplot(111)
ax.cla()
ax.set_xscale('log')
ax.set_yscale('log')
scatter = ax.scatter( basinsizes_km2, mean_precip_mmyr_PRISM, c=mean_stormfract, marker='o', edgecolors='none', cmap='viridis' )
X,Y,Z = loggrid(basinsizes_km2, mean_precip_mmyr_PRISM, mean_stormfract, 200, 200, True, 5.)
contour = ax.contour( X,Y,Z,4, colors='k', linewidths=3 )
ax.set_xlabel( 'Basin area [km$^2$]', fontsize=16 )
ax.set_ylabel( 'Mean annual precipitation [mm]', fontsize=16 )
cbar = fig.colorbar(scatter)



