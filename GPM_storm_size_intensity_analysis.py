from netCDF4 import Dataset
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import glob
from scipy import sparse
from scipy import ndimage
import cartopy.crs as ccrs
import cartopy.feature as cfeature
plt.ion()

# GPM files
GPMfiles = sorted(glob.glob('/home/andy/Dropbox/GPM_data_all_20181226/*.nc4'))[:10000][::48]

# Initialize with a set of data of the right size
#ncdata = Dataset('/home/awickert/Downloads/3B-HHR-E.MS.MRG.3IMERG.20140312-S000000-E002959.0000.V05B.HDF5.nc')
ncdata = Dataset(GPMfiles[0])
Y,X = np.meshgrid(ncdata.variables['lat'][:], ncdata.variables['lon'][:])
Y = np.flipud(Y.transpose())
X = X.transpose()

# Cell areas
cell_area_km2 = np.cos(Y * np.pi/180.) * np.pi * 12713.6 / 360. * np.pi * 12756 / 360. * 0.1**2

# Calculate storm sizes, intensities, and areas for each cell
# (intensities for each cell is a simpler sum -- don't need storm area)

# PROBLEM: BLOBS TOUCH EDGE. SO THIS MUST BE ONLY A PROOF-OF-CONCEPT
# MUST HAVE LARGER / CONTINUOUS MAP TO DO THIS ACCURATELY

ii = 1
storm_sizes_all = []
storm_rainfall_rates_all = []
storm_sizes_over_cells_all = []

print('Storm size')

for GPMfile in GPMfiles:

    print(GPMfile.split('/')[-1])
    print('%06.2f' %(100 * ii / float(len(GPMfiles))) +'%')
    ii += 1
    
    # Precipitation
    ncdata = Dataset(GPMfile, 'r')
    precip = ncdata.variables['precipitationCal'][:]
    precip = np.flipud(precip.transpose())

    # Storm size stats -- storm = contiguous area of any precip
    blobs, nblobs = ndimage.label( precip > 0 )
    blobsizes = []
    blob_rainfall_rates = []
    blobsizes_over_cells = np.zeros(blobs.shape)
    for i in range(1,nblobs+1):
        blob = (blobs == i)
        blob_area_weighted = blob * cell_area_km2
        blobsizes.append(np.sum( blob_area_weighted ) )
        blob_rainfall_rates.append( np.mean( blob_area_weighted * precip ) )
        blobsizes_over_cells += blob * blobsizes[-1]

    storm_sizes_all.append(blobsizes)
    storm_rainfall_rates_all.append(blob_rainfall_rates)
    storm_sizes_over_cells_all.append(blobsizes_over_cells)

storm_sizes_all = np.array(storm_sizes_all)
storm_rainfall_rates_all = np.array(storm_rainfall_rates_all)
storm_sizes_over_cells_all = np.array(storm_sizes_over_cells_all)

storm_sizes_over_cells_all_0nan = storm_sizes_over_cells_all.copy()
storm_sizes_over_cells_all_0nan[storm_sizes_over_cells_all == 0] = np.nan
mean_storm_size_over_cells = np.nanmean(storm_sizes_over_cells_all_0nan, axis=0)

print('Mean precipitation')
ii = 0
precip_total = np.zeros(precip.shape)
for GPMfile in GPMfiles:

    print(GPMfile.split('/')[-1])
    print('%06.2f' %(100 * ii / float(len(GPMfiles))) +'%')
    ii += 1
    
    # Precipitation
    ncdata = Dataset(GPMfile, 'r')
    precip = ncdata.variables['precipitationCal'][:]
    precip = np.flipud(precip.transpose())

    precip_total +=  precip

mean_precip = precip_total / float(ii) * 24 * 365.25 # [mm/yr]

# Prepare for Cartopy plotting
lats = ncdata['lat'][:]
lons = ncdata['lon'][:]
data_extent = (np.min(lons), np.max(lons), np.min(lats), np.max(lats))

#central_lat = 37.5
#central_lon = -96
extent = [-120, -70, 22, 50]
central_lon = np.mean(extent[:2])
central_lat = np.mean(extent[2:])

fig = plt.figure()
ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
ax.set_extent(extent)
#ax.add_feature(cfeature.LAND)
#ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle='-')
ax.add_feature(cfeature.STATES, linestyle=':')
#ax.add_feature(cfeature.LAKES, alpha=0.5)
#ax.add_feature(cfeature.RIVERS)
#ax.contourf(lons, lats, np.flipud(mean_storm_size_over_cells), transform=ccrs.PlateCarree())
cs = ax.imshow(mean_storm_size_over_cells/1E6, origin='upper', transform=ccrs.PlateCarree(), extent=data_extent, vmax=0.4)
#cs = ax.imshow(np.nanmedian(storm_sizes_over_cells_all_0nan, axis=0)/1E6, origin='upper', transform=ccrs.PlateCarree(), extent=data_extent, vmax=0.4)
cax,kw = matplotlib.colorbar.make_axes(ax,location='bottom',pad=0.05,shrink=0.7)
cbar = fig.colorbar(cs, cax=cax, extend='max', **kw) #extend='both',
cbar.set_label(r'Mean area of storm precipitating on cell [$\times 10^6$ km$^2$]', fontsize=14)
#cbar.set_label(r'Median area of storm precipitating on cell ($\times 10^6$ km$^2$)', fontsize=14)

#plt.title('Mean extent of storm raining on each pixel [km^2]')
#plt.colorbar()


# Average precip

fig = plt.figure()
ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
ax.set_extent(extent)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle='-')
ax.add_feature(cfeature.STATES, linestyle=':')
cs = ax.imshow(mean_precip, origin='upper', transform=ccrs.PlateCarree(), extent=data_extent, vmax=4000)
cax,kw = matplotlib.colorbar.make_axes(ax,location='bottom',pad=0.05,shrink=0.7)
cbar = fig.colorbar(cs, cax=cax, extend='max', **kw) #extend='both',
cbar.set_label(r'Mean precipitation [mm yr$^{-1}$]', fontsize=14)

