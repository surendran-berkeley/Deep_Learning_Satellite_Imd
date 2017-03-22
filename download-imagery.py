import wget
import numpy as np
import pandas as pd
import time
import os
import os.path
import sys
import getopt
from osgeo import gdal, ogr, osr
from scipy import misc, ndimage
import cStringIO
gdal.UseExceptions()
import urllib
from retrying import retry

# Helper function to read a raster file
def read_raster(raster_file):
    """
    Function
    --------
    read_raster

    Given a raster file, get the pixel size, pixel location, and pixel value

    Parameters
    ----------
    raster_file : string
        Path to the raster file

    Returns
    -------
    x_size : float
        Pixel size
    top_left_x_coords : numpy.ndarray  shape: (number of columns,)
        Longitude of the top-left point in each pixel
    top_left_y_coords : numpy.ndarray  shape: (number of rows,)
        Latitude of the top-left point in each pixel
    centroid_x_coords : numpy.ndarray  shape: (number of columns,)
        Longitude of the centroid in each pixel
    centroid_y_coords : numpy.ndarray  shape: (number of rows,)
        Latitude of the centroid in each pixel
    bands_data : numpy.ndarray  shape: (number of rows, number of columns, 1)
        Pixel value
    """
    raster_dataset = gdal.Open(raster_file, gdal.GA_ReadOnly)
    # get project coordination
    proj = raster_dataset.GetProjectionRef()
    bands_data = []
    # Loop through all raster bands
    for b in range(1, raster_dataset.RasterCount + 1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())
        no_data_value = band.GetNoDataValue()
    bands_data = np.dstack(bands_data)
    rows, cols, n_bands = bands_data.shape

    # Get the metadata of the raster
    geo_transform = raster_dataset.GetGeoTransform()
    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = geo_transform

    # Get location of each pixel
    x_size = 1.0 / int(round(1 / float(x_size)))
    y_size = - x_size
    y_index = np.arange(bands_data.shape[0])
    x_index = np.arange(bands_data.shape[1])
    top_left_x_coords = upper_left_x + x_index * x_size
    top_left_y_coords = upper_left_y + y_index * y_size
    # Add half of the cell size to get the centroid of the cell
    centroid_x_coords = top_left_x_coords + (x_size / 2)
    centroid_y_coords = top_left_y_coords + (y_size / 2)

    return (x_size, top_left_x_coords, top_left_y_coords, centroid_x_coords, centroid_y_coords, bands_data)


# Helper function to get the pixel index of the point
def get_cell_idx(lon, lat, top_left_x_coords, top_left_y_coords):
    """
    Function
    --------
    get_cell_idx

    Given a point location and all the pixel locations of the raster file,
    get the column and row index of the point in the raster

    Parameters
    ----------
    lon : float
        Longitude of the point
    lat : float
        Latitude of the point
    top_left_x_coords : numpy.ndarray  shape: (number of columns,)
        Longitude of the top-left point in each pixel
    top_left_y_coords : numpy.ndarray  shape: (number of rows,)
        Latitude of the top-left point in each pixel

    Returns
    -------
    lon_idx : int
        Column index
    lat_idx : int
        Row index
    """
    lon_idx = np.where(top_left_x_coords < lon)[0][-1]
    lat_idx = np.where(top_left_y_coords > lat)[0][-1]
    return lon_idx, lat_idx

# Helper function to read a shapefile
def get_shp_extent(shp_file):
    """
    Function
    --------
    get_shp_extent

    Given a shapefile, get the extent (boundaries)

    Parameters
    ----------
    shp_file : string
        Path to the shapefile

    Returns
    -------
    extent : tuple
        Boundary location of the shapefile (x_min, x_max, y_min, y_max)
    """
    inDriver = ogr.GetDriverByName("ESRI Shapefile")
    inDataSource = inDriver.Open(shp_file, 0)
    inLayer = inDataSource.GetLayer()
    extent = inLayer.GetExtent()
    # x_min_shp, x_max_shp, y_min_shp, y_max_shp = extent
    return extent

# Helper functions to download images from Google Maps API
@retry(wait_exponential_multiplier=1000, wait_exponential_max=3600000)
def save_img(url, file_path, file_name):
    """
    Function
    --------
    save_img

    Given a url of the map, save the image

    Parameters
    ----------
    url : string
        URL of the map from Google Map Static API
    file_path : string
        Folder name of the map
    file_name : string
        File name

    Returns
    -------
    None
    """
    a = urllib.urlopen(url).read()
    b = cStringIO.StringIO(a)
    image = ndimage.imread(b, mode='RGB')
    # when no image exists, api will return an image with the same color.
    # and in the center of the image, it said'Sorry. We have no imagery here'.
    # we should drop these images if large area of the image has the same color.
    if np.array_equal(image[:,:10,:],image[:,10:20,:]):
        pass
    else:
        misc.imsave(file_path + file_name, image[50:450, :, :])

# print-out help / instructions
def help():
    print('\nusage:')
    print('use --keyid= for setting the Google Maps Static API key number.')
    print('use --topidx= for setting the top row to start looping over the country.')
    print('(note: the top-left index should not be altered.)\n')

# store country list
country_list = ['malawi', 'nigeria', 'rwanda', 'tanzania', 'uganda']

# run main
def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["country=", "topidx=", "bottomidx=", "keyid="])
    except getopt.GetoptError:
        sys.exit(2)

    optlist = [opt for opt, arg in opts]
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            help()
            sys.exit(2)
        elif opt == '--country':
            country = arg
            if country not in country_list:
                print('\nCountry %s not available. Specify one of the following countries: %s\n' % (country, country_list))
                sys.exit(2)
        elif opt == '--topidx':
            top_idx_param = int(arg)
        elif opt == '--bottomidx':
            bottom_idx_param = int(arg)
        elif opt == '--keyid':
            key_id = arg

    # print top row index
    if '--topidx' in optlist:
        print('top index row: %s' % top_idx_param)

    # print bottom row index
    if '--bottomidx' in optlist:
        print('bottom index row: %s' % bottom_idx_param)

    # print key
    key = os.getenv('GMAP_API_KEY_%s' % key_id)
    print('key variable: GMAP_API_KEY_%s, key: %s' % (key_id, key))

    # retrieve nightlights data (run only once)
    #night_image_url = 'https://ngdc.noaa.gov/eog/data/web_data/v4composites/F182010.v4.tar'
    #wget.download(night_image_url)

    # this illustrates how you can read the nightlight image
    raster_file = 'data/nightlights/F182010.v4/F182010.v4d_web.stable_lights.avg_vis.tif'
    x_size, top_left_x_coords, top_left_y_coords, centroid_x_coords, centroid_y_coords, bands_data = read_raster(raster_file)

    # save the result in compressed format - see https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html
    #np.savez('data/nightlights/nightlight.npz', top_left_x_coords=top_left_x_coords, top_left_y_coords=top_left_y_coords, bands_data=bands_data)

    # create dataframe; create dict of shapefile paths
    df = pd.DataFrame()
    country_shp_files = {
        'ghana':'GHGE71FL.shp',
        'malawi':'MWGE71FL.shp',
        'nigeria':'NGGE71FL.shp',
        'rwanda':'Sector_Boundary_2012.shp',
        'tanzania':'TZGE7AFL.shp',
        'uganda':'UGGE71FL.shp',
    }

    # Now read in the shapefile for Rwanda and extract the edges of the country
    inShapefile = "data/shapefiles/%s/%s" % ( country, country_shp_files[country] )
    x_min_shp, x_max_shp, y_min_shp, y_max_shp = get_shp_extent(inShapefile)

    # retrieve coordinates
    left_idx, top_idx = get_cell_idx(x_min_shp, y_max_shp, top_left_x_coords, top_left_y_coords )
    right_idx, bottom_idx = get_cell_idx(x_max_shp, y_min_shp, top_left_x_coords, top_left_y_coords)

    # calculate # images (based on row, height)
    num_images = (bottom_idx - top_idx + 1) * (right_idx - left_idx + 1)
    row = {
      'country':country,
      'left_idx':left_idx,
      'top_idx':top_idx,
      'right_idx':right_idx,
      'bottom_idx':bottom_idx,
      'num_images':num_images,
    }
    df = df.append(row, ignore_index=True)

    # Download Daytime Satellite Imagery; retrieve and save images
    # note: set top index (from passed argument)
    if '--topidx' in optlist:
        top_idx = top_idx_param

    # print bottom row index
    if '--bottomidx' in optlist:
        bottom_idx = bottom_idx_param

    m = 1
    for j in xrange(top_idx, bottom_idx + 1):
        for i in xrange(left_idx, right_idx + 1):
            lon = centroid_x_coords[i]
            lat = centroid_y_coords[j]
            url = 'https://maps.googleapis.com/maps/api/staticmap?center="' + str(lat) + ',' + \
                   str(lon) + '"&zoom=16&size=400x500&maptype=satellite&key=' + key
            lightness = bands_data[j, i, 0]
            file_path = '/data/google_image/%s/' % country + str(lightness) + '/'
            if not os.path.isdir(file_path):
                os.makedirs(file_path)
            file_name = str(i) + '_' + str(j) +'.png'
            if not os.path.isfile(os.path.join(file_path, file_name)):
                save_img(url, file_path, file_name)
            if m % 50 == 0:
                print 'm: %s, lightness: %s, file_name: %s, url: %s' % (m, lightness, file_name, url)
            m += 1
            if m == 24900:
                print '\ntop-left corner i:%s, j:%s\nfinal cell i: %s, j:%s\n(N=%s iterations)\n' % (left_idx, top_idx, i, j, m)
                stop

    # print final indices
    print '\ntop-left corner i:%s, j:%s\nfinal cell i: %s, j:%s\n(N=%s iterations)\n' % (left_idx, top_idx, i, j, m)

    # write indices for all countries' shape files
    columns = ['country', 'left_idx', 'top_idx', 'right_idx', 'bottom_idx', 'num_images']
    df = df[columns]
    print df

if __name__ == "__main__":
    main(sys.argv[1:])
