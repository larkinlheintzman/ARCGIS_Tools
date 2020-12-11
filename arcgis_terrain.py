from arcgis.gis import GIS
from arcgis.geocoding import geocode
from arcgis.geometry import Geometry
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from math import gcd
import math
from pyproj import Transformer, Proj, transform
from scipy.interpolate import griddata
from scipy import interpolate
from scipy import ndimage
import time

# This function computes the factor of the argument passed
def factorization(n):
    factors = []
    def get_factor(n):
        x_fixed = 2
        cycle_size = 2
        x = 2
        factor = 1
        while factor == 1:
            for count in range(cycle_size):
                if factor > 1: break
                x = (x * x + 1) % n
                factor = gcd(x - x_fixed, n)

            cycle_size *= 2
            x_fixed = x
        return factor
    while n > 1:
        next = get_factor(n)
        factors.append(next)
        n //= next
    factors = np.sort(factors, axis=None)
    # return [np.prod(factors[:-1]), factors[-1]]
    return factors

def point_rotation(origin, pt, ang):
    # returns the pt rotated about the origin by ang (ang in degrees)
    c = math.cos(math.radians(ang))
    s = math.sin(math.radians(ang))
    # translate to origin
    pt_temp = [pt[0] - origin[0], pt[1] - origin[1]]
    pt_spun = [ pt_temp[0]*c - pt_temp[1]*s, pt_temp[0]*s + pt_temp[1]*c ]
    # translate back to frame
    pt_spun = [pt_spun[0] + origin[0], pt_spun[1] + origin[1]]
    return pt_spun

def meters2lat_lon(xs, ys):
    transformer = Transformer.from_crs(3857, 4326) # meters -> lat lon
    ll_pts = transformer.transform(xs, ys)
    return ll_pts

def lat_lon2meters(lats, lons):
    transformer = Transformer.from_crs(4326, 3857) # lat lon -> meters
    xy_pts = transformer.transform(lats, lons)
    return xy_pts

# def lat_lon2meters(lat_lon_pts):
#     # '''Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913'''
#     originShift = 2 * math.pi * 6378137 / 2.0
#     x_y_pts = []
#     for ll in lat_lon_pts:
#         lat = ll[0]
#         lon = ll[1]
#         mx = lon * originShift / 180.0
#         my = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
#         my = my * originShift / 180.0
#         x_y_pts.append([mx,my])
#     return x_y_pts

def center_geolocation(geolocations):
    """
    Provide a relatively accurate center lat, lon returned as a list pair, given
    a list of list pairs.
    ex: in: geolocations = ((lat1,lon1), (lat2,lon2),)
        out: (center_lat, center_lon)
    """
    x = 0
    y = 0
    z = 0

    for lat, lon in geolocations:
        lat = float(lat)
        lon = float(lon)
        x += math.cos(math.radians(lat)) * math.cos(math.radians(lon))
        y += math.cos(math.radians(lat)) * math.sin(math.radians(lon))
        z += math.sin(math.radians(lat))

    x = float(x / len(geolocations))
    y = float(y / len(geolocations))
    z = float(z / len(geolocations))

    return math.degrees(math.atan2(z, math.sqrt(x * x + y * y))), math.degrees(math.atan2(y, x))

def centroid_calc(points_meters):
    return np.mean(points_meters, axis = 1)


def get_terrain_map(lat_lon = [0,0], sample_dist = 10, extent = 100, heading = 0, show_plot = False, verbosity = False):

    # gis = GIS("pro")
    # gis = GIS(url="http://virginiatech.maps.arcgis.com", client_id="rluxzSWjZS6TfeXs", username="hlarkin3_virginiatech", password="arcgisheintzman97#26640", verify_cert=False)
    gis = GIS(username="larkinheintzman",password="Meepp97#26640")
    if verbosity:
        print("Successfully logged in as: " + gis.properties.user.username)
    elv_map = gis.content.get('58a541efc59545e6b7137f961d7de883')
    elv_layer = elv_map.layers[0]

    if len(lat_lon) > 2:
        print("Error, too many lat long points provided")
        return -1
    xy = lat_lon2meters(lat_lon[0], lat_lon[1])
    cen_pt = list(xy)

    lhc_pt = [cen_pt[0] - extent/2, cen_pt[1] - extent/2] # lefthand corner point
    max_samples = 30 # api limitation!
    if extent > max_samples*sample_dist:
        # multiple calls are required
        fac = factorization(int(extent/sample_dist))
        if len(fac) == 1: # random prime number incomming
            sc = fac[0]
            cc = 1
        else:
            sc = np.max(fac[fac <= max_samples])
            fac = np.delete(fac,np.argmax(fac[fac <= max_samples]))
            cc = np.prod(fac)
    else:
        # single call suffices
        cc = 1
        sc = int(np.round(extent/sample_dist))
    sample_extents = sc * sample_dist

    if verbosity:
        print("total calls: {}".format(np.square(cc)))

    # set max values
    elv_layer.extent['xmin'] = lhc_pt[0]
    elv_layer.extent['xmax'] = lhc_pt[0] + extent
    elv_layer.extent['ymin'] = lhc_pt[1]
    elv_layer.extent['ymax'] = lhc_pt[1] + extent

    # print(elv_layer.extent)
    # house keeping, initialize with empty lists
    x = np.empty([0,sc*cc])
    y = np.empty([0,sc*cc])
    e = np.empty([0,sc*cc])
    data = []
    for j in range(cc): # outter loop for y values
        # lists of values for a single row, reset at every y iteration
        x_row = np.empty([sc,0])
        y_row = np.empty([sc,0])
        e_row = np.empty([sc,0])
        for i in range(cc): # inner loop for x values O_O
            x_values = [np.linspace(lhc_pt[0] + i * sample_extents, lhc_pt[0] + (i + 1 - 1/sc) * sample_extents, sc)]
            y_values = [np.linspace(lhc_pt[1] + j * sample_extents, lhc_pt[1] + (j + 1 - 1/sc) * sample_extents, sc)]
            [X,Y] = np.meshgrid(x_values, y_values)
            # put in rotation here
            geo_points = [point_rotation(origin = [cen_pt[0],cen_pt[1]],pt = [xp,yp],ang = heading) for xv,yv in zip(X,Y) for xp,yp in zip(xv,yv)]
            # print(points)
            g = Geometry({"points": geo_points, "spatialReference": 3857})
            failed = True
            while failed: # keep trying until it works
                try:
                    elv_samples = elv_layer.get_samples(g, sample_count=len(g.coordinates()), out_fields='location,values,value,resolution')
                    failed = False
                except TimeoutError:
                    print("failed call, trying again...")
                    failed = True
            # extract location info JANK INCOMING
            xs = np.array([e['location']['x'] for e in elv_samples], dtype=float).reshape([sc,sc])
            ys = np.array([e['location']['y'] for e in elv_samples], dtype=float).reshape([sc,sc])
            es = np.array([e['values'][0] for e in elv_samples], dtype=float).reshape([sc,sc])
            if not np.all(xs == X) or not np.all(ys == Y):
                # xs = np.array([e['location']['x'] for e in elv_samples], dtype=float)
                # ys = np.array([e['location']['y'] for e in elv_samples], dtype=float)
                qs = np.stack([xs.reshape(len(elv_samples)), ys.reshape(len(elv_samples))]).T
                es = es.reshape(len(elv_samples))
                # data came back in weird ordering, need to re-order
                print("re-ordering data ...")
                es_square = np.zeros([sc,sc])
                for scx in range(sc):
                    for scy in range(sc): # maybe the least efficient way to do this
                        idx = np.where(np.all(np.abs(qs - np.asarray([X[scx,scy],Y[scx,scy]])) <= 1e-9,axis = 1))
                        # print(idx)
                        try:
                            es_square[scx,scy] = es[idx]
                        except ValueError as e:
                            print("hellfire")
                es = es_square
                print("done re-ordering")

            # then just the tuple of all
            data_temp = []
            for s in elv_samples:
                xy_spun = point_rotation(origin = [cen_pt[0],cen_pt[1]],pt = [s['location']['x'],s['location']['y']],ang = -heading) # spin back to original frame
                data_temp.append((xy_spun[0], xy_spun[1],s['values'][0]))
                # data_temp = [(xy_spun[0], xy_spun[1],e['values'][0]) for e in elv_samples]
            # data_temp = [(e['location']['x'],e['location']['y'],e['values'][0]) for e in elv_samples]
            data = data + data_temp
            # append to larger arrays
            x_row = np.append(x_row, xs, axis=1)
            y_row = np.append(y_row, ys, axis=1)
            e_row = np.append(e_row, es, axis=1)
        x = np.append(x, x_row, axis=0)
        y = np.append(y, y_row, axis=0)
        e = np.append(e, e_row, axis=0)

    # flip elevation data up to down to match other layers
    e = np.flipud(e)
    # interpolate terrain to match size/resolution of other layers
    c = np.int(extent/sample_dist)
    scale_factor = 3/20 # factor to get 6.66667m mapping from 1m mapping (1/6.6667)
    scaled_extent = np.ceil(scale_factor*extent).astype(np.int)

    factor = scaled_extent/e.shape[0]
    e_interp = ndimage.zoom(e, factor, order = 3)

    if show_plot:
        # Attaching 3D axis to the figure
        grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
        points = np.array([x, y])
        grid_z = griddata(points.transpose(), z, (grid_x, grid_y), method='cubic', fill_value=np.mean(z))
        print(grid_z)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis')

    if verbosity:
        print("x min: {}".format(np.min(x)))
        print("x max: {}".format(np.max(x)))
        print("x max - min: {}".format(np.max(x)-np.min(x)))
        print("x spacing: {}".format(abs(x[0][0] - x[0][1])))

        print("y min: {}".format(np.min(y)))
        print("y max: {}".format(np.max(y)))
        print("y max - min: {}".format(np.max(y)-np.min(y)))

    return [e,e_interp,x,y,data,lat_lon]
    # savedimg = elv_layer.export_image(bbox=elv_layer.extent, size=[3840,2160], f='image', save_folder='.', save_file='testerino.jpg')

if __name__ == "__main__":


    # anchor_point = [float(ics_pt[0]), float(ics_pt[1])]
    anchor_point = [42.17965, -74.21362]
    extent = 20e3
    sample_dist = int(extent/10)
    heading = 0
    start_time = time.time()
    # save terrain as csv file (this method is pretty slow, but can compensate with interp)
    [e,e_interp,x,y,data,ll_pt] = get_terrain_map(lat_lon=anchor_point,
                                         sample_dist = sample_dist,
                                         extent = extent,
                                         heading = -heading) # because flipping

    plt.imshow(e_interp)
    plt.title('e interp')
    plt.show()

    gdt = np.gradient(e_interp)
    plt.imshow(np.sqrt(gdt[0]**2 + gdt[1]**2))
    plt.title('e interp grad')
    plt.show()

    c = np.int(extent / sample_dist)
    scale_factor = 3 / 20  # factor to get 6.66667m mapping from 1m mapping (1/6.6667)
    scaled_extent = np.ceil(scale_factor * extent).astype(np.int)
    x_start = np.linspace(0, extent, c)
    y_start = np.linspace(0, extent, c)
    X_start, Y_start = np.meshgrid(x_start, y_start)
    Z_start = np.zeros_like(X_start)

    f = interpolate.Rbf(X_start, Y_start, Z_start, e, function='multiquadric')
    x_temp = np.linspace(0,extent,scaled_extent) # get correct size of terrain map
    y_temp = np.linspace(0,extent,scaled_extent)
    X_temp, Y_temp = np.meshgrid(x_temp, y_temp)
    Z_temp = np.zeros_like(X_temp)
    e_interp = f(X_temp, Y_temp, Z_temp)

    plt.imshow(e_interp)
    plt.title('e interp old')
    plt.show()

    gdt = np.gradient(e_interp)
    plt.imshow(np.sqrt(gdt[0]**2 + gdt[1]**2))
    plt.title('e interp grad old')
    plt.show()


    print('done')

    # elv_filename = "map_layers\\elv_data_"+file_id+".csv"
    # if save_files:
    #     np.savetxt(elv_filename,e_interp,delimiter=",", fmt='%f')

