from arcgis_terrain import meters2lat_lon
from arcgis_terrain import lat_lon2meters
import csv
import numpy as np
import matplotlib.pyplot as plt

with open("C:\\Users\\Larkin\\ags_grabber\\DevilsDitch_hikers.csv") as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    data = list(reader)

# plot each track based on order column
done = False
last_idx = 0
start_idx = 0

ap = [38.44706, -78.46993]
ap_meters = lat_lon2meters(ap[0], ap[1])
extent = 20e3
scale_factor = 3/20 # factor to get 6.66667m mapping from 1m mapping (1/6.6667)

order = [np.int(d[0]) for d in data] # order column
lat_lon = np.array([[np.float(d[1]), np.float(d[2])] for d in data])

for i in range(order.count(1)):
    if i == order.count(1)-1: # last iterations
        xy = lat_lon2meters(lat_lon[start_idx:,0], lat_lon[start_idx:,1])
        plt.plot(xy[0],xy[1])

        x_pts = (np.array(xy[0]) - (ap_meters[0] - (extent/2)))*scale_factor # reduces number of interpolants
        y_pts = (np.array(xy[1]) - (ap_meters[1] - (extent/2)))*scale_factor

        np.savetxt('devilsditch_track_meters.csv',np.array([x_pts, y_pts]),delimiter=",", fmt='%f')
    else:
        last_idx = order.index(1,last_idx+1)
        xy = lat_lon2meters(lat_lon[start_idx:last_idx,0], lat_lon[start_idx:last_idx,1])
        plt.plot(xy[0],xy[1])
        start_idx = last_idx
    # plt.show()
    # print(len(xy[0]))

