from arcgis_terrain import meters2lat_lon
from arcgis_terrain import lat_lon2meters
import csv
import numpy as np
import matplotlib.pyplot as plt

# params = [37.67752, -79.33887,  'punchbowl']
# # params = [38.29288, -78.65848,  'brownmountain']
# # params = [38.44706, -78.46993,  'devilsditch']
# # params = [37.99092, -78.52798,  'biscuitrun']
# # params = [37.82520, -79.081910, 'priest']
# # params = [34.12751, -116.93247, 'sanbernardino']
#
# with open("C:\\Users\\Larkin\\ags_grabber\\track_temp.csv") as f:
#     reader = csv.reader(f, delimiter=',')
#     data = []
#     for row in reader:
#         if any(x.strip() for x in row):
#             data.append(row)
#     track = np.array(data).astype(np.float)
#
# ap_meters = lat_lon2meters(params[0], params[1])
extent = 20e3
scale_factor = 3/20 # factor to get 6.66667m mapping from 1m mapping (1/6.6667)
#
# xy = lat_lon2meters(track[:,1], track[:,0])
#
# x_pts = (np.array(xy[0]) - (ap_meters[0] - (extent/2)))*scale_factor # reduces number of interpolants
# y_pts = (np.array(xy[1]) - (ap_meters[1] - (extent/2)))*scale_factor
#
# np.savetxt(params[2] + '_track_meters.csv',np.array([x_pts, y_pts]),delimiter=",", fmt='%f')
# plt.plot(x_pts, y_pts)
# plt.show()

points = [[37.67752, -79.33887,  'punchbowl'],
          [38.29288, -78.65848,  'brownmountain'],
          [38.44706, -78.46993,  'devilsditch'],
          [37.99092, -78.52798,  'biscuitrun'],
          [37.82520, -79.081910, 'priest'],
          [34.12751, -116.93247, 'sanbernardino']]

for pt in points:
    pt_meters = lat_lon2meters(pt[0], pt[1])
    print(pt[2])
    print(meters2lat_lon(pt_meters[0] - extent/2, pt_meters[1] - extent/2))
    print(meters2lat_lon(pt_meters[0] + extent/2, pt_meters[1] - extent/2))
    print(meters2lat_lon(pt_meters[0] + extent/2, pt_meters[1] + extent/2))
    print(meters2lat_lon(pt_meters[0] - extent/2, pt_meters[1] + extent/2))
    print("-----------------")