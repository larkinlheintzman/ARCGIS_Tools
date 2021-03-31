from arcgis_terrain import get_terrain_map
from arcgis.features import FeatureLayer
from arcgis.gis import GIS
from arcgis_terrain import lat_lon2meters
from arcgis_terrain import meters2lat_lon
import time

from arcgis.geometry.filters import envelope_intersects
import arcgis.geometry
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from arcgis_terrain import point_rotation
from scipy import interpolate
from matplotlib import path
import matplotlib.pyplot as plt
import math
import json
import sys
import csv
import os
import glob

import matlab.engine

def trim_extent(x_pts, y_pts, scaled_extent):
    # trim data to only within extents
    rm_mask = np.logical_or(x_pts < 0, x_pts > scaled_extent) # mask to remove points (x axis)
    rm_mask = np.logical_or(rm_mask,np.logical_or(y_pts < 0, y_pts > scaled_extent)) # other axis mask
    x_pts_trimmed = x_pts[np.invert(rm_mask)]
    y_pts_trimmed = y_pts[np.invert(rm_mask)] # trim points
    return [x_pts_trimmed, y_pts_trimmed]

def grab_features(anchor_point, extent, sample_dist = 10, case_name = 'blah', heading = 0, save_files = False, save_to_folder = False, file_id = 'temp', plot_data = False):
    roads_url = "https://carto.nationalmap.gov/arcgis/rest/services/transportation/MapServer/30"
    river_url = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/6"
    riverw_url = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/8"
    water_url = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/9"
    powerlines_url = "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/ArcGIS/rest/services/Electric_Power_Transmission_Lines/FeatureServer/0"
    railroads_url = "https://carto.nationalmap.gov/arcgis/rest/services/transportation/MapServer/35"
    trails_url = "https://partnerships.nationalmap.gov/arcgis/rest/services/USGSTrails/MapServer/0"

    # adding water_url twice, once for boundaries and once for linear features
    # the layer named 'lakes' gets boundary treatment
    url_list = [riverw_url, river_url, roads_url, water_url, powerlines_url, railroads_url, trails_url]
    name_list = ['rivers_bdd', 'rivers', 'roads', 'lakes', 'powerlines', 'railroads', 'trails']
    inac_layers = ['rivers_bdd', 'lakes']

    gis = GIS(username="larkinheintzman",password="Meepp97#26640") # linked my arcgis pro account
    ap_meters = lat_lon2meters(anchor_point[0], anchor_point[1])


    scale_factor = 3/20 # factor to get 6.66667m mapping from 1m mapping (1/6.6667)
    scaled_extent = np.ceil(scale_factor*extent).astype(np.int)
    viz_cnt = 0
    viz_map = np.zeros([scaled_extent,scaled_extent,len(name_list)+len(inac_layers)])

    for i,url in enumerate(url_list):

        # binary map, will use feature coords to populate (one per layer)
        bin_map = np.zeros([scaled_extent,scaled_extent])
        inac_bin_map = np.zeros([scaled_extent,scaled_extent])

        geom = arcgis.geometry.Polygon({'spatialReference': {"wkid" : 3857},
                             'rings': [[
                                [ap_meters[0] - (extent/2), ap_meters[1] + (extent/2)],
                                [ap_meters[0] + (extent/2), ap_meters[1] + (extent/2)],
                                [ap_meters[0] + (extent/2), ap_meters[1] - (extent/2)],
                                [ap_meters[0] - (extent/2), ap_meters[1] - (extent/2)]
                             ]]})

        lyr = FeatureLayer(url = url, gis = gis)
        geom_filter = envelope_intersects(geom, sr=geom['spatialReference'])

        q = []
        query_cnt = 0
        while type(q)==list and query_cnt <= 30: # have to do this because arcgis is sketchy as hell and doesnt always come back
            try:
                print("querying {} layer...".format(name_list[i]))
                query_starttime = time.time()

                q = lyr.query(return_count_only=False, return_ids_only=False, return_geometry=True,
                              out_sr='3857', geometry_filter=geom_filter)

                query_endtime = time.time()

            except (json.decoder.JSONDecodeError, TypeError) as e:
                if type(e) != TypeError:
                    query_cnt = query_cnt + 1
                print("error on query: {}".format(e))
                print("{} layer failed on query, trying again ...".format(name_list[i]))
                # gis.
                gis = GIS(username="larkinheintzman",password="Meepp97#26640") # linked my arcgis pro account
                lyr = FeatureLayer(url=url, gis=gis)

        print("query time {}".format(query_endtime - query_starttime))

        if query_cnt > 30 and not q:
            print("{} layer failed too many times, leaving empty".format(name_list[i]))
            if save_files:
                if save_to_folder:
                    fn = "map_layers\\" + case_name + "\\"+name_list[i]+"_data_"+file_id+".csv"
                    np.savetxt(fn,bin_map,delimiter=",", fmt='%f')
                    if name_list[i] in inac_layers:
                        fn = "map_layers\\" + case_name + "\\" + name_list[i] + "_inac_data_" + file_id + ".csv"
                        np.savetxt(fn, bin_map, delimiter=",", fmt='%f')
                else:
                    fn = "map_layers\\" + name_list[i]+"_data_"+file_id+".csv"
                    np.savetxt(fn,bin_map,delimiter=",", fmt='%f')
                    if name_list[i] in inac_layers:
                        fn = "map_layers\\" + name_list[i] + "_inac_data_" + file_id + ".csv"
                        np.savetxt(fn, bin_map, delimiter=",", fmt='%f')
            continue
        print("{} layer sucessfully queried".format(name_list[i]))

        # re-build into list of x-y values
        # feat_points = []
        query_dict = q.to_dict()
        for j,feat in enumerate(query_dict['features']):

            # print("starting feature {} ...".format(j))

            # pull feature points out of query, they have different keys...
            if 'paths' in feat['geometry'].keys():
                x_pts = [pt[0] for pt in feat['geometry']['paths'][0]]
                y_pts = [pt[1] for pt in feat['geometry']['paths'][0]]
                # plot_points = np.array(feat['geometry']['paths'][0])
            else:
                x_pts = [pt[0] for pt in feat['geometry']['rings'][0]] # arcgis is stupid
                y_pts = [pt[1] for pt in feat['geometry']['rings'][0]]
                # plot_points = np.array(feat['geometry']['rings'][0])

            # re-center on 0,0 at center
            x_pts = (np.array(x_pts) - (ap_meters[0] - (extent/2)))*scale_factor # reduces number of interpolants
            y_pts = (np.array(y_pts) - (ap_meters[1] - (extent/2)))*scale_factor

            # rotate points about origin to establish heading
            [x_pts, y_pts] = point_rotation(origin = [scaled_extent/2,scaled_extent/2],pt = [x_pts, y_pts],ang = heading)

            # quick check for a feature that does not enter the extent
            [x_pts_trimmed, y_pts_trimmed] = trim_extent(x_pts, y_pts, scaled_extent)
            if x_pts_trimmed.shape[0] == 0:
                continue

            # # treat each section of a feature intersecting the extent as separate
            # dists = np.sqrt(np.sum(np.diff(np.array([x_pts, y_pts]).T, axis = 0)**2, axis=1))
            # breaks = list(np.where(dists >= dists.mean() + 5*dists.std())[0])

            # x_pts_full = x_pts # save full coordinate set
            # y_pts_full = y_pts
            # breaks = [-1] + breaks + [x_pts.shape[0]-1]

            # for br in range(len(breaks) - 1):
            #     x_pts = x_pts_full[(breaks[br]+1):(breaks[br+1]+1)]
            #     y_pts = y_pts_full[(breaks[br]+1):(breaks[br+1]+1)]
            #
            #     if x_pts.shape[0] <= 1: # ignore tiny chops
            #         continue

            # if data is too short, add some points in the middle
            # while x_pts.shape[0] < 4:
            #     x_pt = (x_pts[0] + x_pts[1]) / 2  # average between first and second point
            #     y_pt = (y_pts[0] + y_pts[1]) / 2
            #     x_pts = np.insert(x_pts, 1, x_pt)
            #     y_pts = np.insert(y_pts, 1, y_pt)

            # total length of feature ring/path, for interpolation along features
            total_len = np.sum(np.sqrt(np.sum(np.diff(np.array([x_pts, y_pts]).T, axis=0) ** 2, axis=1)))

            interp_starttime = time.time()

            tck, u = interpolate.splprep([x_pts, y_pts], s=0, k=1)  # parametric interpolation
            u_new = np.arange(0, 1 + 1 / total_len, 1 / total_len)  # scaled discretization
            pts_interp = interpolate.splev(u_new, tck)

            interp_endtime = time.time()

            print("{} interpolation took {}".format(j,interp_endtime - interp_starttime))

            x_pts = pts_interp[0]
            y_pts = pts_interp[1]

            if name_list[i] in inac_layers:

                inac_starttime = time.time()

                # do boundary calculation for binary matrix (slow for large bounaries but whatever)
                ring = path.Path(np.array([x_pts, y_pts]).T)

                # test_pts is the rectangular matrix covering ring for boundary calculation

                # trim test_pts rectangle to only consider points within the scaled extent
                [x_pts_trimmed, y_pts_trimmed] = trim_extent(x_pts, y_pts, scaled_extent)

                x_test, y_test = np.meshgrid(np.arange(np.min(x_pts_trimmed), np.max(x_pts_trimmed), 1),
                                             np.arange(np.min(y_pts_trimmed), np.max(y_pts_trimmed), 1))

                test_pts = np.array([x_test.flatten(), y_test.flatten()]).T
                mask = ring.contains_points(test_pts, radius=1)

                # instead of filling gaps, we want to save filled in areas separately
                # so we need to re-create the bin_map here but on inac. points
                x_pts_inac = test_pts[mask,0]
                y_pts_inac = test_pts[mask,1]

                inac_endtime = time.time()
                print("{} inac took {}".format(j,inac_endtime - inac_starttime))

                pts_inac = np.stack([x_pts_inac,y_pts_inac]).T

                # remove points being used as linear features
                for pt in np.stack([x_pts_trimmed,y_pts_trimmed]).T:
                    pts_inac = np.delete(pts_inac, np.where(np.equal(pt,pts_inac).all(1)), axis = 0)

                # binarization step
                pts_inac = np.round(pts_inac).astype(np.int)
                # flip y axis
                pts_inac[:,1] = inac_bin_map.shape[1] - pts_inac[:,1]
                # remove any points outside limits of binary map (fixes round versus ceil issues)
                rm_mask = np.logical_or(pts_inac[:,0] < 0, pts_inac[:,0] >= inac_bin_map.shape[1])
                rm_mask = np.logical_or(rm_mask, np.logical_or(pts_inac[:,1] < 0, pts_inac[:,1] >= inac_bin_map.shape[0]))
                pts_inac = pts_inac[np.invert(rm_mask),:]
                inac_bin_map[pts_inac[:,1], pts_inac[:,0]] = 1  # set indices to 1
                # print("looped inac calculation time = {} sec".format(time.time() - s_time))

            # trim features to scaled extent
            [x_pts, y_pts] = trim_extent(x_pts, y_pts, scaled_extent)
            # binarization step
            x_pts_idx = np.round(x_pts).astype(np.int)
            y_pts_idx = np.round(y_pts).astype(np.int)

            # flip y axis because indices are fliped
            y_pts_idx = bin_map.shape[1] - y_pts_idx
            # remove any points outside limits of binary map (fixes round versus ceil issues)
            rm_mask = np.logical_or(x_pts_idx < 0, x_pts_idx >= bin_map.shape[1])
            rm_mask = np.logical_or(rm_mask, np.logical_or(y_pts_idx < 0, y_pts_idx >= bin_map.shape[0]))
            x_pts_idx = x_pts_idx[np.invert(rm_mask)]
            y_pts_idx = y_pts_idx[np.invert(rm_mask)]
            bin_map[y_pts_idx, x_pts_idx] = 1 # set indices to 1

            # print("done with feature {}".format(j))

        # add to viz map
        if name_list[i] in inac_layers:
            viz_map[:, :, viz_cnt] = inac_bin_map
            viz_cnt = viz_cnt + 1
            if save_files:
                if save_to_folder:
                    fn = "map_layers\\" + case_name + "\\" + name_list[i] + "_inac_data_" + file_id + ".csv"
                    np.savetxt(fn, inac_bin_map, delimiter=",", fmt='%f')
                else:
                    fn = "map_layers\\" + name_list[i] + "_inac_data_" + file_id + ".csv"
                    np.savetxt(fn, inac_bin_map, delimiter=",", fmt='%f')


        viz_map[:, :, viz_cnt] = bin_map
        viz_cnt = viz_cnt + 1
        if save_files:
            if save_to_folder:
                fn = "map_layers\\" + case_name + "\\" + name_list[i]+"_data_"+file_id+".csv"
                np.savetxt(fn,bin_map,delimiter=",", fmt='%f')
            else:
                fn = "map_layers\\" + name_list[i]+"_data_"+file_id+".csv"
                np.savetxt(fn,bin_map,delimiter=",", fmt='%f')


    # save terrain as csv file (this method is pretty slow, but can compensate with interp)
    [e,e_interp,x,y,data,ll_pt] = get_terrain_map(lat_lon=anchor_point,
                                         sample_dist = sample_dist,
                                         extent = extent,
                                         heading = -heading) # because flipping

    if save_files:
        if save_to_folder:
            elv_filename = "map_layers\\" + case_name + "\\elv_data_" + file_id + ".csv"
            np.savetxt(elv_filename,e_interp,delimiter=",", fmt='%f')
        else:
            elv_filename = "map_layers\\elv_data_" + file_id + ".csv"
            np.savetxt(elv_filename,e_interp,delimiter=",", fmt='%f')

    if plot_data:
        plt.imshow(e_interp)
        plt.show()

        # plt_list = []
        # fix stupid names
        # for nme in inac_layers:
        #     name_list.insert(name_list.index(nme), nme+' inac')

        for i in range(viz_map.shape[-1]):
            # row_idx, col_idx = np.where(viz_map[:,:,i] != 0)
            # flip y values
            # col_idx = viz_map.shape[0] - col_idx
            # plt_list.append(go.Scatter(x=row_idx, y=col_idx, mode='markers', name=name_list[i]))
            plt.imshow(viz_map[:,:,i])
            # plt.title(name_list[i])
            plt.show()

        # fig = go.Figure(data=plt_list)
        # fig.show()

if __name__ == "__main__":

    ics = [
        [37.197730, -80.585233,'kentland'],
        [36.891640, -81.524214,'hmpark'],
        # [38.29288, -78.65848,  'brownmountain'],
        # [38.44706, -78.46993,  'devilsditch'],
        # [37.67752, -79.33887,  'punchbowl'],
        # [37.99092, -78.52798,  'biscuitrun'],
        # [37.82520, -79.081910, 'priest'] ,
        # [34.12751, -116.93247, 'sanbernardino'] ,
        # [31.92245, -109.9673,'[31.92245, -109.9673]'],
        # [31.9024, -109.2785,'[31.9024, -109.2785]'],
        # [31.42903, -110.2933,'[31.42903, -110.2933]'],
        # [34.55, -111.6333,'[34.55, -111.6333]'],
        # [34.6, -112.55,'[34.6, -112.55]'],
        # [34.82167, -111.8067,'[34.82167, -111.8067]'],
        # [33.3975, -111.3478,'[33.3975, -111.3478]'],
        # [33.70542, -111.338,'[33.70542, -111.338]'],
        # [31.39708, -111.2064,'[31.39708, -111.2064]'],
        # [32.4075, -110.825,'[32.4075, -110.825]'],
        # [34.89333, -111.8633,'[34.89333, -111.8633]'],
        # [34.94833, -111.795,'[34.94833, -111.795]'],
        # [31.72262, -110.1878,'[31.72262, -110.1878]'],
        # [33.39733, -111.348,'[33.39733, -111.348]'],
        # [34.63042, -112.5553,'[34.63042, -112.5553]'],
        # [34.55977, -111.6539,'[34.55977, -111.6539]'],
        # [34.90287, -111.8131,'[34.90287, -111.8131]'],
        # [34.86667, -111.8833,'[34.86667, -111.8833]'],
        # [32.43543, -110.7893,'[32.43543, -110.7893]'],
        # [32.40917, -110.7098,'[32.40917, -110.7098]'],
        # [35.33068, -111.7111,'[35.33068, -111.7111]'],
        # [32.01237, -109.3157,'[32.01237, -109.3157]'],
        # [31.85073, -109.4219,'[31.85073, -109.4219]'],
        # [34.88683, -111.784,'[34.88683, -111.784]'],
        # [32.41977, -110.7473,'[32.41977, -110.7473]'],
        # [33.60398, -112.5151,'[33.60398, -112.5151]'],
        # [33.3968, -111.3481,'[33.3968, -111.3481]'],
        # [33.52603, -111.3905,'[33.52603, -111.3905]'],
        # [32.33333, -110.8528,'[32.33333, -110.8528]'],
        # [32.33583, -110.9102,'[32.33583, -110.9102]'],
        # [32.337, -110.9167,'[32.337, -110.9167]'],
        # [35.08133, -111.0711,'[35.08133, -111.0711]'],
        # [33.25, -113.5093,'[33.25, -113.5093]'],
        # [31.50572, -110.6762,'[31.50572, -110.6762]'],
        # [34.91667, -111.8,'[34.91667, -111.8]'],
        # [35.1938, -114.057,'[35.1938, -114.057]'],
        # [33.39715, -111.3479,'[33.39715, -111.3479]'],
        # [33.37055, -111.1152,'[33.37055, -111.1152]'],
        # [34.0927, -111.4246,'[34.0927, -111.4246]'],
        # [31.83522, -110.3567,'[31.83522, -110.3567]'],
        # [35.24375, -111.5997,'[35.24375, -111.5997]'],
        # [34.82513, -111.7875,'[34.82513, -111.7875]'],
        # [33.39705, -111.3479,'[33.39705, -111.3479]'],
        # [33.38885, -111.3657,'[33.38885, -111.3657]'],
        # [32.82142, -111.2021,'[32.82142, -111.2021]'],
        # [34.97868, -111.8964,'[34.97868, -111.8964]'],
        # [33.47802, -111.4377,'[33.47802, -111.4377]'],
        # [34.82387, -111.7751,'[34.82387, -111.7751]'],
        # [34.9253, -111.7341,'[34.9253, -111.7341]'],
        # [34.09278, -111.421,'[34.09278, -111.421]'],
        # [36.23878, -112.6892,'[36.23878, -112.6892]'],
        # [31.33447, -109.8186,'[31.33447, -109.8186]'],
        # [36.37473, -106.6795,'[36.37473, -106.6795]'],
        # [42.02893, -74.33659,'[42.02893, -74.33659]'],
        # [41.54819, -80.33056,'[41.54819, -80.33056]'],
        # [42.0097, -74.42595,'[42.0097, -74.42595]'],
        # [42.17965, -74.21362,'[42.17965, -74.21362]'],
        # [42.55121, -73.4874,'[42.55121, -73.4874]'],
        # [42.01362, -80.35002,'[42.01362, -80.35002]'],
        # [42.74271, -73.45475,'[42.74271, -73.45475]'],
        # [42.1549, -74.20523,'[42.1549, -74.20523]'],
        # [42.90324, -73.8047,'[42.90324, -73.8047]'],
        # [44.16065, -73.85545,'[44.16065, -73.85545]'],
        # [43.42736, -74.4481,'[43.42736, -74.4481]'],
        # [44.18, -72.99531,'[44.18, -72.99531]'],
        # [43.52022, -73.59601,'[43.52022, -73.59601]'],
        # [44.12531, -73.78635,'[44.12531, -73.78635]'],
        # [43.73413, -74.25577,'[43.73413, -74.25577]'],
        # [43.06902, -74.48481,'[43.06902, -74.48481]'],
        # [43.8756, -74.43076,'[43.8756, -74.43076]'],
        # [43.41544, -74.4148,'[43.41544, -74.4148]'],
        # [43.42473, -73.73209,'[43.42473, -73.73209]'],
        # [43.59779, -74.55354,'[43.59779, -74.55354]'],
        # [43.4449, -74.4086,'[43.4449, -74.4086]'],
        # [43.4332, -74.41433,'[43.4332, -74.41433]'],
        # [43.4539, -74.52317,'[43.4539, -74.52317]'],
        # [43.63354, -74.55927,'[43.63354, -74.55927]'],
        # [44.19204, -74.26329,'[44.19204, -74.26329]'],
        # [44.31349, -74.56818,'[44.31349, -74.56818]'],
        # [43.42498, -74.41496,'[43.42498, -74.41496]'],
        # [43.33195, -74.49095,'[43.33195, -74.49095]'],
        # [43.95385, -75.15748,'[43.95385, -75.15748]'],
        # [44.12993, -74.58844,'[44.12993, -74.58844]'],
        # [44.28656, -74.61429,'[44.28656, -74.61429]'],
        # [43.51063, -74.57393,'[43.51063, -74.57393]'],
        # [44.19013, -74.81336,'[44.19013, -74.81336]'],
        # [43.65649, -76.00019,'[43.65649, -76.00019]'],
        # [42.42501, -76.47478,'[42.42501, -76.47478]'],
        # [42.31735, -76.47791,'[42.31735, -76.47791]'],
        # [42.34432, -77.47638,'[42.34432, -77.47638]'],
        # [42.25618, -77.78988,'[42.25618, -77.78988]'],
        # [42.39831, -72.88675,'[42.39831, -72.88675]'],
        # [48.1103, -121.4917,'[48.1103, -121.4917]'],
        # [48.64606, -122.4247,'[48.64606, -122.4247]']
        ]

    base_dir = 'C:/Users/Larkin/ags_grabber'

    # for item in ics:
    #     try:
    #         os.mkdir(base_dir + '/map_layers/' + item[2])
    #     except FileExistsError:
    #         pass
    #

    eng = matlab.engine.start_matlab() # engine for running matlab

    for i,ics_pt in enumerate(ics):

        # try:
        #     os.mkdir(base_dir + '/map_layers/' + str(ics_pt[2]))
        # except FileExistsError:
        #     pass

        anchor_point = [float(ics_pt[0]), float(ics_pt[1])]
        extent = 10e3
        save_flag = True
        plot_flag = False
        file_extension = 'temp'

        sample_dist = int(extent/100)
        heading = 0
        start_time = time.time()
        grab_features(anchor_point = anchor_point, extent = extent, sample_dist = sample_dist, case_name = str(ics_pt[2]),
                      heading = heading, save_files = save_flag, save_to_folder = False, file_id = file_extension, plot_data = plot_flag)
        time.sleep(1) # wait for... files to settle?

        # run matlab
        if save_flag:
            res = eng.importmap_py(str(ics_pt[2]), base_dir)

        print("------- total time = {} seconds, iteration {}/{} ------".format(time.time() - start_time,i,len(ics)))
    eng.quit()