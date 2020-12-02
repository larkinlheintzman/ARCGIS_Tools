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

import matlab.engine

def grab_features(anchor_point, extent, sample_dist = 10, heading = 0, save_files = False, file_id = 'temp', plot_data = False):
    roads_url = "https://carto.nationalmap.gov/arcgis/rest/services/transportation/MapServer/30"
    river_url = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/6"
    riverw_url = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/8"
    water_url = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/9"
    powerlines_url = "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/ArcGIS/rest/services/Electric_Power_Transmission_Lines/FeatureServer/0"
    railroads_url = "https://carto.nationalmap.gov/arcgis/rest/services/transportation/MapServer/35"
    trails_url = "https://partnerships.nationalmap.gov/arcgis/rest/services/USGSTrails/MapServer/0"

    # adding water_url twice, once for boundaries and once for linear features
    # the layer named 'lakes' gets boundary treatment
    url_list = [trails_url, river_url, riverw_url, roads_url, water_url, powerlines_url, railroads_url]
    # url_list = [trails_url]
    name_list = ['trails', 'rivers', 'rivers_bdd', 'roads', 'lakes', 'powerlines', 'railroads']
    # name_list = ['trails']
    inac_layers = ['rivers_bdd', 'lakes']
    inac_layers = []

    gis = GIS(username="larkinheintzman",password="Meepp97#26640") # linked my arcgis pro account
    ap_meters = lat_lon2meters(anchor_point[0], anchor_point[1])
    # print(ap_meters)

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
                                [ap_meters[0] - (extent/2), ap_meters[1] - (extent/2)],
                                [ap_meters[0] - (extent/2), ap_meters[1] + (extent/2)],
                                [ap_meters[0] + (extent/2), ap_meters[1] + (extent/2)],
                                [ap_meters[0] + (extent/2), ap_meters[1] - (extent/2)],
                                [ap_meters[0] - (extent/2), ap_meters[1] - (extent/2)]
                             ]]})

        lyr = FeatureLayer(url = url, gis = gis)
        geom_filter = envelope_intersects(geom, sr=geom['spatialReference'])

        q = []
        query_cnt = 0
        while type(q)==list and query_cnt <= 5: # have to do this because arcgis is sketchy as hell and doesnt always come back
            try:
                print("querying {} layer...".format(name_list[i]))
                q = lyr.query(return_count_only=False, return_ids_only=False, return_geometry=True,
                              out_sr='3857', geometry_filter=geom_filter)
            except json.decoder.JSONDecodeError as e:
                query_cnt = query_cnt + 1
                print("error on query: {}".format(e))
                print("{} layer failed on query, trying again ...".format(name_list[i]))
        if query_cnt > 5 and not q:
            print("{} layer failed too many times, breaking!".format(name_list[i]))
            continue
            # if save_files:
            #     fn = "map_layers/"+name_list[i]+"_data_"+file_id+".csv"
            #     np.savetxt(fn,bin_map,delimiter=",", fmt='%f')
        print("{} layer sucessfully queried".format(name_list[i]))

        # re-build into list of x-y values
        # feat_points = []
        query_dict = q.to_dict()
        for j,feat in enumerate(query_dict['features']):

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

            # x_uinterp = x_pts # untrimmed uninterpolated points (but scaled)
            # y_uinterp = y_pts

            # trim data to only within extents (because arcgis cant fuckin' do this)
            rm_mask = np.logical_or(x_pts < 0, x_pts > scaled_extent) # mask to remove points (x axis)
            rm_mask = np.logical_or(rm_mask,np.logical_or(y_pts < 0, y_pts > scaled_extent)) # other axis mask
            x_pts = x_pts[np.invert(rm_mask)]
            y_pts = y_pts[np.invert(rm_mask)] # trim interpolated points

            if x_pts.shape[0] > 1: # if there are still points after trimming

                # if data is too short, add some points in the middle
                while x_pts.shape[0] < 4:
                    x_pt = (x_pts[0] + x_pts[1])/2 # average between first and second point
                    y_pt = (y_pts[0] + y_pts[1])/2
                    x_pts = np.insert(x_pts, 1, x_pt)
                    y_pts = np.insert(y_pts, 1, y_pt)

                # total length of feature ring/path, for interpolation along features
                total_len = np.sum(np.sqrt(np.sum(np.diff(np.array([x_pts, y_pts]).T, axis=0) ** 2, axis=1)))

                tck, u = interpolate.splprep([x_pts, y_pts], s=0, k=1)  # parametric interpolation
                u_new = np.arange(0, 1 + 1 / total_len, 1 / total_len)  # scaled discretization
                pts_interp = interpolate.splev(u_new, tck)

                x_pts = pts_interp[0]
                y_pts = pts_interp[1]

                if name_list[i] in inac_layers:
                    s_time = time.time()
                    # do boundary calculation for binary matrix (slow for large bounaries but whatever)
                    ring = path.Path(np.array([x_pts, y_pts]).T)
                    # TODO: try cv2 here, test for speeeeeeeeeeeeeeeeeeeeeeed

                    # test_pts is the rectangular matrix covering ring for boundary calculation
                    x_test, y_test = np.meshgrid(np.arange(np.min(x_pts), np.max(x_pts), 1) , np.arange(np.min(y_pts), np.max(y_pts), 1))
                    test_pts = np.array([x_test.flatten(), y_test.flatten()]).T
                    mask = ring.contains_points(test_pts, radius=1)

                    # instead of filling gaps, we want to save filled in areas separately
                    # so we need to re-create the bin_map here but on inac. points
                    x_pts_inac = test_pts[mask,0]
                    y_pts_inac = test_pts[mask,1]
                    pts_inac = np.stack([x_pts_inac,y_pts_inac]).T

                    # remove points being used as linear features
                    for pt in np.stack([x_pts,y_pts]).T:
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
                    print("inac calculation time = {} sec".format(time.time() - s_time))

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

        # add to viz map
        if name_list[i] in inac_layers:
            viz_map[:, :, viz_cnt] = inac_bin_map
            viz_cnt = viz_cnt + 1
            if save_files:
                fn = "map_layers/" + name_list[i] + "_inac_data_" + file_id + ".csv"
                np.savetxt(fn, inac_bin_map, delimiter=",", fmt='%f')

        viz_map[:, :, viz_cnt] = bin_map
        viz_cnt = viz_cnt + 1
        if save_files:
            fn = "map_layers/"+name_list[i]+"_data_"+file_id+".csv"
            np.savetxt(fn,bin_map,delimiter=",", fmt='%f')

    # save terrain as csv file (this method is pretty slow, but can compensate with interp)
    [e,x,y,data,ll_pt] = get_terrain_map(lat_lon=anchor_point,
                                         sample_dist = sample_dist,
                                         extent = extent,
                                         heading = -heading) # because flipping
    # flip elevation data up to down to match other layers
    e = np.flipud(e)

    # interpolate terrain to match size/resolution of other layers
    f = interpolate.interp2d(np.arange(0, extent, sample_dist), np.arange(0, extent, sample_dist), e, kind='cubic')
    x_temp = np.linspace(0,extent,scaled_extent) # get correct size of terrain map
    y_temp = np.linspace(0,extent,scaled_extent)
    e_interp = f(x_temp, y_temp)

    elv_filename = "map_layers\\elv_data_"+file_id+".csv"
    if save_files:
        np.savetxt(elv_filename,e_interp,delimiter=",", fmt='%f')

    if plot_data:
        terr_fig = px.imshow(e_interp)
        terr_fig.show()

        plt_list = []
        # fix stupid names
        for nme in inac_layers:
            name_list.insert(name_list.index(nme), nme+' inac')

        for i in range(viz_map.shape[-1]):
            row_idx, col_idx = np.where(viz_map[:,:,i] != 0)
            # flip y values
            col_idx = viz_map.shape[0] - col_idx
            plt_list.append(go.Scatter(x=row_idx, y=col_idx, mode='markers', name=name_list[i]))

        fig = go.Figure(data=plt_list)
        fig.show()

if __name__ == "__main__":

    ics = [
        # [38.29288, -78.65848,  'BrownMtn-hiker'],
        # [38.44706, -78.46993,  'DevilsDitch_hikers'],
        # [37.67752, -79.33887,  'Punchbowl_hiker'],
        # [37.99092, -78.52798,  'BiscuitRun_hikers'],
        # [37.519485, -79.651315,  'temp'],
        # [38.24969, -78.39555,  'Quinque_dementia'],
        [38.55209, -78.32099,  'OldRag'],
        # [38.20656, -78.67878,  'BrownsCove'],
        # [38.02723, -78.45076,  'Charlottesville_dementia'],
        # [34.12751, -116.93247, 'SanBernardinoPeak'] ,
        ]

    base_dir = 'C:/Users/Larkin/ags_grabber'

    eng = matlab.engine.start_matlab() # engine for running matlab

    for i,ics_pt in enumerate(ics):

        anchor_point = [float(ics_pt[0]), float(ics_pt[1])]
        extent = 15e3
        save_flag = True
        plot_flag = False
        file_extension = 'temp'

        sample_dist = int(extent/100)
        heading = 0
        start_time = time.time()
        grab_features(anchor_point = anchor_point, extent = extent, sample_dist = sample_dist,
                      heading = heading, save_files = save_flag, file_id = file_extension, plot_data = plot_flag)
        time.sleep(1) # wait for... files to settle?

        # run matlab
        if save_flag:
            res = eng.importmap_py(str(ics_pt[2]), base_dir)

        print("------- total time = {} seconds, iteration {}/{} ------".format(time.time() - start_time,i,len(ics)))
    eng.quit()