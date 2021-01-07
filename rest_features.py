import requests
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

url = 'https://services3.arcgis.com/GVgbJbqm8hXASVYi/arcgis/rest/services/Trailheads/FeatureServer/0/ query?geometry=-118.7798,34.0259&geometryType=esriGeometryPoint&inSR=4326&distance=2 &units=esriSRUnit_StatuteMile&outFields=*&returnGeometry=true'
url = 'https://api.github.com'
try:
    response = requests.get(url)

    # If the response was successful, no Exception will be raised
    response.raise_for_status()
except HTTPError as http_err:
    print(f'HTTP error occurred: {http_err}')  # Python 3.6
except Exception as err:
    print(f'Other error occurred: {err}')  # Python 3.6
else:
    print('Success!')
    print(response.json())

