# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 20:00:20 2023

@author: quatlab

# Title: JALBTX Lake Michigan Analaysis
# Author: Collin Roland
# Date Created: 20231212
# Summary: Goal is to systematically analyze volume change for all JABLTX pointclouds
# Date Last Modified: 20240327
# To do:  Fix edge cases for incorrect transect orientation, line smoothing across multible linestrings
"""

# %% Import packages

import alphashape
import contextily as cx
import folium
import geopandas as gpd
import glob
import json
import latex
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
import pdal
import pathlib
from pathlib import Path
import pyproj
import rasterio as rio
import rasterio
from rasterio import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio import errors
from rasterstats import zonal_stats
import re
import rioxarray
import scipy.io
import shapely
import shapelysmooth
from shapely import Point, LineString, MultiLineString, MultiPoint, distance, intersects, buffer, prepare
from shapely.ops import split, snap
from whitebox_tools import WhiteboxTools

# plt.rcParams.update({
#     'figure.constrained_layout.use': True,
#     'font.size': 12,
#     'axes.edgecolor': 'black',
#     'xtick.color':    'black',
#     'ytick.color':    'black',
#     'axes.labelcolor':'black',
#     'axes.spines.right':True,
#     'axes.spines.top':  True,
#     'xtick.direction': 'in',
#     'ytick.direction': 'in',
#     'xtick.major.size': 6,
#     'xtick.minor.size': 4,
#     'ytick.major.size': 6,
#     'ytick.minor.size': 4,
#     'xtick.major.pad': 15,
#     'xtick.minor.pad': 15,
#     'ytick.major.pad': 15,
#     'ytick.minor.pad': 15,
#     })

wbt = WhiteboxTools()
wbt.set_whitebox_dir(Path(r'C:\Users\quatlab\Documents\WBT'))
%matplotlib qt5
# %% Set home directory

homestr = r'D:\CJR'
home = Path(r'D:\CJR')

# %% Self-defined functions


def read_file(file):
    """Read in a raster file
    
    Parameters
    -----
    file: (string) path to input file to read
    """
    return(rasterio.open(file))


def reproj_match(infile, match):
    """Reproject a file to match the shape and projection of existing raster. 
    Uses bilinear interpolation for resampling.
    
    Parameters
    ----------
    infile : (string) path to input file to reproject
    match : (string) path to raster with desired shape and projection 
    outfile : (string) path to output file tif
    """
    # open input
    with rasterio.open(infile) as src:
        src_transform = src.transform
        
        # open input to match
        with rasterio.open(match) as match:
            dst_crs = match.crs
            
            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,     # input CRS
                dst_crs,     # output CRS
                match.width,   # input width
                match.height,  # input height 
                *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
            )
            # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "compress":"LZW",
                           "dtype":"float32",
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": match.nodata})
        memfile = MemoryFile()
        # with MemoryFile() as memfile:
        with memfile.open(**dst_kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear)
        try:
            data = memfile.open()
            error_state = False
            return data, error_state
        except rasterio.errors.RasterioIOError as err:
            error_state = True
            data= []
            return data,error_state
            pass
            #with memfile.open() as dataset:  # Reopen as DatasetReader
                #return dataset

                
def read_paths(path,extension):
    """Read the paths of all files in a directory (including subdirectories)
    with a specified extension
    Parameters
    -----
    file: (string) path to input file to read
    extension: (string) file extension of interest
    """
    AllPaths = []
    FileNames = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                FileNames.append(file)
                filepath = subdir+os.sep+file
                AllPaths.append(filepath)
    return(AllPaths,FileNames)


def get_cell_size(str_grid_path):
    with rasterio.open(str(str_grid_path)) as ds_grid:
        cs_x, cs_y = ds_grid.res
    return cs_x


def define_grid_projection(str_source_grid, dst_crs, dst_file):
    print('Defining grid projection:')
    with rasterio.open(str_source_grid, 'r') as src:
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
        })
        arr_src = src.read(1)
        with rasterio.open(dst_file, 'w', **kwargs) as dst:
            dst.write(arr_src, indexes=1)


def reproject_grid_layer(str_source_grid, dst_crs, dst_file, resolution, logger):
    # reproject raster plus resample if needed
    # Resolution is a pixel value as a tuple
    try:
        st = timer()
        with rasterio.open(str_source_grid) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=resolution)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            with rasterio.open(dst_file, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear)
        end = round((timer() - st)/60.0, 2)
        logger.info(f'Reprojected DEM. Time elapsed: {end} mins')
        return dst_file
    except:
        logger.critical(f'{str_source_grid}: failed to reproject.')
        sys.exit(1)


def reproject_vector_layer(in_shp, str_target_proj4, logger):
    print(f'Reprojecting vector layer: {in_shp}')
    proj_shp = in_shp.parent / f'{in_shp.stem}_proj.shp'
    if proj_shp.is_file():
        logger.info(f'{proj_shp} reprojected file already exists\n')
        return str(proj_shp)
    else:
        gdf = gpd.read_file(str(in_shp))
        # fix float64 to int64
        float64_2_int64 = ['NHDPlusID', 'Shape_Area', 'DSContArea', 'USContArea']
        for col in float64_2_int64:
            try:
                gdf[col] = gdf[col].astype(np.int64)
            except KeyError:
                pass
        gdf_proj = gdf.to_crs(str_target_proj4)
        gdf_proj.to_file(str(proj_shp))
        logger.info(f'{proj_shp} successfully reprojected\n')
        return str(proj_shp)


def clip_features_using_grid(
        str_lines_path, output_filename, str_dem_path, in_crs, logger, mask_shp):
    # clip features using HUC mask, if the mask doesn't exist polygonize DEM
    mask_shp = Path(mask_shp)
    if mask_shp.is_file():
        st = timer()
        # whitebox clip
        WBT.clip(str_lines_path, mask_shp, output_filename)
        end = round((timer() - st)/60.0, 2)
        logger.info(f'Streams clipped by {mask_shp}. Time Elapsed: {end} mins')
    else:
        st = timer()
        logger.warning(f'''
        {mask_shp} does not file exists. Creating new mask from DEM.
        This step can be error prone please review the output.
        ''')
        # Polygonize the raster DEM with rasterio:
        with rasterio.open(str(str_dem_path)) as ds_dem:
            arr_dem = ds_dem.read(1)
        arr_dem[arr_dem > 0] = 100
        mask = arr_dem == 100
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(arr_dem, mask=mask, transform=ds_dem.transform))
            )
        poly = list(results)
        poly_df = gpd.GeoDataFrame.from_features(poly)
        poly_df.crs = in_crs
        # poly_df = poly_df[poly_df.raster_val == 100.0]
        # tmp_shp = os.path.dirname(str_dem_path) + "/mask.shp"  # tmp huc mask
        poly_df.to_file(str(mask_shp))
        # whitebox clip
        WBT.clip(str_lines_path, str(mask_shp), output_filename)
        end = round((timer() - st)/60.0, 2)
        logger.info(f'Streams clipped by {mask_shp}. Time Elapsed: {end} mins')


def run_pdal(json_path,bounds,outfile):
    # 
    with open(json_path) as json_file:
        the_json = json.load(json_file)
    #the_json[0]['filename'] = filename_laz
    #the_json[-9]['groups'] = groups
    the_json[-4]['bounds'] = bounds
    the_json[-1]['filename'] = outfile
    pipeline = pdal.Pipeline(json.dumps(the_json))
    try:
        pipeline.execute()
    except RuntimeError as e:
        print(e)


def open_memory_tif(arr, meta):
    from rasterio.io import MemoryFile
    #     with rasterio.Env(GDAL_CACHEMAX=256, GDAL_NUM_THREADS='ALL_CPUS'):
    with MemoryFile() as memfile:
        with memfile.open(**meta) as dataset:
            dataset.write(arr, indexes=1)
        return memfile.open()

def raster_clip(infile, clip_geom, crs_in):
    #Debugging
    # infile= value
    # crs_in = clip_geom.crs
    # clip_geom = clip_geom
    
    infile=infile
    clip_geom = clip_geom.to_crs('EPSG:32616')
    clip_geom = clip_geom.reset_index()
    dem = read_file(infile)
    try:
        out_image, out_transform = mask(dem,[clip_geom.geometry[0]], nodata=dem.meta['nodata'],crop=True)
        error_state = False
        out_meta = dem.meta
        out_meta.update({"crs": dem.crs,
                           "compress":"LZW",
                           "dtype":"float32",
                           "transform": out_transform,
                           "width": out_image.shape[2],
                           "height": out_image.shape[1]})
        clip_kwargs = out_meta
        memfile_clip = MemoryFile()
        with memfile_clip.open(**clip_kwargs) as dst:
            dst.write(out_image)
        return memfile_clip,error_state
    except ValueError as err:
        memfile_clip=[]
        error_state=True
        return memfile_clip,error_state
        pass
    
def densify_geometry(line_geometry, step, crs=None):
    # crs: epsg code of a coordinate reference system you want your line to be georeferenced with
    # step: add a vertice every step in whatever unit your coordinate reference system use.
    length_m=line_geometry.length # get the length
    xy=[] # to store new tuples of coordinates
    for distance_along_old_line in np.arange(0,int(length_m),step): 
        point = line_geometry.interpolate(distance_along_old_line) # interpolate a point every step along the old line
        xp,yp = point.x, point.y # extract the coordinates
        xy.append((xp,yp)) # and store them in xy list
    new_line=LineString(xy) # Here, we finally create a new line with densified points.  
    if crs != None:  #  If you want to georeference your new geometry, uses crs to do the job.
        new_line_geo=gpd.geoseries.GeoSeries(new_line,crs=crs) 
        return new_line_geo
    else:
        return new_line
    
def gen_xsec(point, angle, poslength, neglength, step, merged_dem, crs=None):
    '''
    point - Tuple (x, y)
    angle - Angle you want your end point at in degrees.
    length - Length of the line you want to plot.
    
    Will plot the line on a 10 x 10 plot.
    '''
    #
    step = step
    crs=crs
    xsec_angle = angle
    # unpack the first point
    x = point.iloc[0].xy[0][0]
    y = point.iloc[0].xy[1][0] 
    # find the end point
    endy = y + poslength * math.cos(math.radians(xsec_angle[0]))
    endx = x + poslength * math.sin(math.radians(xsec_angle[0]))
    end_point = Point(endx, endy)   
    # find the start point
    starty = y + neglength * math.cos(math.radians(xsec_angle[0]+180))
    startx = x + neglength * math.sin(math.radians(xsec_angle[0]+180))
    start_point = Point(startx,starty)
    # Figure out which direction is landward
    xsec_line = LineString([start_point, end_point])
    xsec_line = densify_geometry(xsec_line, step, crs=crs)
    start_coords = xsec_line.geometry[0].coords[0:int(neglength)]
    start_elevs = np.array([sample[0] for sample in merged_dem.sample(start_coords)])
    start_elevs[start_elevs < 0.0] = np.nan
    start_elev = np.nanmedian(start_elevs)
    if np.isnan(start_elev):
        start_elev = 100.0
    # print("Start elev=",start_elev)
    end_coords = xsec_line.geometry[0].coords[int(neglength):int(neglength + poslength)]
    end_elevs = np.array([sample[0] for sample in merged_dem.sample(end_coords)])
    end_elevs[end_elevs < 0.0] = np.nan
    end_elev = np.nanmedian(end_elevs)   
    if np.isnan(end_elev):
        end_elev = 100.0
    # print("End elev=",end_elev)
    if start_elev >= end_elev:
        xsec_angle_mod = xsec_angle[0]+180
         
         # find the end point
        endy2 = y + poslength * math.cos(math.radians(xsec_angle_mod))
        endx2 = x + poslength * math.sin(math.radians(xsec_angle_mod))
        end_point2 = Point(endx2, endy2) 
         
        # find the start point
        starty2 = y + neglength * math.cos(math.radians(xsec_angle_mod+180))
        startx2 = x + neglength * math.sin(math.radians(xsec_angle_mod+180))
        start_point2 = Point(startx2,starty2)
    
        # generate a line from points
        xsec_line2 = LineString([start_point2, end_point2])
        xsec_line2 = densify_geometry(xsec_line2, step, crs=crs)
        start_coords = xsec_line2.geometry[0].coords[0:int(neglength)]
        start_elevs = np.array([sample[0] for sample in merged_dem.sample(start_coords)])
        start_elevs[start_elevs < 0.0] = np.nan
        start_elev2 = np.nanmedian(start_elevs)
        if np.isnan(start_elev2):
            start_elev2 = 100.0
        # print("Start elev=",start_elev)
        end_coords = xsec_line2.geometry[0].coords[int(neglength):int(neglength + poslength)]
        end_elevs = np.array([sample[0] for sample in merged_dem.sample(end_coords)])
        end_elevs[end_elevs < 0.0] = np.nan
        end_elev2 = np.nanmedian(end_elevs) 
        if np.isnan(end_elev2):
            end_elev2 = 100.0
        if (start_elev2 < end_elev2):
            xsec_line = xsec_line2
    # densify line to specified resolution
    # Debugging
    # fix, ax = plt.subplots(1)
    # ax.set_aspect('equal')
    # ax.plot(shoreline_sing3.oords.xy[0], shoreline_sing3.coords.xy[1])
    # # ax.plot(smooth_baseline.coords.xy[0], smooth_baseline.coords.xy[1])
    # merge_shorelines.plot(ax=ax)
    # xsec_line.plot(ax=ax)
    # ax.scatter(start_point.x, start_point.y, color='black')
    # ax.scatter(end_point.x, end_point.y, color='red')

    # xsec_line2.plot(ax=ax, color='red')
    # ax.plot(xsec_line.coords.xy[0], xsec_line.coords.xy[1])
    return xsec_line, start_point, end_point

def replace_line_end(adjacent_line, line, snap_tolerance):
    '''
    adjacent_line = Shapely Linestring for line whose endpoint that will be snapped to line
    line = Shapely Linestring for line that is being snapped to
    snap_tolerance = float, distance in meters that snaps can occur across
    
    Snaps start/endpoint of a linestring and regenerates line
    '''
    # Debugging
    # adjacent_line = seg_1
    # line = seg_2
    # snap_tolerance = 20.0
    start_adj = shapely.Point(adjacent_line.coords.xy[0][0], adjacent_line.coords.xy[1][0])
    end_adj = shapely.Point(adjacent_line.coords.xy[0][-1], adjacent_line.coords.xy[1][-1])
    adj_near = shapely.ops.nearest_points(line, adjacent_line.boundary)[1]
    adj_near_snap = shapely.snap(adj_near, line, tolerance = snap_tolerance)
    distance = adj_near.distance([start_adj, end_adj])
    adj_line_points = [shapely.Point(i) for i in adjacent_line.coords]
    if adj_near.distance(start_adj) < adj_near.distance(end_adj):
        adj_line_points[0] = adj_near_snap
    else:
        adj_line_points[-1] = adj_near_snap
    adjacent_line = shapely.LineString(adj_line_points)
    return adjacent_line
     
def gen_xsec_wrap(shoreline_clip, outdir, tile_name, poslength, neglength, xsec_spacing, simp_tolerance, step, merged_dem, crs):
    '''
    line - geoseries with a linestring geometry
    point - Tuple (x, y)
    angle - Angle you want your end point at in degrees.
    length - Length of the line you want to plot.
   
    Will plot the line on a 10 x 10 plot.
    ''' 
    # Smooth baseline
    merge_shorelines = shapely.unary_union(shoreline_clip.geometry)
    if type(merge_shorelines.geom_type) == 'MultiLineString':
        merge_shorelines = shapely.ops.linemerge(merge_shorelines)
        merge_shorelines = gpd.GeoSeries(data = merge_shorelines, crs='EPSG:32616')
    else:
        merge_shorelines = gpd.GeoSeries(data = merge_shorelines, crs='EPSG:32616')
    if type(merge_shorelines.geometry[0].geom_type) == 'MultiLineString':
        merge_shorelines = merge_shorelines.geometry.explode(ignore_index=True)
        snap_tolerance = 20.0
        for count in range(0, len(merge_shorelines)):
            # Debugging
            # count = 0
            segment = merge_shorelines.iloc[count]
            segment_mod = merge_shorelines.iloc[count]
            segment_distances = segment.distance([i for i in merge_shorelines if i != segment])
            segment_distances = np.insert(segment_distances, count, 1000.0)
            touching_segments = np.where(segment_distances < snap_tolerance)[0].tolist()
            if len(touching_segments) > 0:
                for i in touching_segments:
                    segment_mod = replace_line_end(segment_mod, merge_shorelines.iloc[i], segment_distances[i]+2.0)
                merge_shorelines.iloc[count] = segment_mod
        # fig, ax = plt.subplots(1)
        # ax.set_aspect('equal')
        # for i in range(0, len(merge_shorelines)):
        #     merge_shorelines.iloc[[i]].plot(ax=ax, linestyle='dashed', color=np.random.rand(3,))
        merge_shorelines = shapely.unary_union(merge_shorelines.geometry)
        merge_shorelines = shapely.ops.linemerge(merge_shorelines)
        merge_shorelines = gpd.GeoSeries(data=merge_shorelines, crs='EPSG:32616')
    merge_shorelines = merge_shorelines.geometry.explode(index_parts=True)
    for count in range(0, len(merge_shorelines[0])):
        geom = merge_shorelines[0][count]
        geom = shapelysmooth.chaikin_smooth(geom.simplify(simp_tolerance), iters = 7)
        merge_shorelines[0][count] = geom
    for count_1 in range(0,len(merge_shorelines)):
        # Debugging
        # count_1 = 1
        # count_4 = 0
        shoreline_sing_1 = gpd.GeoSeries(data=merge_shorelines.iloc[count_1], crs=crs)
        if shoreline_sing_1.geometry.geom_type[0]=='MultiLineString':
            shoreline_sing_2 = shoreline_sing_1.geometry.explode(index_parts=True)
        else:
            shoreline_sing_2 = shoreline_sing_1
        for count_4 in range(0,len(shoreline_sing_2)):
            shoreline_sing_3 = gpd.GeoSeries(data=shoreline_sing_2.iloc[count_4], crs=crs)
            if (shoreline_sing_3.geometry.length[0]>=(2*spacing)):   
                # simplify_baseline = shoreline_sing_3.simplify(simp_tolerance)
                # simplify_baseline = simplify_baseline.reset_index()
                # smooth_baseline = shapelysmooth.catmull_rom_smooth(simplify_baseline.geometry[0],alpha=0.9)
                # fig,ax = plt.subplots(1,1)
                # ax.set_aspect('equal')
                # shoreline_sing_1.plot(ax=ax)
                # plt.plot(smooth_baseline.coords.xy[0],smooth_baseline.coords.xy[1],color='red')
                # for geom in shoreline_clip[0:5]:
                #     ax.plot(geom.coords.xy[0], geom.coords.xy[0], color='black')
                # Create cross-section points
                num_points = []
                num_points = int(shoreline_sing_3.iloc[0].length / spacing)
                lenSpace = np.linspace(spacing, shoreline_sing_3.iloc[0].length, num_points)
                tempPointList_xsec = []
                tempLineList_xsec = []
                tempStartPoints_xsec = []
                tempEndPoints_xsec = []
                for space in lenSpace:
                    tempPoint_xsec = (shoreline_sing_3.interpolate(space)) #.tolist()[0]
                    tempPointList_xsec.append(tempPoint_xsec) 
                # Calculate bearing at cross-section points
                transformer = pyproj.Transformer.from_crs(32616, 6318)
                geodesic = pyproj.Geod(ellps='WGS84')
                temp_angle = []
                for count_2,value in enumerate(tempPointList_xsec):
                    if count_2>0:
                        lat1,long1 = transformer.transform(tempPointList_xsec[count_2-1].x,tempPointList_xsec[count_2-1].y)
                        lat2,long2 = transformer.transform(tempPointList_xsec[count_2].x,tempPointList_xsec[count_2].y)
                        fwd_azimuth,back_azimuth,distance = geodesic.inv(long1,lat1,long2,lat2)
                        orth_angle = back_azimuth-90
                        temp_angle.append(orth_angle)
                    if count_2==0:
                        count_2=1
                        lat1,long1 = transformer.transform(tempPointList_xsec[count_2-1].x,tempPointList_xsec[count_2-1].y)
                        lat2,long2 = transformer.transform(tempPointList_xsec[count_2].x,tempPointList_xsec[count_2].y)
                        fwd_azimuth,back_azimuth,distance = geodesic.inv(long1,lat1,long2,lat2)
                        orth_angle = back_azimuth-90
                        temp_angle.append(orth_angle)
                # Generate cross-section lines
                xsec_lines = []
                xsec_starts = []
                xsec_ends = []
                for count_3, value in enumerate(temp_angle):
                    # Debugging
                    # count_3 = 120
                    # value = temp_angle[count_3]
                    [xsec_line, start_point, end_point] = gen_xsec(tempPointList_xsec[count_3], value, poslength, neglength, step, merged_dem, crs=crs)
                    xsec_lines.append(xsec_line.iloc[0])
                    xsec_starts.append(start_point)
                    xsec_ends.append(end_point)  
                # Convert to GDF
                xsec_lines = gpd.GeoSeries(xsec_lines,crs=crs)
                xsec_starts = gpd.GeoSeries(xsec_starts,crs=crs)
                xsec_ends = gpd.GeoSeries(xsec_ends,crs=crs)
                xsec_shoreline = gpd.GeoSeries(shoreline_sing_3.geometry, crs=crs)   
                # Write outputs
                tmp_start = os.path.join(outdir,'StartPoints',(tile_name+'_start_points_'+str(count_1)+'_'+str(count_4)+'.shp'))
                tmp_end = os.path.join(outdir,'EndPoints',(tile_name+'_end_points_'+str(count_1)+'_'+str(count_4)+'.shp'))
                tmp_lines = os.path.join(outdir,'Transects',(tile_name+'_transects_'+str(count_1)+'_'+str(count_4)+'.shp'))
                tmp_shore = os.path.join(outdir,'Shorelines',(tile_name+'_shoreline_'+str(count_1)+'_'+str(count_4)+'.shp'))
                xsec_lines.to_file(tmp_lines)
                xsec_starts.to_file(tmp_start)
                xsec_ends.to_file(tmp_end)
                xsec_shoreline.to_file(tmp_shore)
                # plt.close()
                # Debugging
                # fix, ax = plt.subplots(1)
                # ax.set_aspect('equal')
                # ax.plot(shoreline_clip.iloc[4].coords.xy[0], shoreline_clip.iloc[4].coords.xy[1])
                # ax.plot(smooth_baseline.coords.xy[0], smooth_baseline.coords.xy[1])
                # ax.plot(xsec_line.coords.xy[0], xsec_line.coords.xy[1]))
                # xsec_line.plot(ax=ax)


def fix_index(gdf):
    gdf["row_id"] = gdf.index
    gdf.reset_index(drop=True, inplace=True)
    gdf.set_index("row_id", inplace=True)
    return (gdf)


def linestring_to_points(feature, line):
    return {feature: line.coords}


def make_xsec_points(transects, merged_dem):
    # Debugging
    transects = transects
    all_points = pd.DataFrame()
    for count,transect in enumerate(transects.geometry):
        points = [Point(coord[0], coord[1]) for coord in transect.coords]
        x = [coord[0] for coord in transect.coords]
        y = [coord[1] for coord in transect.coords]
        ID = [count]*len(points)
        near_dist = [distance(points[0],point) for point in points]
        elevation = [x[0] for x in merged_dem.sample(transect.coords)]
        
        #elevation = [np.nan if i<100 else i for i in elevation]
        temp_df = pd.DataFrame({'ID_1':ID,'RASTERVALU':elevation,'NEAR_DIST':near_dist,'Easting':x,'Northing':y})
        all_points= pd.concat([all_points,temp_df])
    all_points = all_points.reset_index(drop=True)
    all_points['FID']= all_points.index
    all_points = all_points[['FID','ID_1','RASTERVALU','NEAR_DIST','Easting','Northing']]
    return(all_points)
 
       
def temp_merge(dem_files):
    mem = MemoryFile()
    merge(dem_files, dst_path=mem.name)
    merged_dem = rasterio.open(mem)
    return(merged_dem)

def rasterize_points_2012(count, tiles2009, laz2009_root, dem2009_root_buffer, json_pipeline, json_pipeline_mod):
    # Select a 2009 tile, find adjacent 2009 tiles
    tile2009_sing = tiles2009.iloc[count] # pull out a single 2009 tile
    tiles_2009_adj = tiles2009[shapely.intersects(buffer(tile2009_sing.geometry, 50), tiles2009.geometry).values] # buffer tile by 50 and find intersecting tiles
    merge_tile_2009_paths = [os.path.join(laz2009_root, i) for i in tiles_2009_adj['Name']] 
    # calculate bounds
    bounds = buffer(tile2009_sing.geometry, 50).bounds
    bounds_json = str("("+"["+str(bounds[0])+","+str(+bounds[2])+"]"+","+"["+str(bounds[1])+","+str(bounds[3])+"]"+")")
    # Process 2009 LAZ
    filename_laz = os.path.join(laz2009_root,tile2009_sing['Name'])
    filename_tif = os.path.join(dem2009_root_buffer,os.path.splitext(os.path.basename(filename_laz))[0]+'.tif') 
    # Writing merged JSON 2009
    with open(json_pipeline) as json_file:
        the_json = json.load(json_file)
        the_json[-6]["expression"]="(Classification == 2) || (Classification == 11)"
    with open(json_pipeline_mod,'w',encoding='utf_8') as f:
        f.write("[\n")
        for i,val in enumerate(merge_tile_2009_paths):
            basename = os.path.splitext(val)[0][:]
            basename = re.escape(str(basename))
            string=str(basename+'.laz')
            #string = re.escape(string)
            if (i<(len(merge_tile_2009_paths)-1)):
                f.write('\t'+"\""+string+"\""+",\n")
            if (i==(len(merge_tile_2009_paths)-1)):
                f.write('\t'+"\""+string+"\",")
        for i,val1 in enumerate(the_json):
            part = val1
            json_str = str(part)
            json_str = json_str.replace('{','')
            json_str = json_str.replace(' ','')
            json_str = json_str.replace('}','')
            json_str = json_str.replace('\'','\"')
            # i is counter for json, j is counter for json part
            # Case for non-ending JSON block with a single line
            if (i<(len(the_json)-1)) & (len(part)<2):
                json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t},')
                f.write(json_str)
            # Case for non-ending JSON block with multiple lines
            if (i<(len(the_json)-1)) & (len(part)>1):
                json_str = json_str.replace(',',',\n\t\t')
                json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t},')
                f.write(json_str)
            # Case for ending JSON block with a single line
            if (i==(len(the_json)-1)) & (len(part)<2):
                json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t}\n]')
                f.write(json_str)
            # Case for ending JSON block with multiple lines
            if (i==(len(the_json)-1)) & (len(part)>1):
                json_str = json_str.replace(',',',\n\t\t')
                json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t}\n]')
                f.write(json_str)
    run_pdal(json_pipeline_mod,bounds_json,filename_tif)
    print("Finished with:",tile2009_sing['Name']) 
# %% Read tile indexes and other filenames/paths

def rasterize_points_generic(count, tiles2009, laz2009_root, dem2009_root_buffer, json_pipeline, json_pipeline_mo):
    # Select a 2009 tile, find adjacent 2009 tiles
    tile2009_sing = tiles2009.iloc[count] # pull out a single 2009 tile
    tiles_2009_adj = tiles2009[shapely.intersects(buffer(tile2009_sing.geometry, 50), tiles2009.geometry).values] # buffer tile by 50 and find intersecting tiles
    merge_tile_2009_paths = [os.path.join(laz2009_root, i) for i in tiles_2009_adj['Name']] 
    # calculate bounds
    bounds = buffer(tile2009_sing.geometry, 50).bounds
    bounds_json = str("("+"["+str(bounds[0])+","+str(+bounds[2])+"]"+","+"["+str(bounds[1])+","+str(bounds[3])+"]"+")")
    # Process 2009 LAZ
    filename_laz = os.path.join(laz2009_root,tile2009_sing['Name'])
    filename_tif = os.path.join(dem2009_root_buffer,os.path.splitext(os.path.basename(filename_laz))[0]+'.tif') 
    # Writing merged JSON 2009
    with open(json_pipeline) as json_file:
        the_json = json.load(json_file)
        the_json[-6]["expression"]="(Classification == 2) || (Classification == 11)"
    with open(json_pipeline_mod,'w',encoding='utf_8') as f:
        f.write("[\n")
        for i,val in enumerate(merge_tile_2009_paths):
            basename = os.path.splitext(val)[0][:]
            basename = re.escape(str(basename))
            string=str(basename+'.laz')
            #string = re.escape(string)
            if (i<(len(merge_tile_2009_paths)-1)):
                f.write('\t'+"\""+string+"\""+",\n")
            if (i==(len(merge_tile_2009_paths)-1)):
                f.write('\t'+"\""+string+"\",")
        for i,val1 in enumerate(the_json):
            part = val1
            json_str = str(part)
            json_str = json_str.replace('{','')
            json_str = json_str.replace(' ','')
            json_str = json_str.replace('}','')
            json_str = json_str.replace('\'','\"')
            # i is counter for json, j is counter for json part
            # Case for non-ending JSON block with a single line
            if (i<(len(the_json)-1)) & (len(part)<2):
                json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t},')
                f.write(json_str)
            # Case for non-ending JSON block with multiple lines
            if (i<(len(the_json)-1)) & (len(part)>1):
                json_str = json_str.replace(',',',\n\t\t')
                json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t},')
                f.write(json_str)
            # Case for ending JSON block with a single line
            if (i==(len(the_json)-1)) & (len(part)<2):
                json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t}\n]')
                f.write(json_str)
            # Case for ending JSON block with multiple lines
            if (i==(len(the_json)-1)) & (len(part)>1):
                json_str = json_str.replace(',',',\n\t\t')
                json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t}\n]')
                f.write(json_str)
    run_pdal(json_pipeline_mod,bounds_json,filename_tif)
    print("Finished with:",tile2009_sing['Name']) 
os.chdir(home)
tiles2012 = gpd.read_file(r'.\JABLTX_2012\gl2012_usace_lakemichigan_index.shp')
tiles2012 = tiles2012.to_crs('EPSG:32616')
tiles2020 = gpd.read_file(r'.\JABLTX_2020\usace2020_lake_mich_il_in_mi_wi_index.shp')
tiles2020 = tiles2020.to_crs('EPSG:32616')


laz2012_root = r'D:\CJR\JABLTX_2012\LAZ'
dem2012_root = r'D:\CJR\JABLTX_2012\DEM'
dem2012_root_buffer = r'D:\CJR\JABLTX_2012\DEM\buffer'
laz2020_root = r'D:\CJR\JABLTX_2020\LAZ'
dem2020_root = r'D:\CJR\JABLTX_2020\DEM'
dem2020_root_buffer = r'D:\CJR\JABLTX_2020\DEM\buffer'
dem_fill_2012_root = r'D:\CJR\JABLTX_2012\DEMfill'
dem_fill_2020_root = r'D:\CJR\JABLTX_2020\DEMfill'
json2012 = homestr+r'\PDAL\lake_mi_jalbtx_2012_pipeline_delaunay.json'
json2012_mod = homestr+r'\PDAL\lake_mi_jalbtx_2012_pipeline_delaunay_mod.json'
json2020 = homestr+r'\PDAL\lake_mi_jalbtx_2020_pipeline_delaunay.json'
json2020_mod = homestr+r'\PDAL\lake_mi_jalbtx_2020_pipeline_delaunay_mod.json'

# Write tiles to folium map for viewing
# os.chdir(homestr+'\Folium')
# tile_map = tiles2012.explore(name="2012 tiles")
# tile_map = tiles2020.explore(m=tile_map, color="red",name="2020 tiles")
# folium.LayerControl().add_to(tile_map)
# tile_map.save("tiles.html")

# %% Rasterize
    
def rasterize_points(count, tiles2009, laz2009_root, tiles2019, laz2019_root, dem2009_root_buffer, dem2019_root_buffer, dem_fill_2009_root, dem_fill_2019_root, json_pipeline, json_pipeline_mod):
    # Debugging
    # count = 100
    # tiles2009 = tiles2012
    # laz2009_root = laz2012_root
    # tiles2019 = tiles2020
    # laz2019_root = laz2020_root
    # dem2009_root_buffer = dem2012_root_buffer
    # dem2019_root_buffer = dem2020_root_buffer
    # dem_fill_2009_root = dem_fill_2012_root
    # dem_fill_2019_root = dem_fill_2020_root
    # json_pipeline = json2020
    # json_pipeline_mod = json2020_mod
    # Select a 2009 tile, find adjacent 2009 tiles
    tile2009_sing = tiles2009.iloc[count] # pull out a single 2009 tile
    tiles_2009_adj = tiles2009[shapely.intersects(buffer(tile2009_sing.geometry, 50), tiles2009.geometry).values] # buffer tile by 50 and find intersecting tiles
    merge_tile_2009_paths = [os.path.join(laz2009_root, i) for i in tiles_2009_adj['Name']]
    # calculate bounds
    bounds = buffer(tile2009_sing.geometry, 50).bounds
    bounds_json = str("("+"["+str(bounds[0])+","+str(+bounds[2])+"]"+","+"["+str(bounds[1])+","+str(bounds[3])+"]"+")")
    # identify adjacent 2019 tiles using the buffered 2009 tile
    tile2019_intersect = tiles2019[shapely.intersects(buffer(tile2009_sing.geometry, 50), tiles2019.geometry).values]
    tile2019_intersect = tile2019_intersect.reset_index()
    tile2019_intersect = tile2019_intersect.drop(['index','Index'],axis=1)
    merge_tile_2019_paths = [os.path.join(laz2019_root,i[3:]) for i in tile2019_intersect['Name']]
    # Process LAZ files
    if len(tile2019_intersect)>0:
        # Process 2009 LAZ
        filename_laz = os.path.join(laz2009_root,tile2009_sing['Name'])
        filename_tif = os.path.join(dem2009_root_buffer,os.path.splitext(os.path.basename(filename_laz))[0]+'.tif')
        # Writing merged JSON 2009
        with open(json_pipeline) as json_file:
            the_json = json.load(json_file)
            the_json[-6]["expression"]="(Classification == 2) || (Classification == 11)"
        with open(json_pipeline_mod,'w',encoding='utf_8') as f:
            f.write("[\n")
            for i,val in enumerate(merge_tile_2009_paths):
                basename = os.path.splitext(val)[0][:]
                basename = re.escape(str(basename))
                string=str(basename+'.laz')
                #string = re.escape(string)
                if (i<(len(merge_tile_2009_paths)-1)):
                    f.write('\t'+"\""+string+"\""+",\n")
                if (i==(len(merge_tile_2009_paths)-1)):
                    f.write('\t'+"\""+string+"\",")
            for i,val1 in enumerate(the_json):
                part = val1
                json_str = str(part)
                json_str = json_str.replace('{','')
                json_str = json_str.replace(' ','')
                json_str = json_str.replace('}','')
                json_str = json_str.replace('\'','\"')
                # i is counter for json, j is counter for json part
                # Case for non-ending JSON block with a single line
                if (i<(len(the_json)-1)) & (len(part)<2):
                    json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t},')
                    f.write(json_str)
                # Case for non-ending JSON block with multiple lines
                if (i<(len(the_json)-1)) & (len(part)>1):
                    json_str = json_str.replace(',',',\n\t\t')
                    json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t},')
                    f.write(json_str)
                # Case for ending JSON block with a single line
                if (i==(len(the_json)-1)) & (len(part)<2):
                    json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t}\n]')
                    f.write(json_str)
                # Case for ending JSON block with multiple lines
                if (i==(len(the_json)-1)) & (len(part)>1):
                    json_str = json_str.replace(',',',\n\t\t')
                    json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t}\n]')
                    f.write(json_str)
        with open(json_pipeline_mod) as json_file:
            the_json = json.load(json_file) 
        run_pdal(json_pipeline_mod,bounds_json,filename_tif)
        basename = os.path.splitext(os.path.basename(filename_laz))[0]+'.tif'
        basename = basename[14:]
        basename = str(os.path.basename(merge_tile_2019_paths[0])[0:8]+'_'+basename)
        filename_tif = os.path.join(dem2019_root_buffer,basename)  
        # Writing merged JSON 2019
        with open(json_pipeline) as json_file:
            the_json = json.load(json_file)
            the_json[-2]["max_triangle_edge_length"]=100
            the_json[-6]["expression"]="(Classification == 2) || (Classification == 29)"
        with open(json_pipeline_mod,'w',encoding='utf_8') as f:
            f.write("[\n")
            for i,val in enumerate(merge_tile_2019_paths):
                basename = os.path.splitext(val)[0][:]
                basename = re.escape(str(basename))
                string=str(basename[0:-6]+'.copc.laz')
                #string = re.escape(string)
                if (i<(len(merge_tile_2019_paths)-1)):
                    f.write('\t'+"\""+string+"\""+",\n")
                if (i==(len(merge_tile_2019_paths)-1)):
                    f.write('\t'+"\""+string+"\",")
            for i,val1 in enumerate(the_json):
                part = val1
                json_str = str(part)
                json_str = json_str.replace('{','')
                json_str = json_str.replace(' ','')
                json_str = json_str.replace('}','')
                json_str = json_str.replace('\'','\"')
                # i is counter for json, j is counter for json part
                # Case for non-ending JSON block with a single line
                if (i<(len(the_json)-1)) & (len(part)<2):
                    json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t},')
                    f.write(json_str)
                # Case for non-ending JSON block with multiple lines
                if (i<(len(the_json)-1)) & (len(part)>1):
                    json_str = json_str.replace(',',',\n\t\t')
                    json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t},')
                    f.write(json_str)
                # Case for ending JSON block with a single line
                if (i==(len(the_json)-1)) & (len(part)<2):
                    json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t}\n]')
                    f.write(json_str)
                # Case for ending JSON block with multiple lines
                if (i==(len(the_json)-1)) & (len(part)>1):
                    json_str = json_str.replace(',',',\n\t\t')
                    json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t}\n]')
                    f.write(json_str)
        with open(json_pipeline_mod) as json_file:
            the_json = json.load(json_file) 
        run_pdal(json_pipeline_mod,bounds_json,filename_tif)
        # Clip DEMs
        # clip_geom = gpd.GeoDataFrame(geometry=gpd.GeoSeries(tile2009_sing.geometry),crs='EPSG:6344')
        # filename_tif = os.path.join(dem2009_root,os.path.splitext(os.path.basename(filename_laz))[0]+'.tif')
        # [clip2009,errorState] = raster_clip(filename_tif,clip_geom,clip_geom.crs)
        # filename_tif = os.path.join(dem2009_root, 'clip',os.path.splitext(os.path.basename(filename_laz))[0]+'.tif')
        # DEM = read_file(clip2009).read(1,masked=True)
        # DEM.shape = [1,np.shape(DEM)[0],np.shape(DEM)[1]]
        # with rasterio.open(filename_tif, "w", **read_file(clip2009).meta) as dest:
        #     dest.write(DEM)
        # out_name = os.path.join(dem_fill_2009_root,os.path.splitext(os.path.basename(filename_laz))[0]+'.tif')
        # wbt.fill_missing_data(filename_tif,out_name,filter=15,weight=5.0,no_edges=True)
        # clip2009 = None
        # dest = None
        # filename_tif = None
        # DEM = None
        
        # filename_tif_2019 = os.path.join(dem2019_root,basename)  
        # [clip2019, errorState] = raster_clip(filename_tif_2019, clip_geom, clip_geom.crs)
        # filename_tif = os.path.join(dem2019_root, 'clip',basename)
        # DEM = read_file(clip2019).read(1,masked=True)
        # DEM.shape = [1,np.shape(DEM)[0],np.shape(DEM)[1]]
        # with rasterio.open(filename_tif, "w", **read_file(clip2019).meta) as dest:
        #     dest.write(DEM)
        # out_name = os.path.join(dem_fill_2019_root,basename)
        # wbt.fill_missing_data(filename_tif,out_name,filter=15,weight=5.0,no_edges=True)
        # clip2019 = None
        # dest = None
        # filename_tif = None
        # DEM = None
    else:
        print("No overlapping tiles")
    print("Finished with:",tile2009_sing['Name'])   

# from joblib import Parallel, delayed
# Parallel(n_jobs=4)(delayed(rasterize_points(count,tiles2009, laz2009_root, tiles2019, laz2019_root, dem2009_root, dem2019_root, json_pipeline, json_pipeline_mod) for count in range(0,len(tiles2009-1))))    

for count in range(0,len(tiles2012)-1):
    rasterize_points(count, tiles2012, laz2012_root, tiles2020, laz2020_root, dem2012_root_buffer, dem2020_root_buffer, dem_fill_2012_root, dem_fill_2020_root, json_pipeline, json_pipeline_mod)
# %% Clip DEMs


[dem_paths_2009, dem_filenames_2009] = read_paths(dem2012_root_buffer,'.tif')
for count, value in enumerate(dem_paths_2009):
    # Debugging 
    # count = 0
    # value = dem_paths_2009[count]
    
    substring = dem_filenames_2009[count][:-4]
    tile2009_sing = [tiles2012[tiles2012['Name']==i] for i in tiles2012['Name'] if i[:-4] in substring][0]
    clip_geom = gpd.GeoDataFrame(geometry=gpd.GeoSeries(tile2009_sing.geometry),crs='EPSG:32616')
    [clip2009,errorState] = raster_clip(value, clip_geom, clip_geom.crs)
    if clip2009==[]:
        print("No overlapping data ",count)
    else:
        filename_tif = os.path.join(dem2012_root, 'clip', dem_filenames_2009[count])
        DEM = read_file(clip2009).read(1,masked=True)
        DEM.shape = [1,np.shape(DEM)[0],np.shape(DEM)[1]]
        with rasterio.open(filename_tif, "w", **read_file(clip2009).meta) as dest:
            dest.write(DEM)
    clip2009 = None
    dest = None
    filename_tif = None
    DEM = None
 
[dem_paths_2019, dem_filenames_2019] = read_paths(dem2020_root_buffer,'.tif')
for count, value in enumerate(dem_paths_2019):
    # Debugging 
    # count = 0
    # value = dem_paths_2009[count]
    
    substring = dem_filenames_2019[count][8:-4]
    tile2009_sing = [tiles2012[tiles2012['Name']==i] for i in tiles2012['Name'] if i[8:-4] in substring][0]
    clip_geom = gpd.GeoDataFrame(geometry=gpd.GeoSeries(tile2009_sing.geometry),crs='EPSG:32616')
    [clip2019,errorState] = raster_clip(value,clip_geom,clip_geom.crs)
    if clip2019==[]:
        print("No overlapping data ",count)
    else:
        filename_tif = os.path.join(dem2019_root, 'clip', dem_filenames_2019[count])
        DEM = read_file(clip2019).read(1,masked=True)
        DEM.shape = [1,np.shape(DEM)[0],np.shape(DEM)[1]]
        with rasterio.open(filename_tif, "w", **read_file(clip2019).meta) as dest:
            dest.write(DEM)
    clip2019 = None
    dest = None
    filename_tif = None
    DEM = None
# %% Interpolate missing raster data

dem_fill_2012_root = r'D:\CJR\JABLTX_2012\DEMfill'
dem_fill_2020_root = r'D:\CJR\JABLTX_2020\DEMfill'
[dem_paths_2012,dem_filenames_2012] = read_paths(r'D:\CJR\JABLTX_2012\DEM','.tif')
[dem_paths_2020,dem_filenames_2020] = read_paths(r'D:\CJR\JABLTX_2020\DEM','.tif')

os.chdir(dem_fill_2012_root)
for count,value in enumerate(dem_paths_2012):
    out_name = os.path.join(dem_fill_2012_root,dem_filenames_2012[count])
    wbt.fill_missing_data(value,out_name,filter=31,weight=5.0,no_edges=False)
    
os.chdir(dem_fill_2020_root)
for count,value in enumerate(dem_paths_2020):
    out_name = os.path.join(dem_fill_2020_root,dem_filenames_2020[count])
    wbt.fill_missing_data(value,out_name,filter=31,weight=5.0,no_edges=False)  

# %% Generate DODs

[dem_paths_2012, dem_filenames_2012] = read_paths(dem_fill_2012_root,'.tif')
[dem_paths_2020, dem_filenames_2020] = read_paths(dem_fill_2020_root,'.tif')
dod_path = Path(r'D:\CJR\lake_michigan_dod\tiles')
for count,value in enumerate(dem_paths_2012):
    template = dem_filenames_2012[count][-20:]
    if count==62:
        template = dem_filenames_2012[count][-17:] 
    filepath_2020 = [i for i in dem_paths_2020 if template in i]
    reproj2020, error_state = reproj_match(infile=filepath_2020[0],match=value)
    DEM_2012 = read_file(value)
    DOD = reproj2020.read(1,masked=True)-DEM_2012.read(1,masked=True)
    DOD.shape = [1,np.shape(DOD)[0],np.shape(DOD)[1]]
    outname_base = dem_filenames_2012[count][8:]
    outname = str("DOD"+outname_base)
    outname = os.path.join(dod_path,outname)
    with rasterio.open(outname, "w", **DEM_2012.meta) as dest:
        dest.write(DOD)
    print(count)
# %% Generate transects using 2012 data

[dem_paths_2012, dem_filenames_2012] = read_paths(dem_fill_2012_root,'.tif')
[dem_paths_2020, dem_filenames_2020] = read_paths(dem_fill_2020_root,'.tif')
   
outdir = r"D:\CJR\lake_michigan_bluff_delineation\2012"
os.chdir(outdir)
os.chdir(r'..')
shoreline_clip_poly = gpd.read_file(r'wisconsin_shoreline_clip.shp')
shoreline_clip_poly = shoreline_clip_poly.to_crs('EPSG:32616')
shoreline = gpd.read_file(Path(homestr+r'.\LakeSuperior_BluffDelin\great_lakes_hardened_shorelines_cliff_delin_edit.shp'))
shoreline = shoreline.to_crs('EPSG:32616')
shoreline = gpd.clip(shoreline.geometry, shoreline_clip_poly.geometry)
tiles2012 = tiles2012.to_crs('EPSG:32616')
for count in range(0, len(tiles2012)-1):
    print('Working on tile ',count,' of',len(tiles2012),' .')
    # Debugging
    # count = 44
    # Define parameters
    tile2012_sing = tiles2012.iloc[count]  # pull out a single 2009 tile
    bounds = tile2012_sing.geometry.bounds
    shoreline_clip = gpd.clip(shoreline.geometry, tile2012_sing.geometry)
    if len(shoreline_clip)>0:
        # if len(shoreline_sig)>1:
        #     lines = [i for i in shoreline_sing.geometry]
        #     multi_line = MultiLineString(lines)
        #     shoreline_sing_mod = shapely.ops.linemerge(multi_line)
        tile_name = os.path.splitext(tile2012_sing.Name)[0]
        xsec_spacing = 5
        poslength = 150.0
        neglength = 60.0
        spacing = xsec_spacing
        simp_tolerance = 20
        step = 1.0
        crs = 32616
        
        merge_tiles = tiles2012[shapely.intersects(buffer(tile2012_sing.geometry, 200), tiles2012.geometry).values]
        merge_tiles = [os.path.join(dem_fill_2012_root, (os.path.splitext(i)[0]+'.tif')) for i in merge_tiles['Name']]
        merge_tiles = list(set(dem_paths_2012).intersection(set(merge_tiles)))
        merged_dem = temp_merge(merge_tiles)
        gen_xsec_wrap(shoreline_clip, outdir, tile_name, poslength, neglength, xsec_spacing, simp_tolerance, step, merged_dem, crs=crs)
        os.chdir(os.path.join(outdir,'Transects'))
        search_pattern=str(tile_name+'*.shp')
        transect_files = (glob.glob(search_pattern))
        transects = pd.DataFrame()
        for file in transect_files:
            transect = gpd.read_file(file)
            transects = pd.concat([transects,transect]) 
        # transect_file = os.path.join(outdir,'Transects',(tile_name+'_transects.shp'))
        xsec_points = make_xsec_points(transects,merged_dem)
        xsec_points.to_csv(os.path.join(outdir,'delineation_points_text',(tile_name+'_points.txt')),index=False)
        dem_merge = None
# %% Generate 2020 transect data
outdir = r'D:\CJR\lake_michigan_bluff_delineation\2020'
os.chdir(outdir)
os.chdir(r'..')
shoreline_clip_poly = gpd.read_file(r'wisconsin_shoreline_clip.shp')
shoreline_clip_poly = shoreline_clip_poly.to_crs('EPSG:32616')
shoreline = gpd.read_file(Path(homestr+r'.\LakeSuperior_BluffDelin\great_lakes_hardened_shorelines_cliff_delin_edit.shp'))
shoreline = shoreline.to_crs('EPSG:32616')
shoreline = gpd.clip(shoreline.geometry, shoreline_clip_poly.geometry)
tiles2012 = tiles2012.to_crs('EPSG:32616')
tiles2020 = tiles2020.to_crs('EPSG:32616')
for count in range(0, len(tiles2012)-1):
    # Debugging
    # count = 44
    tile2012_sing = tiles2012.iloc[count]  # pull out a single 2009 tile
    bounds = tile2012_sing.geometry.bounds
    shoreline_clip = gpd.clip(shoreline.geometry, tile2012_sing.geometry)
    if len(shoreline_clip)>0:
        tile_name = os.path.splitext(tile2012_sing.Name)[0]
        merge_tiles = tiles2012[shapely.intersects(buffer(tile2012_sing.geometry, 200), tiles2012.geometry).values]
        merge_tiles = [str(os.path.splitext(i)[0][14:]+'.tif') for i in merge_tiles['Name']]
        merge_tile_paths = []
        for filepath in dem_paths_2020:
            if any(merge_tile in filepath for merge_tile in merge_tiles):
                merge_tile_paths.append(filepath)
        merged_dem = temp_merge(merge_tile_paths)
        os.chdir(r'D:\CJR\lake_michigan_bluff_delineation\2012\Transects')
        search_pattern=str(tile_name+'*.shp')
        transect_files = (glob.glob(search_pattern))
        transects = pd.DataFrame()
        for file in transect_files:
            transect = gpd.read_file(file)
            transects = pd.concat([transects,transect]) 
        # transect_file = os.path.join(outdir,'Transects',(tile_name+'_transects.shp'))
        xsec_points = make_xsec_points(transects,merged_dem)
        xsec_points.to_csv(os.path.join(outdir,'delineation_points_text',('2020' + tile_name[14:] + '_points.txt')),index=False)
        dem_merge = None

# %% Create merged transects

os.chdir(r'D:\CJR\lake_michigan_bluff_delineation\2012\Transects')
all_transects = pd.DataFrame()
for number, fileName in enumerate(glob.glob('*.shp')):
    transects = gpd.read_file(fileName)
    all_transects = pd.concat([all_transects, transects])
os.chdir(os.path.abspath(os.path.join(os.getcwd(), r'..\..')))
all_transects.to_file(f'lake_michigan_transects.shp')
# %% Generate full 2012 DEMs (not just 2020 overlap tiles)

laz2012_root = r'D:\CJR\JABLTX_2012\LAZ'
dem2012_root_buffer = r'D:\CJR\JABLTX_2012\DEM\buffer_full'
json_pipeline = homestr+r'\PDAL\lake_mi_jalbtx_2012_pipeline_delaunay.json'
json_pipeline_mod = homestr+r'\PDAL\lake_mi_jalbtx_2012_pipeline_delaunay_mod.json'

for count in range(0,len(tiles2012)):
    rasterize_points_2012(count, tiles2012, laz2012_root, dem2012_root_buffer, json_pipeline, json_pipeline_mod)

# CLean up directory
dem2012_root_buffer_olap = r'D:\CJR\JABLTX_2012\DEM\buffer'
[dem_paths_2012, dem_filenames_2012] = read_paths(dem2012_root_buffer,'.tif')
[dem_paths_2012_olap, dem_filenames_2012_olap] = read_paths(dem2012_root_buffer_olap,'.tif')

redundant_dems = [i for i in dem_filenames_2012 if i in dem_filenames_2012_olap]
os.chdir(dem2012_root_buffer)
for i in redundant_dems:
    Path.unlink(i)
    
# Clip down to tile boundaries
dem2012_root = r'D:\CJR\JABLTX_2012\DEM'
[dem_paths_2009, dem_filenames_2009] = read_paths(dem2012_root_buffer,'.tif')
for count, value in enumerate(dem_paths_2009):
    # Debugging 
    # count = 0
    # value = dem_paths_2009[count]
    
    substring = dem_filenames_2009[count][:-4]
    tile2009_sing = [tiles2012[tiles2012['Name']==i] for i in tiles2012['Name'] if i[:-4] in substring][0]
    clip_geom = gpd.GeoDataFrame(geometry=gpd.GeoSeries(tile2009_sing.geometry),crs='EPSG:32616')
    [clip2009, errorState] = raster_clip(value, clip_geom, clip_geom.crs)
    if clip2009==[]:
        print("No overlapping data ",count)
    else:
        filename_tif = os.path.join(dem2012_root, 'clip_full', dem_filenames_2009[count])
        DEM = read_file(clip2009).read(1,masked=True)
        DEM.shape = [1,np.shape(DEM)[0],np.shape(DEM)[1]]
        with rasterio.open(filename_tif, "w", **read_file(clip2009).meta) as dest:
            dest.write(DEM)
    clip2009 = None
    dest = None
    filename_tif = None
    DEM = None
    
# Interpolate missing data
dem_fill_2012_root = r'D:\CJR\JABLTX_2012\DEMfill_full'
[dem_paths_2012,dem_filenames_2012] = read_paths(r'D:\CJR\JABLTX_2012\DEM\clip_full','.tif')

os.chdir(dem_fill_2012_root)
for count,value in enumerate(dem_paths_2012):
    out_name = os.path.join(dem_fill_2012_root,dem_filenames_2012[count])
    wbt.fill_missing_data(value,out_name,filter=31,weight=5.0,no_edges=False)
# %% Generate full 2020 DEMs (not just 2012 overlap tiles)

laz2020_root = r'D:\CJR\JABLTX_2020\LAZ'
dem2020_root_buffer = r'D:\CJR\JABLTX_2020\DEM\buffer_full'
json_pipeline = homestr+r'\PDAL\lake_mi_jalbtx_2020_pipeline_delaunay.json'
json_pipeline_mod = homestr+r'\PDAL\lake_mi_jalbtx_2020_pipeline_delaunay_mod.json'

# for count in range(0,len(tiles2012)):
#     rasterize_points_2012(count, tiles2012, laz2012_root, dem2012_root_buffer, json_pipeline, json_pipeline_mod)
    
# %% Generate transects using 2012 data for nearshore analysis

dem_fill_2012_root_olap = r'D:\CJR\JABLTX_2012\DEMfill'
[dem_paths_2012_olap, dem_filenames_2012_olap] = read_paths(dem_fill_2012_root_olap,'.tif')
dem_fill_2012_root = r'D:\CJR\JABLTX_2012\DEMfill_full'
[dem_paths_2012, dem_filenames_2012] = read_paths(dem_fill_2012_root,'.tif')

outdir = r"D:\CJR\lake_michigan_bluff_delineation\2012_nearshore"
os.chdir(outdir)
os.chdir(r'..')
# shoreline_clip_poly = gpd.read_file(r'wisconsin_shoreline_clip.shp')
# shoreline_clip_poly = shoreline_clip_poly.to_crs('EPSG:32616')
shoreline = gpd.read_file(r'D:\CJR\lake_michigan_bluff_delineation\great_lakes_hardened_shorelines_lake_mi_nearshore_edit.shp')
shoreline = shoreline.to_crs('EPSG:32616')
# shoreline = gpd.clip(shoreline.geometry, shoreline_clip_poly.geometry)
tiles2012 = tiles2012.to_crs('EPSG:32616')

for count in range(0, len(tiles2012)):
    print('Working on tile ',count,' of',len(tiles2012),' .')
    # Debugging
    if (count == 167):
        # count=0
            # Define parameters
        tile2012_sing = tiles2012.iloc[[count]]  # pull out a single 2009 tile
        bounds = tile2012_sing.geometry.bounds
        
        # Erase overlapping portions of previous tiles to avoid duplicate transects
        tile2012_prior = tiles2012.iloc[0:count-1]
        if count>0:
            tile2012_sing_clean = tile2012_sing.overlay(tile2012_prior, how='difference')  
        else:
            tile2012_sing_clean = tile2012_sing
        bounds = tile2012_sing_clean.geometry.bounds
    
        
        shoreline_clip = gpd.clip(shoreline.geometry, tile2012_sing_clean.geometry)
    
        if len(shoreline_clip)>0:
            # if len(shoreline_sig)>1:
            #     lines = [i for i in shoreline_sing.geometry]
            #     multi_line = MultiLineString(lines)
            #     shoreline_sing_mod = shapely.ops.linemerge(multi_line)
            tile_name = os.path.splitext(tile2012_sing.Name.iloc[0])[0]
            xsec_spacing = 5
            poslength = 250.0
            neglength = 500.0
            spacing = xsec_spacing
            simp_tolerance = 50
            step = 1.0
            crs = 32616
            tile2012_sing_buffer = shapely.buffer(tile2012_sing.geometry, 500)
            tile2012_sing_buffer = gpd.GeoDataFrame(geometry=tile2012_sing_buffer, crs='EPSG:32616')
            merge_tiles = gpd.sjoin(tiles2012, tile2012_sing_buffer)
            merge_tiles_paths = [os.path.join(dem_fill_2012_root, (os.path.splitext(i)[0]+'.tif')) for i in merge_tiles['Name']]
            merge_tiles_paths_olap = [os.path.join(dem_fill_2012_root_olap, (os.path.splitext(i)[0]+'.tif')) for i in merge_tiles['Name']]
            merge_tiles = list(set(dem_paths_2012).intersection(set(merge_tiles_paths)))
            merge_tiles_olap = list(set(dem_paths_2012_olap).intersection(set(merge_tiles_paths_olap)))
            merge_tiles.extend(merge_tiles_olap)
            merged_dem = temp_merge(merge_tiles)
            gen_xsec_wrap(shoreline_clip, outdir, tile_name, poslength, neglength, xsec_spacing, simp_tolerance, step, merged_dem, crs=crs)
            os.chdir(os.path.join(outdir,'Transects'))
            search_pattern=str(tile_name+'*.shp')
            transect_files = (glob.glob(search_pattern))
            transects = pd.DataFrame()
            for file in transect_files:
                transect = gpd.read_file(file)
                transects = pd.concat([transects,transect]) 
            # transect_file = os.path.join(outdir,'Transects',(tile_name+'_transects.shp'))
            xsec_points = make_xsec_points(transects,merged_dem)
            xsec_points.to_csv(os.path.join(outdir,'delineation_points_text',(tile_name+'_points.txt')),index=False)
            dem_merge = None

# %% Fix transect ID's

os.chdir(os.path.join(outdir,'Transects'))
transect_filenames = glob.glob('*.shp')
for count in range(0, len(tiles2012)):
    # count=167
    tile2012_sing = tiles2012.iloc[[count]]  # pull out a single 2009 tile
    tile_name = os.path.splitext(tile2012_sing.Name.iloc[0])[0]
    search_pattern=str(tile_name+'*.shp')
    transect_files = (glob.glob(search_pattern))
    if len(transect_files) > 0:
        transect_numbers = [i[-9:-5] for i in transect_files]
        transect_numbers = [int((re.findall(r'\d+', i))[0]) for i in transect_numbers]
        transect_df = pd.DataFrame(columns=['Filenames', 'transect_group'])
        transect_df['Filenames'] = transect_files
        transect_df['transect_group'] = transect_numbers
        transect_df = transect_df.sort_values('transect_group')
        transect_df = transect_df.reset_index()
        transect_df = transect_df.drop('index', axis = 1)
        all_transects = pd.DataFrame()
        for count in range(0, len(transect_df)):
            transects = gpd.read_file(transect_df['Filenames'][count])
            all_transects = pd.concat([all_transects, transects])
        all_transects = all_transects.reset_index()    
        all_transects = all_transects.drop('index', axis = 1)
        all_transects['FID'] = all_transects.index
        all_transects['tile_name'] = tile_name
        all_transects.to_file(os.path.join(outdir,'Transects_mod',(tile_name+'_transects.shp')))


        
            


# %% Merge 2012 nearshore transects
os.chdir(os.path.join(outdir,'Transects'))
all_transects = pd.DataFrame()
for number, fileName in enumerate(glob.glob('*.shp')):
    transects = gpd.read_file(fileName)
    all_transects = pd.concat([all_transects, transects])
os.chdir(os.path.abspath(os.path.join(os.getcwd(), r'..')))
all_transects.to_file(f'lake_michigan_2012_nearshore_transects.shp')