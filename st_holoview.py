import fiona
import numpy as np
import os
import rioxarray as rio
import xarray as xr
import holoviews as hv
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st


# helpful for holoviews
# https://towardsdatascience.com/displaying-a-gridded-dataset-on-a-web-based-map-ad6bbe90247f

cdict = {'red': [(0.0, 0.0078, 0.0078),
                 (0.06249, 0.0078, 0.0078),
                 (0.0625, 0.0, 0.0),
                 (0.34375, 1.0, 1.0),
                 (0.46875, 1.0, 1.0),
                 (0.8125, 1.0, 1.0),
                 (0.81251, 0.85, 0.85),
                 (1.0, 0.85, 0.85)],
        'green': [(0.0, 0.0078, 0.0078),
                 (0.06249, 0.0078, 0.0078),
                 (0.0625, 0.58, 0.58),
                 (0.34375, 1.0, 1.0),
                 (0.46875, 0.0, 0.0),
                 (0.8125, 0.0, 0.0),
                 (0.81251, 0.85, 0.85),
                 (1.0, 0.85, 0.85)],
        'blue': [(0.0, 1.0, 1.0),
                 (0.06249, 1.0, 1.0),
                 (0.0625, 0.0, 0.0),
                 (0.34375, 0.0, 0.0),
                 (0.46875, 0.0, 0.0),
                 (0.8125, 0.0, 0.0),
                 (0.81251, 0.85, 0.85),
                 (1.0, 0.85, 0.85)]
        }

@st.cache(allow_output_mutation=True)
def get_polygons():
    with fiona.open('data/cb_2018_nation.geojson', 'r') as in_file:
        nation_shps = [feature['geometry'] for feature in in_file]

    with fiona.open('data/cb_2018_states.geojson', 'r') as in_file:
        state_shps = [feature['geometry'] for feature in in_file]

    with fiona.open('data/protrac.geojson', 'r') as in_file:
        protrac_shps = [feature['geometry'] for feature in in_file]

    final_xs = []
    final_ys = []
    polygons_to_draw = []

    for shape in nation_shps:
        for multipolygon in shape['coordinates']:
            for polygon in multipolygon:
                    draw_data = {'x': [point[0] for point in polygon if isinstance(point, tuple)],
                                    'y': [point[1] for point in polygon if isinstance(point, tuple)],
                                    'value': 0}
                    polygons_to_draw.append(draw_data)

    for shape in state_shps:
        for polygon in shape['coordinates']:
            if isinstance(polygon[0], tuple):
                draw_data = {'x': [point[0] for point in polygon],
                                'y': [point[1] for point in polygon],
                                'value': 0}
                polygons_to_draw.append(draw_data)
            else:
                for points in polygon:
                    draw_data = {'x': [point[0] for point in points],
                                    'y': [point[1] for point in points],
                                    'value': 0}
                    polygons_to_draw.append(draw_data)

    for shape in protrac_shps:
        for polygon in shape['coordinates']:
            if isinstance(polygon[0], tuple):
                draw_data = {'x': [point[0] for point in polygon],
                                'y': [point[1] for point in polygon],
                                'value': 0}
                polygons_to_draw.append(draw_data)
            else:
                for points in polygon:
                    draw_data = {'x': [point[0] for point in points],
                                    'y': [point[1] for point in points],
                                    'value': 0}
                    polygons_to_draw.append(draw_data)

    hv_polygon = hv.Path(polygons_to_draw, vdims='value')

    return hv_polygon



# def convert_coords(xc, yc):

    
#     xy1 = [(x, yc[0]) for x in xc]
#     xy2 = [(xc[0], y) for y in yc]
    
#     inProj = Proj(init='epsg:2847')
#     outProj = Proj(init='epsg:4326')
#     x1,y1 = -11705274.6374,4826473.6922
#     # x2,y2 = transform(inProj,outProj,x1,y1)
#     _xy1 = [transform(inProj,outProj,u[0],u[1]) for u in xy1]
#     _xy2 = [transform(inProj,outProj,u[0],u[1]) for u in xy2]

#     new_xc = 100000 * np.array([u[0] for u in _xy1])
#     new_yc = 100000 * np.array([u[1] for u in _xy2])
    
#     return new_xc, new_yc




def create_xarray(data, xmin, ymax, nx, ny, dx, dy):
    # Warning ! Load coordinates in epsg 3857 since this is the projection system of holoviews (Tricky !)
    xcoords = np.load('data/xcoords.npy')
    ycoords = np.load('data/ycoords.npy')


    # xcoords = np.array([xmin + dx/2 + i*dx for i in range(nx)])
    # ycoords = np.array([ymax - dy/2 - i*dy for i in range(ny)])
    # new_xc, new_yc = convert_coords(xcoords, ycoords)
    # print(new_xc, new_yc)
    da = xr.DataArray(data, coords=[ycoords, xcoords],dims=['y', 'x'])
    da.rio.write_nodata(-9999, inplace=True)
    # da.rio.write_crs(4326, inplace=True)
    # ds_proj = da.rio.reproject('epsg:4326')
    # da = da.sortby(["y", "x"])
    print(da.rio.crs)
    # da.assign_coords(x=ycoords, y=xcoords)
    return da

def create_hv_plot(da, well_display, property, contours_display):
    rio_data = da
    in_data = rio_data.where(rio_data != rio_data.rio.nodata)
     
    image_height, image_width = 1200, 1200
    map_height, map_width = image_height, 2000
    left, bottom, right, top = in_data.rio.bounds()
    
    key_dims = ['x', 'y']
    value_dimension = 'value'

    colormap_to_use = 'Plasma'
    clipping = {'NaN': '#00000000'}
    
    if property == 'Standard Thermal Stress':
        colormap_to_use = LinearSegmentedColormap('sts', segmentdata=cdict, N=256)
        clipping = {'NaN': '#00000000', 'min': 'blue', 'max': 'gray'}
    elif property == "Temperature":
        colormap_to_use = 'Plasma'
        clipping = {'NaN': '#00000000'}
    elif property == "Depth":
        colormap_to_use = 'RdBu'
        clipping = {'NaN': '#00000000'}
        
    
    
    hv.extension('bokeh', logo=False)
    hv.opts.defaults(hv.opts.Image(cmap=colormap_to_use,
                        height=image_height,
                        width=image_width,
                        colorbar=True,
                        tools=['hover'], active_tools=['wheel_zoom'],
                        clipping_colors=clipping),
        hv.opts.Tiles(
            # active_tools=['wheel_zoom'],
            height=map_height, width=map_width)
    )
    
    hv_dataset = hv.Dataset(in_data, vdims=value_dimension, kdims=key_dims)
    

    hv_image = hv.Image(hv_dataset)
    if property == 'Standard Thermal Stress':
        hv_redim = hv_image.redim.range(value=(90, 250), x=(left, right), y=(bottom, top))
    else:
        hv_redim = hv_image.redim.range(value=(np.min(da), np.max(da)), x=(left, right), y=(bottom, top))

    if well_display:
        coords = np.load('data/well_coords.npy', allow_pickle=True)

    hv_polygon = get_polygons()
        
    hv_tiles_osm = hv.element.tiles.CartoLight()
    if contours_display and property == 'Standard Thermal Stress':
        contours = hv.operation.contours(hv_redim, levels= np.arange(100, 200 , 10))
    elif contours_display and property == 'Temperature':
        contours = hv.operation.contours(hv_redim, levels= np.arange(np.min(da), np.max(da) , 20))
    elif contours_display and property == 'Depth':
        contours = hv.operation.contours(hv_redim, levels= np.arange(np.min(da), np.max(da) , 500))
    else:
        contours = hv.operation.contours(hv_redim, levels= 10)

    if not contours_display:
        contours = hv.operation.contours(hv_redim, levels=0)
    if well_display:
        hv_points = hv.Points({'x': [point[0] for point in coords], 'y': [point[1] for point in coords]})
        hv_combo = hv_tiles_osm * hv_redim.opts(clipping_colors=clipping) * hv_polygon.opts(line_color='black') * hv_points.opts(color='black', size=1) * contours
    else:
        hv_combo = hv_tiles_osm * hv_redim.opts(clipping_colors=clipping) * hv_polygon.opts(line_color='black') * contours

    return hv_combo
