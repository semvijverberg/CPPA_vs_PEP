#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:18:13 2018

@author: semvijverberg
"""

#import plotting
import func_mcK
import matplotlib.pyplot as plt


def clustering_temporal(data, ex, n_clusters, tfreq, region):
    import sklearn.cluster as cluster
    import xarray as xr
    import numpy as np
    import os
    input = data.squeeze()
#    input = varhotdays
#    n_clusters = 4
#    tfreq = 'daily'
#    methods = ['KMeans', 'AgglomerativeClustering', 'hierarchical']
#    linkage = ['complete', 'average']
#    ex['distmetric'] = 'jaccard'
#    ex['clusmethod'] = methods[1] ; ex['linkage'] = linkage
    

    def clustering_plotting(cluster_method, input):    

        def time_station_shape(array, timesteps):
            X_vec = np.reshape( np.nan_to_num(array.values), (timesteps, array.longitude.size*array.latitude.size) )
            return X_vec

#        mask_time_station = time_station_shape(region_values.mask, 1)
#        mask_time_station = np.array(np.tile(mask_time_station, (data.time.size,1)), dtype=int)
#        output_lonlat = np.array(mask_time_station[0,:].copy(), dtype=int)
        # input shape for clustering
        X_time_station = time_station_shape(input, input.time.size)
        # only keep the land gridcells
#        X_st_land = X_time_station[mask_time_station == 1]
#        X_st_land = X_st_land.reshape( (data.time.size, n_land_stations )  )

#        out_clus = cluster_method.fit(X_st_land)
#        labels = out_clus.labels_

        out_clus = cluster_method.fit(X_time_station)
        labels = out_clus.labels_       
        output = input.copy()
        labels_clusters = xr.DataArray(labels, [input.coords['time'][:]], name='time')
        labels_dates = xr.DataArray(input.time.values, [input.coords['time'][:]], name='time')
        output['cluster'] = labels_clusters
        output['time_date'] = labels_dates
        output = output.set_index(time=['cluster','time_date'])
        output.name = '{}_tf{}_{}'.format(data.name, tfreq, ex['clusmethod'][:4])
        Nocluster = {}
        group_clusters = output.groupby('cluster').mean(dim='time', keep_attrs=True)
#        functions.quicksave_ncdf(output, cls, path=folder, name=output.name)

#        group_clusters = group_clusters.drop('mask')
        labels = []
        for n in range(0,n_clusters):
            folder = os.path.join(ex['fig_path'],'Clustering_temporal/',
                              '{}deg_tfreq{}'.format(ex['grid_res'],tfreq),
                              name_method +'_'+ region)
            if os.path.isdir(folder):
                pass
            else:
                os.makedirs(folder)
            perc_of_cluster = str(100*float(output.sel(cluster=n).time_date.size)/float(input.time.size))[:2]+'%'
            Nocluster['cluster_{}'.format(n)] = perc_of_cluster
            cluster_n = group_clusters.sel(cluster=n)
            cluster_n.name = '{}_cl_{}of{}_{}_tf{}_{}'.format(data.name, n, n_clusters, 
                              perc_of_cluster, tfreq, name_method)
            labels.append('cluster {}, {}'.format(n, perc_of_cluster))
#            func_mcK.xarray_plot(cluster_n, path=folder, saving=True)
        # updating name (name of saved figure)
        group_clusters.name = '{}_{}_clusters_tf{}_{}'.format(data.name, n_clusters, 
                              tfreq, name_method)
        # updating labels (name of title in figure)
        group_clusters['cluster'] = xr.DataArray(labels, dims='cluster')
        group_clusters.attrs['units'] = 'Kelvin'
        # plotting
        file_name = os.path.join(folder, group_clusters.name)
        title = group_clusters.name + ' - T95 McKinnon data - ERA-I SST'
        kwrgs = dict( {'vmin' : -3*group_clusters.std().values, 'vmax' : 3*group_clusters.std().values, 'title' : title, 'clevels' : 'notdefault',
                       'map_proj' : ex['map_proj'], 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
        func_mcK.finalfigure(group_clusters, file_name, kwrgs) 

        return output, group_clusters
    
    algorithm = cluster.__dict__[ex['clusmethod']]
    if ex['clusmethod'] == 'KMeans':
        cluster_method = algorithm(n_clusters)
        name_method = ex['clusmethod']
        output = clustering_plotting(cluster_method, input)
    if ex['clusmethod'] == 'AgglomerativeClustering':
        # linkage_method -- 'average', 'centroid', 'complete', 'median', 'single',
        #                   'ward', or 'weighted'. See 'doc linkage' for more info.
        #                   'average' is standard.
        if np.size(ex['linkage']) == 1:
            link = ex['linkage']
            name_method = 'Aggl' + '_' + link[:3]
            print(name_method)
            cluster_method = algorithm(linkage=link, n_clusters=n_clusters)
            output, group_clusters = clustering_plotting(cluster_method, input)
        else:
            for link in ex['linkage']:
                name_method = 'Aggl' + '_' + link[:3]
                print(name_method)
                cluster_method = algorithm(linkage=link, n_clusters=n_clusters)
                output, group_clusters = clustering_plotting(cluster_method, input)
#    folder = os.path.join(cls.base_path, 'Clustering_temporal', method)
#    quicksave_ncdf(input, cls, path=folder, name=input.name)
    output.attrs['units'] = 'clusters, n = {}'.format(n_clusters)
    return output, group_clusters


#def clustering_spatial(data, ex, n_clusters, region):
#    import numpy as np
#    import xarray as xr
#    import os
#    import sklearn.cluster as cluster
#    input = data.squeeze()
#    def clustering_plotting(cluster_method, input, n_clusters):
#        region_values = input.where(input.mask == True)
#        output = region_values[0].copy()
#        def station_time_shape(array, timesteps):
#            X_vec = np.reshape( array.values, (timesteps, array.longitude.size*array.latitude.size) )
#            X_station_time = np.swapaxes(X_vec, 1,0 )
#            return X_station_time
#        # reshape array to [station, time] (to create dendogram timeseries)
#        
#        mask_station_time = station_time_shape(region_values.mask, 1)
#        mask_station_time = np.array(np.tile(mask_station_time, (1,data.time.size)), dtype=int)
#        output_lonlat = np.array(mask_station_time[:,0].copy(), dtype=int)
#        indic_station_time = station_time_shape(region_values, region_values.time.size)
#        # only keep the land gridcells
#        indic_st_land = indic_station_time[mask_station_time == 1]
#        n_land_stations = output_lonlat[output_lonlat==1].size
#        indic_st_land = indic_st_land.reshape( (n_land_stations, data.time.size)  )
#
##        X = StandardScaler().fit_transform(X_station_time)
#
#        
#        out_clus = cluster_method.fit(indic_st_land)
#        labels = out_clus.labels_
#        # plug in the land gridcell with the labels
#        land_gcs = np.argwhere(output_lonlat == 1)[:,0]
#        output_lonlat[land_gcs] = labels
#        output_lonlat[land_gcs] = output_lonlat[land_gcs] + 1
#        output_lonlat = np.reshape(np.array(output_lonlat), region_values[0].shape)
#        output.values = output_lonlat
#        output.name = ex['name'] + '_' + str(n_clusters) + '_' + name_method
#        
#        folder = os.path.join(ex['fig_path'],'Clustering_spatial/', 
#                              '{}deg_tfreq{}'.format(ex['grid_res'],ex['tfreq']))
#        if os.path.isdir(folder) != True : os.makedirs(folder)
##        savepath = os.path.join(folder, output.name)
#        xarray_mask_plot(output, path=folder, saving=True)
#        return output   
#    
#    algorithm = cluster.__dict__[ex['clusmethod']]
#    if ex['clusmethod'] == 'KMeans':
#        cluster_method = algorithm(n_clusters)
#        name_method = ex['clusmethod']
#        output = clustering_plotting(cluster_method, input, n_clusters)
#    if ex['clusmethod'] == 'AgglomerativeClustering':
#        # linkage_method -- 'average', 'centroid', 'complete', 'median', 'single',
#        #                   'ward', or 'weighted'. See 'doc linkage' for more info.
#        #                   'average' is standard.
#        if np.size(ex['linkage']) == 1:
#            link = ex['linkage']
#            name_method = 'AgglomerativeClustering' + '_' + link
#            print(name_method)
#            cluster_method = algorithm(linkage=link, n_clusters=n_clusters, 
#                                           affinity=ex['distmetric'])
#            output = clustering_plotting(cluster_method, input, n_clusters)
#        else:
#            for link in ex['linkage']:
#                name_method = 'AgglomerativeClustering' + '_' + link
#                print(name_method)
#                cluster_method = algorithm(linkage=link, n_clusters=n_clusters, 
#                                           affinity=ex['distmetric'])
#                output = clustering_plotting(cluster_method, input, n_clusters)
#                
##    folder = os.path.join(cls.base_path, 'Clustering_spatial', method)
##    quicksave_ncdf(input, cls, path=folder, name=input.name)
#    output.attrs['units'] = 'clusters, n = {}'.format(n_clusters) 
#    return output
#
#def save_figure(data, path):
#    import os
#    import matplotlib.pyplot as plt
##    if 'path' in locals():
##        pass
##    else:
##        path = '/Users/semvijverberg/Downloads'
#    if path == 'default':
#        path = '/Users/semvijverberg/Downloads'
#    else:
#        path = path
#    import datetime
#    today = datetime.datetime.today().strftime("%d-%m-%y_%H'%M")
#    if type(data.name) is not type(None):
#        name = data.name.replace(' ', '_')
#    if 'name' in locals():
#        print('input name is: {}'.format(name))
#        name = name + '.jpeg'
#        pass
#    else:
#        name = 'fig_' + today + '.jpeg'
#    print(('{} to path {}'.format(name, path)))
#    plt.savefig(os.path.join(path,name), format='jpeg', dpi=300, bbox_inches='tight')
#
#
#
#def xarray_mask_plot(data, path='default', saving=False):
#    # from plotting import save_figure
#    import matplotlib.pyplot as plt
#    import cartopy.crs as ccrs
#    import numpy as np
#    plt.figure(figsize=(10,6))
#    input = data.where(data.mask==True, drop=True)
#    input.name = data.name
##    if len(input.longitude[np.where(input.longitude > 180)[0]]) != 0:
##        input = convert_longitude(input)
##    else:
##        pass
#    if input.ndim != 2:
#        print("number of dimension is {}, printing first element of first dimension".format(np.squeeze(input).ndim))
#        input = input[0]
#    else:
#        pass
##    proj = ccrs.Orthographic(central_longitude=input.longitude.mean().values, 
##                             central_latitude=input.latitude.mean().values)
#    proj = ccrs.Miller(central_longitude=240)
#    ax = plt.axes(projection=proj)
#    ax.coastlines()
##    colors = plt.cm.rainbow(np.linspace(0,1,data.max()))
#    cmap = plt.cm.tab10
#    levels = np.arange(0.5, input.max()+1.5)
#    clevels = np.arange(input.min(), input.max()+1E-9)
#    input.values = np.nan_to_num(input.values)
##    clevels = np.insert(clevels, 0, 0)
#    plotting = input.where(input.mask==True)
#    plotting = input
#    plotting.plot.pcolormesh(ax=ax, levels=levels,
#                             transform=ccrs.PlateCarree(), cmap=cmap,
#                             add_colorbar=True, cbar_kwargs={'ticks':clevels})
#    if saving == True:
#        save_figure(input, path=path)
#    plt.show()

#def convert_longitude(data):
#    import numpy as np
#    import xarray as xr
#    lon_above = data.longitude[np.where(data.longitude > 180)[0]]
#    lon_normal = data.longitude[np.where(data.longitude <= 180)[0]]
#    # roll all values to the right for len(lon_above amount of steps)
#    data = data.roll(longitude=len(lon_above))
#    # adapt longitude values above 180 to negative values
#    substract = lambda x, y: (x - y)
#    lon_above = xr.apply_ufunc(substract, lon_above, 360)
#    convert_lon = xr.concat([lon_above, lon_normal], dim='longitude')
#    data['longitude'] = convert_lon
#    return data

from matplotlib.colors import Normalize
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        import numpy as np
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def PlateCarree_timesteps(data, path='default', type='abs', cbar_mode='compare', region='U.S.', saving=False):
#    path='default'; type='abs'; cbar_mode='compare'; region='U.S.'; saving=False
    #%%
    import numpy as np
    import cartopy.crs as ccrs
    import cartopy.feature as cfeat
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import matplotlib.ticker as mticker
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec  
    if len(data['time']) <= 4:
        input = data
        pass
    else:
        "select less time steps to plot"
        input = data.isel(time=[0,1,2,3])
    fig = plt.figure()
    for i in input['time'].values:
        print(i)
        rows = 2 if len(input['time']) / 2. > 1 else 1
        columns = int(len(input['time'])/rows + len(input['time']) % float(rows))
        gs = gridspec.GridSpec(rows, columns)
        fig_grid = np.concatenate([input['time'].values, np.array([np.datetime64('1900-01-01')])]) if len(input['time']) % float(rows) != 0 else input['time'].values
        r,c = np.where(fig_grid.reshape((gs._nrows,gs._ncols)) == i)[0:2]
        proj = ccrs.PlateCarree()
        # proj = ccrs.Mollweide()
        ax = fig.add_subplot(gs[int(r),int(c)], projection=proj)
        ax.add_feature(cfeat.COASTLINE)
        # ax.background_img(name='ne_shaded')
        # ax.add_feature(cfeat.LAND); # ax.add_feature(cfeat.OCEAN) # ax.add_feature(cfeat.LAKES, alpha=0.5)
        ax.add_feature(cfeat.BORDERS, linestyle=':')
        if region == 'global':
            values_region = input
        elif region != 'global':
            values_region, region_coords = func_mcK.find_region(input, region=region)    
            ax.set_xlim(region_coords[0], region_coords[1])  # west lon, east_lon
            ax.set_ylim(region_coords[2], region_coords[3])  # south_lat, north_lat           
        if cbar_mode == 'compare':
            pass
        elif cbar_mode == 'individual':
            print(cbar_mode)
            values_region = values_region.sel(time=i)
        if type=='norm':
            std_region = np.std(values_region).values
            min_region = np.min(values_region/std_region).values ; max_region = np.max(values_region/std_region).values
            plottable = np.squeeze(input.sel(time=i)/std_region)
            unit = r"$\sigma$ "+"= {:}".format(round(std_region,1))
        elif type == 'abs':
            min_region = np.min(values_region) ; max_region = np.max(values_region)
            plottable = np.squeeze(input.sel(time=i))
            unit = plottable.attrs['units']
        if plottable['longitude'][0] == 0. and plottable['longitude'][-1] - 360 < 5.:
#            plottable = extend_longitude(plottable)
            plottable = func_mcK.convert_longitude(plottable)
        #Check if anomaly (around 0) field
        if abs(plottable.mean())/( 4 * plottable.std() ) < 2:
            norm = MidpointNormalize(midpoint=0, vmin=min_region, vmax=max_region)
        lons, lats = np.meshgrid(plottable['longitude'].values, plottable['latitude'].values)
        # norm = colors.BoundaryNorm(boundaries=np.linspace(round(min_region),round(max_region),11), ncolors=256)
        map = ax.pcolormesh(lons, lats, np.squeeze(plottable), transform=ccrs.PlateCarree(), norm=norm, cmap=plt.cm.RdBu_r)
        ax.set_title(np.str(i).split(':')[0]) ; 
        plt.colorbar(map, ax=ax, orientation='horizontal', 
                    use_gridspec=True, fraction = 0.1, pad=0.15, label=unit)
        # cb.set_label('label', rotation=0, position=(0.5, 0.5))
#        ax.tick_params(axis='both')
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        lat_steps = int(abs(region_coords[2] - region_coords[3])/3)
        yticks = np.arange(region_coords[2], region_coords[3]+5, lat_steps)
        gl.ylocator = mticker.FixedLocator(yticks)
        lon_steps = int(abs(region_coords[0] - region_coords[1])/3)
        xticks = np.arange(region_coords[0], region_coords[1]+10, lon_steps)
        gl.xlocator = mticker.FixedLocator(xticks)
        gl.xlabels_top = False ; gl.xformatter = LONGITUDE_FORMATTER
        gl.ylabels_right= False ; gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size':8} ; gl.ylabel_style = {'size':8}
        #%%
    if saving == True:
        save_figure(input, path=path)
    plt.show()