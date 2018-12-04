#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 09:47:10 2018

@author: semvijverberg
"""
#%%
import os
import numpy as np
import xarray as xr
import generate_varimax
import func_mcK
import matplotlib.pyplot as plt

#base_path = "/Users/semvijverberg/surfdrive/Data_ERAint/"
#path_pp  = os.path.join(base_path, 'input_pp/')
#
#load_filename = 'sst_1979-2017_2mar_31aug_dt-1days_2.5deg.nc'
#folder_name = path_pp
#varname = 'sst'
#
#geo_object = generate_varimax.load_data(load_filename, folder_name, varname)

def varimax_PCA_Sem(xarray, max_comps):
    geo_object = xarray
    lats = geo_object.latitude.values
    lons = geo_object.longitude.values
    geo_data = xarray.values
    flattened = np.reshape(np.array(geo_data), 
                           (geo_data.shape[0], np.prod(geo_data.shape[1:])))
    nonmask_flat = np.where(np.array(flattened)[0] != 0.)[0]
    nonmasked = flattened[:,nonmask_flat]
    
    
    # convert to nonmasked array
    data = nonmasked
    truncate_by = 'max_comps'
    max_comps=60
    fraction_explained_variance=0.9
    verbosity=0
    
    var_standard = generate_varimax.get_varimax_loadings_standard(data=data,
                        truncate_by = truncate_by, 
                        max_comps=max_comps,
                        fraction_explained_variance=fraction_explained_variance,
                        verbosity=verbosity,
                        )
    #%%
    # Plug in nonmasked values into the original lonlat with mask
    npcopy = np.array(flattened[:max_comps])
    npcopy[:,nonmask_flat] = np.swapaxes(var_standard['weights'],1,0)
    nplonlat = np.reshape(npcopy, (npcopy.shape[0], lats.size, lons.size)) 
                                                   
    
    # convert to xarray
    
    xrarray_output = xr.DataArray(nplonlat, coords=[np.arange(0,max_comps), lats, 
                                 lons], 
                                dims=['modes', 'latitude','longitude'], 
                                name='rot_pca_standard')
    return xrarray_output
                                         
xrarray = xrarray_output.sel(modes=slice(0,10))
modes = xrarray.modes.values
xrarray.attrs['units'] = 'Kelvin (absolute values)'
file_name = os.path.join(ex['fig_path'], 
             'Varimax standard - mode {}-{}.png'.format(modes[0], modes[-1]))
title = 'Varimax standard - mode'
kwrgs = dict( {'vmin' : -3*xrarray.std().values, 'vmax' : 3*xrarray.std().values, 'title' : title, 'clevels' : 'notdefault',
               'map_proj' : map_proj, 'cmap' : plt.cm.RdBu_r, 'column' : 2} )
func_mcK.finalfigure(xrarray, file_name, kwrgs) 
