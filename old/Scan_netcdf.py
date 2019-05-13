#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 09:39:10 2019

@author: semvijverberg
"""
import os, sys
os.chdir('/Users/semvijverberg/surfdrive/Scripts/Extracting_precursor/')
script_dir = os.getcwd()
if sys.version[:1] == '3':
    from importlib import reload as rel
import func_mcK
from ROC_score import ROC_score_wrapper
import numpy as np
import xarray as xr 
import pandas as pd
import cartopy.crs as ccrs
xrplot = func_mcK.xarray_plot
import matplotlib.pyplot as plt


# scan netdf
base_path = "/Users/semvijverberg/surfdrive/Data_ERAint/"
exp_folder = ''
path_raw = os.path.join(base_path, 'input_raw')
path_pp  = os.path.join(base_path, 'input_pp')


ex = dict({'path_pp' : path_pp,
           'filename' : 'sm2_1979-2017_2jan_31okt_dt-1days_2.5deg.nc'})
    
    
ex['region'] = 'Northern'
    
# full globe - full time series
varfullgl = func_mcK.import_array(ex['filename'], ex)
# select region
Prec_reg = func_mcK.find_region(varfullgl, region=ex['region'])[0]
dates = pd.to_datetime(Prec_reg.time.values)

#%%
ex['rollingmean'] = 1
if ex['rollingmean'] != 1:
    # Smoothen precursor time series by applying rolling mean
    Prec_reg = func_mcK.rolling_mean_time(Prec_reg, ex)
    
ex['RV_months']  = [6,7,8]
# only selecting RV_months 
RV_period = []
for mon in ex['RV_months']:
    # append the indices of each year corresponding to your RV period
    RV_period.insert(-1, np.where(dates.month == mon)[0] )
RV_period = [x for sublist in RV_period for x in sublist]
RV_period.sort()
ex['RV_period'] = RV_period
datesRV = dates[RV_period]


Prec_reg_time = Prec_reg.sel(time=datesRV)
norm = Prec_reg_time / Prec_reg_time.std(dim='time')
#ex['path_pp'] = '/Users/semvijverberg/surfdrive/Scripts/rasterio/'
#mask = func_mcK.import_array('landseamask_2.5deg.nc', ex)
#mask_reg = func_mcK.find_region(mask, region=ex['region'])[0]
#mask = (('latitude', 'longitude'), mask_reg)

#norm.coords['mask'] = mask
norm = norm.where(abs(norm.values) < 3*norm.std().values)
for i in np.linspace(0,datesRV.size-50,10):
    for_plot = norm.isel(time=int(i))    
#    for_plot.coords['mask'] = mask
    xarray_plot(for_plot)

#%%
norm = ts_3d / ts_3d.std(dim='time')
comp = norm.sel(time=event_train)
comp = comp.where(abs(comp.values) < 3.5*comp.std().values)
for i in np.linspace(0,comp.time.size-10,15):
    for_plot = comp.isel(time=int(i))    
#    for_plot.coords['mask'] = mask
    xarray_plot(for_plot)

xarray_plot(comp.mean(dim='time')/comp.std(dim='time'))
