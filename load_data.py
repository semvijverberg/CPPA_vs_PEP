#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 09:39:41 2019

@author: semvijverberg
"""
import os
import xarray as xr
import pandas as pd
import numpy as np
import func_CPPA



def load_data(ex):
    #'Mckinnonplot', 'U.S.', 'U.S.cluster', 'PEPrectangle', 'Pacific', 'Whole', 'Northern', 'Southern'
    def oneyr(datetime):
        return datetime.where(datetime.year==datetime.year[0]).dropna()
    
    if ex['load_mcK'][0] == '1':
        # Load in mckinnon Time series
        ex['RVname'] = 'T95' + ex['load_mcK'][1:]
        ex['name']   = 'sst_NOAA'
        ex['startyear'] = 1982 ; 
        if ex['load_mcK'][1:] == 'bram':
            T95name = 'T95_Bram_McK.csv' ; lpyr=True
        else:
            T95name = 'PEP-T95TimeSeries.txt'; lpyr=False
        RVtsfull, datesmcK = read_T95(T95name, ex) 
        ex['endyear'] = int(RVtsfull[-1].time.dt.year)
        datesRV = func_CPPA.make_datestr(datesmcK, ex,
                                        ex['startyear'], ex['endyear'], lpyr=lpyr)
        filename_precur = ('{}_1982-2017_2jan_31aug_dt-1days_{}deg'
                        '.nc'.format(ex['name'], ex['grid_res']))
    else:
        # load ERA-i Time series
        print('\nimportRV_1dts is true, so the 1D time serie given with name \n'
                  '{} is imported.'.format(ex['RVts_filename']))
        filename = os.path.join(ex['RV1d_ts_path'], ex['RVts_filename'])
        dicRV = np.load(filename,  encoding='latin1').item()
        RVtsfull = dicRV['RVfullts95']
        ex['mask'] = dicRV['RV_array']['mask']
        func_CPPA.xarray_plot(dicRV['RV_array']['mask'])
        RVhour   = RVtsfull.time[0].dt.hour.values
        datesRV = func_CPPA.make_datestr(pd.to_datetime(RVtsfull.time.values), ex, 
                                        ex['startyear'], ex['endyear'])
        # add RVhour to daily dates
        datesRV = datesRV + pd.Timedelta(int(RVhour), unit='h')
        filename_precur = '{}_{}-{}_2jan_31okt_dt-1days_{}deg.nc'.format(ex['name'],
                           ex['startyear'], ex['endyear'], ex['grid_res'])
        filename_precur = '{}_1979-2017_1jan_31dec_daily_{}deg.nc'.format(ex['name'],
                           ex['grid_res'])
        ex['endyear'] = int(datesRV[-1].year)
    
    # Selected Time series of T95 ex['sstartdate'] until ex['senddate']
    RVts = RVtsfull.sel(time=datesRV)
    ex['n_oneyr'] = oneyr(datesRV).size
    
    RV_ts, datesmcK = func_CPPA.time_mean_bins(RVts, ex)
    #expanded_time = func_mcK.expand_times_for_lags(datesmcK, ex)
    
    if ex['mcKthres'] == 'mcKthres':
        # binary time serie when T95 exceeds 1 std
        ex['hotdaythres'] = RV_ts.mean(dim='time').values + RV_ts.std().values
    else:
        percentile = ex['mcKthres']
        ex['hotdaythres'] = np.percentile(RV_ts.values, percentile)
        ex['mcKthres'] = '{}'.format(percentile)
    
    # Load in external ncdf
    filename = '{}_1979-2017_1mar_31dec_dt-1days_{}deg.nc'.format(ex['name'],
                ex['grid_res'])
    #filename_precur = 'sm2_1979-2017_2jan_31okt_dt-1days_{}deg.nc'.format(ex['grid_res'])
    #path = os.path.join(ex['path_raw'], 'tmpfiles')
    # full globe - full time series
    varfullgl = func_CPPA.import_array(filename_precur, ex, path='pp')
    
    
    # Converting Mckinnon timestemp to match xarray timestemp
    #expandeddaysmcK = func_mcK.to_datesmcK(expanded_time, expanded_time[0].hour, varfullgl.time[0].dt.hour)
    # region mckinnon - expanded time series
    #Prec_reg = func_mcK.find_region(varfullgl.sel(time=expandeddaysmcK), region=ex['region'])[0]
    Prec_reg = func_CPPA.find_region(varfullgl, region=ex['region'])[0]
    
    if ex['tfreq'] != 1:
        Prec_reg, datesvar = func_CPPA.time_mean_bins(Prec_reg, ex)

    
    Prec_reg = Prec_reg.to_array().squeeze()

    
    ## filter out outliers 
    if ex['name'][:2]=='sm':
        Prec_reg = Prec_reg.where(Prec_reg.values < 5.*Prec_reg.std(dim='time').values)
    
    if ex['add_lsm'] == True:
        base_path_lsm = '/Users/semvijverberg/surfdrive/Scripts/rasterio/'
        mask = func_CPPA.import_array(ex['mask_file'].format(ex['grid_res']), ex,
                                     base_path_lsm)
        mask_reg = func_CPPA.find_region(mask, region=ex['region'])[0]
        mask_reg = mask_reg.to_array().squeeze()
        mask = (('latitude', 'longitude'), mask_reg.values)
        Prec_reg.coords['mask'] = mask
        Prec_reg.values = Prec_reg * mask_reg
        


    ex['n_yrs'] = len(set(RV_ts.time.dt.year.values))
    ex['n_conv'] = ex['n_yrs'] 
    return RV_ts, Prec_reg, ex

def read_T95(T95name, ex):
    filepath = os.path.join(ex['path_pp'], T95name)
    if filepath[-3:] == 'txt':
        data = pd.read_csv(filepath)
        datelist = []
        values = []
        for r in data.values:
            year = int(r[0][:4])
            month = int(r[0][5:7])
            day = int(r[0][7:11])
            string = '{}-{}-{}'.format(year, month, day)
            values.append(float(r[0][10:]))
            datelist.append( pd.Timestamp(string) )
    elif filepath[-3:] == 'csv':
        data = pd.read_csv(filepath, sep='\t')
        datelist = []
        values = []
        for r in data.iterrows():
            year = int(r[1]['Year'])
            month = int(r[1]['Month'])
            day =   int(r[1]['Day'])
            string = '{}-{}-{}T00:00:00'.format(year, month, day)
            values.append(float(r[1]['T95(degC)']))
            datelist.append( pd.Timestamp(string) )
    dates = pd.to_datetime(datelist)
    RVts = xr.DataArray(values, coords=[dates], dims=['time'])
    return RVts, dates