#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 16:29:47 2018

@author: semvijverberg
"""
import os, sys
if os.path.isdir("/Users/semvijverberg/surfdrive/"):
    basepath = "/Users/semvijverberg/surfdrive/"
else:
    basepath = "/home/semvij/"
os.chdir(os.path.join(basepath, 'Scripts/CPPA/CPPA'))


import xarray as xr
import pandas as pd
import numpy as np
from netCDF4 import num2date
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib as mpl
from shapely.geometry.polygon import LinearRing
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker
import datetime, calendar
from sklearn.cluster import DBSCAN
import scipy 
import func_CPPA
from ROC_score import ROC_score

flatten = lambda l: [item for sublist in l for item in sublist]



def main(RV_ts, Prec_reg, ex):
    #%%
    if (ex['method'] == 'no_train_test_split') : ex['n_conv'] = 1
    if ex['method'][:5] == 'split' : ex['n_conv'] = 1
    if ex['method'][:6] == 'random' : ex['tested_yrs'] = []
    if ex['method'][:6] == 'random' : ex['n_conv'] = int(ex['n_yrs'] / int(ex['method'][6:]))
    if ex['method'] == 'iter': ex['n_conv'] = ex['n_yrs'] 
    
    if ex['ROC_leave_n_out'] == True or ex['method'] == 'no_train_test_split': 
        print('leave_n_out set to False')
        ex['leave_n_out'] = False

    Prec_train_mcK = find_region(Prec_reg, region=ex['region'])[0]

    rmwhere, window = ex['rollingmean']
    if rmwhere == 'all' and window != 1:
        Prec_reg = func_CPPA.rolling_mean_time(Prec_reg, ex, center=False)
    
    train_test_list  = []
    l_ds_PEP         = []        
    
    for n in range(ex['n_conv']):
        train_all_test_n_out = (ex['ROC_leave_n_out'] == True) & (n==0) 
        ex['n'] = n
        # do single run    
        # =============================================================================
        # Create train test set according to settings 
        # =============================================================================
        train, test, ex = train_test_wrapper(RV_ts, Prec_reg, ex)       
        # =============================================================================
        # Calculate Precursor
        # =============================================================================
        if train_all_test_n_out == True:
            # only train once on all years if ROC_leave_n_out == True
            ds_mcK = mcKmean(Prec_train_mcK, train, ex)             
        # Force Leave_n_out validation even though pattern is based on whole dataset
        if (ex['ROC_leave_n_out'] == True) & (ex['n']==0):
            # start selecting leave_n_out
            ex['leave_n_out'] = True
            train, test, ex['test_years'] = func_CPPA.rand_traintest(RV_ts, Prec_reg, 
                                              ex)  
        
        elif train_all_test_n_out == False:
            # train each time on only train years
            # train each time on only train years
            ds_mcK = mcKmean(Prec_train_mcK, train, ex)  
            
        l_ds_PEP.append(ds_mcK)        
        
        # appending tuple
        train_test_list.append( (train, test) )
        
    ex['train_test_list'] = train_test_list    
    
    
    #%%

    return l_ds_PEP, ex



# =============================================================================
# =============================================================================
# Wrapper functions
# =============================================================================
# =============================================================================
def mcKmean(Prec_train_mcK, train, ex):
    

    lats = Prec_train_mcK.latitude
    lons = Prec_train_mcK.longitude
    
    array = np.zeros( (len(ex['lags']), len(lats), len(lons)) )
    pattern = xr.DataArray(data=array, coords=[ex['lags'], lats, lons], 
                          dims=['lag','latitude','longitude'], name='mcK_Comp_diff_lags',
                          attrs={'units':'Kelvin'})

    
    
    event_train = func_CPPA.Ev_timeseries(train['RV'], ex['event_thres'], ex)[0].time

    for lag in ex['lags']:
        idx = ex['lags'].index(lag)
        
        events_min_lag = func_dates_min_lag(event_train, lag)[1]
 

        pattern_atlag = Prec_train_mcK.sel(time=events_min_lag).mean(dim='time')
        pattern[idx] = pattern_atlag 
        

    ds_mcK = xr.Dataset( {'pattern' : pattern} )
    return ds_mcK


def only_spatcov_wrapper(l_ds, RV_ts, Prec_reg, ex):
    #%%
    ex['score'] = []
    ex['test_ts_prec'] = np.zeros( len(ex['lags']) , dtype=list)
    ex['test_RV'] = np.zeros( len(ex['lags']) , dtype=list)
    var_reg_mcK = find_region(Prec_reg, region=ex['regionmcK'])[0]
    
    for n in range(len(ex['train_test_list'])):
        ex['n'] = n
        
        test =ex['train_test_list'][n][1]
        ex['test_year'] = list(set(test['RV'].time.dt.year.values))
        
        
        ds = l_ds[n]
        
        
        ex = ROC_score_only_spatcov(test, ds, var_reg_mcK, ex)
    #%%
    return ex

def ROC_score_only_spatcov(test, ds, var_reg_mcK, ex):
    #%%
    # =============================================================================
    # calc ROC scores
    # =============================================================================
    ROC  = np.zeros(len(ex['lags']))
    FP_TP    = np.zeros(len(ex['lags']), dtype=list)
    ROC_boot = np.zeros(len(ex['lags']), dtype=list)
    
    if 'n_boot' not in ex.keys():
        n_boot = 0
    else:
        n_boot = ex['n_boot']
    
    for lag_idx, lag in enumerate(ex['lags']):
        
        idx = ex['lags'].index(lag)
        dates_test = pd.to_datetime(test['RV'].time.values)
        # select antecedant SST pattern to summer days:
        dates_min_lag = dates_test - pd.Timedelta(int(lag), unit='d')


        var_test_mcK = var_reg_mcK.sel(time=dates_min_lag)    
        var_patt_mcK = find_region(ds['pattern'].sel(lag=lag), region=ex['regionmcK'])[0]
        crosscorr_mcK = func_CPPA.cross_correlation_patterns(var_test_mcK, 
                                                            var_patt_mcK)

        
        if (
            ex['leave_n_out'] == True and ex['method'] == 'iter'
            or ex['ROC_leave_n_out'] or ex['method'][:6] == 'random'
            ):
            if ex['n'] == 0:
                ex['test_RV'][idx]          = test['RV'].values
                ex['test_ts_prec'][lag_idx]  = crosscorr_mcK.values
            else:
                ex['test_RV'][idx]     = np.concatenate( [ex['test_RV'][idx], test['RV'].values] )  
                ex['test_ts_prec'][lag_idx] = np.concatenate( [ex['test_ts_prec'][lag_idx], crosscorr_mcK.values] )
                
        
            if  ex['n'] == ex['n_conv']-1:
                if lag_idx == 0:
                    print('Calculating ROC scores\nDatapoints precursor length '
                      '{}\nDatapoints RV length {}'.format(len(ex['test_ts_prec'][0]),
                       len(ex['test_RV'][0])))
                    

                ts_pred  = ((ex['test_ts_prec'][lag_idx]-np.mean(ex['test_ts_prec'][lag_idx]))/ \
                                          (np.std(ex['test_ts_prec'][lag_idx]) ) )                 


                if lag > 30:
                    obs_array = pd.DataFrame(ex['test_RV'][0])
                    obs_array = obs_array.rolling(7, center=True, min_periods=1).mean()
                    threshold = (obs_array.mean() + obs_array.std()).values
                    events_idx = np.where(obs_array > threshold)[0]
                else:
                    events_idx = np.where(ex['test_RV'][0] > ex['event_thres'])[0]
                y_true = func_CPPA.Ev_binary(events_idx, len(ex['test_RV'][0]),  ex['min_dur'], 
                                         ex['max_break'], grouped=False)
                y_true[y_true!=0] = 1
        
                if 'use_ts_logit' in ex.keys() and ex['use_ts_logit'] == True:
                    ROC[lag_idx], FP, TP, ROC_boot[lag_idx] = ROC_score(ts_pred, y_true,
                                           n_boot=n_boot, win=0, n_yrs=ex['n_yrs'])
                else:
                    ROC[lag_idx], FP, TP, ROC_boot[lag_idx] = ROC_score(ts_pred, y_true,
                                           n_boot=n_boot, win=0, n_yrs=ex['n_yrs'])
                
                print('\n*** ROC score for {} lag {} ***\n\nPEP {:.2f} '
                ' ±{:.2f} 2*std random events\n\n'.format(ex['region'], 
                  lag, ROC[idx], np.percentile(ROC_boot[lag_idx], 99)))
            
            
        elif ex['leave_n_out'] == False or ex['method'][:5] == 'split':
            if idx == 0:
                print('performing hindcast')
            ROC[lag_idx], FP, TP, ROC_boot[lag_idx] = ROC_score(ts_pred, y_true,
                                           n_boot=n_boot, win=0, n_yrs=ex['n_yrs'])
            

            
            print('\n*** ROC score for {} lag {} ***\n\nPEP {:.2f} '
            ' ±{:.2f} 2*std random events\n\n'.format(ex['region'], 
              lag, ROC[idx], np.percentile(ROC_boot[lag_idx], 99)))

        

    ex['score'].append([ ROC, ROC_boot, FP_TP])
    #%%
    return ex


def train_test_wrapper(RV_ts, Prec_reg, ex):
    #%%
    now = datetime.datetime.now()
    rmwhere, window = ex['rollingmean']
    if ex['leave_n_out'] == True and ex['method'][:6] == 'random':
        train, test, ex['test_years'] = func_CPPA.rand_traintest(RV_ts, Prec_reg, 
                                          ex)
        
        general_folder = '{}_leave_{}_out_{}_{}_tf{}_{}p_{}deg_{}nyr_{}rm{}_{}'.format(
                            ex['method'], ex['leave_n_years_out'], ex['startyear'], ex['endyear'],
                          ex['tfreq'], ex['event_percentile'], ex['grid_res'],
                          ex['n_oneyr'], 
                          window, rmwhere, 
                          now.strftime("%Y-%m-%d"))

    elif ex['method']=='no_train_test_split':
        print('Training on all years')
        Prec_train_idx = np.arange(0, Prec_reg.time.size) #range(len(full_years)) if full_years[i] in rand_train_years]
        train = dict( { 'Prec'  : Prec_reg,
                        'Prec_train_idx' : Prec_train_idx,
                        'RV'    : RV_ts,
                        'events': func_CPPA.Ev_timeseries(RV_ts, ex['event_thres'], ex)[0]})
        test = train.copy()

    
        general_folder = 'hindcast_{}_{}_tf{}_{}p_{}deg_{}nyr_{}rm_{}'.format(
                          ex['startyear'], ex['endyear'],
                          ex['tfreq'], ex['event_percentile'], ex['grid_res'],
                          ex['n_oneyr'], 
                          ex['rollingmean'], 
                          now.strftime("%Y-%m-%d"))
        
    else:
        train, test, ex['test_years'] = func_CPPA.rand_traintest(RV_ts, Prec_reg, 
                                          ex)
    

        general_folder = '{}_{}_{}_tf{}_{}p_{}deg_{}nyr_{}rm{}_{}'.format(
                            ex['method'], ex['startyear'], ex['endyear'],
                          ex['tfreq'], ex['event_percentile'], ex['grid_res'],
                          ex['n_oneyr'], 
                           window, rmwhere, 
                          now.strftime("%Y-%m-%d"))
        
                       
                          


        
        ex['test_years'] = 'all_years'

    
    subfolder         = 'lags{}Ev{}d{}p'.format(ex['lags'], ex['min_dur'], 
                             ex['max_break'])
    subfolder = subfolder.replace(' ' ,'')
    ex['CPPA_folder'] = os.path.join(general_folder, subfolder)
    ex['output_dic_folder'] = os.path.join(ex['figpathbase'], ex['CPPA_folder'])
    

    #%%
    
    return train, test, ex
        

def find_region(data, region='Mckinnonplot'):
    if region == 'Mckinnonplot':
        west_lon = -240; east_lon = -40; south_lat = -10; north_lat = 80

    elif region ==  'U.S.soil':
        west_lon = -130; east_lon = -60; south_lat = 0; north_lat = 60
    elif region ==  'U.S.cluster':
        west_lon = -100; east_lon = -70; south_lat = 20; north_lat = 50
    elif region ==  'PEPrectangle':
        west_lon = -215; east_lon = -130; south_lat = 20; north_lat = 50
    elif region ==  'Pacific':
        west_lon = -215; east_lon = -120; south_lat = 19; north_lat = 60
    elif region ==  'Whole':
        west_lon = -360; east_lon = -1; south_lat = -80; north_lat = 80
    elif region ==  'Northern':
        west_lon = -360; east_lon = -1; south_lat = -10; north_lat = 80
    elif region ==  'Southern':
        west_lon = -360; east_lon = -1; south_lat = -80; north_lat = -10
    elif region ==  'Tropics':
        west_lon = -360; east_lon = -1; south_lat = -15; north_lat = 30 
    elif region ==  'elnino3.4':
        west_lon = -170; east_lon = -120; south_lat = -5; north_lat = 5 
#    elif region == 'for_soil':
        

    region_coords = [west_lon, east_lon, south_lat, north_lat]
    import numpy as np
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return int(idx)
#    if data.longitude.values[-1] > 180:
#        all_values = data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360+east_lon))
#        lon_idx = np.arange(find_nearest(data['longitude'], 360 + west_lon), find_nearest(data['longitude'], 360+east_lon))
#        lat_idx = np.arange(find_nearest(data['latitude'],north_lat),find_nearest(data['latitude'],south_lat),1)
        
    if west_lon <0 and east_lon > 0:
        # left_of_meridional = np.array(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(0, east_lon)))
        # right_of_meridional = np.array(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360)))
        # all_values = np.concatenate((np.reshape(left_of_meridional, (np.size(left_of_meridional))), np.reshape(right_of_meridional, np.size(right_of_meridional))))
        lon_idx = np.concatenate(( np.arange(find_nearest(data['longitude'], 360 + west_lon), len(data['longitude'])),
                              np.arange(0,find_nearest(data['longitude'], east_lon), 1) ))
        
        north_idx = find_nearest(data['latitude'],north_lat)
        south_idx = find_nearest(data['latitude'],south_lat)
        if north_idx > south_idx:
            lat_idx = np.arange(south_idx,north_idx,1)
            all_values = data.sel(latitude=slice(south_lat, north_lat), 
                                  longitude=(data.longitude > 360 + west_lon) | (data.longitude < east_lon))
        elif south_idx > north_idx:
            lat_idx = np.arange(north_idx,south_idx,1)
            all_values = data.sel(latitude=slice(north_lat, south_lat), 
                                  longitude=(data.longitude > 360 + west_lon) | (data.longitude < east_lon))
    if west_lon < 0 and east_lon < 0:
        lon_idx = np.arange(find_nearest(data['longitude'], 360 + west_lon), find_nearest(data['longitude'], 360+east_lon))
        
        north_idx = find_nearest(data['latitude'],north_lat)
        south_idx = find_nearest(data['latitude'],south_lat)
        if north_idx > south_idx:
            lat_idx = np.arange(south_idx,north_idx,1)
            all_values = data.sel(latitude=slice(south_lat, north_lat), 
                                  longitude=slice(360+west_lon, 360+east_lon))
        elif south_idx > north_idx:
            lat_idx = np.arange(north_idx,south_idx,1)
            all_values = data.sel(latitude=slice(north_lat, south_lat), 
                                  longitude=slice(360+west_lon, 360+east_lon))     
        
#        all_values = data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360+east_lon))
#        lat_idx = np.arange(find_nearest(data['latitude'],north_lat),find_nearest(data['latitude'],south_lat),1)

    return all_values, region_coords


def store_ts_wrapper(l_ds_PEP, RV_ts, Prec_reg_mcK, ex):
    #%%
    ex['output_ts_folder'] = os.path.join(ex['output_dic_folder'], 'timeseries_robwghts')
    if os.path.isdir(ex['output_ts_folder']) != True : os.makedirs(ex['output_ts_folder'])
    
    for n in range(len(ex['train_test_list'])):
        ex['n'] = n
        
        test =ex['train_test_list'][n][1]
        ex['test_year'] = list(set(test['RV'].time.dt.year.values))
        
        print('Storing timeseries using patterns retrieved '
              'without test year(s) {}'.format(ex['test_year']))
        

        ds_mcK = l_ds_PEP[n]
        
        
        store_timeseries(ds_mcK, RV_ts, Prec_reg_mcK, ex)
    #%%
    return

def store_timeseries(ds_mcK, RV_ts, Prec_reg_mcK, ex):
    #%%
    
    for lag in ex['lags']:

        # spatial covariance of whole PEP pattern
        var_patt_PEP = func_CPPA.find_region(ds_mcK['pattern'].sel(lag=lag), region=ex['regionmcK'])[0]
        spatcov_PEP = func_CPPA.cross_correlation_patterns(Prec_reg_mcK, var_patt_PEP).values


        
        # merge data
        columns = ['spatcov_PEP']

             
        dates = pd.to_datetime(Prec_reg_mcK.time.dt.values)
        dates -= pd.Timedelta(dates.hour[0], unit='h')
        df = pd.DataFrame(data = spatcov_PEP[:,None], index=dates, columns=columns) 
        df.index.name = 'date'
        
        name_trainset = 'testyr{}_{}.csv'.format(ex['test_year'], lag)
        df.to_csv(os.path.join(ex['output_ts_folder'], name_trainset ))
        #%%
    return


def func_dates_min_lag(dates, lag):
    dates_min_lag = pd.to_datetime(dates.values) - pd.Timedelta(int(lag), unit='d')
    ### exlude leap days from dates_train_min_lag ###


    # ensure that everything before the leap day is shifted one day back in time 
    # years with leapdays now have a day less, thus everything before
    # the leapday should be extended back in time by 1 day.
    mask_lpyrfeb = np.logical_and(dates_min_lag.month == 2, 
                                         dates_min_lag.is_leap_year
                                         )
    mask_lpyrjan = np.logical_and(dates_min_lag.month == 1, 
                                         dates_min_lag.is_leap_year
                                         )
    mask_ = np.logical_or(mask_lpyrfeb, mask_lpyrjan)
    new_dates = np.array(dates_min_lag)
    new_dates[mask_] = dates_min_lag[mask_] - pd.Timedelta(1, unit='d')
    dates_min_lag = pd.to_datetime(new_dates)   
    # to be able to select date in pandas dataframe
    dates_min_lag_str = [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates_min_lag]                                         
    return dates_min_lag_str, dates_min_lag




def read_T95(T95name, ex):
    filepath = os.path.join(ex['RV1d_ts_path'], T95name)
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

def import_array(filename, ex, path='pp'):
    if path == 'pp':
        file_path = os.path.join(ex['path_pp'], filename)        
    elif path != 'pp':
        file_path = os.path.join(path, filename) 
    ncdf = xr.open_dataset(file_path, decode_cf=True, decode_coords=True, decode_times=False)
    variables = list(ncdf.variables.keys())
    strvars = [' {} '.format(var) for var in variables]
    var = [var for var in strvars if var not in ' time time_bnds longitude latitude '][0] 
#    marray = np.squeeze(ncdf.to_array(file_path).rename(({file_path: var})))
    numtime = ncdf['time']
    dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])
    dates = pd.to_datetime(dates)
#    print('temporal frequency \'dt\' is: \n{}'.format(dates[1]- dates[0]))
    ncdf['time'] = dates
    return ncdf

def save_figure(data, path):
    import os
    import matplotlib.pyplot as plt
#    if 'path' in locals():
#        pass
#    else:
#        path = '/Users/semvijverberg/Downloads'
    if path == 'default':
        path = '/Users/semvijverberg/Downloads'
    else:
        path = path
    import datetime
    today = datetime.datetime.today().strftime("%d-%m-%y_%H'%M")
    if type(data.name) is not type(None):
        name = data.name.replace(' ', '_')
    if 'name' in locals():
        print('input name is: {}'.format(name))
        name = name + '.jpeg'
        pass
    else:
        name = 'fig_' + today + '.jpeg'
    print(('{} to path {}'.format(name, path)))
    plt.savefig(os.path.join(path,name), format='jpeg', dpi=300, bbox_inches='tight')
    
def area_weighted(xarray):
    # Area weighted, taking cos of latitude in radians     
    coslat = np.cos(np.deg2rad(xarray.coords['latitude'].values)).clip(0., 1.)
    area_weights = np.tile(coslat[..., np.newaxis],(1,xarray.longitude.size))
#    xarray.values = xarray.values * area_weights 

    return xr.DataArray(xarray.values * area_weights, coords=xarray.coords, 
                           dims=xarray.dims)
    
def convert_longitude(data):
    import numpy as np
    import xarray as xr
    lon_above = data.longitude[np.where(data.longitude > 180)[0]]
    lon_normal = data.longitude[np.where(data.longitude <= 180)[0]]
    # roll all values to the right for len(lon_above amount of steps)
    data = data.roll(longitude=len(lon_above))
    # adapt longitude values above 180 to negative values
    substract = lambda x, y: (x - y)
    lon_above = xr.apply_ufunc(substract, lon_above, 360)
    if lon_normal[0] == 0.:
        convert_lon = xr.concat([lon_above, lon_normal], dim='longitude')
    else:
        convert_lon = xr.concat([lon_normal, lon_above], dim='longitude')
    data['longitude'] = convert_lon
    return data



def to_datesmcK(datesmcK, to_hour, from_hour):
    
    dt_hours = from_hour - to_hour
    matchdaysmcK = datesmcK + pd.Timedelta(int(dt_hours), unit='h')
    return xr.DataArray(matchdaysmcK, dims=['time'])




# =============================================================================
# =============================================================================
# Plotting functions
# =============================================================================
# =============================================================================
    
def extend_longitude(data):
    import xarray as xr
    import numpy as np
    plottable = xr.concat([data, data.sel(longitude=data.longitude[:1])], dim='longitude').to_dataset(name="ds")
    plottable["longitude"] = np.linspace(0,360, len(plottable.longitude))
    plottable = plottable.to_array(dim='ds')
    return plottable

def xarray_plot(data, path='default', name = 'default', saving=False):
    # from plotting import save_figure
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import numpy as np
    if type(data) == type(xr.Dataset()):
        data = data.to_array().squeeze()
#    original
    fig = plt.figure( figsize=(10,6) ) 
    if len(data.longitude[np.where(data.longitude > 180)[0]]) != 0:
        if data.longitude.where(data.longitude==0).dropna(dim='longitude', how='all') == 0.:
            print('hoi')   
            data = convert_longitude(data)
    else:
        pass
    if data.ndim != 2:
        print("number of dimension is {}, printing first element of first dimension".format(np.squeeze(data).ndim))
        data = data[0]
    else:
        pass
    if 'mask' in list(data.coords.keys()):
        cen_lon = data.where(data.mask==True, drop=True).longitude.mean()
        data = data.where(data.mask==True, drop=True)
    else:
        cen_lon = data.longitude.mean().values
    proj = ccrs.PlateCarree(central_longitude=cen_lon)
    ax = fig.add_subplot(111, projection=proj)
    ax.coastlines(facecolor='grey')
    vmin = np.round(float(data.min())-0.01,decimals=2) 
    vmax = np.round(float(data.max())+0.01,decimals=2) 
    vmin = -max(abs(vmin),vmax) ; vmax = max(abs(vmin),vmax)
    # ax.set_global()
    if 'mask' in list(data.coords.keys()):
        plot = data.copy().where(data.mask==True).plot.pcolormesh(ax=ax, cmap=plt.cm.RdBu_r,
                             transform=ccrs.PlateCarree(), add_colorbar=True,
                             vmin=vmin, vmax=vmax,
                             cbar_kwargs={'orientation' : 'horizontal'})
    else:
        plot = data.plot.pcolormesh(ax=ax, cmap=plt.cm.RdBu_r,
                             transform=ccrs.PlateCarree(), add_colorbar=True,
                             vmin=vmin, vmax=vmax, 
                             cbar_kwargs={'orientation' : 'horizontal'})
    if saving == True:
        save_figure(data, path=path)
    plt.show()
    



def plot_oneyr_events(xarray, ex, test_year, folder, saving=False):
    #%%
    if ex['event_percentile'] == 'std':
        # binary time serie when T95 exceeds 1 std
        threshold = xarray.mean(dim='time').values + xarray.std().values
    else:
        percentile = ex['event_percentile']
        threshold = np.percentile(xarray.values, percentile)
    
    testyear = xarray.where(xarray.time.dt.year == test_year).dropna(dim='time', how='any')
    freq = pd.Timedelta(testyear.time.values[1] - testyear.time.values[0])
    plotpaper = xarray.sel(time=pd.DatetimeIndex(start=testyear.time.values[0], 
                                                end=testyear.time.values[-1], 
                                                freq=freq ))

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    plotpaper.plot(ax=ax, color='blue', linewidth=3, label='ERA-I T95')
    plt.axhline(y=threshold, color='blue', linewidth=2 )
    plt.fill_between(plotpaper.time.values, threshold, plotpaper, where=(plotpaper.values > threshold),
                 interpolate=True, color="crimson", label="ERA-I hot days")
    if ex['load_mcK'][0] != '1':
        T95name = 'PEP-T95TimeSeries.txt'
        T95, datesmcK = read_T95(T95name, ex)
        datesRV_mcK = func_CPPA.make_datestr(datesmcK, ex, 1982, 2015)
        T95RV = T95.sel(time=datesRV_mcK)
        if ex['mcKthres'] == 'mcKthres':
            # binary time serie when T95 exceeds 1 std
            threshold = T95RV.mean(dim='time').values + T95RV.std().values
        else:
            percentile = ex['mcKthres']
            threshold = np.percentile(T95RV.values, percentile)
        testyear = datesRV_mcK.where(datesRV_mcK.year == test_year).dropna()
        freq = pd.Timedelta(testyear[1] - testyear[0])
        plot_T95 = T95RV.sel(time=pd.DatetimeIndex(start=testyear[0], 
                                                end=testyear[-1], 
                                                freq=freq ))
        plot_T95.plot(ax=ax, linewidth=2, color='orange', linestyle='--', label='GHCND T95')
        plt.axhline(y=threshold, color='orange', linestyle='--', linewidth=2)
        plt.fill_between(plot_T95.time.values, threshold, plot_T95, where=(plot_T95.values > threshold),
                 interpolate=True, color="orange", alpha=0.5, label="GHCND hot days")
    ax.legend(fontsize='x-large', fancybox=True, facecolor='grey',
              frameon=True, framealpha=0.3)
    ax.set_title('T95 timeseries and hot days events in eastern U.S.', fontsize=18)
    ax.set_ylabel('Temperature anomalies [K]', fontsize=15)
    ax.set_xlabel('')
    #%%
    if saving == True:
        filename = os.path.join(folder, 'ts_{}'.format(test_year))
        plt.savefig(filename+'.png', dpi=300)






def plot_oneyr_events_allRVts(ex, test_year, folder, saving=False):
    #%%
    
    def add_RVts(xarray, ex):
        if ex['event_percentile'] == 'std':
            # binary time serie when T95 exceeds 1 std
            threshold = xarray.mean(dim='time').values + xarray.std().values
        else:
            percentile = ex['event_percentile']
            threshold = np.percentile(xarray.values, percentile)
        
    
        testyear = xarray.where(xarray.time.dt.year == test_year).dropna(dim='time', how='any')
        freq = pd.Timedelta(testyear.time.values[1] - testyear.time.values[0])
        plotpaper = xarray.sel(time=pd.DatetimeIndex(start=testyear.time.values[0], 
                                                    end=testyear.time.values[-1], 
                                                    freq=freq ))
    
    
        ax = fig.add_subplot(111)
        plotpaper.plot(ax=ax, color=ex['line_color'], linewidth=3, linestyle= ex['line_style'], label=ex['RVts_name'])
        plt.axhline(y=threshold, color=ex['line_color'], linewidth=2, linestyle= ex['line_style'], alpha=ex['alpha'] )
        plt.fill_between(plotpaper.time.values, threshold, plotpaper, where=(plotpaper.values > threshold),
                     interpolate=True, color=ex['line_color'], alpha=ex['alpha'])
        ax.legend(fontsize='x-large', fancybox=True, facecolor='grey',
                  frameon=True, framealpha=0.3)
        ax.set_title('T95 timeseries and hot day events of eastern U.S. cluster', fontsize=18)
        ax.set_ylabel('Temperature anomalies [K]', fontsize=15)
        ax.set_xlabel('')
        return fig, ax
    
    fig = plt.figure(figsize=(15, 5))
    
    
    for T95name in ['T95_Bram_McK.csv']:
#        ex['RV1d_ts_path'] = 
        if T95name == 'T95_Bram_McK.csv':
            lpyr = True
            ex['RVts_name'] = 'T95 GHCND'
            ex['line_color']= 'green'
            ex['alpha']     = 0.4
            ex['line_style'] = 'dashdot'
        elif T95name == 'PEP-T95TimeSeries.txt':
            lpyr = False
            ex['RVts_name'] = 'T95 GHCND'
            ex['line_color']= 'blue'
            ex['alpha']     = 0.4
            ex['line_style'] = 'dashed'


        RVtsfull, datesmcK = read_T95(T95name, ex) 
        datesRV = func_CPPA.make_datestr(datesmcK, ex,
                                            1999, 2015, lpyr=lpyr)
        xarray = RVtsfull.sel(time=datesRV)
        
        fig, ax = add_RVts(xarray, ex)
    
    for reanalysis in ['ERAint', 'ERA5']:
#        ex['RV1d_ts_path'] = 
        if reanalysis == 'ERA5':
            ex['RVts_filename'] = "era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1.npy"
            ex['RVts_name'] =  "T95 ERA-5"
            ex['line_color']= 'red'
            ex['line_style'] = 'solid'
            ex['alpha']     = 0.6
        elif reanalysis == 'ERAint':
            ex['RVts_filename'] = 'ERAint_t2mmax_US_1979-2017_averAggljacc0.75d_tf1_n4__to_t2mmax_US_tf1.npy'
            ex['RVts_name'] = "T95 ERA-int"
            ex['line_color']= 'blue'
            ex['line_style'] = 'dashdot'
            ex['alpha']     = 0.5


        filename = os.path.join(ex['RV1d_ts_path'], ex['RVts_filename'])
        dicRV = np.load(filename,  encoding='latin1').item()
        RVtsfull = dicRV['RVfullts95']
        RVhour   = RVtsfull.time[0].dt.hour.values
        datesRV = func_CPPA.make_datestr(pd.to_datetime(RVtsfull.time.values), ex, 
                                        1999, 2015, lpyr=False)
        datesRV = datesRV + pd.Timedelta(int(RVhour), unit='h')

        xarray = RVtsfull.sel(time=datesRV)
        
        fig, ax = add_RVts(xarray, ex)
    

    
    #%%
    if saving == True:
        filename = os.path.join(folder, 'ts_{}'.format(test_year))
        plt.savefig(filename+'.png', dpi=300)


