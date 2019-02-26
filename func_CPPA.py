#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 16:29:47 2018

@author: semvijverberg
"""
import os
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
flatten = lambda l: [item for sublist in l for item in sublist]


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
        datesRV = make_datestr(datesmcK, ex,
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
        xarray_plot(dicRV['RV_array']['mask'])
        RVhour   = RVtsfull.time[0].dt.hour.values
        datesRV = make_datestr(pd.to_datetime(RVtsfull.time.values), ex, 
                                        ex['startyear'], ex['endyear'])
        # add RVhour to daily dates
        datesRV = datesRV + pd.Timedelta(int(RVhour), unit='h')
        filename_precur = '{}_{}-{}_2jan_31okt_dt-1days_{}deg.nc'.format(ex['name'],
                           ex['startyear'], ex['endyear'], ex['grid_res'])
        filename_precur = '{}_1979-2017_2jan_31okt_dt-1days_{}deg.nc'.format(ex['name'],
                           ex['grid_res'])
        ex['endyear'] = int(datesRV[-1].year)
    
    # Selected Time series of T95 ex['sstartdate'] until ex['senddate']
    RVts = RVtsfull.sel(time=datesRV)
    ex['n_oneyr'] = oneyr(datesRV).size
    
    RV_ts, datesmcK = time_mean_bins(RVts, ex)
    #expanded_time = func_mcK.expand_times_for_lags(datesmcK, ex)
    
    if ex['mcKthres'] == 'mcKthres':
        # binary time serie when T95 exceeds 1 std
        ex['hotdaythres'] = RV_ts.mean(dim='time').values + RV_ts.std().values
    else:
        percentile = ex['mcKthres']
        ex['hotdaythres'] = np.percentile(RV_ts.values, percentile)
        ex['mcKthres'] = '{}'.format(percentile)
    
    # Load in external ncdf
    filename = '{}_1979-2017_2mar_31okt_dt-1days_{}deg.nc'.format(ex['name'],
                ex['grid_res'])
    #filename_precur = 'sm2_1979-2017_2jan_31okt_dt-1days_{}deg.nc'.format(ex['grid_res'])
    #path = os.path.join(ex['path_raw'], 'tmpfiles')
    # full globe - full time series
    varfullgl = import_array(filename_precur, ex, path='pp')
    
    
    # Converting Mckinnon timestemp to match xarray timestemp
    #expandeddaysmcK = func_mcK.to_datesmcK(expanded_time, expanded_time[0].hour, varfullgl.time[0].dt.hour)
    # region mckinnon - expanded time series
    #Prec_reg = func_mcK.find_region(varfullgl.sel(time=expandeddaysmcK), region=ex['region'])[0]
    Prec_reg = find_region(varfullgl, region=ex['region'])[0]
    
    if ex['tfreq'] != 1:
        Prec_reg, datesvar = time_mean_bins(Prec_reg, ex)
    
    if ex['rollingmean'] != 1:
        # Smoothen precursor time series by applying rolling mean
        Prec_reg = rolling_mean_time(Prec_reg, ex)
    
    Prec_reg = Prec_reg.to_array().squeeze()
    
    
    ## filter out outliers 
    if ex['name'][:2]=='sm':
        Prec_reg = Prec_reg.where(Prec_reg.values < 5.*Prec_reg.std(dim='time').values)
    
    if ex['add_lsm'] == True:
        base_path_lsm = '/Users/semvijverberg/surfdrive/Scripts/rasterio/'
        mask = import_array(ex['mask_file'].format(ex['grid_res']), ex,
                                     base_path_lsm)
        mask_reg = find_region(mask, region=ex['region'])[0]
        mask_reg = mask_reg.to_array().squeeze()
        mask = (('latitude', 'longitude'), mask_reg.values)
        Prec_reg.coords['mask'] = mask
        Prec_reg.values = Prec_reg * mask_reg
        

    #ex['lags_idx'] = [12, 18, 24, 30]  
    #ex['lags'] = [l*ex['tfreq'] for l in ex['lags_idx'] ]
    ex['n_yrs'] = len(set(RV_ts.time.dt.year.values))
    ex['n_conv'] = ex['n_yrs'] 
    return RV_ts, Prec_reg, ex




def main(RV_ts, Prec_reg, ex):
    if (ex['leave_n_out'] == False) and ex['ROC_leave_n_out'] == False : ex['n_conv'] = 1
    if ex['method'][:5] == 'split' : ex['n_conv'] = 1
    if ex['ROC_leave_n_out'] == True: 
        print('leave_n_out set to False')
        ex['leave_n_out'] = False

    
    train_test_list = []
    l_ds_CPPA        = []
    l_ds_PEP        = []        
    
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
            ds_Sem = extract_precursor(Prec_reg, train, test, ex)    
            if ex['wghts_accross_lags'] == True:
                ds_Sem['pattern'] = filter_autocorrelation(ds_Sem, ex)
                
            ds_mcK = mcKmean(train, ex)             
        # Force Leave_n_out validation even though pattern is based on whole dataset
        if (ex['ROC_leave_n_out'] == True) & (ex['n']==0):
            # start selecting leave_n_out
            ex['leave_n_out'] = True
            train, test, ex['test_years'] = rand_traintest(RV_ts, Prec_reg, 
                                              ex)    
        
#        elif (ex['leave_n_out'] == True) & (ex['ROC_leave_n_out'] == False):
        elif (ex['ROC_leave_n_out'] == False) or ex['method'][:5] == 'split':
            # train each time on only train years
            ds_Sem = extract_precursor(Prec_reg, train, test, ex)    
            
            if ex['wghts_accross_lags'] == True:
                ds_Sem['pattern'] = filter_autocorrelation(ds_Sem, ex)
                
            ds_mcK = mcKmean(Prec_reg, train, ex)  
            

        l_ds_CPPA.append(ds_Sem)
        l_ds_PEP.append(ds_mcK)        
        
        # appending tuple
        train_test_list.append( (train, test) )
        
    ex['train_test_list'] = train_test_list
    
    l_ds_CPPA = grouping_regions_similar_coords(l_ds_CPPA, ex, 
                     grouping = 'group_accros_tests_single_lag', eps=10)
    
    store_ts_wrapper(l_ds_CPPA, l_ds_PEP, RV_ts, Prec_reg, ex)

#    if (ex['leave_n_out']==False) & (ex['ROC_leave_n_out']==False):
#        # only need single run
#        break
#    if (ex['method'][:5]=='split'):
#        # only need single run
#        break
#    if (ex['method'][:6] == 'random'):
#        if n == ex['n_stop']:
#            ex['n_conv'] = ex['n_stop']
#            break
    return l_ds_CPPA, l_ds_PEP, ex



# =============================================================================
# =============================================================================
# Wrapper functions
# =============================================================================
# =============================================================================

def mcKmean(Prec_reg, train, ex):
    
    Prec_train = Prec_reg.isel(time=train['Prec_train_idx'])
    Prec_train_mcK = find_region(Prec_train, region=ex['region'])[0]
    dates_train = to_datesmcK(train['RV'].time, train['RV'].time.dt.hour[0], 
                                           Prec_train.time[0].dt.hour)
#        time = Prec_train_mcK.time
    lats = Prec_train_mcK.latitude
    lons = Prec_train_mcK.longitude
    pthresholds = np.linspace(1, 9, 9, dtype=int)
    
    array = np.zeros( (len(ex['lags']), len(lats), len(lons)) )
    pattern = xr.DataArray(data=array, coords=[ex['lags'], lats, lons], 
                          dims=['lag','latitude','longitude'], name='mcK_Comp_diff_lags',
                          attrs={'units':'Kelvin'})
    array = np.zeros( (len(ex['lags']), len(dates_train)) )
    pattern_ts = xr.DataArray(data=array, coords=[ex['lags'], dates_train], 
                          dims=['lag','time'], name='mcK_mean_ts_diff_lags',
                          attrs={'units':'Kelvin'})
    
    array = np.zeros( (len(ex['lags']), len(pthresholds)) )
    pattern_p = xr.DataArray(data=array, coords=[ex['lags'], pthresholds], 
                          dims=['lag','percentile'], name='mcK_mean_p_diff_lags')
    for lag in ex['lags']:
        idx = ex['lags'].index(lag)
        event_train = Ev_timeseries(train['RV'], ex['hotdaythres'], ex).time
        event_train = to_datesmcK(event_train, event_train.dt.hour[0], Prec_train_mcK.time[0].dt.hour)
        events_min_lag = event_train - pd.Timedelta(int(lag), unit='d')
        dates_train_min_lag = dates_train - pd.Timedelta(int(lag), unit='d')
        
        ### exlude leap days ###
        # no leap days in dates_train_min_lag
        noleapdays = ((dates_train_min_lag.dt.month == 2) & (dates_train_min_lag.dt.day == 29))==False
        # only keep noleapdays
        dates_train_min_lag = dates_train_min_lag[noleapdays].dropna(dim='time', how='all')
        # also kick out the corresponding events
        dates_train = dates_train[noleapdays].dropna(dim='time', how='all')
        
       # no leap days in events_min_lag
        noleapdays = ((events_min_lag.dt.month == 2) & (events_min_lag.dt.day == 29))==False
        # only keep noleapdays
        events_min_lag = events_min_lag[noleapdays].dropna(dim='time', how='all') 
        # also kick out the corresponding events
        event_train = event_train[noleapdays].dropna(dim='time', how='all')   


        pattern_atlag = Prec_train_mcK.sel(time=events_min_lag).mean(dim='time')
        pattern[idx] = pattern_atlag 
        ts_3d = Prec_train_mcK.sel(time=dates_train_min_lag) 
        
        
        crosscorr = cross_correlation_patterns(ts_3d, pattern_atlag)
        crosscorr['time'] = pattern_ts.time
        pattern_ts[idx] = crosscorr
        # Percentile values based on training dataset
        p_pred = []
        for p in pthresholds:	
            p_pred.append(np.percentile(crosscorr.values, p*10))
        pattern_p[idx] = p_pred
    ds_mcK = xr.Dataset( {'pattern' : pattern, 'ts' : pattern_ts, 'perc' : pattern_p} )
    return ds_mcK


def train_test_wrapper(RV_ts, Prec_reg, ex):
    #%%
    now = datetime.datetime.now()
    if ex['leave_n_out'] == True and ex['method'][:6] == 'random':
        train, test, ex['test_years'] = rand_traintest(RV_ts, Prec_reg, 
                                          ex)
        
        general_folder = '{}_leave_{}_out_{}_{}_tf{}_{}p_{}deg_{}nyr_{}tperc_{}tc_{}rm_{}'.format(
                            ex['method'], ex['leave_n_years_out'], ex['startyear'], ex['endyear'],
                          ex['tfreq'], ex['mcKthres'], ex['grid_res'],
                          ex['n_oneyr'], 
                          ex['perc_map'], ex['comp_perc'], ex['rollingmean'], 
                          now.strftime("%Y-%m-%d"))
      
        
    if ex['leave_n_out'] == True and ex['method'][:5] == 'split':
        train, test, ex['test_years'] = rand_traintest(RV_ts, Prec_reg, 
                                          ex)
    

        general_folder = '{}_{}_{}_tf{}_{}p_{}deg_{}nyr_{}tperc_{}tc_{}rm_{}'.format(
                            ex['method'], ex['startyear'], ex['endyear'],
                          ex['tfreq'], ex['mcKthres'], ex['grid_res'],
                          ex['n_oneyr'], 
                          ex['perc_map'], ex['comp_perc'], ex['rollingmean'], 
                          now.strftime("%Y-%m-%d"))
        

        
    if ex['leave_n_out']==True and ex['method']=='iter': # and ex['ROC_leave_n_out']==False
        train, test, ex['test_years'] = rand_traintest(RV_ts, Prec_reg, 
                                          ex)

        general_folder = '{}_{}_{}_tf{}_{}p_{}deg_{}nyr_{}tperc_{}tc_{}rm_{}'.format(
                            ex['method'], ex['startyear'], ex['endyear'],
                          ex['tfreq'], ex['mcKthres'], ex['grid_res'],
                          ex['n_oneyr'], 
                          ex['perc_map'], ex['comp_perc'], ex['rollingmean'], 
                          now.strftime("%Y-%m-%d"))
                          
                          
    elif ex['leave_n_out'] == False:
        train = dict( { 'Prec'  : Prec_reg,
                        'RV'    : RV_ts,
                        'events': Ev_timeseries(RV_ts, ex['hotdaythres'], ex)})
        test = train.copy()


        general_folder = 'hindcast_{}_{}_tf{}_{}p_{}deg_{}nyr_{}tperc_{}tc_{}rm_{}'.format(
                          ex['startyear'], ex['endyear'],
                          ex['tfreq'], ex['mcKthres'], ex['grid_res'],
                          ex['n_oneyr'], 
                          ex['perc_map'], ex['comp_perc'], ex['rollingmean'], 
                          now.strftime("%Y-%m-%d"))

        
        ex['test_years'] = 'all_years'
        print('Training on all years')
    
    subfolder         = 'lags{}Ev{}d{}p_pmd{}'.format(ex['lags'], ex['min_dur'], 
                             ex['max_break'], ex['prec_reg_max_d'])
    subfolder = subfolder.replace(' ' ,'')
    ex['CPPA_folder'] = os.path.join(general_folder, subfolder)
    ex['output_dic_folder'] = os.path.join(ex['figpathbase'], ex['CPPA_folder'])
    

    #%%
    
    return train, test, ex
        

def extract_precursor(Prec_reg, train, test, ex):
    #%%
    Prec_train = Prec_reg.isel(time=train['Prec_train_idx'])
    lats = Prec_train.latitude
    lons = Prec_train.longitude
    
    array = np.zeros( (len(ex['lags']), len(lats), len(lons)) )
    pattern_CPPA = xr.DataArray(data=array, coords=[ex['lags'], lats, lons], 
                          dims=['lag','latitude','longitude'], name='communities_composite',
                          attrs={'units':'Kelvin'})


    array = np.zeros( (len(ex['lags']), len(lats), len(lons)) )
    pat_num_CPPA = xr.DataArray(data=array, coords=[ex['lags'], lats, lons], 
                          dims=['lag','latitude','longitude'], name='commun_numb_init', 
                          attrs={'units':'Precursor regions'})
    
    array = np.zeros( (len(ex['lags']), len(lats), len(lons)) )
    std_train_lag = xr.DataArray(data=array, coords=[ex['lags'], lats, lons], 
                          dims=['lag','latitude','longitude'], name='std_train_min_lag', 
                          attrs={'units':'std [-]'})

    pat_num_CPPA.name = 'commun_numbered'
    
    weights     = pattern_CPPA.copy()
    weights.name = 'weights'
   

#    Actors_ts_GPH = [[] for i in ex['lags']] #!
    
#    x = 0
    for lag in ex['lags']:
#            i = ex['lags'].index(lag)
        idx = ex['lags'].index(lag)
        event_train = Ev_timeseries(train['RV'], ex['hotdaythres'], ex).time
        event_train = to_datesmcK(event_train, event_train.dt.hour[0], 
                                           Prec_train.time[0].dt.hour)
        events_min_lag = event_train - pd.Timedelta(int(lag), unit='d')
        
        dates_train = to_datesmcK(train['RV'].time, train['RV'].time.dt.hour[0], 
                                           Prec_train.time[0].dt.hour)
        dates_train_min_lag = dates_train - pd.Timedelta(int(lag), unit='d')
        
        ### exlude leap days ###
        # no leap days in dates_train_min_lag
        noleapdays = ((dates_train_min_lag.dt.month == 2) & (dates_train_min_lag.dt.day == 29))==False
        # only keep noleapdays
        dates_train_min_lag = dates_train_min_lag[noleapdays].dropna(dim='time', how='all')
        # also kick out the corresponding events
        dates_train = dates_train[noleapdays].dropna(dim='time', how='all')
        
       # no leap days in events_min_lag
        noleapdays = ((events_min_lag.dt.month == 2) & (events_min_lag.dt.day == 29))==False
        # only keep noleapdays
        events_min_lag = events_min_lag[noleapdays].dropna(dim='time', how='all') 
        # also kick out the corresponding events
        event_train = event_train[noleapdays].dropna(dim='time', how='all')   

        
        
        event_idx = [list(dates_train.values).index(E) for E in event_train.values]
        binary_events = np.zeros(dates_train.size)    
        binary_events[event_idx] = 1
        std_train_lag[idx] = Prec_train.sel(time=dates_train_min_lag).std(dim='time')
        ts_3d = Prec_train.sel(time=dates_train_min_lag)/std_train_lag[idx]
        #%%
        # extract precursor regions composite approach
        composite_p1, xrnpmap_p1, wghts_at_lag = extract_regs_p1(events_min_lag,
                                                     ts_3d, std_train_lag[idx], ex)  
#        composite_p1.plot() 
#        xrnpmap_p1.plot()
     

        pattern_CPPA[idx] = composite_p1.where(composite_p1.mask == True)
        
        pat_num_CPPA[idx] = xrnpmap_p1
    
        weights[idx] = wghts_at_lag

        #%%        
    ds_Sem = xr.Dataset( {'pattern_CPPA' : pattern_CPPA, 'pat_num_CPPA' : pat_num_CPPA,
                          'weights' : weights, 'std_train_min_lag' : std_train_lag } )
                          
                          
    
    return ds_Sem


# =============================================================================
# =============================================================================
# Core functions
# =============================================================================
# =============================================================================

def extract_regs_p1(events_min_lag, ts_3d, std_train_lag, ex):
    #%% 
    x=0
#    T, pval, mask_sig = Welchs_t_test(sample, full, alpha=0.01)
#    threshold = np.reshape( mask_sig, (mask_sig.size) )
#    mask_threshold = threshold 
#    plt.figure()
#    plt.imshow(mask_sig)
    # divide train set into train-feature and train-weights part:

    lats = ts_3d.latitude
    lons = ts_3d.longitude
    comp_years = list(events_min_lag.time.dt.year.values)


    ts_3d_train_n = ts_3d.copy()


    all_yrs_set = list(set(ts_3d.time.dt.year.values))    
    ex['n_ch1'] = 1
    chunks = [all_yrs_set[i:(i+ex['n_ch1'])] for i in range(int(len(all_yrs_set)))]
    years_n_out = [2, 3, 4, 5]
    for n_out in years_n_out:    
        chunks, count = create_chunks(all_yrs_set, n_out, chunks)
  


    count = np.zeros( (len(all_yrs_set)) )
    for yr in all_yrs_set:
        idx = all_yrs_set.index(yr)
        count[idx] = np.sum( [chnk.count(yr) for chnk in chunks] )

    iter_regions = np.zeros( (len(chunks), ts_3d[0].size))
    
    for idx in range(len(chunks)):
        yrs = chunks[idx]#[all_yrs_set.index(yr) for yr in yrs]
        yrs_trainfeat =  [i for i in all_yrs_set if i not in yrs] 
        
        # exclude years in event time series
        one_out_idx_ev = [i for i in range(len(comp_years) ) if comp_years[i] in yrs_trainfeat]
        event_one_out = events_min_lag.isel( time = one_out_idx_ev)
        comp_subset = ts_3d_train_n.sel(time=event_one_out)
        
        # region which are more constrained compared to whole training set 
        # are given higher wgths
#        wgths_constrain = (comp_subset.std(dim='time')/std_train_lag)
        # normalized over whole training set
        
        mean = comp_subset.mean(dim='time') #/ comp_subset.std(dim='time')
        

        threshold = mean.quantile(ex['perc_map']/100).values
        nparray = np.reshape(np.nan_to_num(mean.values), (mean.size))
#            threshold = np.percentile(nparray, ex['perc_map'])
        mask_threshold = np.array(abs(nparray) < ( threshold ), dtype=int)
        Corr_Coeff = np.ma.MaskedArray(nparray, mask=mask_threshold)
        Regions_lag_i = define_regions_and_rank_new(Corr_Coeff, lats.values, lons.values, ex)
        iter_regions[idx] = Regions_lag_i
#        if idx % 10 == 0.:
#            plt.figure(figsize=(10,15)) ; plt.imshow(np.reshape(np.array(Regions_lag_i,dtype=int), (lats.size, lons.size)))


    mask_reg_all_1 = (iter_regions != 0.)
    reg_all_1 = iter_regions.copy()
    reg_all_1[mask_reg_all_1] = 1
#    plt.figure(figsize=(10,15)) ; plt.imshow(np.reshape(np.sum(reg_all_1, axis=0), (lats.size, lons.size))) ; plt.colorbar()
    mask_final = ( np.sum(reg_all_1, axis=0) < int(ex['comp_perc'] * len(chunks)))
#    plt.figure(figsize=(10,15)) ; plt.imshow(np.reshape(np.array(mask_final,dtype=int), (lats.size, lons.size))) 
#    %%
    weights = np.sum(reg_all_1, axis=0)
    weights[mask_final==True] = 0.
#    plt.figure(figsize=(10,15)) ; plt.imshow(np.reshape(weights, (lats.size, lons.size)))
    composite_p1 = ts_3d.sel(time=events_min_lag).mean(dim='time')
    nparray_comp = np.reshape(np.nan_to_num(composite_p1.values), (composite_p1.size))
    Corr_Coeff = np.ma.MaskedArray(nparray_comp, mask=mask_final)
    lat_grid = composite_p1.latitude.values
    lon_grid = composite_p1.longitude.values


    # retrieve regions sorted in order of 'strength'
    # strength is defined as an area weighted values in the composite
    Regions_lag_i = define_regions_and_rank_new(Corr_Coeff, lat_grid, lon_grid, ex)
    
    assert np.sum(Regions_lag_i) != 0., ('No regions detected with these criteria.')
        
    
    # Regions are iteratively counted starting from first lag (R0) to last lag R(-1)
    # adapt numbering of different communities/Regions to account for 
    # multiple variables/lags
    if Regions_lag_i.max()> 0:
        n_regions_lag_i = int(Regions_lag_i.max()) 	
        A_r = np.reshape(Regions_lag_i, (lat_grid.size, lon_grid.size))
        A_r + x
    x = A_r.max() 

    # if there are less regions that are desired, the n_strongest is lowered
    if n_regions_lag_i <= ex['n_strongest']:
        ex['upd_n_strongest'] = n_regions_lag_i
               
    
    # reshape to latlon grid
    npmap = np.reshape(Regions_lag_i, (lat_grid.size, lon_grid.size))
    mask_strongest = (npmap!=0) 
    npmap[mask_strongest==False] = 0
    xrnpmap_init = composite_p1.copy()
    xrnpmap_init.values = npmap
    
    mask = (('latitude', 'longitude'), mask_strongest)
    composite_p1.coords['mask'] = mask
    xrnpmap_init.coords['mask'] = mask
    xrnpmap_init = xrnpmap_init.where(xrnpmap_init.mask==True)

    sum_count = np.reshape(weights, (lat_grid.size, lon_grid.size))
    weights = sum_count / np.max(sum_count)
#    ts_3d_train_n.values = ts_3d_train_n.values * weights
#    ts_regions_lag_i, sign_ts_regions = spatial_mean_regions(Regions_lag_i, 
#                                         regions_for_ts, ts_3d_train_n, Corr_Coeff)[:2]

#    plt.figure()
#    xrnpmap_init.plot.contourf(cmap=plt.cm.tab10)   
#    composite_p1.plot.contourf()  
#    list_region_info = [Regions_lag_i, ts_regions_lag_i, sign_ts_regions, weights]
    #%%
    return composite_p1, xrnpmap_init, weights



def spatial_mean_regions(Regions_lag_i, regions_for_ts, ts_3d, npmean):
    #%%
    n_time   = ts_3d.time.size
    lat_grid = ts_3d.latitude
    lon_grid = ts_3d.longitude
    regions_for_ts = list(regions_for_ts)
    
    actbox = np.reshape(ts_3d.values, (n_time, 
                  lat_grid.size*lon_grid.size))  
    
    # get lonlat array of area for taking spatial means 
    lons_gph, lats_gph = np.meshgrid(lon_grid, lat_grid)
    cos_box = np.cos(np.deg2rad(lats_gph))
    cos_box_array = np.repeat(cos_box[None,:], actbox.shape[0], 0)
    cos_box_array = np.reshape(cos_box_array, (cos_box_array.shape[0], -1))
    

    # this array will be the time series for each feature
    ts_regions_lag_i = np.zeros((actbox.shape[0], len(regions_for_ts)))
    
    # track sign of eacht region
    sign_ts_regions = np.zeros( len(regions_for_ts) )
    
    # std regions
    std_regions     = np.zeros( (len(regions_for_ts)) )
    
    # composite needed for sign
    meanbox = np.reshape(npmean, (lat_grid.size*lon_grid.size))
    
    if Regions_lag_i.shape == actbox[0].shape:
        Regions = Regions_lag_i
    elif Regions_lag_i.shape == (lat_grid.shape[0], lon_grid.shape[0]):
        Regions = np.reshape(Regions_lag_i, (Regions_lag_i.size))
    # calculate area-weighted mean over features
    for r in regions_for_ts:
        idx = regions_for_ts.index(r)
        # start with empty lonlat array
        B = np.zeros(Regions.shape)
        # Mask everything except region of interest
        B[Regions == r] = 1	
        # Calculates how values inside region vary over time
        ts_regions_lag_i[:,idx] = np.nanmean(actbox[:, B == 1] * cos_box_array[:, B == 1], axis =1)
        # get sign of region
        sign_ts_regions[idx] = np.sign(np.mean(meanbox[B==1]))
#    print(sign_ts_regions)
        
    std_regions = np.std(ts_regions_lag_i, axis=0)
    #%%
    return ts_regions_lag_i, sign_ts_regions, std_regions


def store_ts_wrapper(l_ds_CPPA, l_ds_PEP, RV_ts, Prec_reg, ex):

    for n in range(len(ex['train_test_list'])):
        ex['n'] = n
        
        test =ex['train_test_list'][n][1]
        ex['test_year'] = list(set(test['RV'].time.dt.year.values))
        
        print('Storing timeseries using patterns retrieved '
              'without test year(s) {}'.format(ex['test_year']))
        
        ds_Sem = l_ds_CPPA[n]
        ds_mcK = l_ds_PEP[n]
        
        
        store_timeseries(ds_mcK, ds_Sem, RV_ts, Prec_reg, ex)

    return

def store_timeseries(ds_mcK, ds_Sem, RV_ts, Prec_reg, ex):
#%%
    
    ex['output_ts_folder'] = os.path.join(ex['output_dic_folder'], 'timeseries')
    if os.path.isdir(ex['output_ts_folder']) != True : os.makedirs(ex['output_ts_folder'])

    
    for lag in ex['lags']:
        idx = ex['lags'].index(lag)
        dates = to_datesmcK(RV_ts.time, RV_ts.time.dt.hour[0], 
                                           RV_ts.time[0].dt.hour)
        dates_min_lag = dates - pd.Timedelta(int(lag), unit='d')
        ### exlude leap days ###
        # no leap days in dates_train_min_lag
        noleapdays = ((dates_min_lag.dt.month == 2) & (dates_min_lag.dt.day == 29))==False
        # only keep noleapdays
        dates_lag = dates_min_lag[noleapdays].dropna(dim='time', how='all')
        # also kick out the corresponding events
        dates = dates[noleapdays].dropna(dim='time', how='all')
     
        
        
        np.warnings.filterwarnings('ignore')
        mask_regions = ds_Sem['pat_num_CPPA'].sel(lag=lag).values >= 1
        # Make time series for whole period
#        Prec_trainsel = Prec_reg.isel(time=train['Prec_train_idx'])
        # actor/precursor full 3d timeseries at RV period minus lag, normalized over std within dates_train_min_lag
        ts_3d_n = Prec_reg.sel(time=dates_lag)/ds_Sem['std_train_min_lag'][idx]
        
        # ts_3d is given more weight to robust precursor regions
        ts_3d_nw = ts_3d_n  * ds_Sem['weights'].sel(lag=lag)
        mask_notnan = (np.product(np.isnan(ts_3d_nw.values),axis=0)==False) # nans == False
        mask = mask_notnan * mask_regions
        # normal mean of extracted regions
        composite_p1 = ds_Sem['pattern_CPPA'].sel(lag=lag).where(mask==True)
        ts_3d_nw     = ts_3d_nw.where(mask==True)

        
        Regions_lag_i = ds_Sem['pat_num_CPPA'][idx].squeeze().values
        regions_for_ts = np.unique(Regions_lag_i[~np.isnan(Regions_lag_i)])
#        mean_n = composite_p1/ts_3d.std(dim='time')
        npmean        = composite_p1.values
        ts_regions_lag_i, sign_ts_regions = spatial_mean_regions(Regions_lag_i, 
                                regions_for_ts, ts_3d_nw, npmean)[:2]
        check_nans = np.where(np.isnan(ts_regions_lag_i))
        if check_nans[0].size != 0:
            print('{} nans found in time series of region {}, dropping this region.'.format(
                    check_nans[0].size, 
                    regions_for_ts[check_nans[1]]))
            regions_for_ts = np.delete(regions_for_ts, check_nans[1])
            ts_regions_lag_i = np.delete(ts_regions_lag_i, check_nans[1], axis=1)
            sign_ts_regions  = np.delete(sign_ts_regions, check_nans[1], axis=0)
        
        name_trainset = 'testyr{}_{}.csv'.format(ex['test_year'], lag)
        spatcov_CPPA = cross_correlation_patterns(ts_3d_nw, ds_Sem['pattern_CPPA'][idx])
        ts_3d_PEP = find_region(Prec_reg.sel(time=dates_lag), region=ex['regionmcK'])[0]
        var_patt_PEP = find_region(ds_mcK['pattern'].sel(lag=lag), region=ex['regionmcK'])[0]
        spatcov_PEP = cross_correlation_patterns(ts_3d_PEP, var_patt_PEP)
        columns = list(regions_for_ts)
        columns.insert(0, 'spatcov_CPPA')
        columns.insert(0, 'spatcov_PEP')
        data = np.concatenate([spatcov_PEP.values[:,None], spatcov_CPPA.values[:,None], ts_regions_lag_i], axis=1)

        df = pd.DataFrame(data = data, index=pd.to_datetime(spatcov_CPPA.time.values), columns=columns) 
        df.index.name = 'date'
                          
        df.to_csv(os.path.join(ex['output_ts_folder'], name_trainset ))
    return


def rand_traintest(RV_ts, Prec_reg, ex):
    #%%
    all_years = np.arange(ex['startyear'], ex['endyear']+1)
    
    # conditions failed initally assumed True
    a_conditions_failed = True
    while a_conditions_failed == True:
        
        a_conditions_failed = False
        # Divide into random sampled 25 year for train & rest for test
    #        n_years_sampled = int((ex['endyear'] - ex['startyear']+1)*0.66)
        if ex['method'][:6] == 'random':
            size_train = int(np.percentile(range(len(all_years)), int(ex['method'][6:])))
            size_test  = len(all_years) - size_train
            ex['leave_n_years_out'] = size_test
            ex['n_stop'] = int(len(all_years) / size_test) + 33
            rand_test_years = np.random.choice(all_years, ex['leave_n_years_out'], replace=False)
        elif ex['method'] == 'iter':
            ex['leave_n_years_out'] = 1
            if ex['n'] >= ex['n_yrs']:
                n = ex['n'] - ex['n_yrs']
            else:
                n = ex['n']
            rand_test_years = [all_years[n]]
        elif ex['method'][:5] == 'split':
            size_train = int(np.percentile(range(len(all_years)), int(ex['method'][5:])))
            size_test  = len(all_years) - size_train
            ex['leave_n_years_out'] = size_test
            print('Using {} years to train and {} to test'.format(size_train, size_test))
            rand_test_years = all_years[-size_test:]
            
    
            
        # test duplicates
        a_conditions_failed = (len(set(rand_test_years)) != ex['leave_n_years_out'])
        # Update random years to be selected as test years:
    #        initial_years = [yr for yr in initial_years if yr not in random_test_years]
        rand_train_years = [yr for yr in all_years if yr not in rand_test_years]
        

        full_years  = list(Prec_reg.time.dt.year.values)
        RV_years  = list(RV_ts.time.dt.year.values)
        
    #            RV_dates_train_idx = [i for i in range(len(RV_dates)) if RV_dates[i] in rand_train_years]
        Prec_train_idx = [i for i in range(len(full_years)) if full_years[i] in rand_train_years]
        RV_train_idx = [i for i in range(len(RV_years)) if RV_years[i] in rand_train_years]
        
    #            RV_dates_test_idx = [i for i in range(len(RV_dates)) if RV_dates[i] in rand_test_years]
        Prec_test_idx = [i for i in range(len(full_years)) if full_years[i] in rand_test_years]
        RV_test_idx = [i for i in range(len(RV_years)) if RV_years[i] in rand_test_years]
        
        
        # This is inconveniently big to put into dictionary
        RV_train = RV_ts.isel(time=RV_train_idx)
        
    #    if len(RV_dates_test_idx) 
    #            dates_test = matchdatesRV.isel(time=RV_dates_test_idx)
        Prec_test = Prec_reg.isel(time=Prec_test_idx)
        RV_test = RV_ts.isel(time=RV_test_idx)
        
        event_train = Ev_timeseries(RV_train, ex['hotdaythres'], ex).time
        event_test = Ev_timeseries(RV_test, ex['hotdaythres'], ex).time
        
        test_years = [yr for yr in list(set(RV_years)) if yr in rand_test_years]
        
        ave_events_pyr = (len(event_train) + len(event_test))/len(all_years)
        exp_events     = int(ave_events_pyr) * len(rand_test_years)
        tolerance      = 0.2 * exp_events
        diff           = abs(len(event_test) - exp_events)
        
        print('test year is {}, with {} events'.format(test_years, len(event_test)))
        if diff > tolerance and ex['method']=='random': 
            print('not a representative sample, drawing new sample')
            a_conditions_failed = True
                   
        
        train = dict( {    'RV'             : RV_train,
                           'events'         : event_train,
                           'Prec_train_idx' : Prec_train_idx})
        test = dict( {     'Prec'           : Prec_test,
                           'RV'             : RV_test,
                           'events'         : event_test,
                           'Prec_test_idx'  : Prec_test_idx})
    #%%
    return train, test, test_years

def filter_autocorrelation(ds_Sem, ex):
    n_lags = len(ex['lags'])
    n_lats = ds_Sem['pattern'].latitude.size
    n_lons = ds_Sem['pattern'].longitude.size
    ex['n_steps'] = len(ex['lags'])
    weights = np.zeros( (n_lags, n_lats, n_lons) )
    xrweights = ds_Sem['pattern'].copy()
    xrweights.values = weights
    for lag in ex['lags']:
        data = np.nan_to_num(ds_Sem['pattern'].sel(lag=lag).values)
        mask = np.ma.masked_array(data, dtype=bool)
        idx = ex['lags'].index(lag)
        weights[idx] = mask
        xrweights[idx].values = mask
    weights = np.sum(weights, axis=0)
    return weights * ds_Sem['pattern']


def Welchs_t_test(sample, full, alpha):
    np.warnings.filterwarnings('ignore')
    mask = (sample[0] == 0.).values
#    mask = np.reshape(mask, (mask.size))
    n_space = full.latitude.size*full.longitude.size
    npfull = np.reshape(full.values, (full.time.size, n_space))
    npsample = np.reshape(sample.values, (sample.time.size, n_space))
    
#    npsample = npsample[np.broadcast_to(mask==False, npsample.shape)] 
#    npsample = np.reshape(npsample, (sample.time.size, 
#                                     int(npsample.size/sample.time.size) ))
#    npfull   = npfull[np.broadcast_to(mask==False, npfull.shape)] 
#    npfull = np.reshape(npfull, (full.time.size, 
#                                     int(npfull.size/full.time.size) ))
       
    T, pval = scipy.stats.ttest_ind(npsample, npfull, axis=0, 
                                equal_var=False, nan_policy='omit')
    pval = np.reshape(pval, (full.latitude.size, full.longitude.size))
    T = np.reshape(T, (full.latitude.size, full.longitude.size))
    mask_sig = (pval > alpha) 
    mask_sig[mask] = True
    return T, pval, mask_sig

def merge_neighbors(lsts):
  sets = [set(lst) for lst in lsts if lst]
  merged = 1
  while merged:
    merged = 0
    results = []
    while sets:
      common, rest = sets[0], sets[1:]
      sets = []
      for x in rest:
        if x.isdisjoint(common):
          sets.append(x)
        else:
          merged = 1
          common |= x
      results.append(common)
    sets = results
  return sets


def create_chunks(all_yrs_set, n_out, chunks):
    '''yr_prior are years which have priority to be part of the chunk '''
    # determine priority in random sampling
    # count what years are the least taken out of the composite
    def find_priority(chunks, all_yrs_set):
        count = np.zeros( (len(all_yrs_set)) )
        for yr in all_yrs_set:
            idx = all_yrs_set.index(yr)
            count[idx] = np.sum( [chnk.count(yr) for chnk in chunks] )
        list_idx_1 = list(np.argwhere(count == count.min()))
        list_idx_2 = list(np.argwhere(count != count.max()))
        
        all_yrs = all_yrs_set.copy()    
        yr_prior_1 = [all_yrs[int(i)] for i in list_idx_1]
        yr_prior_2 = [all_yrs[int(i)] for i in list_idx_2 if i not in list_idx_1]
        return yr_prior_1, yr_prior_2, count
    
    yr_prior_1, yr_prior_2, count = find_priority(chunks, all_yrs_set)
    
    
    for yr in all_yrs_set:    
        # yr is always going to be part of chunk
        if len(yr_prior_1) != 0 and yr in yr_prior_1:
            yr_prior_1.remove(yr)
        
        # resplenish every time half the years have passed
        if all_yrs_set.index(yr) < 0.5*len(all_yrs_set) or len(yr_prior_1) == 0:
            yr_prior_1, yr_prior_2 = find_priority(chunks, all_yrs_set)[:2]
            if len(yr_prior_1) != 0 and yr in yr_prior_1:
                yr_prior_1.remove(yr)
        
        years_to_add = [[yr]]
        n_choice = n_out - 1
        
        if len(yr_prior_1) >= n_choice:
            
            # yr of for loop iteration is always in list, 
            # give priority to yr_prior_1
            
            yrs_to_list  = list(np.random.choice(yr_prior_1, n_choice, replace=False))
            years_to_add.append(yrs_to_list)
#                years_to_add = flatten(years_to_add)
            # the year that is added has now reduced priority
            yr_prior_1 = [yr for yr in yr_prior_1 if yr not in yrs_to_list]
            if len(flatten(years_to_add)) != n_out:
                print(yr_prior_1)
                print(yr_prior_2)
                print(years_to_add)
                print('first')
                break
            
        elif len(yr_prior_1) < n_choice:# and len(yr_prior_1) != 0:
            
            yr_prior_1, yr_prior_2 = find_priority(chunks, all_yrs_set)[:2]
            if len(yr_prior_1) != 0 and yr in yr_prior_1:
                yr_prior_1.remove(yr)
            
            n_out_part = n_choice - len(yr_prior_1)
            
            # if there are still sufficient years left in yr_prior_1, just pick them
            if n_out_part < len(yr_prior_1):
                yrs_to_list  = list(np.random.choice(yr_prior_1, n_out_part, replace=False))
            # if not, add what is left in yr_prior_1
            else:
                yrs_to_list = yr_prior_1
            years_to_add.append(yrs_to_list)
#                years_to_add = flatten(years_to_add)
            
            # the year that is added has now reduced priority
            yr_prior_1 = [yr for yr in yr_prior_1 if yr not in yrs_to_list]
            
            # what is left is sampled from yr_prior_2
            
            n_out_left = n_out - len(flatten(years_to_add))
            
            # ensure that there will be years in prior_2
            if len(yr_prior_2) < n_out_left and len(yr_prior_2) != 0:
                # add years that are left in yr_prior_2
                years_to_add.append(yr_prior_2)
                n_out_left = n_out - len(years_to_add)
#                    years_to_add = flatten(years_to_add)
            elif len(yr_prior_2) == 0 or n_out_left != 0:
                # create new yr_prior_2
                yr_prior_2 = [yr for yr in all_yrs_set.copy() if yr not in flatten(years_to_add)]
            
                yrs_to_list  = list(np.random.choice(yr_prior_2, n_out_left, replace=False))
                years_to_add.append( yrs_to_list )
#                    years_to_add = flatten(years_to_add)

            
            if len(flatten(years_to_add)) != n_out:
                print('second')
                print(yr_prior_1)
                print(yr_prior_2)
                print(n_out_left)
                print(n_out)
                print(years_to_add)
                break
            
        chunks.append( flatten(years_to_add) )
        
            
    yr_prior_1, yr_prior_2, count = find_priority(chunks, all_yrs_set)
  
    return chunks, count


def define_regions_and_rank_new(Corr_Coeff, lat_grid, lon_grid, ex):
    #%%
    '''
	takes Corr Coeffs and defines regions by strength

	return A: the matrix whichs entries correspond to region. 1 = strongest, 2 = second strongest...
    '''
#    print('extracting features ...\n')

	
	# initialize arrays:
	# A final return array 
    A = np.ma.copy(Corr_Coeff)
#    A = np.ma.zeros(Corr_Coeff.shape)
	#========================================
	# STEP 1: mask nodes which were never significantly correlatated to index (= count=0)
	#========================================
	
	#========================================
	# STEP 2: define neighbors for everey node which passed Step 1
	#========================================

    indices_not_masked = np.where(A.mask==False)[0].tolist()

    lo = lon_grid.shape[0]
    la = lat_grid.shape[0]
	
	# create list of potential neighbors:
    N_pot=[[] for i in range(A.shape[0])]

	#=====================
	# Criteria 1: must bei geographical neighbors:
    n_between = ex['prec_reg_max_d']
	#=====================
    for i in indices_not_masked:
        neighb = []
        def find_neighboors(i, lo):
            n = []	
    
            col_i= i%lo
            row_i = i//lo
    
    		# knoten links oben
            if i==0:	
                n= n+[lo-1, i+1, lo ]
    
    		# knoten rechts oben	
            elif i== lo-1:
                n= n+[i-1, 0, i+lo]
    
    		# knoten links unten
            elif i==(la-1)*lo:
                n= n+ [i+lo-1, i+1, i-lo]
    
    		# knoten rechts unten
            elif i == la*lo-1:
                n= n+ [i-1, i-lo+1, i-lo]
    
    		# erste zeile
            elif i<lo:
                n= n+[i-1, i+1, i+lo]
    	
    		# letzte zeile:
            elif i>la*lo-1:
                n= n+[i-1, i+1, i-lo]
    	
    		# erste spalte
            elif col_i==0:
                n= n+[i+lo-1, i+1, i-lo, i+lo]
    	
    		# letzt spalte
            elif col_i ==lo-1:
                n= n+[i-1, i-lo+1, i-lo, i+lo]
    	
    		# nichts davon
            else:
                n = n+[i-1, i+1, i-lo, i+lo]
            return n
        
        for t in range(n_between+1):
            direct_n = find_neighboors(i, lo)
            if t == 0:
                neighb.append(direct_n)
            if t == 1:
                for n in direct_n:
                    ind_n = find_neighboors(n, lo)
                    neighb.append(ind_n)
        n = list(set(flatten(neighb)))
        if i in n:
            n.remove(i)
        
	
	#=====================
	# Criteria 2: must be all at least once be significanlty correlated 
	#=====================	
        m =[]
        for j in n:
            if j in indices_not_masked:
                m = m+[j]
		
		# now m contains the potential neighbors of gridpoint i

	
	#=====================	
	# Criteria 3: sign must be the same for each step 
	#=====================				
        l=[]
	
        cc_i = A.data[i]
        cc_i_sign = np.sign(cc_i)
		
	
        for k in m:
            cc_k = A.data[k]
            cc_k_sign = np.sign(cc_k)
		

            if cc_i_sign *cc_k_sign == 1:
                l = l +[k]

            else:
                l = l
			
            if len(l)==0:
                l =[]
                A.mask[i]=True	
    			
            elif i not in l: 
                l = l + [i]	
		
		
            N_pot[i]=N_pot[i] + l	



	#========================================	
	# STEP 3: merge overlapping set of neighbors
	#========================================
    Regions = merge_neighbors(N_pot)
	
	#========================================
	# STEP 4: assign a value to each region
	#========================================
	

	# 2) combine 1A+1B 
    B = np.abs(A)
	
	# 3) calculate the area size of each region	
	
    Area =  [[] for i in range(len(Regions))]
	
    for i in range(len(Regions)):
        indices = np.array(list(Regions[i]))
        indices_lat_position = indices//lo
        lat_nodes = lat_grid[indices_lat_position[:]]
        cos_nodes = np.cos(np.deg2rad(lat_nodes))		
		
        area_i = [np.sum(cos_nodes)]
        Area[i]= Area[i]+area_i
	
	#---------------------------------------
	# OPTIONAL: Exclude regions which only consist of less than n nodes
	# 3a)
	#---------------------------------------	
	
    # keep only regions which are larger then the mean size of the regions
    if ex['min_n_gc'] == 'mean':
        n_nodes = int(np.mean([len(r) for r in Regions]))
    else:
        n_nodes = ex['min_n_gc']
    
    R=[]
    Ar=[]
    for i in range(len(Regions)):
        if len(Regions[i])>=n_nodes:
            R.append(Regions[i])
            Ar.append(Area[i])
	
    Regions = R
    Area = Ar	
	
	
	
	# 4) calcualte region value:
	
    C = np.zeros(len(Regions))
	
    Area = np.array(Area)
    for i in range(len(Regions)):
        C[i]=Area[i]*np.mean(B[list(Regions[i])])


	
	
	# mask out those nodes which didnot fullfill the neighborhood criterias
    A.mask[A==0] = True	
		
		
	#========================================
	# STEP 5: rank regions by region value
	#========================================
	
	# rank indices of Regions starting with strongest:
    sorted_region_strength = np.argsort(C)[::-1]
	
	# give ranking number
	# 1 = strongest..
	# 2 = second strongest
    
    # create clean array
    Regions_lag_i = np.zeros(A.data.shape)
    for i in range(len(Regions)):
        j = list(sorted_region_strength)[i]
        Regions_lag_i[list(Regions[j])]=i+1
    
    Regions_lag_i = np.array(Regions_lag_i, dtype=int)
    #%%
    return Regions_lag_i



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

def Ev_timeseries(xarray, threshold, ex):  
    #%%
    tfreq_RVts = pd.Timedelta((xarray.time[1]-xarray.time[0]).values)
    min_dur = ex['min_dur'] ; max_break = ex['max_break']  + 1
    min_dur = pd.Timedelta(min_dur, 'd') / tfreq_RVts
    max_break = pd.Timedelta(max_break, 'd') / tfreq_RVts
    Ev_ts = xarray.where( xarray.values > threshold) 
    Ev_dates = Ev_ts.dropna(how='all', dim='time').time
    event_idx = [list(xarray.time.values).index(E) for E in Ev_dates.values]
    peak_o_thresh = np.zeros((Ev_ts.size))
    ev_num = 1
    # group events inter event time less than max_break
    for i in range(Ev_dates.size):
        if i < Ev_dates.size-1:
            curr_ev = event_idx[i]
            next_ev = event_idx[i+1]
        elif i == Ev_dates.size-1:
            curr_ev = event_idx[i]
            next_ev = event_idx[i-1]
                 
        if abs(next_ev - curr_ev) <= max_break:
            peak_o_thresh[curr_ev] = ev_num
        elif abs(next_ev - curr_ev) > max_break:
            peak_o_thresh[curr_ev] = ev_num
            ev_num += 1

    # remove events which are too short
    for i in np.arange(1, max(peak_o_thresh)+1):
        No_ev_ind = np.where(peak_o_thresh==i)[0]
        # if shorter then min_dur, then not counted as event
        if No_ev_ind.size < min_dur:
            peak_o_thresh[No_ev_ind] = 0
#             all other events are subtracted one
#            peak_o_thresh[np.where(peak_o_thresh > i)[0]] -= 1
    if np.sum(peak_o_thresh) < 1:
        Events = Ev_ts.where(peak_o_thresh > 0 ).dropna(how='all', dim='time').time
        pass
    else:
        peak_o_thresh[peak_o_thresh == 0] = np.nan
        Ev_labels = xr.DataArray(peak_o_thresh, coords=[Ev_ts.coords['time']])
        Ev_dates = Ev_labels.dropna(how='all', dim='time').time
        
#        Ev_dates = Ev_ts.time.copy()     
#        Ev_dates['Ev_label'] = Ev_labels    
#        Ev_dates = Ev_dates.groupby('Ev_label').max().values
#        Ev_dates.sort()
        Events = xarray.sel(time=Ev_dates)
    
    #%%
    return Events

def timeseries_tofit_bins(xarray, ex):
    datetime = pd.to_datetime(xarray['time'].values)
    one_yr = datetime.where(datetime.year == datetime.year[0]).dropna(how='any')
    
    seldays_pp = pd.DatetimeIndex(start=one_yr[0], end=one_yr[-1], 
                                freq=(datetime[1] - datetime[0]))
    end_day = one_yr.max() 
    # after time averaging over 'tfreq' number of days, you want that each year 
    # consists of the same day. For this to be true, you need to make sure that
    # the selday_pp period exactly fits in a integer multiple of 'tfreq'
    temporal_freq = np.timedelta64(ex['tfreq'], 'D') 
    fit_steps_yr = (end_day - seldays_pp.min() ) / temporal_freq
    # line below: The +1 = include day 1 in counting
    start_day = (end_day - (temporal_freq * np.round(fit_steps_yr, decimals=0))) \
                + np.timedelta64(1, 'D') 
    
    def make_datestr_2(datetime, start_yr):
        breakyr = datetime.year.max()
        datesstr = [str(date).split('.', 1)[0] for date in start_yr.values]
        nyears = (datetime.year[-1] - datetime.year[0])+1
        startday = start_yr[0].strftime('%Y-%m-%dT%H:%M:%S')
        endday = start_yr[-1].strftime('%Y-%m-%dT%H:%M:%S')
        firstyear = startday[:4]
        datesdt = start_yr
        def plusyearnoleap(curr_yr, startday, endday, incr):
            startday = startday.replace(firstyear, str(curr_yr+incr))
            endday = endday.replace(firstyear, str(curr_yr+incr))
            next_yr = pd.DatetimeIndex(start=startday, end=endday, 
                            freq=(datetime[1] - datetime[0]))
            # excluding leap year again
            noleapdays = (((next_yr.month==2) & (next_yr.day==29))==False)
            next_yr = next_yr[noleapdays].dropna(how='all')
            return next_yr
        
        for yr in range(0,nyears-1):
            curr_yr = yr+datetime.year[0]
            next_yr = plusyearnoleap(curr_yr, startday, endday, 1)
            datesdt = np.append(datesdt, next_yr)
#            print(len(next_yr))
#            nextstr = [str(date).split('.', 1)[0] for date in next_yr.values]
#            datesstr = datesstr + nextstr
#            print(nextstr[0])
            
            upd_start_yr = plusyearnoleap(next_yr.year[0], startday, endday, 1)

            if next_yr.year[0] == breakyr:
                break
        datesdt = pd.to_datetime(datesdt)
        return datesdt, upd_start_yr
    
    start_yr = pd.DatetimeIndex(start=start_day, end=end_day, 
                                freq=(datetime[1] - datetime[0]))
    # exluding leap year from cdo select string
    noleapdays = (((start_yr.month==2) & (start_yr.day==29))==False)
    start_yr = start_yr[noleapdays].dropna(how='all')
    datesdt, next_yr = make_datestr_2(datetime, start_yr)
    months = dict( {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',
                         8:'aug',9:'sep',10:'okt',11:'nov',12:'dec' } )
    startdatestr = '{} {}'.format(start_day.day, months[start_day.month])
    enddatestr   = '{} {}'.format(end_day.day, months[end_day.month])
    print('adjusted time series to fit bins: \nFrom {} to {}'.format(
                startdatestr, enddatestr))
    adj_array = xarray.sel(time=datesdt)
    return adj_array, datesdt
    

def time_mean_bins(xarray, ex):
    datetime = pd.to_datetime(xarray['time'].values)
    one_yr = datetime.where(datetime.year == datetime.year[0]).dropna(how='any')
    
    if one_yr.size % ex['tfreq'] != 0:
        possible = []
        for i in np.arange(1,20):
            if 214%i == 0:
                possible.append(i)
        print('Error: stepsize {} does not fit in one year\n '
                         ' supply an integer that fits {}'.format(
                             ex['tfreq'], one_yr.size))   
        print('\n Stepsize that do fit are {}'.format(possible))
        print('\n Will shorten the \'subyear\', so that the temporal'
              ' frequency fits in one year')
        xarray, datetime = timeseries_tofit_bins(xarray, ex)
        one_yr = datetime.where(datetime.year == datetime.year[0]).dropna(how='any')
          
    else:
        pass
    fit_steps_yr = (one_yr.size)  / ex['tfreq']
    bins = list(np.repeat(np.arange(0, fit_steps_yr), ex['tfreq']))
    n_years = (datetime.year[-1] - datetime.year[0]) + 1
    for y in np.arange(1, n_years):
        x = np.repeat(np.arange(0, fit_steps_yr), ex['tfreq'])
        x = x + fit_steps_yr * y
        [bins.append(i) for i in x]
    label_bins = xr.DataArray(bins, [xarray.coords['time'][:]], name='time')
    label_dates = xr.DataArray(xarray.time.values, [xarray.coords['time'][:]], name='time')
    xarray['bins'] = label_bins
    xarray['time_dates'] = label_dates
    xarray = xarray.set_index(time=['bins','time_dates'])
    
    half_step = ex['tfreq']/2.
    newidx = np.arange(half_step, datetime.size, ex['tfreq'], dtype=int)
    newdate = label_dates[newidx]
    

    group_bins = xarray.groupby('bins').mean(dim='time', keep_attrs=True)
    group_bins['bins'] = newdate.values
    dates = pd.to_datetime(newdate.values)
    return group_bins.rename({'bins' : 'time'}), dates

def expand_times_for_lags(datetime, ex):
    expanded_time = []
    for yr in set(datetime.year):
        one_yr = datetime.where(datetime.year == yr).dropna(how='any')
        start_mcK = one_yr[0]
        #start day shifted half a time step
        half_step = ex['tfreq']/2.
#        origshift = np.arange(half_step, datetime.size, ex['tfreq'], dtype=int)
        start_mcK = start_mcK - np.timedelta64(int(half_step+0.49), 'D')
        last_day = '{}{} {}:00:00'.format(yr, ex['senddate'][4:], datetime[0].hour)
        end_mcK   = pd.to_datetime(last_day)
#        adj_year = pd.DatetimeIndex(start=start_mcK, end=end_mcK, 
#                                    freq=(datetime[1] - datetime[0]), 
#                                    closed = None).values
        steps = len(one_yr)
        shift_start = start_mcK - (steps) * np.timedelta64(ex['tfreq'], 'D')
        adj_year = pd.DatetimeIndex(start=shift_start, end=end_mcK, 
                                    freq=pd.Timedelta( '1 days'), 
                                    closed = None).values
        [expanded_time.append(date) for date in adj_year]
    
    return pd.to_datetime(expanded_time)

def make_datestr(dates, ex, startyr, endyr, lpyr=False):
    
    sstartdate = str(startyr) + '-' + ex['startperiod']
    senddate   = str(startyr) + '-' + ex['endperiod']
    
    start_yr = pd.DatetimeIndex(start=sstartdate, end=senddate, 
                                freq=(dates[1] - dates[0]))
    if lpyr==True and calendar.isleap(startyr):
        start_yr -= pd.Timedelta( '1 days')
    else:
        pass
    breakyr = endyr
    datesstr = [str(date).split('.', 1)[0] for date in start_yr.values]
    nyears = (endyr - startyr)+1
    startday = start_yr[0].strftime('%Y-%m-%dT%H:%M:%S')
    endday = start_yr[-1].strftime('%Y-%m-%dT%H:%M:%S')
    firstyear = startday[:4]
    def plusyearnoleap(curr_yr, startday, endday, incr):
        startday = startday.replace(firstyear, str(curr_yr+incr))
        endday = endday.replace(firstyear, str(curr_yr+incr))
        
        next_yr = pd.DatetimeIndex(start=startday, end=endday, 
                        freq=(dates[1] - dates[0]))
        if lpyr==True and calendar.isleap(curr_yr+incr):
            next_yr -= pd.Timedelta( '1 days')
        elif lpyr == False:
            # excluding leap year again
            noleapdays = (((next_yr.month==2) & (next_yr.day==29))==False)
            next_yr = next_yr[noleapdays].dropna(how='all')
        return next_yr
    

    for yr in range(0,nyears):
        curr_yr = yr+startyr
        next_yr = plusyearnoleap(curr_yr, startday, endday, 1)
        nextstr = [str(date).split('.', 1)[0] for date in next_yr.values]
        datesstr = datesstr + nextstr

        if next_yr.year[0] == breakyr:
            break
    datesmcK = pd.to_datetime(datesstr)
    return datesmcK

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
    # Area weighted     
    coslat = np.cos(np.deg2rad(xarray.coords['latitude'].values)).clip(0., 1.)
    area_weights = np.tile(np.sqrt(coslat)[..., np.newaxis],(1,xarray.longitude.size))
    xarray.values = xarray.values * area_weights 
    return xarray
    
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


def rolling_mean_xr(xarray, win):
    closed = int(win/2)
    flatarray = xarray.values.flatten()
    ext_array = np.insert(flatarray, 0, flatarray[-closed:])
    ext_array = np.insert(ext_array, 0, flatarray[:closed])
    
    df = pd.DataFrame(ext_array)
    std = xarray.where(xarray.values!=0.).std().values
#    scipy.signal.gaussian(win, std)
    rollmean = df.rolling(win, center=True, 
                          win_type='gaussian').mean(std=std).dropna()
    
    # replace values with smoothened values
    new_xarray = xarray.copy()
    new_values = np.reshape(rollmean.squeeze().values, xarray.shape)
    # ensure LSM mask
    mask = np.array((xarray.values!=0.),dtype=int)
    new_xarray.values = (new_values * mask)

    return new_xarray

def rolling_mean_time(xarray_or_file, ex):
    #%%
#    xarray_or_file = Prec_reg
#    array = np.zeros(60)
#    array[-30:] = 1
#    xarray_or_file = xr.DataArray(array, coords=[np.arange(60)], dims=['time'])
    
    if type(xarray_or_file) == str:
        file_path = os.path.join(ex['path_pp'], xarray_or_file)        
        ds = xr.open_dataset(file_path, decode_cf=True, decode_coords=True, decode_times=False)
        ds_rollingmean = ds.rolling(time=ex['rollingmean'], center=True, min_periods=1).mean()
        
        new_fname = 'rm{}_'.format(ex['rollingmean']) + xarray_or_file
        file_path = os.path.join(ex['path_pp'], new_fname)
        ds_rollingmean.to_netcdf(file_path, mode='w')
        print('saved netcdf as {}'.format(new_fname))
        print('functions returning None')
        xr_rolling_mean = None
    else:
        # Taking rolling window mean with the closing on the right, 
        # meaning that we are taking the mean over the past at the index/label 
        xr_rolling_mean = xarray_or_file.rolling(time=ex['rollingmean'], center=True, 
                                                 min_periods=1).mean()
        lat = 35
        lon = 360-90
        
        def find_nearest(array, value):
            idx = (np.abs(array - value)).argmin()
            return int(idx)
        
        lat_idx = find_nearest(xarray_or_file['latitude'], lat)
        lon_idx = find_nearest(xarray_or_file['longitude'], lon)
        
        singlegc = xarray_or_file.isel(latitude=lat_idx, 
                                      longitude=lon_idx) 
        
        year = 2012
        singlegc_oneyr = singlegc.where(singlegc.time.dt.year == year).dropna(dim='time', how='all')
        dates = pd.to_datetime(singlegc_oneyr.time.values)
        plt.figure(figsize=(10,6))
        plt.plot(dates, singlegc_oneyr.to_array().squeeze())

        singlegc = xr_rolling_mean.isel(latitude=lat_idx, 
                                      longitude=lon_idx) 
        singlegc_oneyr = singlegc.where(singlegc.time.dt.year == year).dropna(dim='time', how='all')
        plt.plot(dates, singlegc_oneyr.to_array().squeeze())
    #%%
    return xr_rolling_mean

def to_datesmcK(datesmcK, to_hour, from_hour):
    
    dt_hours = from_hour - to_hour
    matchdaysmcK = datesmcK + pd.Timedelta(int(dt_hours), unit='h')
    return xr.DataArray(matchdaysmcK, dims=['time'])


def find_region(data, region='Mckinnonplot'):
    if region == 'Mckinnonplot':
        west_lon = -240; east_lon = -40; south_lat = -10; north_lat = 80

    elif region ==  'U.S.soil':
        west_lon = -130; east_lon = -60; south_lat = 0; north_lat = 60
    elif region ==  'U.S.cluster':
        west_lon = -100; east_lon = -70; south_lat = 20; north_lat = 50
    elif region ==  'PEPrectangle':
        west_lon = -215; east_lon = -125; south_lat = 19; north_lat = 50
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


def cross_correlation_patterns(full_timeserie, pattern):
#    full_timeserie = var_test_mcK
#    pattern = ds_mcK['pattern'].sel(lag=lag)
    mask = np.ma.make_mask(np.isnan(pattern.values)==False)
    
    n_time = full_timeserie.time.size
    n_space = pattern.size
    

#    mask_pattern = np.tile(mask_pattern, (n_time,1))
    # select only gridcells where there is not a nan
    full_ts = np.nan_to_num(np.reshape( full_timeserie.values, (n_time, n_space) ))
    pattern = np.nan_to_num(np.reshape( pattern.values, (n_space) ))

    mask_pattern = np.reshape( mask, (n_space) )
    full_ts = full_ts[:,mask_pattern]
    pattern = pattern[mask_pattern]
    
    crosscorr = np.zeros( (n_time) )
    spatcov   = np.zeros( (n_time) )
    covself   = np.zeros( (n_time) )
#    corrself  = np.zeros( (n_time) )
    for t in range(n_time):
        # Corr(X,Y) = cov(X,Y) / ( std(X)*std(Y) )
        # cov(X,Y) = E( (x_i - mu_x) * (y_i - mu_y) )
        crosscorr[t] = np.correlate(full_ts[t], pattern)
        M = np.stack( (full_ts[t], pattern) )
        spatcov[t] = np.cov(M)[0,1] #/ (np.sqrt(np.cov(M)[0,0]) * np.sqrt(np.cov(M)[1,1]))
#        sqrt( Var(X) ) = sigma_x = std(X)
#        spatcov[t] = np.cov(M)[0,1] / (np.std(full_ts[t]) * np.std(pattern))        
        covself[t] = np.mean( (full_ts[t] - np.mean(full_ts[t])) * (pattern - np.mean(pattern)) )
#        corrself[t] = covself[t] / (np.std(full_ts[t]) * np.std(pattern))
    dates_test = full_timeserie.time
#    corrself = xr.DataArray(corrself, coords=[dates_test.values], dims=['time'])
    
#    # standardize
#    corrself -= corrself.mean(dim='time')
    
    # cov xarray
    covself = xr.DataArray(covself, coords=[dates_test.values], dims=['time'])
    return covself

def kornshell_with_input(args, ex):
#    stopped working for cdo commands
    '''some kornshell with input '''
#    args = [anom]
    import os
    import subprocess
    cwd = os.getcwd()
    # Writing the bash script:
    new_bash_script = os.path.join(cwd, "bash_script.sh")
#    arg_5d_mean = 'cdo timselmean,5 {} {}'.format(infile, outfile)
    #arg1 = 'ncea -d latitude,59.0,84.0 -d longitude,-95,-10 {} {}'.format(infile, outfile)
    
    bash_and_args = [new_bash_script]
    [bash_and_args.append(arg) for arg in args]
    with open(new_bash_script, "w") as file:
        file.write("#!/bin/sh\n")
        file.write("echo bash script output\n")
        for cmd in range(len(args)):

            print(args[cmd])
            file.write("${}\n".format(cmd+1)) 
    p = subprocess.Popen(bash_and_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                         stderr=subprocess.STDOUT)
                         
    out = p.communicate()
    print(out[0].decode())
    return


def grouping_regions_similar_coords(l_ds, ex, grouping = 'group_accros_tests_single_lag', eps=10):
    '''Regions with similar coordinates are grouped together.
    lower eps indicate higher density necessary to form a cluster. 
    If eps too high, no high density is required and originally similar regions 
    are assigned into different clusters. If eps is too low, such an high density is required that
    it will also start to cluster together regions with similar coordinates.
    '''
    #%%
#    grouping = 'group_accros_tests_single_lag'
#    grouping =  'group_across_test_and_lags'
    # Precursor Regions Dimensions
    all_lags_in_exp = l_ds[0]['pat_num_CPPA'].sel(lag=ex['lags']).lag.values
    lags_ind        = np.reshape(np.argwhere(all_lags_in_exp == ex['lags']), -1)
    PRECURSOR_DATA = np.array([data['pat_num_CPPA'].values for data in l_ds])
    PRECURSOR_DATA = PRECURSOR_DATA[:,lags_ind]
    PRECURSOR_LONGITIUDE = l_ds[0]['pat_num_CPPA'].longitude.values
    PRECURSOR_LATITUDE = l_ds[0]['pat_num_CPPA'].latitude.values
    PRECURSOR_LAGS = l_ds[0]['pat_num_CPPA'].sel(lag=ex['lags']).lag.values
    PRECURSOR_YEARS = np.arange(ex['startyear'],ex['endyear']+1E-9)#[np.unique(test['RV'].time.dt.year) for test in ex['train_test_list'][:][1]][0]
    
    # Precursor Grid (to calculate precursor region centre coordinate)
    PRECURSOR_GRID = np.zeros((len(PRECURSOR_LATITUDE), len(PRECURSOR_LONGITIUDE), 2))
    PRECURSOR_GRID[..., 0], PRECURSOR_GRID[..., 1] = np.meshgrid(PRECURSOR_LONGITIUDE, PRECURSOR_LATITUDE)
    

    precursor_coordinates = []
    
    # Array Containing Precursor Region Indices for each YEAR 
    precursor_indices = np.empty((len(PRECURSOR_YEARS),
                                  len(PRECURSOR_LAGS),
                                  len(PRECURSOR_LATITUDE),
                                  len(PRECURSOR_LONGITIUDE)),
                                 np.float32)
    precursor_indices[:,:,:,:] = np.nan
    precursor_indices_new = precursor_indices.copy()
    
    ex['uniq_regs_lag'] = np.zeros(len(PRECURSOR_LAGS))
    # Array Containing Precursor Region Weights for each YEAR and LAG
    for lag_idx, lag in enumerate(PRECURSOR_LAGS):
        
        indices_across_yrs = np.squeeze(PRECURSOR_DATA[:,lag_idx,:,:])
        
        
        if grouping == 'group_accros_tests_single_lag':
            # regions are given same number across test set, not accross all lags
            precursor_coordinates = []
    
    #    precursor_weights = np.zeros_like(precursor_indices, np.float32)
        min_samples = []
        for test_idx, year in enumerate(PRECURSOR_YEARS):
            indices = indices_across_yrs[test_idx, :, :]
            min_samples.append( np.nanmax(indices) )
            for region_idx in np.unique(indices[~np.isnan(indices)]):
            # evaluate regions
    #            plt.figure()
    #            plt.imshow(indices == region_idx)
                precursor_indices[test_idx, lag_idx, indices == region_idx] = int(region_idx)
                precursor_coordinates.append((
                        [test_idx, lag_idx, int(region_idx)], PRECURSOR_GRID[indices == region_idx].mean(0)))
    
        # Group Similar Precursor Regions Together across years for same lag
        precursor_coordinates_index = np.array([index for index, coord in precursor_coordinates])
        precursor_coordinates_coord = np.array([coord for index, coord in precursor_coordinates])
        
        if grouping == 'group_accros_tests_single_lag':
            # Higher min_samples or lower eps indicate higher density necessary to form a cluster.
            min_s = np.nanmin(min_samples)
            precursor_coordinates_group = DBSCAN(min_samples=min_s, eps=eps).fit_predict(precursor_coordinates_coord) + 2
            
            for (year_idx, lag_idx, region_idx), group in zip(precursor_coordinates_index, precursor_coordinates_group):
                precursor_indices_new[year_idx, lag_idx, precursor_indices[year_idx, lag_idx] == region_idx] = group

    
    if grouping == 'group_across_test_and_lags':
        # Group Similar Precursor Regions Together
        precursor_coordinates_index = np.array([index for index, coord in precursor_coordinates])
        precursor_coordinates_coord = np.array([coord for index, coord in precursor_coordinates])

        precursor_coordinates_group = DBSCAN(eps=eps).fit_predict(precursor_coordinates_coord) + 2
        
        
        precursor_indices_new = np.zeros_like(precursor_indices)
        for (year_idx, lag_idx, region_idx), group in zip(precursor_coordinates_index, precursor_coordinates_group):
    #        print(year_idx, lag_idx, region_idx, group)
            precursor_indices_new[year_idx, lag_idx, precursor_indices[year_idx, lag_idx] == region_idx] = group
    
    
    # replace values in PRECURSOR_DATA
    PRECURSOR_DATA[:,:,:,:] = precursor_indices_new[:,:,:,:]
    ex['uniq_regs_lag'][lag_idx] = max(np.unique(precursor_coordinates_group) )
    
    
    l_ds_new = []
    for test_idx, year in enumerate(PRECURSOR_YEARS):
        single_ds = l_ds[test_idx]
        pattern   = single_ds['pat_num_CPPA']
        
        pattern.values = PRECURSOR_DATA[test_idx]
#        # set the rest to nan
#        pattern = pattern.where(pattern.values != 0.)
        single_ds['pat_num_CPPA_clust'] = pattern
    #    print(test_idx)
    #    plt.figure()
    #    single_ds['pat_num_CPPA'][0].plot()
        # overwrite ds
        l_ds_new.append( single_ds )
    #    plt.figure()
    #    l_ds_new[-1]['pat_num_CPPA'][0].plot()
    
    #plt.figure()
    ex['max_N_regs'] = np.nanmax(PRECURSOR_DATA)
    #%%
    return l_ds_new



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
    ax.coastlines()
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
    
    
def plot_events_validation(pred1, pred2, obs, pt1, pt2, othreshold, test_year=None):
    #%%
#    pred1 = crosscorr_Sem
#    pred2 = crosscorr_mcK
#    obs = RV_ts_test
#    pt1 = Prec_threshold_Sem
#    pt2 = Prec_threshold_mcK
#    othreshold = ex['hotdaythres']
#    test_year = int(crosscorr_Sem.time.dt.year[0])

    
    def predyear(pred, obs):
        if str(type(test_year)) == "<class 'numpy.int64'>" or str(type(test_year)) == "<class 'int'>":
            predyear = pred.where(pred.time.dt.year == test_year).dropna(dim='time', how='any')
            obsyear  = obs.where(obs.time.dt.year == test_year).dropna(dim='time', how='any')
            predyear['time'] = obsyear.time
        elif type(test_year) == type(['list']):
            years_in_obs = list(obs.time.dt.year.values)
            test_years = [i for i in range(len(years_in_obs)) if years_in_obs[i] in test_year]
            # Warning this is wrong #!!!
            predyear = pred.isel(time=test_years)
            obsyear = obs.isel(time=test_years)
        else:
            predyear = pred
            predyear['time'] = obs.time
            obsyear = obs
        return predyear, obsyear
        
    predyear1, obsyear = predyear(pred1, obs)
    predyear2, obsyear = predyear(pred2, obs)

    eventdays = obsyear.where( obsyear.values > othreshold ) 
    eventdays = eventdays.dropna(how='all', dim='time').time
    preddays = predyear1.where(predyear1.values > pt1)
    preddays1 = preddays.dropna(how='all', dim='time').time
    preddays = predyear2.where(predyear2.values > pt2)
    preddays2 = preddays.dropna(how='all', dim='time').time
#    # standardize obsyear
#    othreshold -= obsyear.mean(dim='time').values
#    obsyear    -= obsyear.mean(dim='time')
#    
#    # standardize predyear(s)
#    pthreshold -= predyear.mean(dim='time').values
#    predyear    -= predyear.mean(dim='time')
      
    TP1 = [day for day in preddays1.time.values if day in list(eventdays.values)]
    TP2 = [day for day in preddays2.time.values if day in list(eventdays.values)]
#    pthreshold = ((pthreshold - pred1.mean()) * obsyear.std()/predyear.std()).values
#    predyear = (predyear) * obsyear.std()/predyear.std() 
    plt.figure(figsize = (10,5))
    ax1 = plt.subplot(311)
    ax1.plot(pd.to_datetime(obsyear.time.values), obsyear, label='observed',
             color = 'blue')
    ax1.axhline(y=othreshold, color='blue')
    for days in eventdays.time.values:
        ax1.axvline(x=pd.to_datetime(days), color='blue', alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(312)
    ax2.plot(pd.to_datetime(obsyear.time.values),predyear1, label='Sem pattern ts',
             color='red')
    ax2.axhline(y=pt1, color='red')
    for days in preddays1.time.values:
        ax2.axvline(x=pd.to_datetime(days), color='red', alpha=0.3)
    for days in pd.to_datetime(TP1):
        ax2.axvline(x=pd.to_datetime(days), color='green', alpha=1.)
    ax2.legend()
    # second prediction
    ax3 = plt.subplot(313)
    ax3.plot(pd.to_datetime(obsyear.time.values),predyear2, label='mcK pattern ts',
             color='red')
    ax3.axhline(y=pt2, color='red')
    for days in preddays2.time.values:
        ax3.axvline(x=pd.to_datetime(days), color='red', alpha=0.3)
    for days in pd.to_datetime(TP2):
        ax3.axvline(x=pd.to_datetime(days), color='green', alpha=1.)
    ax3.legend()


def plot_oneyr_events(xarray, ex, test_year, folder, saving=False):
    #%%
    if ex['mcKthres'] == 'mcKthres':
        # binary time serie when T95 exceeds 1 std
        threshold = xarray.mean(dim='time').values + xarray.std().values
    else:
        percentile = ex['mcKthres']
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
        datesRV_mcK = make_datestr(datesmcK, ex, 1982, 2015)
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

def plotting_wrapper(plotarr, ex, filename=None,  kwrgs=None):
#    map_proj = ccrs.Miller(central_longitude=240)  
    folder_name = os.path.join(ex['figpathbase'], ex['exp_folder'])
    if os.path.isdir(folder_name) != True : 
        os.makedirs(folder_name)

    if kwrgs == None:
        kwrgs = dict( {'title' : plotarr.name, 'clevels' : 'notdefault', 'steps':17,
                        'vmin' : -3*plotarr.std().values, 'vmax' : 3*plotarr.std().values, 
                       'cmap' : plt.cm.RdBu_r, 'column' : 1, 'subtitles' : None} )
    else:
        kwrgs = kwrgs
        kwrgs['title'] = plotarr.attrs['title']
        
    if filename != None:
        file_name = os.path.join(ex['figpathbase'], filename)
        kwrgs['savefig'] = True
    else:
        kwrgs['savefig'] = False
        file_name = 'Users/semvijverberg/Downloads/test.png'
    finalfigure(plotarr, file_name, kwrgs)
    

def finalfigure(xrdata, file_name, kwrgs):
    #%%
    map_proj = ccrs.PlateCarree(central_longitude=220)  
    lons = xrdata.longitude.values
    lats = xrdata.latitude.values
    strvars = [' {} '.format(var) for var in list(xrdata.dims)]
    var = [var for var in strvars if var not in ' longitude latitude '][0] 
    var = var.replace(' ', '')
    g = xr.plot.FacetGrid(xrdata, col=var, col_wrap=kwrgs['column'], sharex=True,
                      sharey=True, subplot_kws={'projection': map_proj},
                      aspect= (xrdata.longitude.size) / xrdata.latitude.size, size=3.5)
    figwidth = g.fig.get_figwidth() ; figheight = g.fig.get_figheight()

    lon_tick = xrdata.longitude.values
#    lon_tick[lon_tick > 180] -= 360
    
    longitude_labels = np.linspace(np.min(lon_tick), np.max(lon_tick), 6, dtype=int)
    longitude_labels = np.array(sorted(list(set(np.round(longitude_labels, -1)))))

#    longitude_labels = np.concatenate([ longitude_labels, [longitude_labels[-1]], [180]])
#    longitude_labels = [-150,  -70,    0,   70,  140, 140]
    latitude_labels = np.linspace(xrdata.latitude.min(), xrdata.latitude.max(), 4, dtype=int)
    latitude_labels = sorted(list(set(np.round(latitude_labels, -1))))
    
    g.set_ticks(max_xticks=5, max_yticks=5, fontsize='small')
    g.set_xlabels(label=[str(el) for el in longitude_labels])

    
    if kwrgs['clevels'] == 'default':
        vmin = np.round(float(xrdata.min())-0.01,decimals=2) ; vmax = np.round(float(xrdata.max())+0.01,decimals=2)
        clevels = np.linspace(-max(abs(vmin),vmax),max(abs(vmin),vmax),17) # choose uneven number for # steps
    else:
        vmin=kwrgs['vmin']
        vmax=kwrgs['vmax']
        
        clevels = np.linspace(vmin, vmax,kwrgs['steps'])

    cmap = kwrgs['cmap']
    
    n_plots = xrdata[var].size
    for n_ax in np.arange(0,n_plots):
        ax = g.axes.flatten()[n_ax]
#        print(n_ax)
        plotdata = extend_longitude(xrdata[n_ax]).squeeze().drop('ds')
        im = plotdata.plot.pcolormesh(ax=ax, cmap=cmap,
                               transform=ccrs.PlateCarree(),
                               subplot_kws={'projection': map_proj},
                                levels=clevels, add_colorbar=False)
        ax.coastlines(color='black', alpha=0.5)
        
        ax.set_extent([lons[0], lons[-1], lats[0], lats[-1]], ccrs.PlateCarree())
        if kwrgs['subtitles'] == None:
            pass
        else:
            fontdict = dict({'fontsize'     : 18,
                             'fontweight'   : 'bold'})
            ax.set_title(kwrgs['subtitles'][n_ax], fontdict=fontdict, loc='center')
#        lons = [-5.8, -5.8, -5.5, -5.5]
#        lats = [50.27, 50.48, 50.48, 50.27]
        lons_sq = [-215, -215, -125, -125]
        lats_sq = [50, 19, 19, 50]
        ring = LinearRing(list(zip(lons_sq , lats_sq )))
        ax.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='green',
                          linewidth=3.5)
        
        if 'ax_text' in kwrgs.keys():
            ax.text(0.0, 1.01, kwrgs['ax_text'][n_ax],
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes,
            color='black', fontsize=15)
            
        if map_proj.proj4_params['proj'] in ['merc', 'eqc']:
#            print(True)
            ax.set_xticks(longitude_labels[:-1], crs=ccrs.PlateCarree())
            ax.set_xticklabels(longitude_labels[:-1], fontsize=12)
            lon_formatter = cticker.LongitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            
            ax.set_yticks(latitude_labels, crs=ccrs.PlateCarree())
            ax.set_yticklabels(latitude_labels, fontsize=12)
            lat_formatter = cticker.LatitudeFormatter()
            ax.yaxis.set_major_formatter(lat_formatter)
            ax.grid(linewidth=1, color='black', alpha=0.3, linestyle='--')
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            
        else:
            pass
        
    g.fig.text(0.5, 0.99, kwrgs['title'], fontsize=20,
               fontweight='heavy', transform=g.fig.transFigure,
               horizontalalignment='center',verticalalignment='top')
    
    if 'adj_fig_h' in kwrgs.keys():
        g.fig.set_figheight(figheight*kwrgs['adj_fig_h'], forward=True)
    if 'adj_fig_w' in kwrgs.keys():
        g.fig.set_figwidth(figwidth*kwrgs['adj_fig_w'], forward=True)

    if 'cbar_vert' in kwrgs.keys():
        cbar_vert = (figheight/40)/(n_plots*2) + kwrgs['cbar_vert']
    else:
        cbar_vert = (figheight/40)/(n_plots*2)
    if 'cbar_hght' in kwrgs.keys():
        cbar_hght = (figheight/40)/(n_plots*2) + kwrgs['cbar_hght']
    else:
        cbar_hght = (figheight/40)/(n_plots*2)
    if 'wspace' in kwrgs.keys():
        g.fig.subplots_adjust(wspace=kwrgs['wspace'])
    if 'hspace' in kwrgs.keys():
        g.fig.subplots_adjust(hspace=kwrgs['hspace'])
    if 'extend' in kwrgs.keys():
        extend = kwrgs['extend'][0]
    else:
        extend = 'neither'

    cbar_ax = g.fig.add_axes([0.25, cbar_vert, 
                                  0.5, cbar_hght], label='xbar')

    if 'clim' in kwrgs.keys(): #adjust the range of colors shown in cbar
        cnorm = np.linspace(kwrgs['clim'][0],kwrgs['clim'][1],11)
        vmin = kwrgs['clim'][0]
    else:
        cnorm = clevels
    norm = colors.BoundaryNorm(boundaries=cnorm, ncolors=256)
#    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, orientation='horizontal', 
#                 extend=extend, norm=norm)
    cbar = plt.colorbar(im, cbar_ax, cmap=cmap, orientation='horizontal', 
                 extend=extend, norm=norm)
    if 'cticks_center' in kwrgs.keys():
        cbar.set_ticks(clevels + 0.5)
        cbar.set_ticklabels(clevels+1, update_ticks=True)
        cbar.update_ticks()
    
    if 'extend' in kwrgs.keys():
        if kwrgs['extend'][0] == 'min':
            cbar.cmap.set_under(cbar.to_rgba(vmin))
    cbar.set_label(xrdata.attrs['units'], fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    #%%
    if kwrgs['savefig'] != False:
        g.fig.savefig(file_name ,dpi=250, frameon=True)
    
    return


def figure_for_schematic(reg_all_1, chunks, lats, lons, ex):
    #%%
    map_proj = ccrs.PlateCarree(central_longitude=220) 
    regions = np.reshape(reg_all_1, (reg_all_1.shape[0], lats.size, lons.size) )
    name_chnks = [str(chnk) for chnk in chunks]
    regions = xr.DataArray(regions, coords=[name_chnks, lats, lons], 
                           dims=['yrs_out', 'latitude', 'longitude'])
    folder = os.path.join(ex['figpathbase'], ex['CPPA_folder'], 'schematic_fig/')
    
    plots = 9
    subset = np.linspace(0,regions.yrs_out.size-1,plots, dtype=int)
    regions = regions.isel(yrs_out=subset)
    regions  = regions.sel(latitude=slice(60.,0.))
    regions = regions.sel(longitude=slice(160, 250))
    
    if os.path.isdir(folder) != True : os.makedirs(folder)
    for i in range(len(subset))[::int(plots/3)]:
        i = int(i)
        cmap = plt.cm.YlOrBr
        fig = plt.figure(figsize = (14,9))
        ax = plt.subplot(111, projection=map_proj) 
        plotdata = regions.isel(yrs_out=i).where(regions.isel(yrs_out=i) > 0.)
        plotdata.plot.pcolormesh(ax=ax, cmap=cmap, vmin=0, vmax=plots,
                               transform=ccrs.PlateCarree(),
                               subplot_kws={'projection': map_proj},
                                add_colorbar=False, alpha=0.9)
        ax.coastlines(color='black', alpha=0.8, linewidth=2)
#        list_points = np.argwhere(regions.isel(yrs_out=i).values == 1)
#        x_co = regions.isel(yrs_out=i).longitude.values
#        y_co = regions.isel(yrs_out=i).latitude.values
#        for p in list_points:
#            ax.scatter(x_co[p[1]], y_co[p[0]], marker='$1$', color='black', 
#                      s=70, transform=ccrs.PlateCarree())
    #    ax.set_extent([-110, 150,0,80], crs=ccrs.PlateCarree())
#        ax.set_extent([140, 266,0,60])
#        ax.outline_patch.set_visible(False)
#        ax.background_patch.set_visible(False)
        ax.set_title('')
        title = str(plotdata.yrs_out.values) 
        t = ax.text(0.007, 0.01, 
                    'Composite exluding {}'.format(title),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes,
            color='black', fontsize=25, weight='bold')
        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='red'))
        
        fig.savefig(folder+title+'.pdf',  bbox_inches='tight')
#%%
    # final figure
    fig = plt.figure(figsize = (14,9))
    ax = plt.subplot(111, projection=map_proj)
    robustness = np.sum(regions,axis=0)
    xrdata = regions.isel(yrs_out=i).copy()
    xrdata.values = robustness
    plotdata = xrdata.where(xrdata.values > 0.)
    plotdata.plot.pcolormesh(ax=ax, cmap=cmap,
                               transform=ccrs.PlateCarree(), vmin=0, vmax=plots,
                               subplot_kws={'projection': map_proj},
                                add_colorbar=False, alpha=0.5)
#    xrdata = xrdata.where(xrdata.values > 0.8*plots)
    xrdata.plot.contour(ax=ax, 
                               transform=ccrs.PlateCarree(),
                               colors=['purple'], levels=[0., ex['comp_perc'] * plots, plots],
                               subplot_kws={'projection': map_proj},
                               )
    ax.coastlines(color='black', alpha=0.8, linewidth=2)
    list_points = np.argwhere(xrdata.values > 0)
    x_co = xrdata.longitude.values
    y_co = xrdata.latitude.values
#        list_points = list_points - ex['grid_res']/2.
    for p in list_points:
        value = str(int(xrdata.sel(latitude=y_co[p[0]], longitude=x_co[p[1]]).values))
        ax.scatter(x_co[p[1]], y_co[p[0]], marker='${}$'.format(value), color='black', 
                   s=70, transform=ccrs.PlateCarree())
                  
#    ax.outline_patch.set_visible(False)
#    ax.background_patch.set_visible(False)
    ax.set_title('')
    title = 'Sum of incomplete composites'
    t = ax.text(0.006, 0.992, ('Sum over 9 incomplete composites'),
#                            'black contour line shows the gridcell passing the\n'
#                            '\'Composite robustness threshold\''),
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes,
        color='black', fontsize=20)
    t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='green'))
    
    fig.savefig(folder+title+'.pdf', bbox_inches='tight')
    #%%
    return

    
## Filter out outliers
#n_std_kickout = 7
## outliers are now nan, all values that are lower then threshold are kept (==True)
#Prec_reg = Prec_reg.where(abs(Prec_reg.values) < 10*Prec_reg.std(dim='time').values)
#np_arr = np.array(Prec_reg.values)
##find nans
#
#mask_allways_nan = np.product(np.isnan(np_arr), axis=0)
#np_arr[mask_allways_nan] = 0.
#for attempt in range(20):
#    mask_nan = np.isnan(np_arr)
#    prev_values = np.roll(mask_nan[:], 1, axis = 0)
#    np_arr[mask_nan[:]] = np_arr[prev_values]
#    
#    print(np.argwhere(np.isnan(np_arr)).shape)
#
#idx_nan = np.argwhere(mask_nan)
## replace the nan values by the values in the previous step
#Prec_reg.values[idx_nan] = Prec_reg.values[idx_nan - [1, 0, 0]]