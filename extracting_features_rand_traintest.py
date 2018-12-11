#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:40:40 2018

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
import scipy

base_path = "/Users/semvijverberg/surfdrive/Data_ERAint/"
exp_folder = ''
path_raw = os.path.join(base_path, 'input_raw')
path_pp  = os.path.join(base_path, 'input_pp')
if os.path.isdir(path_raw) == False : os.makedirs(path_raw) 
if os.path.isdir(path_pp) == False: os.makedirs(path_pp)
map_proj = ccrs.Miller(central_longitude=240)  


ex = dict(
     {'grid_res'     :       2.5,
     'startyear'    :       1982,
     'endyear'      :       2015,
     'base_path'    :       base_path,
     'path_raw'     :       path_raw,
     'path_pp'      :       path_pp,
     'sstartdate'   :       '1982-06-24',
     'senddate'     :       '1982-08-21',
     'map_proj'     :       map_proj,
     'figpathbase'     :       "/Users/semvijverberg/surfdrive/McKinRepl/T95_sst_NOAA",
     'tfreq'        :       1,
     'RV_name'      :       'T95',
     'name'         :       'sst_NOAA',
     'wghts_accross_lags':  False,
     'wghts_std_anom':      False,
     'splittrainfeat':      False}
     )

if os.path.isdir(ex['figpathbase']) == False: os.makedirs(ex['figpathbase'])

#'Mckinnonplot', 'U.S.', 'U.S.cluster', 'PEPrectangle', 'Pacific', 'Whole', 'Southern'



ex['region'] = 'Whole'
print(ex['region'])

# Load in mckinnon Time series
T95name = 'PEP-T95TimeSeries.txt'
mcKtsfull, datesmcK = func_mcK.read_T95(T95name, ex)
datesmcK_daily = func_mcK.make_datestr(datesmcK, ex)

# Selected Time series of T95 ex['sstartdate'] until ex['senddate']
mcKts = mcKtsfull.sel(time=datesmcK_daily)

# Load in external ncdf
ex['name'] = 'sst_NOAA'
#filename = '{}_1979-2017_2mar_31aug_dt-1days_2.5deg.nc'.format(ex['name'])
filename = '{}_1982-2017_2jan_31aug_dt-1days_2.5deg.nc'.format(ex['name'])
# full globe - full time series
varfullgl = func_mcK.import_array(filename, ex)

# filter out outliers of sst
if ex['name']=='sst':
    varfullgl.where(varfullgl.values < 3.5*varfullgl.std().values)



# take means over bins over tfreq days
mcKts, datesmcK = func_mcK.time_mean_bins(mcKts, ex)
RV_ts = mcKts

def oneyr(datetime):
    return datetime.where(datetime.year==datetime.year[0]).dropna()

expanded_time = func_mcK.expand_times_for_lags(datesmcK, ex)
# Converting Mckinnon timestemp to match xarray timestemp
expandeddaysmcK = func_mcK.to_datesmcK(expanded_time, expanded_time[0].hour, varfullgl.time[0].dt.hour)
# region mckinnon - expanded time series
Prec_reg = func_mcK.find_region(varfullgl.sel(time=expandeddaysmcK), region=ex['region'])[0]
Prec_reg, datesvar = func_mcK.time_mean_bins(Prec_reg, ex)

matchdaysmcK = func_mcK.to_datesmcK(datesmcK, datesmcK[0].hour, Prec_reg.time[0].dt.hour)


# binary time serie when T95 exceeds 1 std
hotdaythreshold = mcKts.mean(dim='time').values + mcKts.std().values

## plotting same figure as in paper
func_mcK.plot_oneyr_events(mcKts, hotdaythreshold, 2012)

    
#%%
    


       
                
# If method == 'random' - Run until ROC has converged
ex['leave_n_out'] = True ; ex['method'] = 'iter'

ex['ROC_leave_n_out'] = False
ex['leave_n_years_out'] = 1


ex['score_per_run'] = []

ex['lags'] = [0, 6]  
ex['min_detection'] = 5
ex['hotdaythres'] = hotdaythreshold
ex['n_strongest'] = 15 
ex['n_std'] = 1.5   
ex['n_yrs'] = len(set(RV_ts.time.dt.year.values))
ex['n_conv'] = 2 #ex['n_yrs'] 
ex['toler'] = 0.010
if ex['leave_n_out'] == True and ex['method'] == 'iter':
    ex['test_ts_mcK'] = np.zeros( len(ex['lags']) , dtype=list)
    ex['test_ts_Sem'] = np.zeros( len(ex['lags']) , dtype=list)
    ex['test_RV'] = np.zeros( len(ex['lags']) , dtype=list)
Convergence = False ; Conv_mcK = False ; Conv_Sem = False



# Purely train-test based on iterating over all years:
    
lats = Prec_reg.latitude
lons = Prec_reg.longitude
array = np.zeros( (ex['n_yrs'], len(ex['lags']), len(lats), len(lons)) )
patterns = xr.DataArray(data=array, coords=[range(ex['n_yrs']), ex['lags'], lats, lons], 
                      dims=['n_tests', 'lag','latitude','longitude'], name='n_tests_patterns',
                      attrs={'units':'Kelvin'})
    
for n in range(ex['n_conv']):
    ex['n'] = n
    n_lags = len(ex['lags'])
    # do single run
    
    
    # =============================================================================
    # Calculate Precursor
    # =============================================================================
    test, train, ds_mcK, ds_Sem, ex = func_mcK.find_precursor(RV_ts, Prec_reg, ex)

    
    # =============================================================================
    # Calculate ROC score
    # =============================================================================
    ex = ROC_score_wrapper(test, train, ds_mcK, ds_Sem, ex)
    
    
    score_per_run = ex['score_per_run']
    events_per_year = [score_per_run[i][1] for i in range(len(score_per_run))]
    l_ds_mcK       = [score_per_run[i][2] for i in range(len(score_per_run))]
    l_ds_Sem       = [score_per_run[i][3] for i in range(len(score_per_run))]
    ran_ROCS       = [score_per_run[i][4] for i in range(len(score_per_run))]

    
    patterns[n,:,:,:] = ds_Sem['pattern']
    ex['n'] += 1
    
    
    
#%%

   
##%% Run code with ex settings
#if __name__ == "__main__":
#    for n in range(ex['n_conv']):
#        ex['n'] = n
#        n_lags = len(ex['lags'])
#        # do single run
#        
#        
#        # =============================================================================
#        # Calculate Precursor
#        # =============================================================================
#        test, train, ds_mcK, ds_Sem, ex = func_mcK.find_precursor(RV_ts, Prec_reg, ex)
#    
#        
#        # =============================================================================
#        # Calculate ROC score
#        # =============================================================================
#        ex['score_per_run'] = ROC_score_wrapper(test, train, ds_mcK, ds_Sem, ex)
#        
#        
#        score_per_run = ex['score_per_run']
#        events_per_year = [score_per_run[i][1] for i in range(len(score_per_run))]
#        l_ds_mcK       = [score_per_run[i][2] for i in range(len(score_per_run))]
#        l_ds_Sem       = [score_per_run[i][3] for i in range(len(score_per_run))]
#        ran_ROCS       = [score_per_run[i][4] for i in range(len(score_per_run))]
#    
#        
#        patterns[n,:,:,:] = ds_Sem['pattern']
#        ex['n'] += 1
    
#    
#ex['n'] = 0
#while Convergence == False:
#    n_lags = len(ex['lags'])
#    # do single run
#    # =============================================================================
#    # Calculate Precursor
#    # =============================================================================
#    test, train, ds_mcK, ds_Sem, ex = func_mcK.find_precursor(RV_ts, Prec_reg, ex)
#
#    
#    # =============================================================================
#    # Calculate ROC score
#    # =============================================================================
#    ex['score_per_run'] = ROC_score_wrapper(test, train, ds_mcK, ds_Sem, ex)
#    score_per_run = ex['score_per_run']
#    events_per_year = [score_per_run[i][1] for i in range(len(score_per_run))]
#    l_ds_mcK       = [score_per_run[i][2] for i in range(len(score_per_run))]
#    l_ds_Sem       = [score_per_run[i][3] for i in range(len(score_per_run))]
#    ran_ROCS       = [score_per_run[i][4] for i in range(len(score_per_run))]
#    
##    conv_std = [np.std(ran_ROCS[:n]) for n in range(len(ran_ROCS))]
##    plt.plot([np.std(ran_ROCS[:n]) for n in range(len(ran_ROCS))])
#    def check_last_n_means(ds, n, ex):
#        diff_last_n_means = np.zeros( (len(ex['lags'])) )
#        ROC_at_lags = []
#        means = []
#        for lag in ex['lags']:
#            idx = ex['lags'].index(lag)
#            n_scores = len(ds)
#            
#            ROC_single_lag = []
#            mean = []
#            diff_means = []
#            for n in range(n_scores):
#                value = float(ds[n]['score'].sel(lag=lag).values)
#                ROC_single_lag.append(value)
#                mean.append(np.mean(ROC_single_lag))
#            # calculating difference between last ex['n_conv'] mean values of runs
#            diff_means.append([abs(mean[-n]-np.mean(mean[-ex['n_conv']:])) for n in range(ex['n_conv']-1)] )
#            diff_means_mean = np.mean(diff_means)
#            diff_last_n_means[idx] = diff_means_mean
#            ROC_at_lags.append(ROC_single_lag)
#            means.append( mean )
#        return diff_last_n_means, ROC_at_lags, diff_means, means
#    
#    if ex['n'] >= ex['n_conv'] and ex['method'] == 'iter' and ex['leave_n_out'] == True:
#        Convergence = True
#    
#    # Check convergence of ROC score mcKinnon
#    if ex['n'] >= ex['n_conv'] and ex['method'] == 'random' and ex['leave_n_out'] == True:
#        std_ran = np.std(ran_ROCS)
#        diff_last_n_means, mean_at_lags, diff_means, means = check_last_n_means(l_ds_mcK, ex['n'], ex)
#        scores_mcK = np.round(np.mean(mean_at_lags,axis=1),2)
#        std_mcK    = np.round(np.std(mean_at_lags,axis=1),2)
#        print('\nMean score of mcK {} ± {} 2*std'.format(scores_mcK,std_mcK))
#        
#        # calculating std between last ex['n_conv'] mean values of runs
#        last_std = np.std(means, axis = 1)
#        check1 = np.zeros( n_lags )
#        check2 = np.zeros( n_lags )
#        for l in range( n_lags):
#            if last_std[l] < std_ran:
#                check1[l] = True
#            else:
#                check1[l] = False
#            if diff_last_n_means[l] < ex['toler']:
#                check2[l] = True
#            else:
#                check2[l] = False
#        check = np.append(check1, check2)
#        all_True = np.ones( ( 2*len(ex['lags']) ),dtype=bool)
#        if np.equal(check, all_True).all():
#            Conv_mcK = True
#            print('\nConvergence mcK is True')
#    
#    
#    # Check convergence of ROC score Sem
#    if ex['n'] >= ex['n_conv'] and ex['method'] == 'random' and ex['leave_n_out'] == True:
#        std_ran = np.std(ran_ROCS)
#        diff_last_n_means, mean_at_lags, diff_means, means = check_last_n_means(l_ds_Sem, ex['n'], ex)
#        scores_Sem = np.round(np.mean(mean_at_lags,axis=1),2)
#        std_Sem    = np.round(np.std(mean_at_lags,axis=1),2)
#        print('\nMean score of Sem {} ± {} 2*std'.format(scores_Sem,std_Sem))
#
#        # calculating std between last ex['n_conv'] mean values of runs
#        last_std = np.std(means, axis = 1)
#        check1 = np.zeros( n_lags )
#        check2 = np.zeros( n_lags )
#        for l in range( n_lags):
#            if last_std[l] < std_ran:
#                check1[l] = True
#            else:
#                check1[l] = False
#            if diff_last_n_means[l] < ex['toler']:
#                check2[l] = True
#            else:
#                check2[l] = False
#        check = np.append(check1, check2)
#        all_True = np.ones( ( 2*len(ex['lags']) ),dtype=bool)
#        if np.equal(check, all_True).all():
#            Conv_Sem = True
#            print('\nConvergence Sem is True')
#    
#    if (Conv_mcK, Conv_Sem) == (True,True):
#        Convergence = True
#    
#    if Convergence == True:
#        print('\n**Converged after {} runs**\n\n\n'.format(ex['n']))
#        text = ['\n**Converged after {} runs**\n\n\n']
#    if ex['n'] == ex['n_conv']:
#        Convergence = True
#        print('Reached max_conv at {}'.format(ex['n']))
#        
#    ex['n'] += 1
    
    
#%%

#events_per_year = [score_per_run[i][1] for i in range(len(score_per_run))]
#score_mcK       = [score_per_run[i][2] for i in range(len(score_per_run))]
#score_Sem       = [score_per_run[i][3] for i in range(len(score_per_run))]
#plt.scatter(events_per_year, score_mcK)
#plt.scatter(events_per_year, score_Sem)
    
#%%

    
# ============================= ===============================================
#   # Saving figures in exp_folder
# =============================================================================
#    # ROC scores of this run
#    ROCsmcK = [round(mcK_ROCS[-(i+1)],2) for i in range(len(ex['lags']))][::-1]
#    ROCsSem = [round(Sem_ROCS[-(i+1)],2) for i in range(len(ex['lags']))][::-1]
#    exp_folder = '{}_{:.2f}_{:.2f}'.format(test_years, np.mean(ROCsmcK),
#                 np.mean(ROCsSem))
#    ex['fig_path'] = os.path.join(ex['exp_folder'], exp_folder)
#    if os.path.isdir(ex['fig_path']) == False: os.makedirs(ex['fig_path'])
#    # Plotting mcK
#    ds_mcK['pattern'].attrs['units'] = 'Kelvin (absolute values)'
#    title = 'PEP\ntest years : {}\n{}'.format(test_years, ROCsmcK)
#    ds_mcK['pattern'].attrs['title'] = title
#    plotting_wrapper(ds_mcK['pattern'], 'PEP') 
#    
#    # Plotting CPD
#    ds_Sem['pattern'].attrs['units'] = 'Weighted Kelvin (absolute values)'
#    file_name = 'CPD'
#    title = 'CPD\ntest years : {}\n{}'.format(test_years, ROCsSem)
#    ds_Sem['pattern'].attrs['title'] = title
#    plotting_wrapper(ds_Sem['pattern'], file_name) 
    
    

        
#%%

#ds_mcK       = score_per_run[-1][2] 
#ds_Sem       = score_per_run[-1][3] 
#
#for lag in ex['lags']:
#    idx = ex['lags'].index(lag)
#
#    # select antecedant SST pattern to summer days:
#    dates_min_lag = matchdaysmcK - pd.Timedelta(int(lag), unit='d')
#    var_full_mcK = func_mcK.find_region(Prec_reg, region='PEPrectangle')[0]
#    full_timeserie_regmck = var_full_mcK.sel(time=dates_min_lag)
#    full_timeserie = Prec_reg.sel(time=dates_min_lag)
#    
#    # select test event predictand series
#    RV_ts_test = mcKts
#    crosscorr_mcK = func_mcK.cross_correlation_patterns(full_timeserie_regmck, 
#                                                ds_mcK['pattern'].sel(lag=lag))
#    crosscorr_Sem = func_mcK.cross_correlation_patterns(full_timeserie, 
#                                                ds_Sem['pattern'].sel(lag=lag))
#    n_boot = 5
#    ROC_mcK, ROC_boot_mcK = ROC_score(crosscorr_mcK, RV_ts_test,
#                                  ex['hotdaythres'], lag, n_boot, ds_mcK['perc'])
#    ROC_Sem, ROC_boot_Sem = ROC_score(crosscorr_Sem, RV_ts_test,
#                                  ex['hotdaythres'], lag, n_boot, ds_Sem['perc'])
#    
##    ROC_mcK, ROC_boot_mcK = ROC_score(crosscorr_mcK, RV_ts_test,
##                                  ex['hotdaythres'], lag, n_boot, 'default')
##    ROC_Sem, ROC_boot_Sem = ROC_score(crosscorr_Sem, RV_ts_test,
##                                  ex['hotdaythres'], lag, n_boot, 'default')
#    
#    ROC_std = 2 * np.std([ROC_boot_mcK, ROC_boot_Sem])
#    print('\n*** ROC score for {} lag {} ***\n\nMck {:.2f} \t Sem {:.2f} '
#        '\t ±{:.2f} 2*std random events'.format(region, 
#          lag, ROC_mcK, ROC_Sem, ROC_std))
#test_year = list(np.arange(2000, 2005))
#func_mcK.plot_events_validation(crosscorr_Sem, RV_ts_test, Prec_threshold, 
#                                ex['hotdaythres'], test_year)
      
#foldername = 'communities_Marlene'

#kwrgs = dict( {'vmin' : 0, 'vmax' : ex['n_strongest'], 
#                   'clevels' : 'notdefault', 'steps':ex['n_strongest']+1,
#                   'map_proj' : map_proj, 'cmap' : plt.cm.Dark2, 'column' : 2} )
#plotting_wrapper(commun_num, foldername, kwrgs=kwrgs)

  

