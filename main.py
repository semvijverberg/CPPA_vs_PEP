#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 10:31:42 2018

@author: semvijverberg
"""

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
xarray_plot = func_mcK.xarray_plot

base_path = "/Users/semvijverberg/surfdrive/Data_ERAint/"
exp_folder = ''
path_raw = os.path.join(base_path, 'input_raw')
path_pp  = os.path.join(base_path, 'input_pp')
if os.path.isdir(path_raw) == False : os.makedirs(path_raw) 
if os.path.isdir(path_pp) == False: os.makedirs(path_pp)


ex = dict(
     {'grid_res'    :       2.5,
     'startyear'    :       1979,
     'endyear'      :       2017,
     'base_path'    :       base_path,
     'path_raw'     :       path_raw,
     'path_pp'      :       path_pp,
     'sstartdate'   :       '06-24', #'1982-06-24',
     'senddate'     :       '08-31', #'1982-08-22',
     'figpathbase'  :       "/Users/semvijverberg/surfdrive/McKinRepl/",
     'RV1d_ts_path' :       "/Users/semvijverberg/surfdrive/MckinRepl/RVts2.5",
     'RVts_filename':       "t2mmax_1979-2017_averAggljacc0.75d_tf1_n6__to_t2mmax_tf1.npy",
     'tfreq'        :       1,
     'load_mcK'     :       False,
     'RV_name'      :       'T2mmax',
     'name'         :       'sst',
     'leave_n_out'  :       True,
     'ROC_leave_n_out':     False,
     'method'       :       'iter',
     'wghts_std_anom':      True,
     'wghts_accross_lags':  False,
     'splittrainfeat':      False,
     'use_ts_logit' :       True,
     'logit_valid'  :       True,
     'pval_logit_first':    0.10,
     'pval_logit_final':    0.05,
     'new_model_sel':       False,
     'mcKthres'     :       'mcKthres'}  # 'mcKthres'
     )


ex['region'] = 'Northern'
ex['regionmcK'] = 'PEPrectangle'
ex['exppathbase'] = '{}_{}_{}_{}'.format(ex['RV_name'],ex['name'],
                      ex['region'], ex['regionmcK'])
ex['figpathbase'] = os.path.join(ex['figpathbase'], ex['exppathbase'])
if os.path.isdir(ex['figpathbase']) == False: os.makedirs(ex['figpathbase'])

ex['sstartdate'] = str(ex['startyear']) + '-' + ex['sstartdate']
ex['senddate'] = str(ex['startyear']) + '-' + ex['senddate']




#if (ex['leave_n_out'] == True) & (ex['ROC_leave_n_out'] == False):
#    ex['figpathbase'] = os.path.join(ex['figpathbase'], '{}_{}_logit{}_'
#                      'model_Sem'.format(ex['RV_name'], ex['name'], ex['use_ts_logit']))
#elif (ex['ROC_leave_n_out'] == True):
#    ex['figpathbase'] = os.path.join(ex['figpathbase'], 
#                              '{}_{}_train_all_test_iter'.format(ex['RV_name'], 'name'))
#elif ex['leave_n_out'] == False:
#        ex['figpathbase'] = os.path.join(ex['figpathbase'], 
#                              '{}_{}_hindcast'.format(ex['RV_name'], 'name'))
##ex_dic_path = "T95_sst_NOAA_default_settings.npy"
#ex_dic_path = "ERA_T2mmax_sst_default_settings.npy"
#ex = np.load(ex_dic_path, encoding='latin1').item()




#'Mckinnonplot', 'U.S.', 'U.S.cluster', 'PEPrectangle', 'Pacific', 'Whole', 'Northern', 'Southern'
def oneyr(datetime):
    return datetime.where(datetime.year==datetime.year[0]).dropna()




if ex['load_mcK'] == True:
    # Load in mckinnon Time series
    T95name = 'PEP-T95TimeSeries.txt'
    RVtsfull, datesmcK = func_mcK.read_T95(T95name, ex)
    datesRV = func_mcK.make_datestr(datesmcK, ex)
    filename_precur = ('{}_1982-2017_2jan_31aug_dt-1days_{}deg'
                    '.nc'.format(ex['name'], ex['grid_res']))
else:
    # load ERA-i Time series
    ex['sstartdate'] = ex['sstartdate'].replace(ex['sstartdate'][:4], 
      str(ex['startyear']) )
    ex['senddate'] = ex['senddate'].replace(ex['senddate'][:4], 
      str(ex['startyear']) )
    print('\nimportRV_1dts is true, so the 1D time serie given with name \n'
              '{} is imported.'.format(ex['RVts_filename']))
    filename = os.path.join(ex['RV1d_ts_path'], ex['RVts_filename'])
    dicRV = np.load(filename,  encoding='latin1').item()
    RVtsfull = dicRV['RVfullts95']
    ex['mask'] = dicRV['RV_array']['mask']
    xarray_plot(dicRV['RV_array']['mask'])
    RVhour   = RVtsfull.time[0].dt.hour.values
    datesRV = func_mcK.make_datestr(pd.to_datetime(RVtsfull.time.values), ex)
    # add RVhour to daily dates
    datesRV = datesRV + pd.Timedelta(int(RVhour), unit='h')
    filename_precur = '{}_{}-{}_2jan_31aug_dt-1days_2.5deg.nc'.format(ex['name'],
                                            ex['startyear'], ex['endyear'])
    ex['endyear'] = int(RVtsfull.time.dt.year[-1])

# Selected Time series of T95 ex['sstartdate'] until ex['senddate']
RVts = RVtsfull.sel(time=datesRV)
ex['n_oneyr'] = oneyr(datesRV).size

# Load in external ncdf
#filename = '{}_1979-2017_2mar_31aug_dt-1days_2.5deg.nc'.format(ex['name'])

# full globe - full time series
varfullgl = func_mcK.import_array(filename_precur, ex)

## filter out outliers of sst
#if ex['name']=='sst':
#    varfullgl.where(varfullgl.values < 3.5*varfullgl.std().values)



RV_ts, datesmcK = func_mcK.time_mean_bins(RVts, ex)
expanded_time = func_mcK.expand_times_for_lags(datesmcK, ex)
# Converting Mckinnon timestemp to match xarray timestemp
#expandeddaysmcK = func_mcK.to_datesmcK(expanded_time, expanded_time[0].hour, varfullgl.time[0].dt.hour)
# region mckinnon - expanded time series
#Prec_reg = func_mcK.find_region(varfullgl.sel(time=expandeddaysmcK), region=ex['region'])[0]
Prec_reg = func_mcK.find_region(varfullgl, region=ex['region'])[0]
if ex['tfreq'] != 1:
    Prec_reg, datesvar = func_mcK.time_mean_bins(Prec_reg, ex)


if ex['mcKthres'] == 'mcKthres':
    # binary time serie when T95 exceeds 1 std
    ex['hotdaythres'] = RV_ts.mean(dim='time').values + RV_ts.std().values
else:
    percentile = ex['mcKthres']
    ex['hotdaythres'] = np.percentile(RV_ts.values, percentile)
    ex['mcKthres'] = '{}p'.format(percentile)


#ex['lags_idx'] = [12, 18, 24, 30]  
#ex['lags'] = [l*ex['tfreq'] for l in ex['lags_idx'] ]
ex['plot_ts'] = False
# [0, 5, 10, 15, 20, 30, 40, 50]
ex['lags'] = [0, 5, 10, 15, 20, 30, 40, 50] #[10, 20, 30, 50] # [0, 5, 10, 15, 20, 30, 40, 50] # [60, 70, 80] # [0, 6, 12, 18]  # [24, 30, 40, 50] # [60, 80, 100]
ex['min_detection'] = 5
ex['leave_n_years_out'] = 5
ex['n_strongest'] = 15 
ex['perc_map'] = 95
ex['comp_perc'] = 0.8
ex['n_yrs'] = len(set(RV_ts.time.dt.year.values))
ex['n_conv'] = ex['n_yrs'] 
if ex['leave_n_out'] == True and ex['method'] == 'iter' or ex['ROC_leave_n_out']:
    ex['test_ts_mcK'] = np.zeros( len(ex['lags']) , dtype=list)
    ex['test_ts_Sem'] = np.zeros( len(ex['lags']) , dtype=list)
    ex['test_RV'] = np.zeros( len(ex['lags']) , dtype=list)


## plotting same figure as in paper
func_mcK.plot_oneyr_events(RV_ts, ex['hotdaythres'], 2012, ex)


print_ex = ['RV_name', 'name', 'grid_res', 'startyear', 'endyear', 
            'sstartdate', 'senddate', 'n_conv', 'leave_n_out',
            'n_oneyr', 'method', 'ROC_leave_n_out', 'wghts_std_anom', 
            'wghts_accross_lags', 'splittrainfeat', 'n_strongest',
            'perc_map', 'tfreq', 'lags', 'n_yrs', 'hotdaythres',
            'pval_logit_first', 'pval_logit_final',
            'mcKthres', 'new_model_sel', 'perc_map', 'comp_perc',
            'logit_valid', 'use_ts_logit', 'region', 'regionmcK']


max_key_len = max([len(i) for i in print_ex])
for key in print_ex:
    key_len = len(key)
    expand = max_key_len - key_len
    key_exp = key + ' ' * expand
    printline = '\'{}\'\t\t{}'.format(key_exp, ex[key])
    print(printline)

#np.save(os.path.join(script_dir, 'ERA_{}_{}_default_settings.npy'.format(
#        ex['RV_name'], ex['name'])), ex)

#%% Run code with ex settings
#ex['n_conv'] = 3

def main(RV_ts, Prec_reg, ex):
    if (ex['leave_n_out'] == False) and ex['ROC_leave_n_out'] == False : ex['n_conv'] = 1
    if ex['ROC_leave_n_out'] == True: 
        print('leave_n_out set to False')
        ex['leave_n_out'] = False
    

    
    ex['score_per_run'] = []
    # Purely train-test based on iterating over all years:
    lats = Prec_reg.latitude
    lons = Prec_reg.longitude
    array = np.zeros( (ex['n_conv'], len(ex['lags']), len(lats), len(lons)) )
    patterns_Sem = xr.DataArray(data=array, coords=[range(ex['n_conv']), ex['lags'], lats, lons], 
                          dims=['n_tests', 'lag','latitude','longitude'], 
                          name='{}_tests_patterns_Sem'.format(ex['n_conv']), attrs={'units':'Kelvin'})
    Prec_mcK = func_mcK.find_region(Prec_reg, region=ex['regionmcK'])[0][0]
    lats = Prec_mcK.latitude
    lons = Prec_mcK.longitude
    array = np.zeros( (ex['n_conv'], len(ex['lags']), len(lats), len(lons)) )
    patterns_mcK = xr.DataArray(data=array, coords=[range(ex['n_conv']), ex['lags'], lats, lons], 
                          dims=['n_tests', 'lag','latitude','longitude'], 
                          name='{}_tests_patterns_mcK'.format(ex['n_conv']), attrs={'units':'Kelvin'})
            
    for n in range(ex['n_conv']):
        train_all_test_n_out = (ex['ROC_leave_n_out'] == True) & (n==0) 
        ex['n'] = n

        # do single run
        
        # =============================================================================
        # Create train test set according to settings 
        # =============================================================================
        train, test, ex = func_mcK.train_test_wrapper(RV_ts, Prec_reg, ex)
        
        # =============================================================================
        # Calculate Precursor
        # =============================================================================
        if train_all_test_n_out == True:
            # only train once on all years if ROC_leave_n_out == True
            ds_mcK, ds_Sem, ex = func_mcK.find_precursor(RV_ts, Prec_reg, 
                                                                      train, test, ex)
        # Force Leave_n_out validation even though pattern is based on whole dataset
        if (ex['ROC_leave_n_out'] == True) & (ex['n']==0):
            # start selecting leave_n_out
            ex['leave_n_out'] = True
            train, test, ex['test_years'] = func_mcK.rand_traintest(RV_ts, Prec_reg, 
                                              ex)
            
            foldername = 'Pattern_full_leave_{}_out_validation_{}_{}_tf{}_{}'.format(
                    ex['leave_n_years_out'], ex['startyear'], ex['endyear'], 
                    ex['tfreq'],ex['lags'])
        
            ex['exp_folder'] = os.path.join(ex['figpathbase'],foldername)
        
#        elif (ex['leave_n_out'] == True) & (ex['ROC_leave_n_out'] == False):
        elif (ex['ROC_leave_n_out'] == False):
            # train each time on only train years
            ds_mcK, ds_Sem, ex = func_mcK.find_precursor(RV_ts, Prec_reg, 
                                                         train, test, ex)

        # =============================================================================
        # Make prediction based on logit model found in 'extract_precursor'
        # =============================================================================
        if (ex['new_model_sel'] == False) and (ex['use_ts_logit'] == True):
            ds_Sem = func_mcK.timeseries_for_test(ds_Sem, test, ex)
#            print(ds_Sem['ts_prediction'][0])
    
        if (ex['new_model_sel'] == True) and (ex['use_ts_logit'] == True):
            ds_Sem = func_mcK.NEW_timeseries_for_test(ds_Sem, test, ex)
            # ds_Sem['ts_prediction'].plot()
            
        # =============================================================================
        # Calculate ROC score
        # =============================================================================
        ex = ROC_score_wrapper(test, train, ds_mcK, ds_Sem, ex)
        l_ds_Sem       = [ex['score_per_run'][i][3] for i in range(len(ex['score_per_run']))]
        l_ds_mcK       = [ex['score_per_run'][i][2] for i in range(len(ex['score_per_run']))]
        patterns_Sem[n,:,:,:] = l_ds_Sem[n]['pattern']
        patterns_mcK[n,:,:,:] = l_ds_mcK[n]['pattern']
        
        
#        ex['n'] += 1
        if (ex['leave_n_out'] == False) & (ex['ROC_leave_n_out'] == False):
            # only need single run
            break
    
        
    return ex, patterns_Sem, patterns_mcK

        
ex, patterns_Sem, patterns_mcK = main(RV_ts, Prec_reg, ex)




# save ex setting in text file
folder = os.path.join(ex['figpathbase'], ex['exp_folder'])
assert (os.path.isdir(folder) != True), print('Overwrite?\n{}'.format(folder))
                                       


if os.path.isdir(folder) != True : os.makedirs(folder)
#folder = '/Users/semvijverberg/Downloads'
txtfile = os.path.join(folder, 'experiment_settings.txt')
with open(txtfile, "w") as text_file:
    max_key_len = max([len(i) for i in print_ex])
    for key in print_ex:
        key_len = len(key)
        expand = max_key_len - key_len
        key_exp = key + ' ' * expand
        printline = '\'{}\'\t\t{}'.format(key_exp, ex[key])
        print(printline, file=text_file)

      
output_dic_folder = folder
filename = 'output_main_dic'
if os.path.isdir(output_dic_folder) != True : os.makedirs(output_dic_folder)
to_dict = dict( {'ex'       :   ex,
                 'patterns_Sem'  :  patterns_Sem,
                 'patterns_mcK'  :  patterns_mcK} )
np.save(os.path.join(output_dic_folder, filename+'.npy'), to_dict)

#%%
events_per_year = [ex['score_per_run'][i][1] for i in range(len(ex['score_per_run']))]
l_ds_mcK        = [ex['score_per_run'][i][2] for i in range(len(ex['score_per_run']))]
l_ds_Sem        = [ex['score_per_run'][i][3] for i in range(len(ex['score_per_run']))]
ran_ROCS        = [ex['score_per_run'][i][4] for i in range(len(ex['score_per_run']))]
score_mcK       = np.round(ex['score_per_run'][-1][2]['score'], 2)
score_Sem       = np.round(ex['score_per_run'][-1][3]['score'], 2)


# mcKinnon composite mean plot
filename = os.path.join(ex['exp_folder'], 'mcKinnon mean composite_tf{}_{}'.format(
            ex['tfreq'], ex['lags']))
mcK_mean = patterns_mcK.mean(dim='n_tests')
mcK_mean.name = 'Composite mean: ROC {}'.format(score_mcK.values)
mcK_mean.attrs['units'] = 'Kelvin'
func_mcK.plotting_wrapper(mcK_mean, filename, ex)

if (ex['leave_n_out'] == True) and (ex['ROC_leave_n_out'] == False):
    # mcKinnon std plot
    filename = os.path.join(ex['exp_folder'], 'mcKinnon std composite_tf{}_{}'.format(
                ex['tfreq'], ex['lags']))
    mcK_std = patterns_mcK.std(dim='n_tests')
    mcK_std.name = 'Composite std: ROC {}'.format(score_mcK.values)
    mcK_std.attrs['units'] = 'Kelvin'
    func_mcK.plotting_wrapper(mcK_std, filename, ex)

# Sem plot
mean_n_patterns = patterns_Sem.mean(dim='n_tests')
mean_n_patterns.attrs['units'] = 'mean over {} runs'.format(ex['n_conv'])
mean_n_patterns.name = 'ROC {}'.format(score_Sem.values)
filename = os.path.join(ex['exp_folder'], 'mean_over_{}_tests'.format(ex['n_conv']) )
func_mcK.plotting_wrapper(mean_n_patterns, filename, ex)

#%%
if ex['load_mcK'] == False:
    xarray_plot(dicRV['RV_array']['mask'], path=folder, name='RV_mask', saving=True)
## plotting same figure as in paper
func_mcK.plot_oneyr_events(RV_ts, ex['hotdaythres'], 2012, folder, saving=True)

#%% Counting times gridcells were extracted
if ex['leave_n_out']:
    n_lags = patterns_Sem.sel(n_tests=0).lag.size
    n_lats = patterns_Sem.sel(n_tests=0).latitude.size
    n_lons = patterns_Sem.sel(n_tests=0).longitude.size
    
    pers_patt = patterns_Sem.sel(n_tests=0).copy()
    arrpatt = np.nan_to_num(patterns_Sem.values)
    mask_patt = (arrpatt != 0)
    arrpatt[mask_patt] = 1
    wghts = np.zeros( (n_lags, n_lats, n_lons) )
#    plt.imshow(arrpatt[0,0]) ; plt.colorbar()
    for l in ex['lags']:
        i = ex['lags'].index(l)
        wghts[i] = np.sum(arrpatt[:,i,:,:], axis=0)
    pers_patt.values = wghts
    
    pers_patt.attrs['units'] = 'persistence pattern over {} runs'.format(ex['n_conv'])
    pers_patt.attrs['title'] = 'ROC {}'.format(score_Sem.values)
    filename = os.path.join(ex['exp_folder'], 'counting_times_extracted_over_{}_tests'.format(ex['n_conv']) )
    kwrgs = dict( {'title' : pers_patt.name, 'clevels' : 'notdefault', 'steps':17,
                    'vmin' : pers_patt.min().values, 'vmax' : pers_patt.max().values, 
                   'cmap' : plt.cm.gist_heat_r, 'column' : 2} )
    func_mcK.plotting_wrapper(pers_patt, filename, ex, kwrgs=kwrgs)
#%% Weighing features if there are extracted every run (training set)
# weighted by persistence of pattern over
if ex['leave_n_out']:
    # weighted by persistence (all years == wgt of 1, less is below 1)
    mean_n_patterns = patterns_Sem.mean(dim='n_tests') * wghts/np.max(wghts)
    # only keep gridcells that were extracted 50% of the test years
    pers_patt_filter = patterns_Sem.sel(n_tests=0).copy().drop('n_tests')
    ex['persistence_criteria'] = int(0.5 * ex['n_conv'])
    mask_pers = (wghts >= ex['persistence_criteria'])
#    mean_n_patterns.coords['mask'] = (('lag', 'latitude','longitude'), mask_pers)
#    mean_n_patterns.values = np.array(mask_pers,dtype=int) * mean_n_patterns
    if mean_n_patterns.sum().values != 0.:
        mean_n_patterns.attrs['units'] = 'weighted by persistence - less than {} out of  {} masked'.format(
                                        ex['persistence_criteria'], ex['n_conv'])
        mean_n_patterns.attrs['title'] = 'weighted by persistence - less than {} masked'.format(
                                        ex['persistence_criteria'])
        mean_n_patterns.name = 'ROC {}'.format(score_Sem.values)
        filename = os.path.join(ex['exp_folder'], 'weighted by persistence'
                                '- less than {} out of  {} masked'.format(
                                        ex['persistence_criteria'], ex['n_conv']))
#        kwrgs = dict( {'title' : mean_n_patterns.name, 'clevels' : 'default', 'steps':17,
#                        'vmin' : -3*mean_n_patterns.std().values, 'vmax' : 3*mean_n_patterns.std().values, 
#                       'cmap' : plt.cm.RdBu_r, 'column' : 2} )
        func_mcK.plotting_wrapper(mean_n_patterns, filename, ex, kwrgs=None)


#%% Initial regions from only composite extraction:


if ex['leave_n_out']:
    subfolder = os.path.join(ex['exp_folder'], 'intermediate_results')
    total_folder = os.path.join(ex['figpathbase'], subfolder)
    if os.path.isdir(total_folder) != True : os.makedirs(total_folder)
    years = range(ex['startyear'], ex['endyear'])
    for n in np.arange(0, ex['n_conv'], 6, dtype=int): 
        yr = years[n]
        pattern_num_init = l_ds_Sem[n]['pattern_num_init']
        


        pattern_num_init.attrs['title'] = ('{} - initial regions extracted from '
                              'composite approach'.format(yr))
        filename = os.path.join(subfolder, pattern_num_init.attrs['title'].replace(
                                ' ','_')+'.png')
        for_plt = pattern_num_init.copy()
        for_plt.values = for_plt.values-0.5
        kwrgs = dict( {'title' : for_plt.attrs['title'], 'clevels' : 'notdefault', 
                       'steps' : for_plt.max()+2,
                       'vmin' : 0, 'vmax' : for_plt.max().values+0.5, 
                       'cmap' : plt.cm.tab10, 'column' : 2} )
        
        func_mcK.plotting_wrapper(for_plt, filename, ex, kwrgs=kwrgs)
        
        if ex['logit_valid'] == True:
            pattern_num = l_ds_Sem[n]['pattern_num']
            pattern_num.attrs['title'] = ('{} - regions that were kept after logit regression '
                                         'pval < {}'.format(yr, ex['pval_logit_final']))
            filename = os.path.join(subfolder, pattern_num.attrs['title'].replace(
                                    ' ','_')+'.png')
            for_plt = pattern_num.copy()
            for_plt.values = for_plt.values-0.5
            kwrgs = dict( {'title' : for_plt.attrs['title'], 'clevels' : 'notdefault', 
                           'steps' : for_plt.max()+2,
                           'vmin' : 0, 'vmax' : for_plt.max().values+0.5, 
                           'cmap' : plt.cm.tab10, 'column' : 2} )
            
            func_mcK.plotting_wrapper(for_plt, filename, ex, kwrgs=kwrgs)
        
        
    
    
# =============================================================================
# =============================================================================
# =============================================================================
# # # 
# =============================================================================
# =============================================================================
# =============================================================================



#%% Load data
import numpy as np
import os
import xarray as xr
output_dic_folder = ('/Users/semvijverberg/surfdrive/MckinRepl/T2mmax_sst_Northern_PEPrectangle/iter_1979_2017_tf1_lags[0, 5, 10, 15, 20, 30, 40, 50]_mcKthres_2.5deg_92nyr_0.05_94tperc_0.8tc_2019-01-15')

filename = 'output_main_dic'

dic = np.load(os.path.join(output_dic_folder,filename+'.npy'),  encoding='latin1').item()
ex = dic['ex']
patterns_Sem = dic['patterns_Sem']
#patterns_mcK = dic['patterns_mcK']


l_ds_mcK        = [ex['score_per_run'][i][2] for i in range(len(ex['score_per_run']))]
lats = l_ds_mcK[0]['pattern'].latitude
lons = l_ds_mcK[0]['pattern'].longitude
array = np.zeros( (ex['n_conv'], len(ex['lags']), len(lats), len(lons)) )
patterns_mcK = xr.DataArray(data=array, coords=[range(ex['n_conv']), ex['lags'], lats, lons], 
                          dims=['n_tests', 'lag','latitude','longitude'], 
                          name='{}_tests_patterns_mcK'.format(ex['n_conv']), attrs={'units':'Kelvin'})
for n in range(ex['n_conv']):
    patterns_mcK[n,:,:,:] = l_ds_mcK[n]['pattern']


 




#%%
        
        
#for n in range(ex['n_conv']):
#    ex['n'] = n
#    n_lags = len(ex['lags'])
#    # do single run
#    
#    
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
#    
#    
#    score_per_run = ex['score_per_run']
#    events_per_year = [score_per_run[i][1] for i in range(len(score_per_run))]
#    l_ds_mcK       = [score_per_run[i][2] for i in range(len(score_per_run))]
#    l_ds_Sem       = [score_per_run[i][3] for i in range(len(score_per_run))]
#    ran_ROCS       = [score_per_run[i][4] for i in range(len(score_per_run))]
#
#    
#    patterns[n,:,:,:] = ds_Sem['pattern']