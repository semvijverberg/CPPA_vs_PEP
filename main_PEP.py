"""
Created on Mon Dec 10 10:31:42 2018

@author: semvijverberg
"""

import os, sys


if os.path.isdir("/Users/semvijverberg/surfdrive/"):
    basepath = "/Users/semvijverberg/surfdrive/"
    data_base_path = basepath
else:
    basepath = "/home/semvij/"
    data_base_path = "/p/projects/gotham/semvij/"
    
os.chdir(os.path.join(basepath, 'Scripts/CPPA_vs_PEP/'))
script_dir = os.getcwd()
sys.path.append(script_dir)
if sys.version[:1] == '3':
    from importlib import reload as rel

import numpy as np
import xarray as xr 
import pandas as pd
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt
#import scipy
import func_PEP
import load_data

from ROC_score import plotting_timeseries




datafolder = 'era5'
path_pp  = os.path.join(data_base_path, 'Data_'+datafolder +'/input_pp') # path to netcdfs
if os.path.isdir(path_pp) == False: os.makedirs(path_pp)


# =============================================================================
# General Settings
# =============================================================================
ex = {'datafolder'  :       datafolder,
      'grid_res'    :       1.0,
     'startyear'    :       1979,
     'endyear'      :       2018,
     'path_pp'      :       path_pp,
     'startperiod'   :       '06-24', #'1982-06-24',
     'endperiod'     :       '08-22', #'1982-08-22',
     'figpathbase'  :       os.path.join(basepath, 'McKinRepl/'),
     'RV1d_ts_path' :       os.path.join(basepath, 'MckinRepl/RVts'),
     'RVts_filename':       datafolder+"_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1.npy",
     'RV_name'      :       'T2mmax',
     'name'         :       'sst',
     'add_lsm'      :       False,
     'region'       :       'Northern',
     'regionmcK'    :       'PEPrectangle',
     'lags'         :       [0, 15, 30, 50], #[0, 5, 10, 15, 20, 30, 40, 50, 60], #[5, 15, 30, 50] #[10, 20, 30, 50] 
     'plot_ts'      :       True,
     }
# =============================================================================
# Settings for event timeseries
# =============================================================================
ex['tfreq']             =       1
ex['max_break']         =       0   
ex['min_dur']           =       1
ex['event_percentile']  =       'std'
# =============================================================================
# Settins for PEP
# =============================================================================
ex['load_mcK']          =       '1bram' # or '1bram' for extended or '1' for mcK ts
ex['filename_precur']   =       '{}_{}-{}_1jan_31dec_daily_{}deg.nc'.format(
                                ex['name'],ex['startyear'],ex['endyear'],ex['grid_res'])
ex['rollingmean']       =       ('CPPA', 1)
# =============================================================================
# Settings for validation     
# =============================================================================
ex['leave_n_out']       =       True
ex['ROC_leave_n_out']   =       False
ex['method']            =       'iter' #'iter' or 'no_train_test_split' or split#8 or random3  
ex['n_boot']            =       1000
# =============================================================================
# load data (write your own function load_data(ex) )
# =============================================================================
RV_ts, Prec_reg, ex = load_data.load_data(ex)

ex['exppathbase'] = '{}_PEP_{}_{}_{}'.format(datafolder, ex['RV_name'],ex['name'],
                      ex['regionmcK'])
ex['figpathbase'] = os.path.join(ex['figpathbase'], ex['exppathbase'])
if os.path.isdir(ex['figpathbase']) == False: os.makedirs(ex['figpathbase'])


print_ex = ['RV_name', 'name', 'max_break',
            'min_dur', 'grid_res', 'startyear', 'endyear', 
            'startperiod', 'endperiod', 'leave_n_out',
            'n_oneyr', 'method', 'ROC_leave_n_out',
            'tfreq', 'lags', 'n_yrs', 
            'rollingmean', 'event_percentile',
            'event_thres', 
            'region', 'regionmcK',
            'add_lsm', 'n_boot']

def printset(print_ex=print_ex, ex=ex):
    max_key_len = max([len(i) for i in print_ex])
    for key in print_ex:
        key_len = len(key)
        expand = max_key_len - key_len
        key_exp = key + ' ' * expand
        printline = '\'{}\'\t\t{}'.format(key_exp, ex[key])
        print(printline)


printset()
n = 1
#%% Run code with ex settings



l_ds_PEP, ex = func_PEP.main(RV_ts, Prec_reg, ex)


output_dic_folder = ex['output_dic_folder']


# save ex setting in text file

if os.path.isdir(output_dic_folder):
    answer = input('Overwrite?\n{}\ntype y or n:\n\n'.format(output_dic_folder))
    if 'n' in answer:
        assert (os.path.isdir(output_dic_folder) != True)
    elif 'y' in answer:
        pass

if os.path.isdir(output_dic_folder) != True : os.makedirs(output_dic_folder)

# save output in numpy dictionary
filename = 'output_main_dic'
if os.path.isdir(output_dic_folder) != True : os.makedirs(output_dic_folder)
to_dict = dict( { 'ex'      :   ex,
                 'l_ds_PEP' : l_ds_PEP} )
np.save(os.path.join(output_dic_folder, filename+'.npy'), to_dict)  

# write output in textfile
print_ex.append('output_dic_folder')
txtfile = os.path.join(output_dic_folder, 'experiment_settings.txt')
with open(txtfile, "w") as text_file:
    max_key_len = max([len(i) for i in print_ex])
    for key in print_ex:
        key_len = len(key)
        expand = max_key_len - key_len
        key_exp = key + ' ' * expand
        printline = '\'{}\'\t\t{}'.format(key_exp, ex[key])
        print(printline)
        print(printline, file=text_file)



#%% Generate output in console



filename = 'output_main_dic'
dic = np.load(os.path.join(output_dic_folder, filename+'.npy'),  encoding='latin1').item()

# load settings
ex = dic['ex']
l_ds_PEP = dic['l_ds_PEP']


ex['n_boot'] = 1000

# write output in textfile
predict_folder = 'PEP'
ex['exp_folder'] = os.path.join(ex['CPPA_folder'], predict_folder)
predict_folder = os.path.join(ex['figpathbase'], ex['exp_folder'])
if os.path.isdir(predict_folder) != True : os.makedirs(predict_folder)

txtfile = os.path.join(predict_folder, 'experiment_settings.txt')
with open(txtfile, "w") as text_file:
    max_key_len = max([len(i) for i in print_ex])
    for key in print_ex:
        key_len = len(key)
        expand = max_key_len - key_len
        key_exp = key + ' ' * expand
        printline = '\'{}\'\t\t{}'.format(key_exp, ex[key])
        print(printline, file=text_file)
        print(printline)


# =============================================================================
# perform prediciton        
# =============================================================================


#ex, l_ds_CPPA = func_pred.make_prediction(l_ds_CPPA, l_ds_PEP, Prec_reg, ex)

ex['exp_folder'] = os.path.join(ex['CPPA_folder'], 'PEP')
ex = func_PEP.only_spatcov_wrapper(l_ds_PEP, RV_ts, Prec_reg, ex)


score_AUC        = np.round(ex['score'][-1][0], 2)
ROC_str      = ['{} days - ROC score {}'.format(ex['lags'][i], score_AUC[i]) for i in range(len(ex['lags'])) ]
ROC_boot = [np.round(np.percentile(ex['score'][-1][1][i],99), 2) for i in range(len(ex['lags']))]

ex['score_AUC']   = score_AUC
ex['ROC_boot_99'] = ROC_boot

filename = 'output_main_dic'
to_dict = dict( { 'ex'      :   ex,
                 'l_ds_PEP' : l_ds_PEP} )
np.save(os.path.join(output_dic_folder, filename+'.npy'), to_dict)  

#%%
# =============================================================================
#   Plotting
# =============================================================================


Prec_mcK = func_PEP.find_region(Prec_reg, region=ex['region'])[0][0]
lats = Prec_mcK.latitude
lons = Prec_mcK.longitude
array = np.zeros( (ex['n_conv'], len(ex['lags']), len(lats), len(lons)) )
patterns_mcK = xr.DataArray(data=array, coords=[range(ex['n_conv']), ex['lags'], lats, lons], 
                      dims=['n_tests', 'lag','latitude','longitude'], 
                      name='{}_tests_patterns_mcK'.format(ex['n_conv']), attrs={'units':'Kelvin'})

for n in range(len(ex['train_test_list'])):
    ex['n'] = n

        
    if (ex['method'][:6] == 'random'):
        if n == ex['n_conv']:
            # remove empty n_tests
            patterns_mcK = patterns_mcK.sel(n_tests=slice(0,ex['n_conv']))
            ex['n_conv'] = ex['n_conv']
    
    patterns_mcK[n,:,:,:] = l_ds_PEP[n]['pattern'].sel(lag=ex['lags'])


    
kwrgs = dict( {'title' : '', 'clevels' : 'notdefault', 'steps':17,
                'vmin' : -0.4, 'vmax' : 0.4, 'subtitles' : ROC_str,
               'cmap' : plt.cm.RdBu_r, 'column' : 1,
               'cbar_vert' : 0.02, 'cbar_hght' : -0.01,
               'adj_fig_h' : 0.9, 'adj_fig_w' : 1., 
               'hspace' : 0.2, 'wspace' : 0.08,
               'title_h' : 0.95} )

# mcKinnon composite mean plot
kwrgs['drawbox'] = True
filename = os.path.join(ex['exp_folder'], 'PEP_mean_composite_tf{}_{}'.format(
            ex['tfreq'], ex['lags']))
mcK_mean = patterns_mcK.mean(dim='n_tests')
mcK_mean.name = 'Composite mean green rectangle'
mcK_mean.attrs['units'] = 'Kelvin'
mcK_mean.attrs['title'] = 'Composite mean'
import func_CPPA
func_CPPA.plotting_wrapper(mcK_mean, ex, filename, kwrgs=kwrgs)



#%% Plot time series events:
folder = '/Users/semvijverberg/Downloads/'
func_PEP.plot_oneyr_events_allRVts(ex, 2012, folder, saving=True)


    
    
#%%
if ex['load_mcK'] == False:
    filename = os.path.join(ex['RV1d_ts_path'], ex['RVts_filename'])
    dicRV = np.load(filename,  encoding='latin1').item()
    folder = os.path.join(ex['figpathbase'], ex['exp_folder'])
    xarray_plot(dicRV['RV_array']['mask'], path=folder, name='RV_mask', saving=True)
    
func_CPPA.plot_oneyr_events(RV_ts, ex, 2012, ex['output_dic_folder'], saving=True)
## plotting same figure as in paper
#for i in range(2005, 2010):
#    func_CPPA.plot_oneyr_events(RV_ts, ex, i, folder, saving=True)




#%% Plotting prediciton time series vs truth:
yrs_to_plot = [1985, 1990, 1995, 2004, 2007, 2012, 2015]
#yrs_to_plot = list(np.arange(ex['startyear'],ex['endyear']+1))
test = ex['train_test_list'][0][1]        
plotting_timeseries(test, yrs_to_plot, ex) 







