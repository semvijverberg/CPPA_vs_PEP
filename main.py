"""
Created on Mon Dec 10 10:31:42 2018

@author: semvijverberg
"""

import os, sys
os.chdir('/Users/semvijverberg/surfdrive/Scripts/CPPA_vs_PEP/')
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
import func_CPPA
import func_pred
import load_data

from ROC_score import plotting_timeseries
xarray_plot = func_CPPA.xarray_plot
xrplot = func_CPPA.xarray_plot



path_pp  = "/Users/semvijverberg/surfdrive/Data_ERAint/input_pp" # path to netcdfs
if os.path.isdir(path_pp) == False: os.makedirs(path_pp)


# =============================================================================
# General Settings
# =============================================================================
ex = {'grid_res'    :       2.5,
     'startyear'    :       1979,
     'endyear'      :       2017,
     'path_pp'      :       path_pp,
     'startperiod'   :       '06-24', #'1982-06-24',
     'endperiod'     :       '08-22', #'1982-08-22',
     'figpathbase'  :       "/Users/semvijverberg/surfdrive/McKinRepl/",
     'RV1d_ts_path' :       "/Users/semvijverberg/surfdrive/MckinRepl/RVts2.5",
     'RVts_filename':       "t2mmax_1979-2017_averAggljacc0.75d_tf1_n4__to_t2mmax_tf1.npy",
     'RV_name'      :       'T2mmax',
     'name'         :       'sst',
     'add_lsm'      :       False,
     'region'       :       'Northern',
     'regionmcK'    :       'PEPrectangle',
     'lags'         :       [0, 5, 10, 15, 20, 30, 40, 50, 60], #[5, 15, 30, 50] #[10, 20, 30, 50] 
     'plot_ts'      :       True,
     }
# =============================================================================
# Settings for event timeseries
# =============================================================================
ex['tfreq']             =       1
ex['max_break']         =       0   
ex['min_dur']           =       1   
ex['load_mcK']          =       '0'
ex['mcKthres']          =       'mcKthres'    
# =============================================================================
# Settins for precursor / CPPA
# =============================================================================
ex['filename_precur']   = '{}_1979-2017_1jan_31dec_daily_{}deg.nc'.format(
                            ex['name'], ex['grid_res'])
ex['rollingmean']       =       ('CPPA', 1)
ex['extra_wght_dur']    =       False
ex['prec_reg_max_d']    =       1
ex['perc_map']          =       95
ex['comp_perc']         =       0.80
ex['min_n_gc']          =       5
ex['wghts_accross_lags']=       False
# =============================================================================
# Settings for validation     
# =============================================================================
ex['leave_n_out']       =       True,
ex['ROC_leave_n_out']   =       False,
ex['method']            =       'iter' #87  



if ex['name'][:2] == 'sm' or ex['name'][:2] == 'st':
    ex['region']     = 'U.S.soil'
    ex['regionmcK']  = 'U.S.soil'
    ex['add_lsm']   = True
    ex['mask_file'] = 'mask_North_America_for_soil{}deg.nc'.format(ex['grid_red'])



RV_ts, Prec_reg, ex = load_data.load_data(ex)

ex['exppathbase'] = '{}_{}_{}_{}'.format(ex['RV_name'],ex['name'],
                      ex['region'], ex['regionmcK'])
ex['figpathbase'] = os.path.join(ex['figpathbase'], ex['exppathbase'])
if os.path.isdir(ex['figpathbase']) == False: os.makedirs(ex['figpathbase'])


print_ex = ['RV_name', 'name', 'load_mcK', 'max_break',
            'min_dur', 'grid_res', 'startyear', 'endyear', 
            'startperiod', 'endperiod', 'n_conv', 'leave_n_out',
            'n_oneyr', 'method', 'ROC_leave_n_out',
            'wghts_accross_lags', 
            'perc_map', 'tfreq', 'lags', 'n_yrs', 'hotdaythres',
            'rollingmean',
            'mcKthres', 'perc_map', 'comp_perc', 'extra_wght_dur',
            'region', 'regionmcK',
            'add_lsm', 'min_n_gc', 'prec_reg_max_d']

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



l_ds_CPPA, l_ds_PEP, ex = func_CPPA.main(RV_ts, Prec_reg, ex)



#ex['output_dic_folder'] = '/Users/semvijverberg/surfdrive/McKinRepl/T2mmax_sst_Northern_PEPrectangle/iter_1979_2017_tf1_mcKthresp_2.5deg_60nyr_95tperc_0.8tc_1rmCPPA_wgtano_2019-03-25/lags[0,5,10,15,20,30,40,50,60]Ev1d0p_pmd0'

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
                 'l_ds_CPPA' : l_ds_CPPA,
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




#%%
args = ['python output_wrapper.py {}'.format(output_dic_folder)]
func_CPPA.kornshell_with_input(args, ex)


#%% Generate output in console



filename = 'output_main_dic'
dic = np.load(os.path.join(output_dic_folder, filename+'.npy'),  encoding='latin1').item()

# load settings
ex = dic['ex']
# load patterns
l_ds_CPPA = dic['l_ds_CPPA']
l_ds_PEP = dic['l_ds_PEP']

# write output in textfile
predict_folder = '{}{}_ts{}'.format(ex['pval_logit_final'], ex['logit_valid'], ex['use_ts_logit'])
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


#%%
# =============================================================================
# perform prediciton        
# =============================================================================


#ex, l_ds_CPPA = func_pred.make_prediction(l_ds_CPPA, l_ds_PEP, Prec_reg, ex)
from ROC_score import ROC_score_wrapper
predict_folder = '{}{}_ts{}'.format(ex['pval_logit_final'], ex['logit_valid'], ex['use_ts_logit'])
ex['exp_folder'] = os.path.join(ex['CPPA_folder'], predict_folder)
ex = func_pred.spatial_cov(RV_ts, ex, key1='spatcov_PEP', key2='spatcov_CPPA')
ex = ROC_score_wrapper(ex)


#%%
# =============================================================================
#   Plotting
# =============================================================================


lats = Prec_reg.latitude
lons = Prec_reg.longitude
array = np.zeros( (ex['n_conv'], len(ex['lags']), len(lats), len(lons)) )
patterns_Sem = xr.DataArray(data=array, coords=[range(ex['n_conv']), ex['lags'], lats, lons], 
                      dims=['n_tests', 'lag','latitude','longitude'], 
                      name='{}_tests_patterns_Sem'.format(ex['n_conv']), attrs={'units':'Kelvin'})
Prec_mcK = func_CPPA.find_region(Prec_reg, region=ex['region'])[0][0]
lats = Prec_mcK.latitude
lons = Prec_mcK.longitude
array = np.zeros( (ex['n_conv'], len(ex['lags']), len(lats), len(lons)) )
patterns_mcK = xr.DataArray(data=array, coords=[range(ex['n_conv']), ex['lags'], lats, lons], 
                      dims=['n_tests', 'lag','latitude','longitude'], 
                      name='{}_tests_patterns_mcK'.format(ex['n_conv']), attrs={'units':'Kelvin'})

for n in range(len(ex['train_test_list'])):
    ex['n'] = n
    if ex['use_ts_logit'] == True:
        name_for_ts = 'logit'
    elif ex['use_ts_logit'] == False:
        name_for_ts = 'CPPA'
        
    if (ex['method'][:6] == 'random'):
        if n == ex['n_stop']:
            # remove empty n_tests
            patterns_Sem = patterns_Sem.sel(n_tests=slice(0,ex['n_stop']))
            patterns_mcK = patterns_mcK.sel(n_tests=slice(0,ex['n_stop']))
            ex['n_conv'] = ex['n_stop']
    
    upd_pattern = l_ds_CPPA[n]['pattern_' + name_for_ts].sel(lag=ex['lags'])
    patterns_Sem[n,:,:,:] = upd_pattern * l_ds_CPPA[n]['std_train_min_lag']
    patterns_mcK[n,:,:,:] = l_ds_PEP[n]['pattern'].sel(lag=ex['lags'])

score_mcK       = np.round(ex['score'][-1][0], 2)
score_Sem       = np.round(ex['score'][-1][1], 2)
ROC_str_mcK      = ['{} days - ROC score {}'.format(ex['lags'][i], score_mcK[i]) for i in range(len(ex['lags'])) ]
ROC_str_Sem      = ['{} days - ROC score {}'.format(ex['lags'][i], score_Sem[i]) for i in range(len(ex['lags'])) ]
# Sem plot 
# share kwargs with mcKinnon plot

    
kwrgs = dict( {'title' : '', 'clevels' : 'notdefault', 'steps':17,
                    'vmin' : -0.5, 'vmax' : 0.5, 'subtitles' : ROC_str_Sem,
                   'cmap' : plt.cm.RdBu_r, 'column' : 1} )

mean_n_patterns = patterns_Sem.mean(dim='n_tests')
mean_n_patterns.attrs['units'] = 'mean over {} runs'.format(ex['n_conv'])
mean_n_patterns.attrs['title'] = 'Composite mean - Objective Precursor Pattern'
mean_n_patterns.name = 'ROC {}'.format(score_Sem)
filename = os.path.join(ex['exp_folder'], 'mean_over_{}_tests'.format(ex['n_conv']) )
func_CPPA.plotting_wrapper(mean_n_patterns, ex, filename, kwrgs=kwrgs)



# mcKinnon composite mean plot
filename = os.path.join(ex['exp_folder'], 'mcKinnon mean composite_tf{}_{}'.format(
            ex['tfreq'], ex['lags']))
mcK_mean = patterns_mcK.mean(dim='n_tests')
kwrgs['subtitles'] = ROC_str_mcK
mcK_mean.name = 'Composite mean green rectangle: ROC {}'.format(score_mcK)
mcK_mean.attrs['units'] = 'Kelvin'
mcK_mean.attrs['title'] = 'Composite mean - Subjective green rectangle pattern'
func_CPPA.plotting_wrapper(mcK_mean, ex, filename, kwrgs=kwrgs)




#%% Robustness of training precursor regions

subfolder = os.path.join(ex['exp_folder'], 'intermediate_results')
total_folder = os.path.join(ex['figpathbase'], subfolder)
if os.path.isdir(total_folder) != True : os.makedirs(total_folder)
years = range(ex['startyear'], ex['endyear'])

#n_land = np.sum(np.array(np.isnan(Prec_reg.values[0]),dtype=int) )
#n_sea = Prec_reg[0].size - n_land
if ex['method'] == 'iter':
    test_set_to_plot = [1990, 2000, 2010, 2012, 2015]
elif ex['method'][:6] == 'random':
    test_set_to_plot = [set(t[1]['RV'].time.dt.year.values) for t in ex['train_test_list'][::5]]
#test_set_to_plot = list(np.arange(0,ex['n_conv'],5))
for yr in test_set_to_plot: 
    n = test_set_to_plot.index(yr)
    Robustness_weights = l_ds_CPPA[n]['weights'].sel(lag=ex['lags'])
    size_trainset = ex['n_yrs'] - ex['leave_n_years_out']
    Robustness_weights.attrs['title'] = ('Robustness\n test yr(s): {}, single '
                            'training set (n={} yrs)'.format(yr,size_trainset))
    Robustness_weights.attrs['units'] = 'Weights [{} ... 1]'.format(ex['comp_perc'])
    filename = os.path.join(subfolder, Robustness_weights.attrs['title'].replace(
                            ' ','_')+'.png')
    for_plt = Robustness_weights.where(Robustness_weights.values != 0).copy()
#    n_pattern = Prec_reg[0].size - np.sum(np.array(np.isnan(for_plt[0]),dtype=int))
    
    if ex['n_conv'] == 1:
        steps = 19
    else:
        steps = 11
    kwrgs = dict( {'title' : for_plt.attrs['title'], 'clevels' : 'notdefault', 
                   'steps' : 11, 'subtitles': ROC_str_Sem, 
                   'vmin' : ex['comp_perc'], 'vmax' : for_plt.max().values+1E-9, 
                   'cmap' : plt.cm.viridis_r, 'column' : 2,
                   'cbar_vert' : 0.05, 'cbar_hght' : 0.01,
                   'adj_fig_h' : 1.25, 'adj_fig_w' : 1., 
                   'hspace' : 0.02, 'wspace' : 0.08} )
    
    func_CPPA.plotting_wrapper(for_plt, ex, filename, kwrgs=kwrgs)
    
    
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

#%% Robustness accross training sets
#    ex['lags'] = [5,15,30,50]

lats = patterns_Sem.latitude
lons = patterns_Sem.longitude
array = np.zeros( (ex['n_conv'], len(ex['lags']), len(lats), len(lons)) )
wgts_tests = xr.DataArray(data=array, 
                coords=[range(ex['n_conv']), ex['lags'], lats, lons], 
                dims=['n_tests', 'lag','latitude','longitude'], 
                name='{}_tests_wghts'.format(ex['n_conv']), attrs={'units':'wghts ['})
for n in range(ex['n_conv']):
    wgts_tests[n,:,:,:] = l_ds_CPPA[n]['weights'].sel(lag=ex['lags'])
    
    
if ex['leave_n_out']:
    n_lags = len(ex['lags'])
    n_lats = patterns_Sem.sel(n_tests=0).latitude.size
    n_lons = patterns_Sem.sel(n_tests=0).longitude.size
    
    pers_patt = patterns_Sem.sel(n_tests=0).sel(lag=ex['lags']).copy()
#    arrpatt = np.nan_to_num(patterns_Sem.values)
#    mask_patt = (arrpatt != 0)
#    arrpatt[mask_patt] = 1
    wghts = np.zeros( (n_lags, n_lats, n_lons) )
#    plt.imshow(arrpatt[0,0]) ; plt.colorbar()
    for l in ex['lags']:
        i = ex['lags'].index(l)
        wghts[i] = np.sum(wgts_tests[:,i,:,:], axis=0)
    pers_patt.values = wghts
    pers_patt = pers_patt.where(pers_patt.values != 0)
    size_trainset = ex['n_yrs'] - ex['leave_n_years_out']
    pers_patt.attrs['units'] = 'No. of times in final pattern [0 ... {}]'.format(ex['n_conv'])
    pers_patt.attrs['title'] = ('Robustness SST pattern\n{} different '
                            'training sets (n={} yrs)'.format(ex['n_conv'],size_trainset))
    filename = os.path.join(ex['exp_folder'], 'Robustness_across_{}_training_tests'.format(ex['n_conv']) )
    vmax = ex['n_conv'] 
    mean = np.round(pers_patt.mean(dim=('latitude', 'longitude')).values, 1)
#    mean = pers_patt.quantile(0.80, dim=('latitude','longitude')).values
    std =  np.round(pers_patt.std(dim=('latitude', 'longitude')).values, 0)
    ax_text = ['mean = {}±{}'.format(mean[l],int(std[l])) for l in range(len(ex['lags']))]
    kwrgs = dict( {'title' : pers_patt.attrs['title'], 'clevels' : 'notdefault', 
                   'steps' : 11, 'subtitles': ROC_str_Sem, 
                   'vmin' : max(0,vmax-20), 'vmax' : vmax, 'clim' : (max(0,vmax-20), vmax),
                   'cmap' : plt.cm.magma_r, 'column' : 2, 'extend':['min','yellow'],
                   'cbar_vert' : 0.05, 'cbar_hght' : 0.01,
                   'adj_fig_h' : 1.25, 'adj_fig_w' : 1., 
                   'hspace' : 0.02, 'wspace' : 0.08,
                   'ax_text': ax_text } )
    func_CPPA.plotting_wrapper(pers_patt, ex, filename, kwrgs=kwrgs)
#%% Weighing features if there are extracted every run (training set)
# weighted by persistence of pattern over
if ex['leave_n_out']:
    kwrgs = dict( {'title' : '', 'clevels' : 'notdefault', 'steps':17,
                    'vmin' : -0.5, 'vmax' : 0.5, 'subtitles' : ROC_str_Sem,
                   'cmap' : plt.cm.RdBu_r, 'column' : 1} )
    # weighted by persistence (all years == wgt of 1, less is below 1)
    mean_n_patterns = patterns_Sem.mean(dim='n_tests') * wghts/np.max(wghts)
    mean_n_patterns['lag'] = ROC_str_Sem

    title = 'Composite mean - Objective Precursor Pattern'#\nweighted by robustness over {} tests'.format(
#                                            ex['n_conv'])
    if mean_n_patterns.sum().values != 0.:
        mean_n_patterns.attrs['units'] = 'Kelvin'
        mean_n_patterns.attrs['title'] = title
                             
        mean_n_patterns.name = 'ROC {}'.format(score_Sem)
        filename = os.path.join(ex['exp_folder'], ('weighted by robustness '
                             'over {} tests'.format(ex['n_conv']) ))
#        kwrgs = dict( {'title' : mean_n_patterns.name, 'clevels' : 'default', 'steps':17,
#                        'vmin' : -3*mean_n_patterns.std().values, 'vmax' : 3*mean_n_patterns.std().values, 
#                       'cmap' : plt.cm.RdBu_r, 'column' : 2} )
        func_CPPA.plotting_wrapper(mean_n_patterns, ex, filename, kwrgs=kwrgs)


#%% Plotting prediciton time series vs truth:
yrs_to_plot = [1985, 1990, 1995, 2004, 2007, 2012, 2015]
#yrs_to_plot = list(np.arange(ex['startyear'],ex['endyear']+1))
test = ex['train_test_list'][0][1]        
plotting_timeseries(test, yrs_to_plot, ex) 


#%% Initial regions from only composite extraction:

lags = ex['lags']
lags = [5] #5,15,30,50]
if ex['leave_n_out']:
    subfolder = os.path.join(ex['exp_folder'], 'intermediate_results')
    total_folder = os.path.join(ex['figpathbase'], subfolder)
    if os.path.isdir(total_folder) != True : os.makedirs(total_folder)
    years = range(ex['startyear'], ex['endyear'])
    for n in np.arange(0, ex['n_conv'], 3, dtype=int): 
        yr = years[n]
        pattern_num_init = l_ds_CPPA[n]['pat_num_CPPA_clust'].sel(lag=lags)
        ROC_str_Sem_ = [ROC_str_Sem[ex['lags'].index(l)] for l in lags]
        


        pattern_num_init.attrs['title'] = ('{} - CPPA regions'.format(yr))
        filename = os.path.join(subfolder, pattern_num_init.attrs['title'].replace(
                                ' ','_')+'.png')
        for_plt = pattern_num_init.copy()
        for_plt.values = for_plt.values-0.5
        
        
        kwrgs = dict( {'title' : for_plt.attrs['title'], 'clevels' : 'notdefault', 
                       'steps' : ex['max_N_regs']+1, 'subtitles': ROC_str_Sem_,
                       'vmin' : 0, 'vmax' : ex['max_N_regs'], 
                       'cmap' : plt.cm.tab20, 'column' : 1,
                       'cbar_vert' : 0.07, 'cbar_hght' : 0.000,
                       'adj_fig_h' : 1.5, 'adj_fig_w' : 1., 
                       'hspace' : -0.03, 'wspace' : 0.08,
                       'cticks_center' : True} )
        
        func_CPPA.plotting_wrapper(for_plt, ex, filename, kwrgs=kwrgs)
        
        if ex['logit_valid'] == True:
            pattern_num = l_ds_CPPA[n]['pat_num_logit']
            pattern_num.attrs['title'] = ('{} - regions that were kept after logit regression '
                                         'pval < {}'.format(yr, ex['pval_logit_final']))
            filename = os.path.join(subfolder, pattern_num.attrs['title'].replace(
                                    ' ','_')+'.png')
            for_plt = pattern_num.copy()
            for_plt.values = for_plt.values-0.5
            kwrgs = dict( {'title' : for_plt.attrs['title'], 'clevels' : 'notdefault', 
                           'steps' : for_plt.max()+2, 'subtitles': ROC_str_Sem_,
                           'vmin' : 0, 'vmax' : for_plt.max().values+0.5, 
                           'cmap' : plt.cm.tab20, 'column' : 2} )
            
            func_CPPA.plotting_wrapper(for_plt, ex, filename, kwrgs=kwrgs)











#
#
#filename = 'output_main_dic'
#dic = np.load(os.path.join(output_dic_folder,filename+'.npy'),  encoding='latin1').item()
#ex = dic['ex']
#l_ds_PEP        = [ex['score_per_run'][i][2] for i in range(len(ex['score_per_run']))]
#l_ds_CPPA        = [ex['score_per_run'][i][3] for i in range(len(ex['score_per_run']))]
#
#ex['use_ts_logit']=False
#
#predict_folder = '{}{}_ts{}'.format(ex['pval_logit_final'], ex['logit_valid'], ex['use_ts_logit'])
#ex['exp_folder'] = os.path.join(ex['CPPA_folder'], predict_folder)
#predict_folder = os.path.join(ex['figpathbase'], ex['exp_folder'])
#if os.path.isdir(predict_folder) != True : os.makedirs(predict_folder)
#
#
#ex, patterns_Sem, patterns_mcK = func_pred.make_prediction(l_ds_CPPA, l_ds_PEP, Prec_reg, ex)
#
#
#
#
## =============================================================================
## Plotting
## =============================================================================
#
##%%
#events_per_year = [ex['score_per_run'][i][1] for i in range(len(ex['score_per_run']))]
#l_ds_PEP        = [ex['score_per_run'][i][2] for i in range(len(ex['score_per_run']))]
#l_ds_CPPA        = [ex['score_per_run'][i][3] for i in range(len(ex['score_per_run']))]
#ran_ROCS        = [ex['score_per_run'][i][4] for i in range(len(ex['score_per_run']))]
#score_mcK       = np.round(ex['score_per_run'][-1][2]['score'], 2)
#score_Sem       = np.round(ex['score_per_run'][-1][3]['score'], 2)
#ROC_str_mcK      = ['{} days - ROC score {}'.format(ex['lags'][i], score_mcK[i].values) for i in range(len(ex['lags'])) ]
#ROC_str_Sem      = ['{} days - ROC score {}'.format(ex['lags'][i], score_Sem[i].values) for i in range(len(ex['lags'])) ]
## Sem plot
## share kwargs with mcKinnon plot
#
#    
#kwrgs = dict( {'title' : '', 'clevels' : 'notdefault', 'steps':17,
#                    'vmin' : -0.5, 'vmax' : 0.5, 'subtitles' : ROC_str_Sem,
#                   'cmap' : plt.cm.RdBu_r, 'column' : 1} )
#
#mean_n_patterns = patterns_Sem.mean(dim='n_tests')
#mean_n_patterns.attrs['units'] = 'mean over {} runs'.format(ex['n_conv'])
#mean_n_patterns.attrs['title'] = 'Composite mean - Objective Precursor Pattern'
#mean_n_patterns.name = 'ROC {}'.format(score_Sem.values)
#filename = os.path.join(ex['exp_folder'], 'mean_over_{}_tests'.format(ex['n_conv']) )
#func_mcK.plotting_wrapper(mean_n_patterns, ex, filename, kwrgs=kwrgs)
#
#
#
## mcKinnon composite mean plot
#filename = os.path.join(ex['exp_folder'], 'mcKinnon mean composite_tf{}_{}'.format(
#            ex['tfreq'], ex['lags']))
#mcK_mean = patterns_mcK.mean(dim='n_tests')
#kwrgs['subtitles'] = ROC_str_mcK
#mcK_mean.name = 'Composite mean green rectangle: ROC {}'.format(score_mcK.values)
#mcK_mean.attrs['units'] = 'Kelvin'
#mcK_mean.attrs['title'] = 'Composite mean - Subjective green rectangle pattern'
#func_mcK.plotting_wrapper(mcK_mean, ex, filename, kwrgs=kwrgs)
#
##if (ex['leave_n_out'] == True) and (ex['ROC_leave_n_out'] == False):
##    # mcKinnon std plot
##    filename = os.path.join(ex['exp_folder'], 'mcKinnon std composite_tf{}_{}'.format(
##                ex['tfreq'], ex['lags']))
##    mcK_std = patterns_mcK.std(dim='n_tests')
##    mcK_std.name = 'Composite std: ROC {}'.format(score_mcK.values)
##    mcK_std.attrs['units'] = 'Kelvin'
##    func_mcK.plotting_wrapper(mcK_std, filename, ex)
#
##%% Robustness of training precursor regions
#
#subfolder = os.path.join(ex['exp_folder'], 'intermediate_results')
#total_folder = os.path.join(ex['figpathbase'], subfolder)
#if os.path.isdir(total_folder) != True : os.makedirs(total_folder)
#years = range(ex['startyear'], ex['endyear'])
#
##n_land = np.sum(np.array(np.isnan(Prec_reg.values[0]),dtype=int) )
##n_sea = Prec_reg[0].size - n_land
#if ex['method'] == 'iter':
#    test_set_to_plot = [1990, 2000, 2010, 2012, 2015]
#elif ex['method'][:6] == 'random':
#    test_set_to_plot = [set(t[1]['RV'].time.dt.year.values) for t in ex['train_test_list'][::5]]
##test_set_to_plot = list(np.arange(0,ex['n_conv'],5))
#for yr in test_set_to_plot: 
#    n = test_set_to_plot.index(yr)
#    Robustness_weights = l_ds_CPPA[n]['weights']
#    size_trainset = ex['n_yrs'] - ex['leave_n_years_out']
#    Robustness_weights.attrs['title'] = ('Robustness\n test yr(s): {}, single '
#                            'training set (n={} yrs)'.format(yr,size_trainset))
#    Robustness_weights.attrs['units'] = 'Weights [{} ... 1]'.format(ex['comp_perc'])
#    filename = os.path.join(subfolder, Robustness_weights.attrs['title'].replace(
#                            ' ','_')+'.png')
#    for_plt = Robustness_weights.where(Robustness_weights.values != 0).copy()
##    n_pattern = Prec_reg[0].size - np.sum(np.array(np.isnan(for_plt[0]),dtype=int))
#    
#    if ex['n_conv'] == 1:
#        steps = 19
#    else:
#        steps = 11
#    kwrgs = dict( {'title' : for_plt.attrs['title'], 'clevels' : 'notdefault', 
#                   'steps' : 11, 'subtitles': ROC_str_Sem, 
#                   'vmin' : ex['comp_perc'], 'vmax' : for_plt.max().values+1E-9, 
#                   'cmap' : plt.cm.viridis_r, 'column' : 2,
#                   'cbar_vert' : 0.05, 'cbar_hght' : 0.01,
#                   'adj_fig_h' : 1.25, 'adj_fig_w' : 1., 
#                   'hspace' : 0.02, 'wspace' : 0.08} )
#    
#    func_mcK.plotting_wrapper(for_plt, ex, filename, kwrgs=kwrgs)
#    
#    
##%%
#if ex['load_mcK'] == False:
#    filename = os.path.join(ex['RV1d_ts_path'], ex['RVts_filename'])
#    dicRV = np.load(filename,  encoding='latin1').item()
#    xarray_plot(dicRV['RV_array']['mask'], path=folder, name='RV_mask', saving=True)
#
#func_mcK.plot_oneyr_events(RV_ts, ex, 2012, folder, saving=True)
### plotting same figure as in paper
##for i in range(2005, 2010):
##    func_mcK.plot_oneyr_events(RV_ts, ex, i, folder, saving=True)
#
##%% Robustness accross training sets
#
#lats = patterns_Sem.latitude
#lons = patterns_Sem.longitude
#array = np.zeros( (ex['n_conv'], len(ex['lags']), len(lats), len(lons)) )
#wgts_tests = xr.DataArray(data=array, 
#                coords=[range(ex['n_conv']), ex['lags'], lats, lons], 
#                dims=['n_tests', 'lag','latitude','longitude'], 
#                name='{}_tests_wghts'.format(ex['n_conv']), attrs={'units':'wghts ['})
#for n in range(ex['n_conv']):
#    wgts_tests[n,:,:,:] = l_ds_CPPA[n]['weights']
#    
#    
#if ex['leave_n_out']:
#    n_lags = patterns_Sem.sel(n_tests=0).lag.size
#    n_lats = patterns_Sem.sel(n_tests=0).latitude.size
#    n_lons = patterns_Sem.sel(n_tests=0).longitude.size
#    
#    pers_patt = patterns_Sem.sel(n_tests=0).copy()
##    arrpatt = np.nan_to_num(patterns_Sem.values)
##    mask_patt = (arrpatt != 0)
##    arrpatt[mask_patt] = 1
#    wghts = np.zeros( (n_lags, n_lats, n_lons) )
##    plt.imshow(arrpatt[0,0]) ; plt.colorbar()
#    for l in ex['lags']:
#        i = ex['lags'].index(l)
#        wghts[i] = np.sum(wgts_tests[:,i,:,:], axis=0)
#    pers_patt.values = wghts
#    pers_patt = pers_patt.where(pers_patt.values != 0)
#    size_trainset = ex['n_yrs'] - ex['leave_n_years_out']
#    pers_patt.attrs['units'] = 'No. of times in final pattern [0 ... {}]'.format(ex['n_conv'])
#    pers_patt.attrs['title'] = ('Robustness\n{} different '
#                            'training sets (n={} yrs)'.format(ex['n_conv'],size_trainset))
#    filename = os.path.join(ex['exp_folder'], 'Robustness_across_{}_training_tests'.format(ex['n_conv']) )
#    vmax = ex['n_conv'] + 1E-9
#    mean = np.round(pers_patt.mean(dim=('latitude', 'longitude')).values, 1)
##    mean = pers_patt.quantile(0.80, dim=('latitude','longitude')).values
#    std =  np.round(pers_patt.std(dim=('latitude', 'longitude')).values, 0)
#    ax_text = ['mean = {}±{}'.format(mean[l],int(std[l])) for l in range(len(ex['lags']))]
#    kwrgs = dict( {'title' : pers_patt.attrs['title'], 'clevels' : 'notdefault', 
#                   'steps' : 16, 'subtitles': ROC_str_Sem, 
#                   'vmin' : 0, 'vmax' : vmax, 'clim' : (max(0,vmax-20), vmax),
#                   'cmap' : plt.cm.magma_r, 'column' : 2, 'extend':['min','yellow'],
#                   'cbar_vert' : 0.05, 'cbar_hght' : 0.01,
#                   'adj_fig_h' : 1.25, 'adj_fig_w' : 1., 
#                   'hspace' : 0.02, 'wspace' : 0.08,
#                   'ax_text': ax_text } )
#    func_mcK.plotting_wrapper(pers_patt, ex, filename, kwrgs=kwrgs)
##%% Weighing features if there are extracted every run (training set)
## weighted by persistence of pattern over
#if ex['leave_n_out']:
#    kwrgs = dict( {'title' : '', 'clevels' : 'notdefault', 'steps':17,
#                    'vmin' : -0.5, 'vmax' : 0.5, 'subtitles' : ROC_str_Sem,
#                   'cmap' : plt.cm.RdBu_r, 'column' : 1} )
#    # weighted by persistence (all years == wgt of 1, less is below 1)
#    mean_n_patterns = patterns_Sem.mean(dim='n_tests') * wghts/np.max(wghts)
#    mean_n_patterns['lag'] = ROC_str_Sem
#
#    title = 'Composite mean - Objective Precursor Pattern'#\nweighted by robustness over {} tests'.format(
##                                            ex['n_conv'])
#    if mean_n_patterns.sum().values != 0.:
#        mean_n_patterns.attrs['units'] = 'Kelvin'
#        mean_n_patterns.attrs['title'] = title
#                             
#        mean_n_patterns.name = 'ROC {}'.format(score_Sem.values)
#        filename = os.path.join(ex['exp_folder'], ('weighted by robustness '
#                             'over {} tests'.format(ex['n_conv']) ))
##        kwrgs = dict( {'title' : mean_n_patterns.name, 'clevels' : 'default', 'steps':17,
##                        'vmin' : -3*mean_n_patterns.std().values, 'vmax' : 3*mean_n_patterns.std().values, 
##                       'cmap' : plt.cm.RdBu_r, 'column' : 2} )
#        func_mcK.plotting_wrapper(mean_n_patterns, ex, filename, kwrgs=kwrgs)
#
#
##%% Plotting prediciton time series vs truth:
#yrs_to_plot = [1985, 1990, 1995, 2004, 2007, 2012, 2015]
##yrs_to_plot = list(np.arange(ex['startyear'],ex['endyear']+1))
#test = ex['train_test_list'][0][1]        
#plotting_timeseries(test, yrs_to_plot, ex) 
#
#
##%% Initial regions from only composite extraction:
#
#
#if ex['leave_n_out']:
#    subfolder = os.path.join(ex['exp_folder'], 'intermediate_results')
#    total_folder = os.path.join(ex['figpathbase'], subfolder)
#    if os.path.isdir(total_folder) != True : os.makedirs(total_folder)
#    years = range(ex['startyear'], ex['endyear'])
#    for n in np.arange(0, ex['n_conv'], 6, dtype=int): 
#        yr = years[n]
#        pattern_num_init = l_ds_CPPA[n]['pat_num_CPPA']
#        
#
#
#        pattern_num_init.attrs['title'] = ('{} - CPPA regions'.format(yr))
#        filename = os.path.join(subfolder, pattern_num_init.attrs['title'].replace(
#                                ' ','_')+'.png')
#        for_plt = pattern_num_init.copy()
#        for_plt.values = for_plt.values-0.5
#        kwrgs = dict( {'title' : for_plt.attrs['title'], 'clevels' : 'notdefault', 
#                       'steps' : for_plt.max()+2, 'subtitles': ROC_str_Sem,
#                       'vmin' : 0, 'vmax' : for_plt.max().values+0.5, 
#                       'cmap' : plt.cm.tab10, 'column' : 2} )
#        
#        func_mcK.plotting_wrapper(for_plt, ex, filename, kwrgs=kwrgs)
#        
#        if ex['logit_valid'] == True:
#            pattern_num = l_ds_CPPA[n]['pat_num_logit']
#            pattern_num.attrs['title'] = ('{} - regions that were kept after logit regression '
#                                         'pval < {}'.format(yr, ex['pval_logit_final']))
#            filename = os.path.join(subfolder, pattern_num.attrs['title'].replace(
#                                    ' ','_')+'.png')
#            for_plt = pattern_num.copy()
#            for_plt.values = for_plt.values-0.5
#            kwrgs = dict( {'title' : for_plt.attrs['title'], 'clevels' : 'notdefault', 
#                           'steps' : for_plt.max()+2, 'subtitles': ROC_str_Sem,
#                           'vmin' : 0, 'vmax' : for_plt.max().values+0.5, 
#                           'cmap' : plt.cm.tab10, 'column' : 2} )
#            
#            func_mcK.plotting_wrapper(for_plt, ex, filename, kwrgs=kwrgs)
#        
#        
#
# 
## =============================================================================
## =============================================================================
## =============================================================================
## # # 
## =============================================================================
## =============================================================================
## =============================================================================
#
#
#
##%% Load data
#answer = input("You sure you want to load data, y or n?\n")
#if answer == 'y':
#    import numpy as np
#    import os
#    import xarray as xr
#    import matplotlib.pyplot as plt
#    output_dic_folder = ('/Users/semvijverberg/surfdrive/MckinRepl/T2mmax_sst_Northern_PEPrectangle/iter_1979_2017_tf1_lags[5, 15, 30, 50]_mcKthresp_2.5deg_60nyr_0.05False_tsFalse_95tperc_0.8tc_1_2019-02-01')
#
#    
#    #
#    filename = 'output_main_dic'
#    #
#    dic = np.load(os.path.join(output_dic_folder,filename+'.npy'),  encoding='latin1').item()
#    ex = dic['ex']
#    if 'patterns_Sem' in dic.keys():
#        patterns_Sem = dic['patterns_Sem']
#        patterns_mcK = dic['patterns_mcK']
#    if 'startperiod' not in ex.keys():
#        ex['startperiod'] = ex['sstartdate'][-5:]
#        ex['endperiod'] = ex['senddate'][-5:]
#
#print_ex = ['RV_name', 'name', 'load_mcK', 'grid_res', 'startyear', 'endyear', 
#            'startperiod', 'endperiod', 'n_conv', 'leave_n_out',
#            'n_oneyr', 'method', 'ROC_leave_n_out', 'wghts_std_anom', 
#            'wghts_accross_lags', 'n_strongest',
#            'perc_map', 'tfreq', 'lags', 'n_yrs', 'hotdaythres',
#            'pval_logit_first', 'pval_logit_final', 'rollingmean',
#            'mcKthres', 'new_model_sel', 'perc_map', 'comp_perc',
#            'logit_valid', 'use_ts_logit', 'region', 'regionmcK',
#            'add_lsm', 'min_n_gc']
#def printset(print_ex=print_ex, ex=ex):
#    max_key_len = max([len(i) for i in print_ex])
#    for key in print_ex:
#        key_len = len(key)
#        expand = max_key_len - key_len
#        key_exp = key + ' ' * expand
#        printline = '\'{}\'\t\t{}'.format(key_exp, ex[key])
#        print(printline)
#
#printset()
##%%
#    
#    
#    
##l_ds_PEP        = [ex['score_per_run'][i][2] for i in range(len(ex['score_per_run']))]
##lats = l_ds_PEP[0]['pattern'].latitude
##lons = l_ds_PEP[0]['pattern'].longitude
##array = np.zeros( (ex['n_conv'], len(ex['lags']), len(lats), len(lons)) )
##patterns_mcK = xr.DataArray(data=array, coords=[range(ex['n_conv']), ex['lags'], lats, lons], 
##                          dims=['n_tests', 'lag','latitude','longitude'], 
##                          name='{}_tests_patterns_mcK'.format(ex['n_conv']), attrs={'units':'Kelvin'})
##for n in range(ex['n_conv']):
##    patterns_mcK[n,:,:,:] = l_ds_PEP[n]['pattern']
#
#
##filename = 'only_patterns_and_weights'
##to_dict = dict( {'patterns_Sem'  :  patterns_Sem} )
##np.save(os.path.join(output_dic_folder, filename+'.npy'), to_dict)
#
#
#
#
##