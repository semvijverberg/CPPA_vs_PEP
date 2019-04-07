#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 14:15:35 2019

@author: semvijverberg
"""
import os, sys
os.chdir('/Users/semvijverberg/surfdrive/Scripts/Extracting_precursor/')
script_dir = os.getcwd()
sys.path.append(script_dir)
if sys.version[:1] == '3':
    from importlib import reload as rel
import func_CPPA
import func_pred
import numpy as np
import pandas as pd
import xarray as xr 
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from ROC_score import ROC_score_wrapper
from ROC_score import plotting_timeseries
xarray_plot = func_CPPA.xarray_plot

#output_dic_folder = '/Users/semvijverberg/surfdrive/McKinRepl/T2mmax_sst_Northern_PEPrectangle/random90_leave_5_out_1979_2017_tf1_lags[0,5,10,15,20,30,40,50]_mcKthresp_2.5deg_60nyr_95tperc_0.8tc_1rm_2019-02-05'

if len(sys.argv) == 2:
    output_dic_folder = sys.argv[1]
elif 'output_dic_folder' in globals():
    output_dic_folder = output_dic_folder
else:
    output_dic_folder = input("paste experiment folder:\n")

filename = 'output_main_dic'
dic = np.load(os.path.join(output_dic_folder,filename+'.npy'),  encoding='latin1').item()
ex = dic['ex']


# =============================================================================
# load data
# =============================================================================
RV_ts, Prec_reg, ex = func_CPPA.load_data(ex)

print_ex = ['RV_name', 'name', 'load_mcK', 'max_break',
            'min_dur', 'grid_res', 'startyear', 'endyear', 
            'startperiod', 'endperiod', 'n_conv', 'leave_n_out',
            'n_oneyr', 'method', 'ROC_leave_n_out', 'wghts_std_anom', 
            'wghts_accross_lags', 'splittrainfeat', 'n_strongest',
            'perc_map', 'tfreq', 'lags', 'n_yrs', 'hotdaythres',
            'pval_logit_first', 'pval_logit_final', 'rollingmean',
            'mcKthres', 'perc_map', 'comp_perc',
            'logit_valid', 'use_ts_logit', 'region', 'regionmcK',
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

# =============================================================================
# Finish load data
# =============================================================================

    
#ex['lags'] = [5,15,30,50]
dic_exp = ({'logit_valid_ts'  :   (True,True), 
            'logit_ts'       :   (True,False),
            'CPPA_spatcov'      :   (False,False),
            'logit_valid_spatcov' :   (False,True)
            })
dic_exp = ({
            'CPPA_spatcov'      :   (False,False),
            })
    
ex['shared_folder'] = ('/Users/semvijverberg/Dropbox/VIDI_Coumou/Paper1_Sem/'
                 'output_summ/iter_1979_2017_tf1_mcKthresp_2.5deg_60nyr'
                 '_95tperc_0.8tc_1rm_2019-02-26/lags[0,5,10,15,20,30,40,50,60]Ev1d0p_pmd1')

df = pd.DataFrame(index=ex['lags'], 
                  columns=['nino3.4', 'nino3.4rm5', 'PEP', 'CPPA_spatcov', 
                           'logit_ts', 'logit_valid_spatcov', 
                           'logit_valid_ts'])
df.index.name = 'lag'
df.to_csv(ex['shared_folder']+'/output_summ.csv')

ex['output_ts_folder'] += ''

#%%


def all_output_wrapper(dic, exp_key='CPPA_spatcov'):
    #%%
    # load settings
    ex = dic['ex']
    # load patterns
    l_ds_CPPA = dic['l_ds_CPPA']
    l_ds_PEP = dic['l_ds_PEP']
    
    # adapt settings for prediction
    ex['use_ts_logit'], ex['logit_valid'] = dic_exp[exp_key] 
    
    
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
#            print(printline)
            print(printline, file=text_file)
            
    
    # =============================================================================
    # Perform predictions
    # =============================================================================
    # by logistic regression
    if ex['use_ts_logit'] == True or ex['logit_valid'] == True:
        l_ds_CPPA, ex = func_pred.logit_fit_new(l_ds_CPPA, RV_ts, ex) 
    # by spatial covariance
    ex = func_pred.spatial_cov(ex)
    
    # =============================================================================
    # Calculate AUC score
    # =============================================================================
    print(exp_key)
    ex = ROC_score_wrapper(ex)
        
    score_mcK       = np.round(ex['score'][-1][0], 2)
    score_Sem       = np.round(ex['score'][-1][1], 2)

    # =============================================================================
    # Store data in output summary
    # =============================================================================
    if ex['lags'] == [0, 5, 10, 15, 20, 30, 40, 50, 60]:
        df = pd.read_csv(ex['shared_folder']+'/output_summ.csv', index_col='lag')
        df['PEP'] = score_mcK
        df[exp_key] = score_Sem
        df.to_csv(ex['shared_folder']+'/output_summ.csv')
    
    for idx, lag in enumerate(ex['lags']):
        df = pd.DataFrame(data=ex['test_ts_Sem'][idx])
        filename_csv = exp_key + '_' + str(lag) + '.csv'
        df.to_csv(os.path.join(predict_folder, filename_csv))

#    # El nino 3.4
#    ex = func_pred.spatial_cov(ex, 'nino3.4', 'nino3.4rm5')
#    
#    # =============================================================================
#    # Calculate AUC score
#    # =============================================================================
#    print(exp_key)
#    ex = ROC_score_wrapper(ex)
#        
#    score_nino          = np.round(ex['score'][-1][0], 2)
#    score_ninorm5       = np.round(ex['score'][-1][1], 2)
#
#    # =============================================================================
#    # Store data in output summary
#    # =============================================================================
#    if ex['lags'] == [0, 5, 10, 15, 20, 30, 40, 50, 60]:
#        df = pd.read_csv(ex['shared_folder']+'/output_summ.csv', index_col='lag')
#        df['nino3.4']   = score_nino
#        df['nino3.4rm5'] = score_ninorm5
#        df.to_csv(ex['shared_folder']+'/output_summ.csv')
#    
    
    #%%
# =============================================================================
#   Plotting
# =============================================================================
#    lags = [0,5,15,30,50]
#    lags = [15]
#    score_Sem = [score_Sem[ex['lags'].index(l)] for l in lags]
#    ex['lags'] = lags
    
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
        patterns_Sem[n,:,:,:] = upd_pattern * l_ds_CPPA[n]['std_train_min_lag'].sel(lag=ex['lags'])
        patterns_mcK[n,:,:,:] = l_ds_PEP[n]['pattern'].sel(lag=ex['lags'])
    
    ROC_str_mcK      = ['{} days - AUC score {}'.format(ex['lags'][i], score_mcK[i]) for i in range(len(ex['lags'])) ]
    ROC_str_Sem      = ['{} days - AUC score {}'.format(ex['lags'][i], score_Sem[i]) for i in range(len(ex['lags'])) ]
    # Sem plot 
    # share kwargs with mcKinnon plot
    
    ROC_str_Sem_ = [ROC_str_Sem[ex['lags'].index(l)] for l in lags]  
    kwrgs = dict( {'title' : '', 'clevels' : 'notdefault', 'steps':17,
                        'vmin' : -0.4, 'vmax' : 0.4, 'subtitles' : ROC_str_Sem_,
                       'cmap' : plt.cm.RdBu_r, 'column' : 1} )
    
    mean_n_patterns = patterns_Sem.mean(dim='n_tests')
    mean_n_patterns.attrs['units'] = 'mean over {} runs'.format(ex['n_conv'])
    mean_n_patterns.attrs['title'] = 'Composite mean - Objective Precursor Pattern'
    mean_n_patterns.name = 'ROC {}'.format(score_Sem)
    filename = os.path.join(ex['exp_folder'], 'mean_over_{}_tests'.format(ex['n_conv']) )
    func_CPPA.plotting_wrapper(mean_n_patterns, ex, filename, kwrgs=kwrgs)
    
    
    
    # mcKinnon composite mean plot
    kwrgs['drawbox'] = True
    filename = os.path.join(ex['exp_folder'], 'mcKinnon mean composite_tf{}_{}'.format(
                ex['tfreq'], ex['lags']))
    mcK_mean = patterns_mcK.mean(dim='n_tests')
    kwrgs['subtitles'] = ROC_str_mcK
    mcK_mean.name = 'Composite mean green rectangle: ROC {}'.format(score_mcK)
    mcK_mean.attrs['units'] = 'Kelvin'
    mcK_mean.attrs['title'] = 'Composite mean - Subjective green rectangle pattern'
    func_CPPA.plotting_wrapper(mcK_mean, ex, filename, kwrgs=kwrgs)
    kwrgs.pop('drawbox')
    
    #if (ex['leave_n_out'] == True) and (ex['ROC_leave_n_out'] == False):
    #    # mcKinnon std plot
    #    filename = os.path.join(ex['exp_folder'], 'mcKinnon std composite_tf{}_{}'.format(
    #                ex['tfreq'], ex['lags']))
    #    mcK_std = patterns_mcK.std(dim='n_tests')
    #    mcK_std.name = 'Composite std: ROC {}'.format(score_mcK)
    #    mcK_std.attrs['units'] = 'Kelvin'
    #    func_CPPA.plotting_wrapper(mcK_std, filename, ex)
    
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
    lags = ex['lags']
#    lags = [10] #5,15,30,50]   
    ROC_str_Sem_ = [ROC_str_Sem[ex['lags'].index(l)] for l in lags]
    
    if ex['leave_n_out']:
        kwrgs = dict( {'title' : '', 'clevels' : 'notdefault', 'steps':17,
                        'vmin' : -0.4, 'vmax' : 0.4, 'subtitles' : ROC_str_Sem_,
                       'cmap' : plt.cm.RdBu_r, 'column' : 1,
                       'cbar_vert' : 0.07, 'cbar_hght' : 0.01,
                       'adj_fig_h' : 1.5, 'adj_fig_w' : 1., 
                       'hspace' : 0.02, 'wspace' : 0.08,} )
        # weighted by persistence (all years == wgt of 1, less is below 1)
        mean_n_patterns = patterns_Sem.mean(dim='n_tests') * wghts/np.max(wghts)
        mean_n_patterns = mean_n_patterns.sel(lag=lags)
        mean_n_patterns['lag'] = ROC_str_Sem_

        title = 'Composite mean - Objective Precursor Pattern'

        if mean_n_patterns.sum().values != 0.:
            mean_n_patterns.attrs['units'] = 'Kelvin'
            mean_n_patterns.attrs['title'] = title
                                 
            mean_n_patterns.name = 'ROC {}'.format(score_Sem)
            filename = os.path.join(ex['exp_folder'], ('wghtrobus_'
                                 '{}_tests_{}'.format(ex['n_conv'], lags) ))
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
    

    if ex['leave_n_out']:
        subfolder = os.path.join(ex['exp_folder'], 'intermediate_results')
        total_folder = os.path.join(ex['figpathbase'], subfolder)
        if os.path.isdir(total_folder) != True : os.makedirs(total_folder)
        years = range(ex['startyear'], ex['endyear'])
        for n in np.arange(0, ex['n_conv'], 3, dtype=int): 
            yr = years[n]
            pattern_num_init = l_ds_CPPA[n]['pat_num_CPPA_clust'].sel(lag=lags)
            
            
    
    
            pattern_num_init.attrs['title'] = ('{} - CPPA regions'.format(yr))
            filename = os.path.join(subfolder, pattern_num_init.attrs['title'].replace(
                                    ' ','_')+'.png')
            for_plt = pattern_num_init.copy()
            for_plt.values = for_plt.values-0.5
            
            
            kwrgs = dict( {'title' : for_plt.attrs['title'], 'clevels' : 'notdefault', 
                           'steps' : ex['max_N_regs']+1, 'subtitles': ROC_str_Sem_,
                           'vmin' : 0, 'vmax' : ex['max_N_regs'], 
                           'cmap' : plt.cm.tab20, 'column' : 1,
                           'cbar_vert' : 0.07, 'cbar_hght' : -0.03,
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

#%%
if __name__ == '__main__':    
    start = time.time()
    num_workers = mp.cpu_count()  
    pool = mp.Pool(num_workers)
    print(dic_exp.keys())
    for exp_key in dic_exp.keys():
        
        pool.apply_async(all_output_wrapper, args= (dic, exp_key))
        
    pool.close()
    pool.join()
    end = time.time()
    print("It took ", end - start)