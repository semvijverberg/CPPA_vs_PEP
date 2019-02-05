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
import func_mcK
import numpy as np
import xarray as xr 
#matplotlib.use('WXAgg',warn=False, force=True)
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from ROC_score import plotting_timeseries

if sys.argv != ['']:
    output_dic_folder = sys.argv[1]
else:
    output_dic_folder = input("paste experiment folder:\n")
#output_dic_folder = '/Users/semvijverberg/surfdrive/McKinRepl/T2mmax_sst_Northern_PEPrectangle/iter_1979_2017_tf1_lags[5,15,30,50]_mcKthresp_2.5deg_60nyr_95tperc_0.8tc_1rm_2019-02-05'
filename = 'output_main_dic'
dic = np.load(os.path.join(output_dic_folder,filename+'.npy'),  encoding='latin1').item()
ex = dic['ex']


# =============================================================================
# load data
# =============================================================================
RV_ts, Prec_reg, ex = func_mcK.load_data(ex)

print_ex = ['RV_name', 'name', 'load_mcK', 'grid_res', 'startyear', 'endyear', 
            'startperiod', 'endperiod', 'n_conv', 'leave_n_out',
            'n_oneyr', 'method', 'ROC_leave_n_out', 'wghts_std_anom', 
            'wghts_accross_lags', 'splittrainfeat', 'n_strongest',
            'perc_map', 'tfreq', 'lags', 'n_yrs', 'hotdaythres',
            'pval_logit_first', 'pval_logit_final', 'rollingmean',
            'mcKthres', 'new_model_sel', 'perc_map', 'comp_perc',
            'logit_valid', 'use_ts_logit', 'region', 'regionmcK',
            'add_lsm', 'min_n_gc']
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
#%%
    
    
dic_exp = ({'ts and valid'  :   (True,True), 
            'only ts'       :   (True,False),
            'only Cov'      :   (False,False),
            'Cov and valid' :   (False,True)
            })




def all_output_wrapper(ex, exp_key='only Cov'):
    #%%
    ex['use_ts_logit'], ex['logit_valid'] = dic_exp[exp_key] 
    
    print(exp_key)
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
            
    
    # load patterns
    l_ds_mcK        = [ex['score_per_run'][i][2] for i in range(len(ex['score_per_run']))]
    l_ds_Sem        = [ex['score_per_run'][i][3] for i in range(len(ex['score_per_run']))]
    # perform prediciton
    ex, patterns_Sem, patterns_mcK = func_mcK.make_prediction(l_ds_Sem, l_ds_mcK, Prec_reg, ex)
    
    
# =============================================================================
#   Plotting
# =============================================================================
    score_mcK       = np.round(ex['score_per_run'][-1][2]['score'], 2)
    score_Sem       = np.round(ex['score_per_run'][-1][3]['score'], 2)
    ROC_str_mcK      = ['{} days - ROC score {}'.format(ex['lags'][i], score_mcK[i].values) for i in range(len(ex['lags'])) ]
    ROC_str_Sem      = ['{} days - ROC score {}'.format(ex['lags'][i], score_Sem[i].values) for i in range(len(ex['lags'])) ]
    # Sem plot
    # share kwargs with mcKinnon plot
    
        
    kwrgs = dict( {'title' : '', 'clevels' : 'notdefault', 'steps':17,
                        'vmin' : -0.5, 'vmax' : 0.5, 'subtitles' : ROC_str_Sem,
                       'cmap' : plt.cm.RdBu_r, 'column' : 1} )
    
    mean_n_patterns = patterns_Sem.mean(dim='n_tests')
    mean_n_patterns.attrs['units'] = 'mean over {} runs'.format(ex['n_conv'])
    mean_n_patterns.attrs['title'] = 'Composite mean - Objective Precursor Pattern'
    mean_n_patterns.name = 'ROC {}'.format(score_Sem.values)
    filename = os.path.join(ex['exp_folder'], 'mean_over_{}_tests'.format(ex['n_conv']) )
    func_mcK.plotting_wrapper(mean_n_patterns, ex, filename, kwrgs=kwrgs)
    
    
    
    # mcKinnon composite mean plot
    filename = os.path.join(ex['exp_folder'], 'mcKinnon mean composite_tf{}_{}'.format(
                ex['tfreq'], ex['lags']))
    mcK_mean = patterns_mcK.mean(dim='n_tests')
    kwrgs['subtitles'] = ROC_str_mcK
    mcK_mean.name = 'Composite mean green rectangle: ROC {}'.format(score_mcK.values)
    mcK_mean.attrs['units'] = 'Kelvin'
    mcK_mean.attrs['title'] = 'Composite mean - Subjective green rectangle pattern'
    func_mcK.plotting_wrapper(mcK_mean, ex, filename, kwrgs=kwrgs)
    
    #if (ex['leave_n_out'] == True) and (ex['ROC_leave_n_out'] == False):
    #    # mcKinnon std plot
    #    filename = os.path.join(ex['exp_folder'], 'mcKinnon std composite_tf{}_{}'.format(
    #                ex['tfreq'], ex['lags']))
    #    mcK_std = patterns_mcK.std(dim='n_tests')
    #    mcK_std.name = 'Composite std: ROC {}'.format(score_mcK.values)
    #    mcK_std.attrs['units'] = 'Kelvin'
    #    func_mcK.plotting_wrapper(mcK_std, filename, ex)
    
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
        Robustness_weights = l_ds_Sem[n]['weights']
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
        
        func_mcK.plotting_wrapper(for_plt, ex, filename, kwrgs=kwrgs)
        
        
    #%%
    if ex['load_mcK'] == False:
        filename = os.path.join(ex['RV1d_ts_path'], ex['RVts_filename'])
        dicRV = np.load(filename,  encoding='latin1').item()
        xarray_plot(dicRV['RV_array']['mask'], path=folder, name='RV_mask', saving=True)
    
    func_mcK.plot_oneyr_events(RV_ts, ex, 2012, ex['folder'], saving=True)
    ## plotting same figure as in paper
    #for i in range(2005, 2010):
    #    func_mcK.plot_oneyr_events(RV_ts, ex, i, folder, saving=True)
    
    #%% Robustness accross training sets
    
    lats = patterns_Sem.latitude
    lons = patterns_Sem.longitude
    array = np.zeros( (ex['n_conv'], len(ex['lags']), len(lats), len(lons)) )
    wgts_tests = xr.DataArray(data=array, 
                    coords=[range(ex['n_conv']), ex['lags'], lats, lons], 
                    dims=['n_tests', 'lag','latitude','longitude'], 
                    name='{}_tests_wghts'.format(ex['n_conv']), attrs={'units':'wghts ['})
    for n in range(ex['n_conv']):
        wgts_tests[n,:,:,:] = l_ds_Sem[n]['weights']
        
        
    if ex['leave_n_out']:
        n_lags = patterns_Sem.sel(n_tests=0).lag.size
        n_lats = patterns_Sem.sel(n_tests=0).latitude.size
        n_lons = patterns_Sem.sel(n_tests=0).longitude.size
        
        pers_patt = patterns_Sem.sel(n_tests=0).copy()
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
        pers_patt.attrs['title'] = ('Robustness\n{} different '
                                'training sets (n={} yrs)'.format(ex['n_conv'],size_trainset))
        filename = os.path.join(ex['exp_folder'], 'Robustness_across_{}_training_tests'.format(ex['n_conv']) )
        vmax = ex['n_conv'] + 1E-9
        mean = np.round(pers_patt.mean(dim=('latitude', 'longitude')).values, 1)
    #    mean = pers_patt.quantile(0.80, dim=('latitude','longitude')).values
        std =  np.round(pers_patt.std(dim=('latitude', 'longitude')).values, 0)
        ax_text = ['mean = {}±{}'.format(mean[l],int(std[l])) for l in range(len(ex['lags']))]
        kwrgs = dict( {'title' : pers_patt.attrs['title'], 'clevels' : 'notdefault', 
                       'steps' : 16, 'subtitles': ROC_str_Sem, 
                       'vmin' : 0, 'vmax' : vmax, 'clim' : (max(0,vmax-20), vmax),
                       'cmap' : plt.cm.magma_r, 'column' : 2, 'extend':['min','yellow'],
                       'cbar_vert' : 0.05, 'cbar_hght' : 0.01,
                       'adj_fig_h' : 1.25, 'adj_fig_w' : 1., 
                       'hspace' : 0.02, 'wspace' : 0.08,
                       'ax_text': ax_text } )
        func_mcK.plotting_wrapper(pers_patt, ex, filename, kwrgs=kwrgs)
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
                                 
            mean_n_patterns.name = 'ROC {}'.format(score_Sem.values)
            filename = os.path.join(ex['exp_folder'], ('weighted by robustness '
                                 'over {} tests'.format(ex['n_conv']) ))
    #        kwrgs = dict( {'title' : mean_n_patterns.name, 'clevels' : 'default', 'steps':17,
    #                        'vmin' : -3*mean_n_patterns.std().values, 'vmax' : 3*mean_n_patterns.std().values, 
    #                       'cmap' : plt.cm.RdBu_r, 'column' : 2} )
            func_mcK.plotting_wrapper(mean_n_patterns, ex, filename, kwrgs=kwrgs)
    
    
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
        for n in np.arange(0, ex['n_conv'], 6, dtype=int): 
            yr = years[n]
            pattern_num_init = l_ds_Sem[n]['pat_num_CPPA']
            
    
    
            pattern_num_init.attrs['title'] = ('{} - CPPA regions'.format(yr))
            filename = os.path.join(subfolder, pattern_num_init.attrs['title'].replace(
                                    ' ','_')+'.png')
            for_plt = pattern_num_init.copy()
            for_plt.values = for_plt.values-0.5
            kwrgs = dict( {'title' : for_plt.attrs['title'], 'clevels' : 'notdefault', 
                           'steps' : for_plt.max()+2, 'subtitles': ROC_str_Sem,
                           'vmin' : 0, 'vmax' : for_plt.max().values+0.5, 
                           'cmap' : plt.cm.tab10, 'column' : 2} )
            
            func_mcK.plotting_wrapper(for_plt, ex, filename, kwrgs=kwrgs)
            
            if ex['logit_valid'] == True:
                pattern_num = l_ds_Sem[n]['pat_num_logit']
                pattern_num.attrs['title'] = ('{} - regions that were kept after logit regression '
                                             'pval < {}'.format(yr, ex['pval_logit_final']))
                filename = os.path.join(subfolder, pattern_num.attrs['title'].replace(
                                        ' ','_')+'.png')
                for_plt = pattern_num.copy()
                for_plt.values = for_plt.values-0.5
                kwrgs = dict( {'title' : for_plt.attrs['title'], 'clevels' : 'notdefault', 
                               'steps' : for_plt.max()+2, 'subtitles': ROC_str_Sem,
                               'vmin' : 0, 'vmax' : for_plt.max().values+0.5, 
                               'cmap' : plt.cm.tab10, 'column' : 2} )
                
                func_mcK.plotting_wrapper(for_plt, ex, filename, kwrgs=kwrgs)

#%%
if __name__ == '__main__':    
    start = time.time()
    num_workers = mp.cpu_count()  
    pool = mp.Pool(num_workers)
    for exp_key in dic_exp.keys():
        
        pool.apply_async(all_output_wrapper, args= (ex, exp_key))
        
    pool.close()
    pool.join()
    end = time.time()
    print("It took ", end - start)