#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:12:41 2019

@author: semvijverberg
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
import func_CPPA
import func_pred
import statsmodels.api as sm
import itertools
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

def get_opt_freq(RV_ts, ex):
    #%%                 
    ex['output_ts_folder'] += '_regions_n'
#    for n in range(len(ex['train_test_list'])):
#        ex['n'] = n
#        
#        train, test = ex['train_test_list'][n]
#        ex['test_year'] = list(set(test['RV'].time.dt.year.values))
#        if ex['use_ts_logit'] == False:
#            print('test year(s) {}, with {} events.'.format(ex['test_year'],
#                                 test['events'].size))
        
    # get RV dates (period analyzed)
    all_RV_dates = func_CPPA.to_datesmcK(RV_ts.time, RV_ts.time.dt.hour[0], 
                                       RV_ts.time[0].dt.hour)
    
    for lag_idx, lag in enumerate(ex['lags']):
        # load in timeseries
        #%%

        csv_train_test_data = 'testyr{}_{}.csv'.format(ex['test_year'], lag)
        path = os.path.join(ex['output_ts_folder'], csv_train_test_data)
        data = pd.read_csv(path, index_col='date')
        
        ### only test data ###
        dates_test_lag = func_pred.func_dates_min_lag(all_RV_dates, lag)[0]
        
        data_RVdates = data.loc[dates_test_lag]
        dfRV = pd.DataFrame(RV_ts.sel(time=all_RV_dates).values, index=data_RVdates.index,
                            columns=['RVts'])
                         
        dfRV = (dfRV - dfRV.mean()) / dfRV.std()
        data_n = (data_RVdates-data_RVdates.mean(axis=0)) / data_RVdates.std(axis=0)
        data_n['RVts'] = dfRV
        
        

#            plt.plot(autocorrelation(data_i[reg+'_i'])[:n_days])
#            plt.plot(autocorrelation(data_n_abs[reg])[:n_days])
#            plt.plot(autocorrelation(dfRV)[:n_days])

        
        one_yr = pd.to_datetime(data_n.index).year == 2012
        N_days = one_yr[one_yr==True].size
        
#        subplots_df(data_n[:10*N_days], 3, {'ylim':'normalize',
#                                    'xlim':(0,10*N_days)})
        
        freq = 1./N_days
        fftfreq, df_fft, df_psd = fft_powerspectrum(data_n, freq)
        
        i = fftfreq > 0
        df_psd = df_psd[i]
        period = np.array(1. / (fftfreq[i] * freq), dtype=int) / N_days
        df_psd.index = period
#        df_psd[i].index = np.array(1. / (fftfreq[i] * freq), dtype=int)
        subplots_df(df_psd, 3, {'ylim':'normalize',
                                    'xlim'  : (0,10),
                                    'title' : 'power spectrum normal timeseries' })
        #%%

        # subseasonal
        freqrange = np.logical_and((abs(fftfreq) < 1.2), (abs(fftfreq) > 0.01))
#        freqrange = (abs(fftfreq) > 1.1)
        key = df_fft.columns[1]
        data = np.zeros_like(df_fft.values, dtype=float)
        for i, key in enumerate(df_fft.columns):
            fft_fr = df_fft[key].values
            fft_fr[freqrange] = 0
            data[:,i] = np.real(sp.fftpack.ifft(fft_fr))
            
            
#        df_fft_bis = df_fft.copy()
#        data = np.real(sp.fftpack.ifft(df_fft_bis))
        df_ts_subs = pd.DataFrame(data, columns=df_fft.columns, index=range(df_fft.shape[0]))
        subplots_df(df_ts_subs.iloc[:5*N_days], 3, {'ylim':'normalize',
                                         'title' : 'timeseries exluding interannual variability' })
        #%%
        fftfreq, df_fft, df_psd = fft_powerspectrum(df_ts_subs, freq)
        
        i = fftfreq > 0
        df_psd = df_psd[i]
        period = np.array(1. / (fftfreq[i] * freq), dtype=int)
        df_psd.index = period
        subplots_df(df_psd, 3, {'ylim':'normalize',
                                        'xlim':(0,50),
                                        'title' : 'powerspectrum exluding interannual variability' })
        #%%
        fftfreq, df_fft, df_psd = fft_powerspectrum(data_n, freq)
        
        i = fftfreq > 0
        df_psd = df_psd[i]
        period = np.array(1. / (fftfreq[i] * freq), dtype=int)
        df_psd.index = period
        subplots_df(df_psd, 3, {'ylim':(0,0.05),
                                        'xlim':(0,50),
                                        'title' : 'powerspectrum including interannual variability' })
    
        #%%
        ac_plot(df_ts_subs, 3, {'title':'autocorrelation ts only subseasonal freq'})
        #%%
        ac_plot(data_n, 3, {'title':'autocorrelation ts unfiltered'})

        #%%
                                   
        
        df_i = df_ts_subs.multiply(abs(df_ts_subs['RVts'].values), axis='index')
        subplots_df(df_i.iloc[:10*N_days], 3, {'ylim':'normalize'})

        fftfreq, df_fft, df_psd = fft_powerspectrum(df_i, freq)
        
        i = fftfreq > 0
        df_psd = df_psd[i]
        period = np.array(1. / (fftfreq[i] * freq), dtype=int)
        df_psd.index = period
        subplots_df(df_psd.iloc[:], 3, {'ylim':'normalize',
                                        'xlim':(0,50)})
            
    #%%

def ac_plot(df, colwrap=3, kwrgs={}):
    if (df.columns.size) % colwrap == 0:
        rows = int(df.columns.size / colwrap)
    elif (df.columns.size) % colwrap != 0:
        rows = int(df.columns.size / colwrap) + 1
    fig, ax = plt.subplots(rows, colwrap, sharex='col', sharey='row',
                           figsize = (10,8))
    for i, ax in enumerate(fig.axes):
        if i >= df.columns.size:
            ax.axis('off')
        else:
            header = df.columns[i]
    
            autocorr_temp = autocorrelation(df['RVts'])[:60]
            autocorr_CPPA = autocorrelation(df[header])[:60]
            ax.plot(autocorr_temp, label='temp')
            ax.plot(autocorr_CPPA, label=header)
            ax.grid(which='major')
            ax.legend()
    if 'title' in kwrgs.keys():
        fig.text(0.5, 1.0, kwrgs['title'], fontsize=18, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)
    return

def autocorrelation(x):
    xp = (x - np.mean(x))/np.std(x)
    result = np.correlate(xp, xp, mode='full')
    return result[int(result.size/2):]/(len(xp))

def fft_np(y, freq):
    yfft = sp.fftpack.fft(y)
    ypsd = np.abs(yfft)**2
    fftfreq = sp.fftpack.fftfreq(len(ypsd), freq)
    ypsd = 2.0/len(y) * ypsd
    return fftfreq, yfft, ypsd

def fft_powerspectrum(df, freq):
    
    df_fft = df[:].copy()
    df_psd = df[:].copy()
    
    list_freq = []
    for reg in df.columns:
        fftfreq, yfft, ypsd = fft_np(np.array(df[reg]), freq)
#        plt.plot(yfft)
        
        df_fft[reg] = yfft[:]
        df_psd[reg] = ypsd[:]
        df_fft.index = fftfreq[:]
        df_psd.index = fftfreq[:]
        i = fftfreq > 0
        idx = np.argmax(ypsd[i])
        text = '{} fft {:.1f}, T = {:.0f}'.format(
                reg,
                fftfreq[idx], 
                1 / (fftfreq[i][idx] * freq))
        list_freq.append(text)
    df_psd.columns = list_freq
    
    return fftfreq, df_fft, df_psd


def subplots_df(df, colwrap=3, kwrgs={}):
    if (df.columns.size) % colwrap == 0:
        rows = int(df.columns.size / colwrap)
    elif (df.columns.size) % colwrap != 0:
        rows = int(df.columns.size / colwrap) + 1
    fig, ax = plt.subplots(rows, colwrap, sharex='col', sharey='row',
                           figsize = (10,8))
    for i, ax in enumerate(fig.axes):
        if i >= df.columns.size:
            ax.axis('off')
        else:
            header = df.columns[i]
            if 'ylim' in kwrgs.keys():
                if kwrgs['ylim'][:4] == 'norm':
                    y = df[header] / df[header].max()
                if type(kwrgs['ylim']) != type(str()):
                    ax.set_ylim(kwrgs['ylim'])
                    y = df[header] / df[header].max()
            else:
                y = df[header]
            ax.plot(df.index, y)
            if 'xlim' in kwrgs.keys():
                ax.set_xlim(kwrgs['xlim'])
            ax.text(0.5, 0.9, header, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)
    if 'title' in kwrgs.keys():
        fig.text(0.5, 1.0, kwrgs['title'], fontsize=18, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)
    return