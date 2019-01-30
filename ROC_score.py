# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Mon Oct 15 17:50:16 2018

@author: semvijverberg
"""
import numpy
import random
import os
import numpy as np
import pandas as pd
import func_mcK
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

def ROC_score_wrapper(test, train, ds_mcK, ds_Sem, ex):
    #%%
    # =============================================================================
    # calc ROC scores
    # =============================================================================
    ROC_Sem  = np.zeros(len(ex['lags']))
    ROC_mcK  = np.zeros(len(ex['lags']))
    ROC_boot = np.zeros(len(ex['lags']))
    for lag in ex['lags']:
        idx = ex['lags'].index(lag)
        dates_test = func_mcK.to_datesmcK(test['RV'].time, test['RV'].time.dt.hour[0], 
                                           test['Prec'].time[0].dt.hour)
        # select antecedant SST pattern to summer days:
        dates_min_lag = dates_test - pd.Timedelta(int(lag), unit='d')
        var_test_mcK = func_mcK.find_region(test['Prec'], region=ex['regionmcK'])[0]
    #    full_timeserie_regmck = var_test_mcK.sel(time=dates_min_lag)
    
        var_test_mcK = var_test_mcK.sel(time=dates_min_lag)
        var_patt_mcK = func_mcK.find_region(ds_mcK['pattern'].sel(lag=lag), region=ex['regionmcK'])[0]
        var_test_reg = test['Prec'].sel(time=dates_min_lag)        
        
        crosscorr_mcK = func_mcK.cross_correlation_patterns(var_test_mcK, 
                                                            var_patt_mcK)
        if ex['use_ts_logit'] == False:
            # weight by robustness of precursors
            var_test_reg = var_test_reg * ds_Sem['weights'].sel(lag=lag)
            crosscorr_Sem = func_mcK.cross_correlation_patterns(var_test_reg, 
                                                            ds_Sem['pattern'].sel(lag=lag))
        elif ex['use_ts_logit'] == True:
            crosscorr_Sem = ds_Sem['ts_prediction'][idx]
#        if idx == 0:
#            print(ex['test_years'])
#            print(crosscorr_Sem.time)
        
        if (
            ex['leave_n_out'] == True and ex['method'] == 'iter'
            or ex['ROC_leave_n_out'] or ex['method'][:6] == 'random'
            ):
            if ex['n'] == 0:
                ex['test_ts_mcK'][idx] = crosscorr_mcK.values 
                ex['test_ts_Sem'][idx] = crosscorr_Sem.values
                ex['test_RV'][idx]  = test['RV'].values
    #                ex['test_RV_Sem'][idx]  = test['RV'].values
            else:
    #                update_ROCS = ex['test_ts_mcK'][idx].append(list(crosscorr_mcK.values))
                ex['test_ts_mcK'][idx] = np.concatenate( [ex['test_ts_mcK'][idx], crosscorr_mcK.values] )
                ex['test_ts_Sem'][idx] = np.concatenate( [ex['test_ts_Sem'][idx], crosscorr_Sem.values] )
                ex['test_RV'][idx] = np.concatenate( [ex['test_RV'][idx], test['RV'].values] )  
                
                
#                if ex['plot_ts'] == True:
#                    ax = plt.subplot(111)
#                    ax.set_ylim(-3,3)
#                    # should normalize with std from training spatial covariance or logit ts
#                    ax.plot(crosscorr_mcK.values/crosscorr_mcK.std().values, label='mcK', color='blue')
#                    ax.plot(crosscorr_Sem.values/crosscorr_Sem.std().values, label='Sem', color='green')
##                    ax.plot(crosscorr_mcK.values/np.std(ex['test_ts_mcK'][idx]), label='mcK', color='blue')
##                    ax.plot(crosscorr_Sem.values/np.std(ex['test_ts_Sem'][idx]), label='Sem', color='green')
#                    norm_test_RV = (test['RV']-np.mean(test['RV']))/(train['RV']-np.mean(train['RV'])).std()
#                    ax.plot(norm_test_RV,alpha=0.4, color='red')
#                    threshold = norm_test_RV.std().values
#                    plt.fill_between(range(norm_test_RV.size), threshold, norm_test_RV.values, 
#                                     where=(norm_test_RV.values > threshold),
#                                     interpolate=True, color="orange", alpha=0.5, label="hot days")
#                    ax.legend(loc='lower left')
#                    plt.text(0.5,0.5, '# events {}'.format(len(test['events']),
#                             horizontalalignment='center',
#                             verticalalignment='center', transform=ax.transAxes))
#                    plt.show()
                    
        
            if  ex['n'] == ex['n_conv']-1 or ex['n'] == ex['n_stop']:
                if idx == 0:
                    print('Calculating ROC scores\nDatapoints precursor length '
                      '{}\nDatapoints RV length {}'.format(len(ex['test_ts_mcK'][0]),
                       len(ex['test_RV'][0])))
                
                # normalize
                ts_pred_mcK  = ((ex['test_ts_mcK'][idx]-np.mean(ex['test_ts_mcK'][idx]))/ \
                                          (np.std(ex['test_ts_mcK'][idx]) ) )
                ts_pred_Sem  = ((ex['test_ts_Sem'][idx]-np.mean(ex['test_ts_Sem'][idx]))/ \
                                          (np.std(ex['test_ts_Sem'][idx]) ) )
                norm_test_RV = ((ex['test_RV'][idx]-np.mean(ex['test_RV'][idx]))/ \
                                          (np.std(ex['test_RV'][idx]) ) )  
                
                
                if ex['plot_ts'] == True:
                    #%%
                    threshold = np.std(norm_test_RV)
                    No_train_yrs = int(train['RV'].size / ex['n_oneyr'])
                    title = ('Prediction time series versus truth, '
                             'with {} training years'.format(No_train_yrs))
                    labels = np.linspace(1,10,norm_test_RV.size, dtype=int)
                    labels[-1] = labels[-2]
                    df = pd.DataFrame(data={'RV':norm_test_RV, 'Sem':ts_pred_Sem, 'mcK':ts_pred_mcK, 'plot':labels} )
                    g = sns.FacetGrid(df, col='plot', col_wrap=3, sharex=False)
                    n_plots = len(g.axes)
                    for n_ax in np.arange(0,n_plots):
                        ax = g.axes.flatten()[n_ax]
                        df_sub = df[df['plot'] == n_ax+1]
                        del df_sub['plot']
                        ax.set_ylim(-3,3)
                        ax.set_xlim(df_sub['mcK'].index[0],df_sub['mcK'].index[-1])
                        # should normalize with std from training spatial covariance or logit ts
                        ax.plot(df_sub['mcK'].index, df_sub['mcK'].values, label='mcK', color='blue')
                        ax.plot(df_sub['Sem'].index, df_sub['Sem'].values, label='Sem', color='green')
                        ax.plot(df_sub['RV'],alpha=0.4, label='Truth', color='red')                   
                        
                        ax.fill_between(df_sub['RV'].index, threshold, df_sub['RV'].values, 
                                         where=(df_sub['RV'].values > threshold),
                                         interpolate=True, color="orange", alpha=0.5, label="hot days")
                        if n_ax+1 == n_plots:
                            ax.legend(loc='lower left')
                    g.fig.text(0.5, 1.02, title, fontsize=15,
                           fontweight='heavy', horizontalalignment='center')
#                        plt.text(0.5,0.5, '# events {}'.format(len(test['events']),
#                                 horizontalalignment='center',
#                                 verticalalignment='center', transform=ax.transAxes))
#                    plt.show()
                       #%%
#  
                
#                Prec_threshold_mcK = np.percentile(ex['test_ts_mcK'][idx], 70)
#                Prec_threshold_Sem = np.percentile(ex['test_ts_Sem'][idx], 70)
#
#                func_mcK.plot_events_validation(ex['test_ts_Sem'][idx], ex['test_ts_mcK'][idx], test['RV'], Prec_threshold_Sem, 
#                                            Prec_threshold_mcK, ex['hotdaythres'], 2000)
                
                n_boot = 10
                ROC_mcK[idx], ROC_boot = ROC_score(ts_pred_mcK, ex['test_RV'][idx],
                                      ex['hotdaythres'], lag, n_boot, ex, 'default')
                if ex['use_ts_logit'] == True:
                    ROC_Sem[idx] = ROC_score(ts_pred_Sem, ex['test_RV'][idx],
                                      ex['hotdaythres'], lag, 0, ex, 'default')[0]
                elif ex['use_ts_logit'] == False:
                    ROC_Sem[idx] = ROC_score(ex['test_ts_Sem'][idx], ex['test_RV'][idx],
                                      ex['hotdaythres'], lag, 0, ex, 'default')[0]
                
                print('\n*** ROC score for {} lag {} ***\n\nMck {:.2f} \t Sem {:.2f} '
                '\t ±{:.2f} 2*std random events\n\n'.format(ex['region'], 
                  lag, ROC_mcK[idx], ROC_Sem[idx], 2*np.std(ROC_boot)))
            
                
        elif ex['leave_n_out'] == True and ex['method'] == 'random':        
                               
            # check detection of precursor:
            Prec_threshold_mcK = ds_mcK['perc'].sel(percentile=60 /10).values[0]
            Prec_threshold_Sem = ds_Sem['perc'].sel(percentile=60 /10).values[0]
            
#            # check if there are any detections
#            Prec_det_mcK = (func_mcK.Ev_timeseries(crosscorr_mcK, 
#                                           Prec_threshold_mcK).size > ex['min_detection'])
#            Prec_det_Sem = (func_mcK.Ev_timeseries(crosscorr_Sem, 
#                                           Prec_threshold_Sem).size > ex['min_detection'])
#            
#    #        # plot the detections
#            func_mcK.plot_events_validation(crosscorr_Sem, crosscorr_mcK, test['RV'], Prec_threshold_Sem, 
#                                            Prec_threshold_mcK, ex['hotdaythres'], 2000)
    
            if Prec_det_mcK == True:
                n_boot = 1
                ROC_mcK[idx], ROC_boot = ROC_score(crosscorr_mcK, test['RV'],
                                      ex['hotdaythres'], lag, n_boot, ex, ds_mcK['perc'])
            else:
                print('Not enough predictions detected, neglecting this predictions')
                ROC_mcK[idx] = ROC_boot = 0.5
    
    
            
            if Prec_det_Sem == True:
                n_boot = 0
                ROC_Sem[idx] = ROC_score(crosscorr_Sem, test['RV'],
                                      ex['hotdaythres'], lag, n_boot, ex, ds_Sem['perc'])[0]
            else:
                print('Not enough predictions detected, neglecting this predictions')
                ROC_Sem = ROC_boot = 0.5
                                  
            
            print('\n*** ROC score for {} lag {} ***\n\nMck {:.2f} \t Sem {:.2f} '
                '\t ±{:.2f} 2*std random events\n\n'.format(ex['region'], 
                  lag, ROC_mcK[idx], ROC_Sem[idx], 2*np.std(ROC_boot)))
            
        elif ex['leave_n_out'] == False or ex['method'][:5] == 'split':
            if idx == 0:
                print('performing hindcast')
            n_boot = 5
            ROC_mcK[idx], ROC_boot = ROC_score(crosscorr_mcK, test['RV'],
                                   ex['hotdaythres'], lag, n_boot, ex, 'default')
            ROC_Sem[idx] = ROC_score(crosscorr_Sem, test['RV'],
                                      ex['hotdaythres'], lag, 0, ex, 'default')[0]
            
            Prec_threshold_Sem = np.percentile(crosscorr_Sem, 70)
            Prec_threshold_mcK = np.percentile(crosscorr_mcK, 70)
            
            
#            func_mcK.plot_events_validation(crosscorr_Sem, crosscorr_mcK, test['RV'], Prec_threshold_Sem, 
#                                            Prec_threshold_mcK, ex['hotdaythres'], 2000)
            
#            func_mcK.plot_events_validation(crosscorr_Sem, crosscorr_mcK, test['RV'], 
#                                            ds_Sem['perc'].sel(percentile=5)), 
#                                            Prec_threshold_mcK, ex['hotdaythres'], 2000)
            
            print('\n*** ROC score for {} lag {} ***\n\nMck {:.2f} \t Sem {:.2f} '
                '\t ±{:.2f} 2*std random events\n\n'.format(ex['region'], 
                  lag, ROC_mcK[idx], ROC_Sem[idx], 2*np.std(ROC_boot)))
    
    #%%
    # store output:
    ds_mcK['score'] = xr.DataArray(data=ROC_mcK, coords=[ex['lags']], 
                      dims=['lag'], name='score_diff_lags',
                      attrs={'units':'-'})
    ds_Sem['score'] = xr.DataArray(data=ROC_Sem, coords=[ex['lags']], 
                      dims=['lag'], name='score_diff_lags',
                      attrs={'units':'-'})
    
    # store mean values of prediciton time serie
        
    
    ex['score_per_run'].append([ex['test_years'], len(test['events']), ds_mcK, ds_Sem, ROC_boot])
    return ex



def single_ROC_score_wrapper(test, trian, ds, ex):
    #%%
    # =============================================================================
    # calc ROC scores
    # =============================================================================
    ROC  = np.zeros(len(ex['lags']))
    ROC_boot = np.zeros(len(ex['lags']))
    
    if ds['pattern'].name[:3] == 'mcK':
        var_test_reg = func_mcK.find_region(test['Prec'], region=ex['regionmcK'])[0]
    else:
        var_test_reg = test['Prec']
        
        
    for lag in ex['lags']:
        idx = ex['lags'].index(lag)
        dates_test = func_mcK.to_datesmcK(test['RV'].time, test['RV'].time.dt.hour[0], 
                                           test['Prec'].time[0].dt.hour)
        # select antecedant SST pattern to summer days:
        dates_min_lag = dates_test - pd.Timedelta(int(lag), unit='d')
        
    #    full_timeserie_regmck = var_test_mcK.sel(time=dates_min_lag)
    
        var_test = var_test_reg.sel(time=dates_min_lag)   
    
        if ds['pattern'].name[:3] == 'mcK':
            pred_ts = func_mcK.cross_correlation_patterns(var_test, 
                                                            ds['pattern'].sel(lag=lag))
        if ex['use_ts_logit'] == False and ds['pattern'].name[:3] != 'mcK':
            # weight by robustness of precursors
            var_test = var_test * ds['weights'].sel(lag=lag)
            pred_ts = func_mcK.cross_correlation_patterns(var_test, 
                                                            ds['pattern'].sel(lag=lag))
        elif ex['use_ts_logit'] == True and ds['pattern'].name[:3] != 'mcK':
            pred_ts = ds['ts_prediction'][idx]
#        if idx == 0:
#            print(ex['test_years'])
#            print(crosscorr_Sem.time)
        
        if (ex['leave_n_out'] == True) and (ex['method'] == 'iter') or (ex['ROC_leave_n_out']):
            if ex['n'] == 0:
                ex['test_ts'][idx] = pred_ts.values 
                ex['test_RV'][idx]  = test['RV'].values
    #                ex['test_RV_Sem'][idx]  = test['RV'].values
            else:
    #                update_ROCS = ex['test_ts_mcK'][idx].append(list(crosscorr_mcK.values))
                ex['test_ts'][idx] = np.concatenate( [ex['test_ts_mcK'][idx], pred_ts.values] )
                ex['test_RV'][idx] = np.concatenate( [ex['test_RV'][idx], test['RV'].values] )  
                
                    
        
            if  ex['n'] == ex['n_conv']-1:
                if idx == 0:
                    print('Calculating ROC scores\nDatapoints precursor length '
                      '{}\nDatapoints RV length {}'.format(len(ex['test_ts_mcK'][0]),
                       len(ex['test_RV'][0])))
                
                # normalize
                ex['test_ts'][idx] = (ex['test_ts'][idx]-np.mean(ex['test_ts'][idx])/ \
                                          np.std(ex['test_ts'][idx]))            
                
#                Prec_threshold_mcK = np.percentile(ex['test_ts_mcK'][idx], 70)
#                Prec_threshold_Sem = np.percentile(ex['test_ts_Sem'][idx], 70)
#
#                func_mcK.plot_events_validation(ex['test_ts_Sem'][idx], ex['test_ts_mcK'][idx], test['RV'], Prec_threshold_Sem, 
#                                            Prec_threshold_mcK, ex['hotdaythres'], 2000)
                
                n_boot = 10
                ROC[idx], ROC_boot = ROC_score(ex['test_ts_mcK'][idx], ex['test_RV'][idx],
                                      ex['hotdaythres'], lag, n_boot, ex, 'default')

                
#                print('\n*** ROC score for {} lag {} ***\n\nMck {:.2f} \t Sem {:.2f} '
#                '\t ±{:.2f} 2*std random events\n\n'.format(ex['region'], 
#                  lag, ROC_mcK[idx], ROC_Sem[idx], 2*np.std(ROC_boot)))
            
                
        elif ex['leave_n_out'] == True and ex['method'] == 'random' :        
                               
            # check detection of precursor:
            Prec_threshold = ds['perc'].sel(percentile=60 /10).values[0]
            
            # check if there are any detections
            Prec_det = (func_mcK.Ev_timeseries(pred_ts, 
                                           Prec_threshold).size > ex['min_detection'])


    
            if Prec_det == True:
                n_boot = 1
                ROC[idx], ROC_boot = ROC_score(pred_ts, test['RV'],
                                      ex['hotdaythres'], lag, n_boot, ex, ds['perc'])
            else:
                print('Not enough predictions detected, neglecting this predictions')
                ROC[idx] = ROC_boot = 0.5
    
                                  
            
            print('\n*** ROC score for {} lag {} ***\n\nMck {:.2f} \t Sem {:.2f} '
                '\t ±{:.2f} 2*std random events\n\n'.format(ex['region'], 
                  lag, ROC[idx], 2*np.std(ROC_boot)))
            
        elif ex['leave_n_out'] == False or ex['method'][:5] == 'split':
            if idx == 0:
                print('performing hindcast')
            n_boot = 5
            ROC[idx], ROC_boot = ROC_score(pred_ts, test['RV'],
                                   ex['hotdaythres'], lag, n_boot, ex, 'default')

            
            Prec_threshold_Sem = np.percentile(pred_ts, 70)
            
            
#            func_mcK.plot_events_validation(crosscorr_Sem, crosscorr_mcK, test['RV'], Prec_threshold_Sem, 
#                                            Prec_threshold_mcK, ex['hotdaythres'], 2000)
            
#            func_mcK.plot_events_validation(crosscorr_Sem, crosscorr_mcK, test['RV'], 
#                                            ds_Sem['perc'].sel(percentile=5)), 
#                                            Prec_threshold_mcK, ex['hotdaythres'], 2000)
            
#            print('\n*** ROC score for {} lag {} ***\n\nMck {:.2f} \t Sem {:.2f} '
#                '\t ±{:.2f} 2*std random events\n\n'.format(ex['region'], 
#                  lag, ROC[idx], 2*np.std(ROC_boot)))
    
    #%%
    # store output:
    ds['score'] = xr.DataArray(data=ROC, coords=[ex['lags']], 
                      dims=['lag'], name='score_diff_lags',
                      attrs={'units':'-'})


    
    ex['score_per_run'].append([ex['test_years'], len(test['events']), ds, ROC_boot])
    return ex



def ROC_score(predictions, observed, thr_event, lag, n_boot, ex, thr_pred='default'):
    #%%
#    predictions = ex['test_ts_mcK'][idx]
#    observed = RV_ts
#    thr_event = ex['hotdaythres']
    
   # calculate ROC scores
    observed = numpy.copy(observed)
    # Standardize predictor time series
#    predictions = predictions - numpy.mean(predictions)
    # P_index = numpy.copy(AIR_rain_index)	
    # Test ROC-score			
    
    TP_rate = numpy.ones((11))
    FP_rate =  numpy.ones((11))
    TP_rate[10] = 0
    FP_rate[10] = 0
    AUC_new = numpy.zeros((n_boot))
    
    #print(fixed_event_threshold) 
    events = numpy.where(observed > thr_event)[0][:]  
    not_events = numpy.where(observed <= thr_event)[0][:]     
    for p in numpy.linspace(1, 9, 9, dtype=int):	
        if str(thr_pred) == 'default':
            p_pred = numpy.percentile(predictions, p*10)
        else:
            p_pred = thr_pred.sel(percentile=p).values[0]
        positives_pred = numpy.where(predictions > p_pred)[0][:]
        negatives_pred = numpy.where(predictions <= p_pred)[0][:]

						
        True_pos = [a for a in positives_pred if a in events]
        False_neg = [a for a in negatives_pred if a in events]
        
        False_pos = [a for a in positives_pred if a  in not_events]
        True_neg = [a for a in negatives_pred if a  in not_events]
        
        True_pos_rate = len(True_pos)/(float(len(True_pos)) + float(len(False_neg)))
        False_pos_rate = len(False_pos)/(float(len(False_pos)) + float(len(True_neg)))
        
        FP_rate[p] = False_pos_rate
        TP_rate[p] = True_pos_rate
        
     
    ROC_score = numpy.abs(numpy.trapz(TP_rate, x=FP_rate ))
    # shuffled ROC
    
    ROC_bootstrap = 0
    for j in range(n_boot):
        
#        # shuffle observations / events
#        old_index = range(0,len(observed),1)
#        sample_index = random.sample(old_index, len(old_index))
        
        # shuffle years, but keep years complete:
        old_index = range(0,len(observed),1)
#        n_yr = ex['n_yrs']
        n_oneyr = int( len(observed) / ex['n_yrs'])
        chunks = [old_index[n_oneyr*i:n_oneyr*(i+1)] for i in range(int(len(old_index)/n_oneyr))]
        # replace lost value because of python indexing 
#        chunks[-1] = range(chunks[-1][0], chunks[-1][-1])
        rand_chunks = random.sample(chunks, len(chunks))
        #print(sample_index)
#        new_observed = np.reshape(observed[sample_index], -1)  
        
        new_observed = []
        for chunk in rand_chunks:
            new_observed.append( observed[chunk] )
        
        new_observed = np.reshape( new_observed, -1 )
        # _____________________________________________________________________________
        # calculate new AUC score and store it
        # _____________________________________________________________________________
        #
    
        new_observed = numpy.copy(new_observed)
        # P_index = numpy.copy(MT_rain_index)	
        # Test AUC-score			
        TP_rate = numpy.ones((11))
        FP_rate =  numpy.ones((11))
        TP_rate[10] = 0
        FP_rate[10] = 0

        events = numpy.where(new_observed > thr_event)[0][:]  
        not_events = numpy.where(new_observed <= thr_event)[0][:]     
        for p in numpy.linspace(1, 9, 9, dtype=int):	
            if str(thr_pred) == 'default':
                p_pred = numpy.percentile(predictions, p*10)
            else:
                p_pred = thr_pred.sel(percentile=p).values[0]
            
            p_pred = numpy.percentile(predictions, p*10)
            positives_pred = numpy.where(predictions > p_pred)[0][:]
            negatives_pred = numpy.where(predictions <= p_pred)[0][:]
    
    						
            True_pos = [a for a in positives_pred if a in events]
            False_neg = [a for a in negatives_pred if a in events]
            
            False_pos = [a for a in positives_pred if a  in not_events]
            True_neg = [a for a in negatives_pred if a  in not_events]
            
            True_pos_rate = len(True_pos)/(float(len(True_pos)) + float(len(False_neg)))
            False_pos_rate = len(False_pos)/(float(len(False_pos)) + float(len(True_neg)))
            
            FP_rate[p] = False_pos_rate
            TP_rate[p] = True_pos_rate
            
            #check
            if len(True_pos+False_neg) != len(events) :
                print("check 136")
            elif len(True_neg+False_pos) != len(not_events) :
                print("check 138")
           
            True_pos_rate = len(True_pos)/(float(len(True_pos)) + float(len(False_neg)))
            False_pos_rate = len(False_pos)/(float(len(False_pos)) + float(len(True_neg)))
            
            FP_rate[p] = False_pos_rate
            TP_rate[p] = True_pos_rate
        
        AUC_score  = numpy.abs(numpy.trapz(TP_rate, FP_rate))
        AUC_new[j] = AUC_score
        AUC_new    = numpy.sort(AUC_new[:])[::-1]
        pval       = (numpy.asarray(numpy.where(AUC_new > ROC_score)).size)/ n_boot
        ROC_bootstrap = AUC_new 
    #%%
    return ROC_score, ROC_bootstrap