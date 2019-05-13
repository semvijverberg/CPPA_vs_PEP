#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:13:39 2019

@author: semvijverberg
"""
import os
import xarray as xr
import numpy as np
import pandas as pd
import func_CPPA
import statsmodels.api as sm
import cartopy.crs as ccrs
import itertools
#import scipy as sp
#import seaborn as sns
#import matplotlib.pyplot as plt


def spatial_cov(ex, key1='spatcov_PEP', key2='spatcov_CPPA'):
    #%%
    ex['test_ts_mcK'] = np.zeros( len(ex['lags']) , dtype=list)
    ex['test_RV'] = np.zeros( len(ex['lags']) , dtype=list)
    ex['test_yrs'] = np.zeros( len(ex['lags']) , dtype=list)
    
    if ex['use_ts_logit'] == False:
        ex['test_ts_Sem'] = np.zeros( len(ex['lags']) , dtype=list)
    
    
    for n in range(len(ex['train_test_list'])):
        ex['n'] = n
        
        test = ex['train_test_list'][n][1]
        ex['test_year'] = list(set(test['RV'].time.dt.year.values))
        if ex['use_ts_logit'] == False:
            print('test year(s) {}'.format(ex['test_year']))
                                 
        
        # get RV dates (period analyzed)
        dates_test = func_CPPA.to_datesmcK(test['RV'].time, test['RV'].time.dt.hour[0], 
                                           test['RV'].time[0].dt.hour)
        
        for lag_idx, lag in enumerate(ex['lags']):
            # load in timeseries
            csv_train_test_data = 'testyr{}_{}.csv'.format(ex['test_year'], lag)
            path = os.path.join(ex['output_ts_folder'], csv_train_test_data)
            data = pd.read_csv(path, index_col='date')
            
#            # match hour
#            hour = pd.to_datetime(data.index[0]).hour
#            dates_test += pd.Timedelta(hour, unit='h')
            ### only test data ###
            dates_test_lag = func_dates_min_lag(dates_test, lag)[0]
            dates_test_lag = [d[:10] for d in dates_test_lag]
            
            
            idx = lag_idx
            if ex['use_ts_logit'] == False:
                # spatial covariance CPPA
                spat_cov_lag_i = data.loc[dates_test_lag][key2]
                
            
                if ex['n'] == 0:
                    ex['test_ts_Sem'][idx] = spat_cov_lag_i.values
        #                ex['test_RV_Sem'][idx]  = test['RV'].values
                else:
                    ex['test_ts_Sem'][idx] = np.concatenate( [ex['test_ts_Sem'][idx], spat_cov_lag_i.values] ) 
            
            # spatial covariance PEP
            spat_cov_lag_i = data.loc[dates_test_lag][key1]

            if ex['n'] == 0:
                ex['test_ts_mcK'][idx] = spat_cov_lag_i.values
                ex['test_RV'][idx]     = test['RV'].values
                ex['test_yrs'][idx]    = test['RV'].time
    #                ex['test_RV_Sem'][idx]  = test['RV'].values
            else:
                ex['test_ts_mcK'][idx] = np.concatenate( [ex['test_ts_mcK'][idx], spat_cov_lag_i.values] )
                ex['test_RV'][idx]     = np.concatenate( [ex['test_RV'][idx], test['RV'].values] )  
                ex['test_yrs'][idx]    = np.concatenate( [ex['test_yrs'][idx], test['RV'].time] )  
    #%%
    return ex


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
    
    
def logit_fit_new(l_ds_CPPA, RV_ts, ex):
    #%%
    if (
        ex['leave_n_out'] == True and ex['method'] == 'iter'
        or ex['ROC_leave_n_out'] or ex['method'][:6] == 'random'
        ):
        ex['test_ts_mcK'] = np.zeros( len(ex['lags']) , dtype=list)
        ex['test_ts_Sem'] = np.zeros( len(ex['lags']) , dtype=list)
        ex['test_RV'] = np.zeros( len(ex['lags']) , dtype=list)
        ex['test_yrs'] = np.zeros( len(ex['lags']) , dtype=list)
    

    def events_(RV_ts, ex):
        events = func_CPPA.Ev_timeseries(RV_ts, ex['hotdaythres'], ex).time
        dates = pd.to_datetime(RV_ts.time.values)
        event_idx = [list(dates.values).index(E) for E in events.values]
        binary_events = np.zeros(dates.size)   
        binary_events[event_idx] = 1
        mask_events = np.array(binary_events, dtype=bool)
        return binary_events, mask_events
    
    for n in range(len(ex['train_test_list'])):
        ex['n'] = n
        
        train, test = ex['train_test_list'][n][0], ex['train_test_list'][n][1]
        ex['test_year'] = list(set(test['RV'].time.dt.year.values))
        print('test year(s) {}, with {} events.'.format(ex['test_year'],
                                 test['events'].size))
        
        # get RV dates (period analyzed)
        dates_test = func_CPPA.to_datesmcK(test['RV'].time, test['RV'].time.dt.hour[0], 
                                           test['RV'].time[0].dt.hour)
        dates_train = func_CPPA.to_datesmcK(train['RV'].time, train['RV'].time.dt.hour[0], 
                                           train['RV'].time[0].dt.hour)

        
        ds_Sem = l_ds_CPPA[n]
        Composite = ds_Sem['pattern_CPPA']
        lats = Composite.latitude.values
        lons = Composite.longitude.values
        # copy pattern info to update
        ds_Sem['pat_num_logit'] = ds_Sem['pat_num_CPPA'].copy()
        ds_Sem['pattern_logit'] = ds_Sem['pattern_CPPA'].copy()
        
        logit_model = []
        ex['ts_train_std'] = [] #np.zeros( (len(ex['lags'])), dtype=list )
        
        
        for lag_idx, lag in enumerate(ex['lags']):
            # load in timeseries
            csv_train_test_data = 'testyr{}_{}.csv'.format(ex['test_year'], lag)
            path = os.path.join(ex['output_ts_folder'], csv_train_test_data)
            data = pd.read_csv(path, index_col='date')
            regions_for_ts = [int(r) for r in data.columns[4:].values]
            
            ### only training data ###
            binary_train, mask_events = events_(train['RV'], ex)
            dates_train_lag = func_dates_min_lag(dates_train, lag)[0]
            ts_regions_train = data.loc[dates_train_lag].iloc[:,3:].values
            sign_ts_regions = np.sign(np.mean(ts_regions_train,axis=0))
            
            
            # Perform training
            odds, regions_kept, combs_kept, logitmodel = train_weights_LogReg(
                    ts_regions_train, sign_ts_regions, regions_for_ts, binary_train, ex)
            
            
            # update regions that were kicked out
            Regions_lag_i = ds_Sem['pat_num_CPPA'][lag_idx].squeeze().values
            Composite_lag = Composite[lag_idx]
            
            upd_regions = np.zeros(Regions_lag_i.shape)
            for reg in regions_kept:
                upd_regions[Regions_lag_i == reg] =  reg
        
            # create map of precursor regions
            npmap = np.ma.reshape(upd_regions, (len(lats), len(lons)))
            mask_strongest = (npmap!=0) 
            npmap[mask_strongest==False] = 0
            xrnpmap = Composite_lag.copy()
            xrnpmap.values = npmap
            
            # update the mask for the composite mean
            mask = (('latitude', 'longitude'), mask_strongest)
            Composite_lag.coords['mask'] = mask
            xrnpmap.coords['mask'] = mask
            xrnpmap = xrnpmap.where(xrnpmap.mask==True)
            Composite_lag = Composite_lag.where(xrnpmap.mask==True)
            
        #    plt.figure()
        #    xrnpmap.plot.contourf(cmap=plt.cm.tab10)
                
            ### only test data ###
            binary_test, mask_events = events_(test['RV'], ex)
            dates_test_lag = func_dates_min_lag(dates_test, lag)[0]
            ts_regions_test = data.loc[dates_test_lag].iloc[:,3:].values


            ts_regions_lag_i = ts_regions_test[:,:] * sign_ts_regions[None,:]
            # use only regions which were not kicked out by logit valid
            idx_regions_kept = [regions_for_ts.index(r) for r in regions_kept]
            ts_regions_lag_i = ts_regions_lag_i[:,idx_regions_kept]
            # normalize time series (as done in the training)        
            X_n = ts_regions_lag_i / ex['ts_train_std'][lag_idx]
            
            ts_pred = logitmodel.predict(X_n)
            
            idx = lag_idx
            if (
                ex['leave_n_out'] == True and ex['method'] == 'iter'
                or ex['ROC_leave_n_out'] or ex['method'][:6] == 'random'
                ):
                if ex['n'] == 0:
#                    ex['test_ts_mcK'][idx] = crosscorr_mcK.values 
                    ex['test_ts_Sem'][idx] = ts_pred
                    ex['test_RV'][idx]     = test['RV'].values
                    ex['test_yrs'][idx]    = test['RV'].time
        #                ex['test_RV_Sem'][idx]  = test['RV'].values
                else:
        #                update_ROCS = ex['test_ts_mcK'][idx].append(list(crosscorr_mcK.values))
    #                ex['test_ts_mcK'][idx] = np.concatenate( [ex['test_ts_mcK'][idx], crosscorr_mcK.values] )
                    ex['test_ts_Sem'][idx] = np.concatenate( [ex['test_ts_Sem'][idx], ts_pred] )
                    ex['test_RV'][idx]     = np.concatenate( [ex['test_RV'][idx], test['RV'].values] )  
                    ex['test_yrs'][idx]    = np.concatenate( [ex['test_yrs'][idx], test['RV'].time] )  
            
#            pattern_p2[idx] = norm_mean * ds_Sem['weights'].sel(lag=lag) 
#            pattern_num_p2[lag_idx] = xrnpmap
            logit_model.append(logitmodel)
            
            ds_Sem['pattern_logit'][lag_idx] = Composite_lag.values
            ds_Sem['pat_num_logit'][lag_idx] = xrnpmap.values
            
    # append logit models    
    ds_Sem.attrs['logitmodel'] = logit_model
    # info of logit fit is now appended to ds_Sem
    l_ds_CPPA[n] = ds_Sem

    #%%
    return l_ds_CPPA, ex







def train_weights_LogReg(ts_regions_train, sign_ts_regions, regions_for_ts, binary_train, ex):
    #%%
#    from sklearn.linear_model import LogisticRegression
#    from sklearn.model_selection import train_test_split
    # I want to the importance of each regions as is shown in the composite plot
    # A region will increase the probability of the events when it the coef_ is 
    # higher then 1. This means that if the ts shows high values, we have a higher 
    # probability of the events. To ease interpretability I will multiple each ts
    # with the sign of the region, to that each params should be above one (higher
    # coef_, higher probability of events)      
    
    
    ex['std_of_initial_ts'] = np.std(ts_regions_train, axis=0) #!!
    X_init = ts_regions_train / np.std(ts_regions_train, axis=0) #!!
    y = binary_train
    

#    # Balance Dataset
#    pos = np.where(binary_train == 1)[0]
#    neg = np.where(binary_train == 0)[0]
#   
##    # Create 75/25 NEG/POS Selection, Balancing Data Imbalance and Data Amount
#    idx = np.random.permutation(np.concatenate([pos, np.random.choice(neg, 3*len(pos), False)]))
##    
#    y = y[idx]
#    X_init = X_init[idx,:]
    
    n_feat = len(np.unique(regions_for_ts))
    n_samples = len(X_init[:,0])
    
    regions          = regions_for_ts
    track_super_sign = regions_for_ts
    track_r_kept     = regions_for_ts
    signs = sign_ts_regions

    init_vs_final_bic = []
    if ex['logit_valid'] == True:
        # =============================================================================
        # Step 1: Preselection 
        # =============================================================================
        # first kick out regions which do not show any relation to the event 
        # i.e. low p-value
        X = X_init
        all_regions_significant = True
        i = 0
         #np.zeros( (len(ex['lags'])) , dtype=list)
        while all_regions_significant:
            X_train = X * signs[None,:]
            y_train = y
    
            logit_model=sm.Logit(y_train,X_train)
            result = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
            if i == 0:
    #            print('initial bic value {}'.format(result.bic))
                init_vs_final_bic.append(result.bic)
            if result.mle_retvals['converged'] == False:
                print('logistic regression did not converge, taking odds of prev'
                      'iteration or - if not present -, all odds (wghts) are equal')
                try:
                    # capture odds array of previous iteration
                    odds = odds
                except NameError as error:
                    # if odds in not defined, then it has never converged and there
                    # are no odds available, set all odds (wghts) to 1
                    odds = np.ones(len(n_feat))
                    
            elif result.mle_retvals['converged'] == True:
                odds = np.exp(result.params)
        #        print('statsmodel logit Odds of event happening conditioned on X (p / (1+p) = exp(params) \n{}\n'.format(odds))
    #            print(result.summary2())
                p_vals = result.pvalues
                mask_really_insig = (p_vals >= ex['pval_logit_first'])
                regions_not_sign = np.where(mask_really_insig)[0]
                        
                # update regions accounted for in fit
                X = np.delete(X, regions_not_sign, axis=1)
                track_r_kept = np.delete(track_r_kept, regions_not_sign)
                
                signs  = np.delete(signs, regions_not_sign, axis=0)
            
                                            
                if len(regions_not_sign) == 0:
                    all_regions_significant = False
            i += 1
    #    print(result.summary2())
    
        # regions to be futher investigated before throwing out
        mask_not_super_sig = (p_vals >= ex['pval_logit_final'])
        idx_not_super_sig = (np.where(mask_not_super_sig)[0])
        track_super_sign = np.delete(track_r_kept, idx_not_super_sig)
        regs_for_interac = [i for i in track_r_kept if i not in track_super_sign]
 
        # =============================================================================
        # Step 2: Check 'intermediate regions' should be kicked out
        # =============================================================================
        # Perform logistic regression on combination of two with the regions that were
        # 'not super sign' (pval < ex['pval_logit_final'])
        
        if len(regs_for_interac) != 0:
            # check if regions deleted do not have an interacting component
            combi = list(itertools.product([0, 1], repeat=n_feat))[1:]
            comb_int = [c for c in combi if np.sum(c)==2]
            weak_p = []
            for c in comb_int:
                idx_f = np.where(np.array(c) == 1)[0]
                # get number of region
                regs = [regions[i] for i in idx_f]
                for r in regs_for_interac:
                    if r in regs:   
        #                print(True)
                        weak_p.append(c) 
            
            # remove duplicates in weak_p
            def remove_duplicates(seq):
                seen = set()
                seen_add = seen.add
                return [x for x in seq if not (x in seen or seen_add(x))]
            
            weak_p = remove_duplicates(weak_p)
            
            X_train_cl = X_init
            X_train_cl = X_train_cl * sign_ts_regions[None,:]
            # add interaction time series
            combregions = []
            ts_regions_interaction = np.zeros( (n_samples, len(weak_p)) )
            for comb in weak_p:
                i = weak_p.index(comb)
                idx_f = np.where(np.array(comb) == 1)[0]
                two_ts = X_train_cl[:,idx_f]
                ts_regions_interaction[:,i] = two_ts[:,0] * two_ts[:,1]
                corresponding_regions = [regions[i] for i in idx_f]
                combregions.append(corresponding_regions)
            
            X_inter = ts_regions_interaction / np.std(ts_regions_interaction, axis=0)
            y_train = y
        
            logit_model=sm.Logit(y_train,X_inter)
            result_inter = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
            p_vals = result_inter.pvalues 
            odds_interac   = np.exp(result_inter.params)
            mask_cregions_keeping = (p_vals <= ex['pval_logit_final']) #* (odds > 1.)
            cregions_sign_idx = list(np.where(mask_cregions_keeping)[0])
            cregions_sign = np.array(combregions)[cregions_sign_idx]
            result_inter.summary2()
            if len(cregions_sign_idx)==0.:
                # delete regions that are not kept after testing if they got good enough
                # p_vals
                idx_not_keeping = [list(track_r_kept).index(i) for i in regs_for_interac]
                track_r_kept    = np.delete(track_r_kept, idx_not_keeping)
            else:
                # =============================================================================
                # test if combination of regions is truly giving a better p_val then when it 
                # is taken as a single region
                # =============================================================================
                
                keep_duetointer = []
                comb_sign_r = np.unique(cregions_sign)
                comb_sign_ind = [i for i in comb_sign_r if i not in regs_for_interac]
                comb_sign_ind_idx = [comb_sign_ind.index(r) for r in comb_sign_ind if r in regions]
                X_invol_inter = X_init
                X_invol_inter = X_invol_inter * sign_ts_regions[None,:]
                X_invol_inter = X_invol_inter[:,comb_sign_ind_idx]
                
                ts_r_int_sign = ts_regions_interaction[:,cregions_sign_idx]
                
                X_sing_inter = np.concatenate((X_invol_inter, ts_r_int_sign), axis=1)
                
                logit_model=sm.Logit(y_train,X_sing_inter)
                result_int_test = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
                odds_sin_inter = np.exp(result_int_test.params)
                p_vals = result_int_test.pvalues 
                n_sign_single = len(comb_sign_ind_idx)
                p_vals_single = p_vals[:n_sign_single]
                odds_single   = odds_sin_inter[:n_sign_single]
                odds_inter    = odds_sin_inter[n_sign_single:]
                p_vals_inter  = p_vals[n_sign_single:]
                for i in range(len(p_vals_inter)):
                    p_val_inter = p_vals_inter[i]
                    # interacting with
                    comb_r = [t for t in cregions_sign[i]]
                    # check p_value of perhaps two regions which were tested for interaction
                    if all(elem in regs_for_interac  for elem in comb_r): 
                        p_val = p_vals_inter[i]
                        if p_val < ex['pval_logit_final'] and odds_inter[i] > 1.0:
    #                        print('p_val inter two weak regions : {}'.format(p_val))
                            keep_duetointer.append(comb_r)
                            # the regions are kept into track_r_kept
                            
                        
                    else:
                        # a 'single' region investigated for interaction with region
                        # which showed not a very significant p_val
                        reg_investigated = [i for i in comb_r if i in regs_for_interac]
                        reg_individual = [i for i in comb_r if i not in regs_for_interac]
                        idx_sign_single = comb_sign_ind.index(reg_individual)
                        p_val_inter  = p_vals_inter[i]
                        p_val_single = p_vals_single[idx_sign_single]
                        odds_int     = odds_inter[i]
                        odds_sin     = odds_single[idx_sign_single]
                        if p_val_inter < p_val_single and odds_int > odds_sin:
    #                        print('p_val_inter : {}\np_val_single {}'.format(
    #                                p_val_inter, p_val_single))
                            if p_val_inter < ex['pval_logit_final']:
                                keep_duetointer.append(reg_investigated)
                            
                keep_duetointer = [i for i in keep_duetointer]
                
                # delete regions that are not kept after testing if they got good enough
                # p_vals
                flat_keeping = list(set([item for sublist in keep_duetointer for item in sublist]))
                idx_not_keeping = [list(track_r_kept).index(r) for r in reg_investigated if r not in flat_keeping]
                track_r_kept    = np.delete(track_r_kept, idx_not_keeping)
            
#                # if it adds likelihood when reg_investiged interacts with reg_single, 
#                # then I keep both reg_investiged and reg_single in as single predictors
#                # if two reg_investigated became significant together, then I will keep 
#                # them in as an 'interacting time series' reg1 * reg2 = predictor.
#                if (regs_for_interac in keep_duetointer) and len(regs_for_interac)==2:
#                    add_pred_inter = [i for i in keep_duetointer if all(np.equal(i, regs_for_interac))][0]
#                    idx_pred_inter = [i for i in range(n_feat) if regions[i] in add_pred_inter]
#                    ts_inter_kept   = ts_regions_train[:,idx_pred_inter[0]] * ts_regions_train[:,idx_pred_inter[1]]
#                    X_inter_kept = ts_inter_kept[:,None] / np.std(ts_inter_kept[:,None], axis = 0)
#    #            
#    #                X_final = np.concatenate( (X_train, X_inter_kept), axis=1)
#    #            else:
#    #                X_final = X_train
    # =============================================================================
    # Step 3 Perform log regression on (individual) ex['ts_train_std']that past step 1 & 2.
    # =============================================================================
    # all regions
    X = X_init
    # select only region in track_regions_kept:
    r_kept_idx = [regions.index(r) for r in regions if r in track_r_kept]
    sign_r_kept = sign_ts_regions[r_kept_idx]

    ex['ts_train_std'].append(np.std(ts_regions_train[:,r_kept_idx], axis=0))
    X_final = X[:,r_kept_idx] * sign_r_kept[None,:]
    logit_model=sm.Logit(y,X_final)
    result = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
    odds   = np.exp(result.params)
    p_vals = result.pvalues 
    #        print(result.summary2())
            
#            odds_new = odds_new[:len(track_r_kept)]
    logitmodel = result
#    print('final bic value {}'.format(logitmodel.bic))
    init_vs_final_bic.append(logitmodel.bic)
        
#    results = pd.DataFrame(columns=['aic', 'aicc', 'bic', 'p_vals'])
#    for comb in combinations:
##        idx = combinations.index(comb)
#        idx_f = np.where(np.array(comb) == 1)[0]
#        X_train = X[:,idx_f] * signs[None,idx_f]
#        y_train = y
#        
#        logit_model=sm.Logit(y_train,X_train)
#        result = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
#        aicc = result.aic + (2*n_feat**2 + 2*n_feat) / (n_samples - n_feat -1)
#        pd_out = pd.Series( [ result.aic, aicc, result.bic, result.pvalues ], 
#                           index=results.columns )
#        results = results.append(pd_out, ignore_index=True)
#    # select model with lowest AIC
#    min_aic = results[results.aic == results.aic.min()].aic.values
#    for idx in range(len(results)):
#        rel_likely = np.exp(min_aic-)
    
    #%%        
    return odds, track_r_kept, sign_r_kept, logitmodel

def pred_gridcells(RV_ts, filename, ex):
    #%%
    from sklearn.metrics import roc_auc_score
    from ROC_score import ROC_score
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    
    
    # load prediction timeseries
    filename_pred = '/Users/semvijverberg/Dropbox/VIDI_Coumou/Paper1_Sem/output_summ/GBR_FULL_50.csv'
    data = pd.read_csv(filename_pred)
    pred = data['values'].values
    # how to define binary event timeseries:
    min_dur = ex['min_dur'] ; max_break = ex['max_break']  
    min_dur = 3 ; max_break = 1
    grouped = True ; win = 0; rmwin = 7

    # load 'observations'
    filename = os.path.join(ex['RV1d_ts_path'], ex['RVts_filename'])
    dicRV = np.load(filename,  encoding='latin1').item()
    obs_array = dicRV['RV_array']
    RVhour = obs_array.time.dt.hour[0].values
    datesRV = func_CPPA.make_datestr(pd.to_datetime(obs_array.time.values), ex, 
                                        ex['startyear'], ex['endyear'])
    obs_array = obs_array.sel(time=datesRV + pd.Timedelta(int(RVhour), unit='h'))
    
    obs_array = obs_array.rolling(time=rmwin, center=True, 
                                                 min_periods=1).mean(dim='time')
#    pred = data['values'].rolling(rmwin, center=True, min_periods=1).mean().values
                                                 

    
    def core_pred_gridcells(obs_array, pred, min_dur, max_break):
        lats = obs_array.latitude
        lons = obs_array.longitude
        n_space = lats.size*lons.size
    
        time = obs_array.time
        
        
        obs_flat  = np.reshape( obs_array.values, (time.size, n_space) )
        output = np.zeros( (n_space) )
        # params defining binary event timeseries
        tfreq_RVts = pd.Timedelta((time[1]-time[0]).values)

        min_dur = pd.Timedelta(min_dur, 'd') / tfreq_RVts
        max_break = pd.Timedelta(max_break, 'd') / tfreq_RVts
        
        
        for gc in range(obs_flat.shape[-1]):
            
            gc_threshold = np.mean(obs_flat[:,gc]) + 1. * np.std(obs_flat[:,gc])
            events_idx = np.where(obs_flat[:,gc] > gc_threshold)[0][:]  

            y_true = func_CPPA.Ev_binary(events_idx, time.size,  min_dur, 
                                         max_break, grouped=grouped)
            y_true[y_true!=0] = 1

            
            if gc == int(obs_flat.shape[1]/2):
                print('approx. {} events in binary event timeseries'.format(
                        y_true[y_true==1].size))
            if y_true[y_true==1].size > 10:
#                output[gc] = roc_auc_score(y_true, pred)
                if win == 1:
                    output[gc] = roc_auc_score(y_true, pred)
#                    print('sklearn {}'.format(roc_auc_score(y_true, pred)))
#                    output[gc] = ROC_score(pred, y_true, n_boot=0, win=win, n_yrs=39)[0]
#                    print(output[gc])
                elif win == 0:
                    
                    output[gc] = roc_auc_score(y_true, pred)

                    if gc in range(obs_flat.shape[-1])[::150]:
                        print('{} sklearn {:.2f}'.format(gc, roc_auc_score(y_true, pred)))
                        print('own ROC {:.2f}'.format(ROC_score(pred, y_true, n_boot=0, win=win, n_yrs=39)[0]))
                        y_true = func_CPPA.Ev_binary(events_idx, time.size,  min_dur, 
                                         max_break, grouped=True)
                        y_true[y_true!=0] = 1
                        print('{} grouped sklearn {:.2f}'.format(gc, roc_auc_score(y_true, pred)))
                        
                    
            else: 
                output[gc] = 0.0
        output = np.reshape(output, (lats.size, lons.size))
        output = xr.DataArray(output, dims=obs_array[0].dims, coords=obs_array[0].coords)    
        return output
        

    
    output = core_pred_gridcells(obs_array, pred, min_dur, max_break)
    output = output.where(obs_array.mask==True)
    #%%
    T95 = obs_array.quantile(0.95, dim=('latitude','longitude'))  
    threshold = T95.mean() + T95.std()
    events_idx = np.where(T95 > threshold)[0][:]  

    y_true = func_CPPA.Ev_binary(events_idx, T95.time.size,  min_dur, 
                                 max_break, grouped=grouped)
    y_true[y_true!=0] = 1
    roc_auc_score(y_true, pred)
    ROC, std = ROC_score(pred, y_true, n_boot=0, win=win, n_yrs=39)
    from sklearn.metrics import average_precision_score
    AP = average_precision_score(y_true, pred)
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, pred)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, pred)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % ROC)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
#    sign_thres = np.percentile(std, 99)
    sign_thres = 0.634
    #%%
    fig, ax = plot_earth(view = "US")

    
    clevels = np.arange(0.45, np.round(output.max().values, 1)+1E-9 , 0.05)
    own_c = ['slategrey', 'green', 'cyan', 'blue', 
                  'purple', 'red', 'firebrick', 'maroon']
    own_colors = own_c[:len(clevels)]
    cmap = colors.ListedColormap(own_colors)

    
    if output.max().values > clevels[-1]:
        extend = 'max'
    else:
        extend = 'neither'
    im = output.plot.pcolormesh(ax=ax, 
                                transform=ccrs.PlateCarree(),
                                levels=clevels,
                                cmap=cmap,
                                add_colorbar=False)
    output.plot.contourf(ax = ax, 
                        transform=ccrs.PlateCarree(),
                        levels=[0,sign_thres, 1.0],
                        hatches='...', alpha=0.0,
                        add_colorbar=False)
    ax.set_title('')
    
    cbar_ax = fig.add_axes([0.25, 0.28, 
                                  0.5, 0.03], label='cbar')    
    norm = colors.BoundaryNorm(boundaries=clevels, ncolors=256)
    cbar = plt.colorbar(im, cbar_ax, cmap=cmap, orientation='horizontal', 
                 extend=extend, norm=norm)

    cbar.set_ticks(clevels[1::2])
    if output.max().values > clevels[-1]:
        cbar.cmap.set_over(own_c[len(own_colors)-1])
    cbar.set_label('AUC score', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    folder = ''
    for s in filename_pred.split('/')[:-1]:
        folder  = folder + s + '/'
    name = filename_pred.split('/')[-1][:-3] + 'dur{}_pause{}_gr{}_rmwin{}.png'.format(
            min_dur, max_break, grouped, rmwin)
    filename = os.path.join(folder, name)
    plt.savefig(filename, dpi=600)
    

    #%%
def plot_earth(view="EARTH", kwrgs={}):
    #%%
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    # Create Big Figure
    fig = plt.figure( figsize=(18,12) ) 

    # create Projection and Map Elements
    projection = ccrs.PlateCarree()
    ax = fig.add_subplot(111, projection=projection)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.add_feature(cfeature.OCEAN, color="white")
    ax.add_feature(cfeature.LAND, color="linen")

    if view == "US":
        ax.set_xlim(-125, -65)
        ax.set_ylim(29, 47)
    elif view == "EAST US":
        ax.set_xlim(-105, -65)
        ax.set_ylim(25, 50)
    elif view == "EARTH":
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
    #%%
    return fig, ax