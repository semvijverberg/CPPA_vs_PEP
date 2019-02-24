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
from ROC_score import ROC_score_wrapper
import func_CPPA



def make_prediction_new(l_ds_Sem, l_ds_mcK, Prec_reg, ex):
    #%%
    if (
        ex['leave_n_out'] == True and ex['method'] == 'iter'
        or ex['ROC_leave_n_out'] or ex['method'][:6] == 'random'
        ):
        ex['test_ts_mcK'] = np.zeros( len(ex['lags']) , dtype=list)
        ex['test_ts_Sem'] = np.zeros( len(ex['lags']) , dtype=list)
        ex['test_RV'] = np.zeros( len(ex['lags']) , dtype=list)
        ex['test_yrs'] = np.zeros( len(ex['lags']) , dtype=list)
    
    ex['score_per_run'] = []

    for n in range(len(ex['train_test_list'])):
        ex['n'] = n
        train, test = ex['train_test_list'][n][0], ex['train_test_list'][n][1]
        ex['test_year'] = list(set(test['RV'].time.dt.year.values))
        print('test year(s) {}, with {} events.'.format(ex['test_year'],
                                 test['events'].size))
        
        ds_Sem = l_ds_Sem[n].sel(lag=ex['lags'])
        ds_mcK = l_ds_mcK[n].sel(lag=ex['lags'])
        # =============================================================================
        # Make prediction based on logit model found in 'extract_precursor'
        # =============================================================================
        if ex['use_ts_logit'] == True or ex['logit_valid'] == True:
            ds_Sem = logit_fit(ds_Sem, Prec_reg, train, ex)
            ds_Sem = timeseries_for_test(ds_Sem, test, ex)
            # info of logit fit is now appended to ds_Sem
            l_ds_Sem[n] = ds_Sem


        # =============================================================================
        # Calculate ROC score
        # =============================================================================
        ex = ROC_score_wrapper(test, train, ds_mcK, ds_Sem, ex)
               
    #%%
    return ex, l_ds_Sem


def make_prediction(l_ds_Sem, l_ds_mcK, Prec_reg, ex):
    #%%
    if (
        ex['leave_n_out'] == True and ex['method'] == 'iter'
        or ex['ROC_leave_n_out'] or ex['method'][:6] == 'random'
        ):
        ex['test_ts_mcK'] = np.zeros( len(ex['lags']) , dtype=list)
        ex['test_ts_Sem'] = np.zeros( len(ex['lags']) , dtype=list)
        ex['test_RV'] = np.zeros( len(ex['lags']) , dtype=list)
        ex['test_yrs'] = np.zeros( len(ex['lags']) , dtype=list)
    
    ex['score_per_run'] = []

    for n in range(len(ex['train_test_list'])):
        ex['n'] = n
        train, test = ex['train_test_list'][n][0], ex['train_test_list'][n][1]
        ex['test_year'] = list(set(test['RV'].time.dt.year.values))
        print('test year(s) {}, with {} events.'.format(ex['test_year'],
                                 test['events'].size))
        
        ds_Sem = l_ds_Sem[n].sel(lag=ex['lags'])
        ds_mcK = l_ds_mcK[n].sel(lag=ex['lags'])
        # =============================================================================
        # Make prediction based on logit model found in 'extract_precursor'
        # =============================================================================
        if ex['use_ts_logit'] == True or ex['logit_valid'] == True:
            ds_Sem = logit_fit(ds_Sem, Prec_reg, train, ex)
            
            ds_Sem = timeseries_for_test(l_ds_Sem[n], test, ex)
            # info of logit fit is now appended to ds_Sem
#             = ds_Sem


        # =============================================================================
        # Calculate ROC score
        # =============================================================================
        ex = ROC_score_wrapper(test, train, ds_mcK, ds_Sem, ex)
               
    #%%
    return ex, l_ds_Sem





def logit_fit_new(l_ds_Sem, RV_ts, ex):

    if (
        ex['leave_n_out'] == True and ex['method'] == 'iter'
        or ex['ROC_leave_n_out'] or ex['method'][:6] == 'random'
        ):
        ex['test_ts_mcK'] = np.zeros( len(ex['lags']) , dtype=list)
        ex['test_ts_Sem'] = np.zeros( len(ex['lags']) , dtype=list)
        ex['test_RV'] = np.zeros( len(ex['lags']) , dtype=list)
        ex['test_yrs'] = np.zeros( len(ex['lags']) , dtype=list)
    
    # Event time series 
    events = func_CPPA.Ev_timeseries(RV_ts, ex['hotdaythres'], ex).time
    dates = pd.to_datetime(RV_ts.time.values)
    event_idx = [list(dates.values).index(E) for E in events.values]
    binary_events = np.zeros(dates.size)   
    binary_events[event_idx] = 1
    mask_events = np.array(binary_events, dtype=bool)
    
    
    
    for n in range(len(ex['train_test_list'])):
        ex['n'] = n
        
        test =ex['train_test_list'][n][1]
        ex['test_year'] = list(set(test['RV'].time.dt.year.values))
        print('test year(s) {}, with {} events.'.format(ex['test_year'],
                                 test['events'].size))
        
        ds_Sem = l_ds_Sem[n]
        Composite = ds_Sem['pattern_CPPA']
        lats = Composite.latitude.values
        lons = Composite.longitude.values
        
        
        logit_model = []
        
        
        for lag_idx, lag in enumerate(ex['lags']):
            # load in timeseries
            csv_train_test_data = 'testyr{}_{}.csv'.format(ex['test_year'], lag)
            path = os.path.join(ex['output_ts_folder'], csv_train_test_data)
            data = pd.read_csv(path)
            regions_for_ts = [int(r[0]) for r in data.columns[3:].values]
            ts_regions_lag_i = data.iloc[:,3:].values
            
            # only training dataset
            all_yrs = list(pd.to_datetime(data.date.values))
            train_yrs = [all_yrs.index(d) for d in all_yrs if d.year not in ex['test_year']]
            ts_regions_train = ts_regions_lag_i[train_yrs,:]
            mask_events_train = mask_events[train_yrs]
            sign_ts_regions = np.sign(np.mean(ts_regions_train[mask_events_train],axis=0))
            binary_events_train = binary_events[train_yrs]
            
            # Perform training
            odds, regions_kept, combs_kept, logitmodel = train_weights_LogReg(
                    ts_regions_train, sign_ts_regions, binary_events_train, ex)
            
            
            # update regions that were kicked out
            r_kept_idx = [i-1 for i in regions_for_ts if i in regions_kept]
            ts_train_std_kept = ex['ts_train_std'][lag_idx][r_kept_idx]
            Regions_lag_i = ds_Sem['pat_num_CPPA'][lag_idx].squeeze().values
            Composite_lag = Composite[lag_idx]
            
            upd_regions = np.zeros(Regions_lag_i.shape)
            for i in range(len(regions_kept)):
                reg = regions_kept[i]
                upd_regions[Regions_lag_i == reg] =  i+1
        
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
                
            # calculating test year
            test_yrs = [all_yrs.index(d) for d in all_yrs if d.year in ex['test_year']]
            ts_regions_test = ts_regions_lag_i[test_yrs,:]

            ts_regions_lag_i = ts_regions_test[:,:] * sign_ts_regions[None,:]
            # normalize time series (as done in the training)        
            X_n = ts_regions_lag_i / ts_train_std_kept
            
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
            
        ds_Sem['pattern_logit'] = Composite_lag
        ds_Sem['pat_num_logit'] = xrnpmap
        ds_Sem.attrs['logitmodel'] = logit_model

    # info of logit fit is now appended to ds_Sem
    l_ds_Sem[n] = ds_Sem

    
    return ex, l_ds_Sem


    
def logit_fit(ds_Sem, Prec_reg, train, ex):
#%%
    
#    ex['output_ts_folder'] = os.path.join(ex['folder'], 'timeseries')
#    if os.path.isdir(ex['output_ts_folder']) != True : os.makedirs(ex['output_ts_folder'])
    lats = Prec_reg.latitude
    lons = Prec_reg.longitude
    
    array = np.zeros( (len(ex['lags']), len(lats), len(lons)) )
    pattern_p2 = xr.DataArray(data=array, coords=[ex['lags'], lats, lons], 
                          dims=['lag','latitude','longitude'], name='communities_composite',
                          attrs={'units':'Kelvin'})
    


    array = np.zeros( (len(ex['lags']), len(lats), len(lons)) )
    pattern_num_p2 = xr.DataArray(data=array, coords=[ex['lags'], lats, lons], 
                          dims=['lag','latitude','longitude'], name='commun_numb_init', 
                          attrs={'units':'Precursor regions'})
    pattern_num_p2.name = 'commun_numbered'
    
#    array = np.zeros( (len(ex['lags'])) )
#    logit_model = xr.DataArray(data=array, coords=[ex['lags']], 
#                          dims=['lag'], name='logitmodel')
                          
    
    logit_model = []
    
    ex['ts_train_std'] = []
    
    combs_reg_kept = np.zeros( len(ex['lags']), dtype=list) 
    
    Actors_ts_GPH = [[] for i in ex['lags']] #!
    
    x = 0
    for lag in ex['lags']:
#            i = ex['lags'].index(lag)
        idx = ex['lags'].index(lag)
        event_train = func_CPPA.Ev_timeseries(train['RV'], ex['hotdaythres'], ex).time
        event_train = func_CPPA.to_datesmcK(event_train, event_train.dt.hour[0], 
                                           train['RV'].time[0].dt.hour)
        events_min_lag = event_train - pd.Timedelta(int(lag), unit='d')
        dates_train = func_CPPA.to_datesmcK(train['RV'].time, train['RV'].time.dt.hour[0], 
                                           train['RV'].time[0].dt.hour)
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
        
        
        np.warnings.filterwarnings('ignore')
        mask_regions = ds_Sem['pat_num_CPPA'].sel(lag=lag).values >= 1
        # training perdiod (excluding info from test part)
        Prec_trainsel = Prec_reg.isel(time=train['Prec_train_idx'])
        # actor/precursor full 3d timeseries at RV period minus lag, normalized over std within dates_train_min_lag
        ts_3d_n = Prec_trainsel.sel(time=dates_train_min_lag)/ds_Sem['std_train_min_lag'][idx]
        
        # ts_3d is given more weight to robust precursor regions
        ts_3d_nw = ts_3d_n * ds_Sem['weights'].sel(lag=lag)
        mask_notnan = (np.product(np.isnan(ts_3d_nw.values),axis=0)==False) # nans == False
        mask = mask_notnan * mask_regions
        # normal mean of extracted regions
        composite_p1 = ds_Sem['pattern_CPPA'].sel(lag=lag).where(mask==True)
        ts_3d_nw     = ts_3d_nw.where(mask==True)

        
        regions_for_ts = np.arange(ds_Sem['pat_num_CPPA'].min(), ds_Sem['pat_num_CPPA'][idx].max()+1E-09)
        Regions_lag_i = ds_Sem['pat_num_CPPA'][idx].squeeze().values
#        mean_n = composite_p1/ts_3d.std(dim='time')
        npmean        = composite_p1.values
        ts_regions_lag_i, sign_ts_regions = func_CPPA.spatial_mean_regions(Regions_lag_i, 
                                regions_for_ts, ts_3d_nw, npmean)[:2]
        check_nans = np.where(np.isnan(ts_regions_lag_i))
        if check_nans[0].size != 0:
            print('{} nans found in time series of region {}, dropping this region.'.format(
                    check_nans[0].size, 
                    regions_for_ts[check_nans[1]]))
            regions_for_ts = np.delete(regions_for_ts, check_nans[1])
            ts_regions_lag_i = np.delete(ts_regions_lag_i, check_nans[1], axis=1)
            sign_ts_regions  = np.delete(sign_ts_regions, check_nans[1], axis=0)
            
#        name_trainset = 'testyr{}_{}_trainset.csv'.format(ex['test_year'], lag)
#        spatcov = func_CPPA.cross_correlation_patterns(ts_3d_nw, ds_Sem['pat_num_CPPA'][idx])
#        columns = list(regions_for_ts)
#        columns.insert(0, 'spatcov')
#        data = np.concatenate([spatcov.values[:,None], ts_regions_lag_i], axis=1)
#        df = pd.DataFrame(data = data, index=pd.to_datetime(spatcov.time.values), columns=columns)
#        df.to_csv(os.path.join(ex['output_ts_folder'], name_trainset ))
        
        lat_grid = composite_p1.latitude.values
        lon_grid = composite_p1.longitude.values
                
        
        # get wgths and only regions that contributed to probability    
        
        odds, regions_kept, combs_kept, logitmodel = train_weights_LogReg(
                    ts_regions_lag_i, sign_ts_regions, binary_events, ex)
    
            
        upd_regions = np.zeros(Regions_lag_i.shape)
        for i in range(len(regions_kept)):
            reg = regions_kept[i]
            upd_regions[Regions_lag_i == reg] =  i+1
    
        # create map of precursor regions
        npmap = np.ma.reshape(upd_regions, (len(lat_grid), len(lon_grid)))
        mask_strongest = (npmap!=0) 
        npmap[mask_strongest==False] = 0
        xrnpmap = composite_p1.copy()
        xrnpmap.values = npmap
        
        # update the mask for the composite mean
        mask = (('latitude', 'longitude'), mask_strongest)
        composite_p1.coords['mask'] = mask
        xrnpmap.coords['mask'] = mask
        xrnpmap = xrnpmap.where(xrnpmap.mask==True)
        norm_mean = composite_p1.where(xrnpmap.mask==True)
        
    #    plt.figure()
    #    xrnpmap.plot.contourf(cmap=plt.cm.tab10)
            
        
        pattern_p2[idx] = norm_mean * ds_Sem['weights'].sel(lag=lag) 
        pattern_num_p2[idx] = xrnpmap
        logit_model.append(logitmodel)
    ds_Sem['pattern_logit'] = pattern_p2
    ds_Sem['pat_num_logit'] = pattern_num_p2
    ds_Sem.attrs['logitmodel'] = logit_model
    combs_reg_kept[idx] = combs_kept 
#    plt.figure() 
#    weighted_mean.plot.contourf()
    #%%
    return ds_Sem 

def timeseries_for_test(ds_Sem, test, ex):
    #%%
    time = test['RV'].time
    
    array = np.zeros( (len(ex['lags']), len(time)) )
    ts_predic = xr.DataArray(data=array, coords=[ex['lags'], time.values], 
                          dims=['lag','time'], name='ts_predict_logit',
                          attrs={'units':'Kelvin'})
    ds_Sem['ts_prediction'] = ts_predic

    
    for lag in ex['lags']:
        idx = ex['lags'].index(lag)
        dates_test = func_CPPA.to_datesmcK(test['RV'].time, test['RV'].time.dt.hour[0], 
                                           test['Prec'].time[0].dt.hour)
        # select antecedant SST pattern to summer days:
        dates_min_lag = dates_test - pd.Timedelta(int(lag), unit='d')
        var_test_mcK = func_CPPA.find_region(test['Prec'], region=ex['regionmcK'])[0]
    #    full_timeserie_regmck = var_test_mcK.sel(time=dates_min_lag)
    
        var_test_mcK = var_test_mcK.sel(time=dates_min_lag)
        var_test_reg_n = test['Prec'].sel(time=dates_min_lag)/ds_Sem['std_train_min_lag'][idx] 
        
        # add more weight based on robustness
        
        
        mean = ds_Sem['pattern_logit'].sel(lag=lag)
        mean = np.reshape(mean.values, (mean.size))

        xrpattern_lag_i = ds_Sem['pat_num_logit'].sel(lag=lag)
        regions_for_ts = np.arange(xrpattern_lag_i.min(), xrpattern_lag_i.max()+1)
        
        var_test_reg_nw = var_test_reg_n * ds_Sem['weights'].sel(lag=lag)
        
        ts_regions_lag_i, sign_ts_regions = func_CPPA.spatial_mean_regions(
                        xrpattern_lag_i.values, regions_for_ts, 
                        var_test_reg_nw, mean)[:2]


        ts_regions_lag_i = ts_regions_lag_i[:,:] * sign_ts_regions[None,:]
        # normalize time series (as done in the training)        
        X_n = ts_regions_lag_i / ex['ts_train_std'][idx]
        
        logit_model_lag_i = ds_Sem.attrs['logitmodel'][idx]
        ts_pred = logit_model_lag_i.predict(X_n)
        ts_predic[idx] = ts_pred
    if ex['n'] != 0:
        # make sure time axis align, otherwise it gives nans
        ds_Sem['ts_prediction']['time'].values = time
    ds_Sem['ts_prediction'] = ts_predic
#    print(ds_Sem['ts_prediction'])
    
    #%%
    return ds_Sem




def train_weights_LogReg(ts_regions_lag_i, sign_ts_regions, binary_events, ex):
    #%%
#    from sklearn.linear_model import LogisticRegression
#    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm
    import itertools
    # I want to the importance of each regions as is shown in the composite plot
    # A region will increase the probability of the events when it the coef_ is 
    # higher then 1. This means that if the ts shows high values, we have a higher 
    # probability of the events. To ease interpretability I will multiple each ts
    # with the sign of the region, to that each params should be above one (higher
    # coef_, higher probability of events)      
    
    
    ex['std_of_initial_ts'] = np.std(ts_regions_lag_i, axis=0) #!!
    X_init = ts_regions_lag_i / np.std(ts_regions_lag_i, axis=0) #!!
    y = binary_events
    

    # Balance Dataset
    pos = np.where(binary_events == 1)[0]
    neg = np.where(binary_events == 0)[0]
#    
#    # Create 75/25 NEG/POS Selection, Balancing Data Imbalance and Data Amount
    idx = np.random.permutation(np.concatenate([pos, np.random.choice(neg, 3*len(pos), False)]))
#    
    y = y[idx]
    X_init = X_init[idx,:]
    
    n_feat = len(ts_regions_lag_i[0])
    n_samples = len(X_init[:,0])
    
    regions          = np.arange(1, n_feat + 1)   
    track_super_sign = np.arange(1, n_feat + 1)   
    track_r_kept     = np.arange(1, n_feat + 1)
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
                # add 1 to get number of region
                idx_f += 1
                for i in regs_for_interac:
                    if i in list(idx_f):   
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
                combregions.append(idx_f+1)
            
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
                comb_sign_r = set(cregions_sign.flatten())
                comb_sign_ind = [i for i in comb_sign_r if i not in regs_for_interac]
                comb_sign_ind_idx = [i-1 for i in comb_sign_ind if i in regions]
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
                idx_not_keeping = [i for i in reg_investigated if i not in flat_keeping]
                track_r_kept    = np.delete(track_r_kept, idx_not_keeping)
            
                # if it adds likelihood when reg_investiged interacts with reg_single, 
                # then I keep both reg_investiged and reg_single in as single predictors
                # if two reg_investigated became significant together, then I will keep 
                # them in as an 'interacting time series' reg1 * reg2 = predictor.
                if (regs_for_interac in keep_duetointer) and len(regs_for_interac)==2:
                    add_pred_inter = [i for i in keep_duetointer if all(np.equal(i, regs_for_interac))][0]
                    idx_pred_inter = [i for i in range(n_feat) if regions[i] in add_pred_inter]
                    ts_inter_kept   = ts_regions_lag_i[:,idx_pred_inter[0]] * ts_regions_lag_i[:,idx_pred_inter[1]]
                    X_inter_kept = ts_inter_kept[:,None] / np.std(ts_inter_kept[:,None], axis = 0)
    #            
    #                X_final = np.concatenate( (X_train, X_inter_kept), axis=1)
    #            else:
    #                X_final = X_train
    # =============================================================================
    # Step 3 Perform log regression on (individual) regions that past step 1 & 2.
    # =============================================================================
    #%%
    # all regions
    X = X_init
    # select only region in track_regions_kept:
    r_kept_idx = [i-1 for i in regions if i in track_r_kept]
    sign_r_kept = sign_ts_regions[None,r_kept_idx]

    ex['ts_train_std'].append(np.std(ts_regions_lag_i[:,r_kept_idx], axis=0))
    X_final = X[:,r_kept_idx] * sign_r_kept
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



#def NEW_train_weights_LogReg(ts_regions_lag_i, sign_ts_regions, binary_events, ex):
#    #%%
##    from sklearn.linear_model import LogisticRegression
##    from sklearn.model_selection import train_test_split
#    import statsmodels.api as sm
#    import itertools
#    n_feat = len(ts_regions_lag_i[0])
#    n_samples = len(ts_regions_lag_i[:,0])
#
#    regions          = np.arange(1, n_feat + 1)   
#    track_super_sign = np.arange(1, n_feat + 1)   
#    track_r_kept     = np.arange(1, n_feat + 1)
#    signs = sign_ts_regions
#    
#    assert n_feat != 0, 'no features found'
#
#    all_comb = list(itertools.product([0, 1], repeat=n_feat))[1:]
#    combs = [i for i in all_comb if (np.sum(i) == 2) or np.sum(i) == 1]
#    
#    
#    X_n = ts_regions_lag_i / np.std(ts_regions_lag_i, axis=0) #!!!
#    X_n = X_n[:,:] * sign_ts_regions[None,:]
#    y_train = binary_events
#    
#    
#    X = np.zeros( (n_samples, len(combs)) )
#    for c in combs:
#        idx = combs.index(c)
#        ind_r = np.where(np.array(c) == 1)[0]
#        X[:,idx] = np.product(X_n[:,ind_r], axis=1)
#     
## =============================================================================
##    # Forward selection
## =============================================================================
#    # First parameter
#    combs_kept = []
#    final_bics = [9E8]
#    bic = []
#    # first comb that add likelihood
#    first_par_decr = True
#    while first_par_decr:
#        odds_list  = []
#        for m in range(X.shape[1]):
#            logit_model = sm.Logit(y_train,X[:,m])
#            result = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
#            bic.append(result.bic)
#            odds_list.append( np.exp(result.params) )
#        min_bic = np.argmin(bic)
##        min_bic = np.argmax(odds_list)
#        # if first par does not decrease the likelihood, then I keep it
#        # otherwise, I am just picking a random area from my regions, that has 
#        # likely nothing to do with the event
#        logit_model = sm.Logit(y_train,X[:,min_bic])
#        
#        
#        result = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
##        if np.exp(result.params) > 1.0:
#        X_for = X[:,min_bic,None]
#        combs_kept.append(combs[min_bic])
#        val_bic = bic[min_bic]
#        X = np.delete(X, min_bic, axis=1)
#        first_par_decr = False
##        else:
##            # Delete from combs
##            X = np.delete(X, min_bic, axis=1)
#        
#    # incrementally add parameters
#    final_bics.append(val_bic)
#    diff = []
#    forward_not_converged = True
#      
#    while forward_not_converged:
#        # calculate bic when adding an extra parameter
#        pd_res = pd.DataFrame(columns=['bic', 'odds'])
#        bic = []
#        odds_list = []
#        for m in range(X.shape[1]):
#            X_ = np.concatenate( (X_for, X[:,m,None]), axis=1)
#            logit_model = sm.Logit( y_train, X_ )
#            result = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
#            odds_added = np.exp(result.params[-1]) 
#            pd_res = pd_res.append( {'bic':result.bic, 'odds':odds_added} , ignore_index=True )
#        
##        sorted_odds = pd_res.sort_values(by=['odds'], ascending=False)
##        # take top 25 % of odds
##        idx = max(1, int(np.percentile(range(X.shape[1]), 30)) )
##        min_bic = sorted_odds[:idx]['bic'].idxmin()
#    
#        # just take param with min_bic
#        min_bic = pd_res['bic'].idxmin()
#        
#        val_bic = pd_res.iloc[min_bic]['bic']
##        print(val_bic)
#        final_bics.append(val_bic)
#            # relative difference vs. initial bic
#        diff_m = (final_bics[-1] - final_bics[-2])/final_bics[1] 
#        threshold_bic = 1E-4 # used to be -1%, i.e. 0.01
##        if diff_m < threshold_bic and np.exp(result.params[-1]) > 1.:
#        threshold_bic = -0.01
#        if diff_m < threshold_bic:
#            # add parameter to model only if they substantially decreased bic
#            X_for = np.concatenate( (X_for, X[:,min_bic,None] ), axis=1)
#            combs_kept.append(combs[min_bic])
#        # delete parameter from list
#        X = np.delete(X, min_bic, axis=1)
#        #reloop, is new bic value really lower then the old one?
#        diff.append( (final_bics[-1] - final_bics[-2])/final_bics[1] )
#        # terminate if condition is not met 4 times in a row
#        test_n_extra = 5
#        if len(diff) > test_n_extra:
#            # difference of last 5 attempts
#            # Decrease in bic should be bigger than 1% of the initial bic value (only first par)
#            diffs = [i for i in diff[-test_n_extra:] if i > threshold_bic]
#            
#            if len(diffs) == test_n_extra:
#                forward_not_converged = False
#        # if all attempts are made, then do not delete the last variable from X
#        if X.shape[1] == 1:
#            forward_not_converged = False
#
#        
#    final_forward = sm.Logit( y_train, X_for )
#    result = final_forward.fit(disp=0, method='newton', tol=1E-8, retall=True)
##    print(result.summary2())
#            
#    
#    
##    plt.plot(diff[1:])
#   #%%
## =============================================================================
##   # Backward selection
## =============================================================================
#    backward_not_converged = True
#    while backward_not_converged and X_for.shape[1] > 1:
#        bicb = []
#        for d in range(X_for.shape[1]):
#            X_one_d = np.delete(X_for, d, axis=1)
#            logit_model = sm.Logit( y_train, X_one_d )
#            result = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
#            bicb.append(result.bic)
#        min_bicb = np.argmin(bicb)
#        val_bicb = bicb[min_bicb]
#        diff = (val_bicb - val_bic) / val_bic
#        # even if val_bicb is not lower, but within 1% of larger model,
#        # then the model with df-1 is kept.
#        if abs(diff) < threshold_bic:
#            X_for = np.delete(X_for, min_bicb, axis=1)
#            combs_kept = [c for c in combs_kept if combs_kept.index(c) != min_bicb]
#        else:
#            backward_not_converged = False
#   
#         
## =============================================================================
##   # Final model
## =============================================================================
#    final_model = sm.Logit( y_train, X_for )
#    result = final_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
#    odds = np.exp(result.params)
#    
#    # quantify strength of each region
#    region_str = np.array(combs_kept, dtype=float)
#    for c in combs_kept:
#        idx = combs_kept.index(c)
#        ind_r = np.where(np.array(c) == 1)[0]
#        region_str[idx][ind_r] = float(odds[idx])
#        
#    pd_str = pd.DataFrame(columns=['B_coeff'], index= track_r_kept)
#    for i in range(len(np.swapaxes(region_str, 1,0))):
#        reg_c = np.swapaxes(region_str, 1,0)[i]
#        idx_r = np.where(reg_c != 0)[0]
#        if len(idx_r) == 0:
#            pass
#        else:
#            pd_str.loc[i+1, 'B_coeff'] = np.product( reg_c[idx_r] )
#    pd_str = pd_str.dropna()
#    
#    
#    B_coeff = pd_str['B_coeff'].values.squeeze().flatten()
#    
#    track_r_kept = pd_str['B_coeff'].index.values
#    final_model = result
#    # update combs_kept, region numbers are changed further up to 1,2,3, etc..
#    del_zeros = np.sum(combs_kept,axis=0) >= 1
#    combs_kept_new = [tuple(np.array(c)[del_zeros]) for c in combs_kept ]
#    # store std of regions that were kept
#    
#    #%%
#    return B_coeff, track_r_kept, combs_kept_new, final_model
#
#
#
#def NEW_timeseries_for_test(ds_Sem, test, ex):
#    #%%
#    time = test['RV'].time
#    
#    array = np.zeros( (len(ex['lags']), len(time)) )
#    ts_predic = xr.DataArray(data=array, coords=[ex['lags'], time], 
#                          dims=['lag','time'], name='ts_predict_logit',
#                          attrs={'units':'Kelvin'})
#    
#    for lag in ex['lags']:
#        idx = ex['lags'].index(lag)
#        dates_test = to_datesmcK(test['RV'].time, test['RV'].time.dt.hour[0], 
#                                           test['Prec'].time[0].dt.hour)
#        # select antecedant SST pattern to summer days:
#        dates_min_lag = dates_test - pd.Timedelta(int(lag), unit='d')
##        var_test_mcK = find_region(test['Prec'], region=ex['regionmcK'])[0]
##    #    full_timeserie_regmck = var_test_mcK.sel(time=dates_min_lag)
##    
##        var_test_mcK = var_test_mcK.sel(time=dates_min_lag)
#        var_test_reg = test['Prec'].sel(time=dates_min_lag) 
#        mean = ds_Sem['pattern'].sel(lag=lag)
#        mean = np.reshape(mean.values, (mean.size))
#
#        xrpattern_lag_i = ds_Sem['pattern_num'].sel(lag=lag)
#        regions_for_ts = np.arange(xrpattern_lag_i.min(), xrpattern_lag_i.max()+1)
#        
#        ts_3d_w = var_test_reg * ds_Sem['weights'].sel(lag=lag)
#        ts_regions_lag_i, sign_ts_regions = spatial_mean_regions(
#                        xrpattern_lag_i.values, regions_for_ts, 
#                        ts_3d_w, mean)[:2]
#        # make all ts positive
#        ts_regions_lag_i = ts_regions_lag_i[:,:] * sign_ts_regions[None,:]
#        # normalize time series (as done in the training)        
#        X_n = ts_regions_lag_i / np.std(ts_regions_lag_i, axis=0)
#        
#        combs_kept_lag_i  = ds_Sem['combs_kept'].values[idx] 
#        ts_new = np.zeros( (time.size, len(combs_kept_lag_i)) )
#        for c in combs_kept_lag_i:
#            i = combs_kept_lag_i.index(c)
#            idx_r = np.where(np.array(c) == 1)[0]
##            idx_r = ind_r - 1
#            ts_new[:,i] = np.product(X_n[:,idx_r], axis=1)
#        
#
#        logit_model_lag_i = ds_Sem['logitmodel'].values[idx]
#        
#        ts_pred = logit_model_lag_i.predict(ts_new)
##        ts_prediction = (ts_pred-np.mean(ts_pred))/ np.std(ts_pred)
#        ts_predic[idx] = ts_pred
#    
#    ds_Sem['ts_prediction'] = ts_predic
#    #%%
#    return ds_Sem
