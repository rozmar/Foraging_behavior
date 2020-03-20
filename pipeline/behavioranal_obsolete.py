
import datajoint as dj
import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
from scipy.optimize import curve_fit
#import decimal
import warnings
import pipeline.lab as lab
import pipeline.experiment as experiment
from pipeline.pipeline_tools import get_schema_name
#from . import get_schema_name
# 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
schema = dj.schema(get_schema_name('behavior-anal'),locals())


#%%
logistic_regression_trials_back = 15
logistic_regression_first_session = 8
logistic_regression_max_bias = .9 # 0 = no bias, 1 = full bias
block_reward_ratio_increment_step = 10
block_reward_ratio_increment_window = 20
block_reward_ratio_increment_max = 200
#%%
def calculate_average_likelihood(local_income,choice,parameters):
    #%
# =============================================================================
#         local_income = np.asarray(df_choices['local_fractional_income'][0].tolist())
#         choice = np.asarray(df_choices['choice_local_fractional_income'][0].tolist())
#         mu = df_psycurve_fractional['sigmoid_fit_mu']
#         sigma = df_psycurve_fractional['sigmoid_fit_sigma']
# =============================================================================
    if parameters['fit_type'] == 'sigmoid':
        prediction = norm.cdf(local_income, parameters['mu'], parameters['sigma'])
    else:
        prediction = local_income*parameters['slope']+parameters['c']
        #prediction = np.polyval([parameters['slope'],parameters['c']],local_income)
    if len(prediction) > 0:
        score = np.zeros(len(prediction))
        score[choice==1]=np.log(prediction[choice==1])
        score[choice==0]=np.log(1-prediction[choice==0])
        score = score[~np.isinf(score)&~np.isnan(score)]
        SCORE = np.exp(sum(score)/len(score))
        return SCORE
    

def calculate_average_likelihood_series(local_income,choice,parameters,local_filter=np.ones(10)):
    local_filter = local_filter/sum(local_filter)
    if parameters['fit_type'] == 'sigmoid':
        prediction = norm.cdf(local_income, parameters['mu'], parameters['sigma'])
    else:
        prediction = np.asarray(local_income)*parameters['slope']+parameters['c']
    
    score = np.zeros(len(prediction))
    score[choice==1]=np.log(prediction[choice==1])
    score[choice==0]=np.log(1-prediction[choice==0])
    score_series = np.exp(np.convolve(score,local_filter,mode = 'same'))
    return score_series

# =============================================================================
## for two lickports    
# def calculate_local_income(df_behaviortrial,filter_now):
#     trialnum_differential = df_behaviortrial['trial']
#     trialnum_fractional = df_behaviortrial['trial']
#     right_choice = (df_behaviortrial['trial_choice'] == 'right').values
#     left_choice = (df_behaviortrial['trial_choice'] == 'left').values
#     right_reward = ((df_behaviortrial['trial_choice'] == 'right')&(df_behaviortrial['outcome'] == 'hit')).values
#     left_reward = ((df_behaviortrial['trial_choice'] == 'left')&(df_behaviortrial['outcome'] == 'hit')).values
#     
#     right_reward_conv = np.convolve(right_reward , filter_now,mode = 'valid')
#     left_reward_conv = np.convolve(left_reward , filter_now,mode = 'valid')
#     
#     right_choice = right_choice[len(filter_now)-1:]
#     left_choice = left_choice[len(filter_now)-1:]
#     trialnum_differential = trialnum_differential[len(filter_now)-1:]
#     
#     choice_num = np.ones(len(left_choice))
#     choice_num[:]=np.nan
#     choice_num[left_choice] = 0
#     choice_num[right_choice] = 1
#     
#     todel = np.isnan(choice_num)
#     right_reward_conv = right_reward_conv[~todel]
#     left_reward_conv = left_reward_conv[~todel]
#     choice_num = choice_num[~todel]
#     trialnum_differential = trialnum_differential[~todel]
#    
#     local_differential_income = right_reward_conv - left_reward_conv
#     choice_local_differential_income = choice_num
#     
#     local_fractional_income = right_reward_conv/(right_reward_conv+left_reward_conv)
#     choice_local_fractional_income = choice_num
#     
#     todel = np.isnan(local_fractional_income)
#     local_fractional_income = local_fractional_income[~todel]
#     choice_local_fractional_income = choice_local_fractional_income[~todel]
#     trialnum_fractional = trialnum_differential[~todel]
#     return local_fractional_income, choice_local_fractional_income, trialnum_fractional, local_differential_income, choice_local_differential_income, trialnum_differential
# =============================================================================

def calculate_local_income_three_ports(df_behaviortrial,filter_now):
#%
    trialnum_differential = df_behaviortrial['trial']
    trialnum_fractional = df_behaviortrial['trial']
    right_choice = (df_behaviortrial['trial_choice'] == 'right').values
    left_choice = (df_behaviortrial['trial_choice'] == 'left').values
    middle_choice = (df_behaviortrial['trial_choice'] == 'middle').values
    right_reward = ((df_behaviortrial['trial_choice'] == 'right')&(df_behaviortrial['outcome'] == 'hit')).values
    left_reward = ((df_behaviortrial['trial_choice'] == 'left')&(df_behaviortrial['outcome'] == 'hit')).values
    middle_reward = ((df_behaviortrial['trial_choice'] == 'middle')&(df_behaviortrial['outcome'] == 'hit')).values
    
    right_reward_conv = np.convolve(right_reward , filter_now,mode = 'valid')
    left_reward_conv = np.convolve(left_reward , filter_now,mode = 'valid')
    middle_reward_conv = np.convolve(middle_reward , filter_now,mode = 'valid')
    
    right_choice = right_choice[len(filter_now)-1:]
    left_choice = left_choice[len(filter_now)-1:]
    middle_choice = middle_choice[len(filter_now)-1:]
    trialnum_differential = trialnum_differential[len(filter_now)-1:]
    
    choice_num = np.ones(len(left_choice))
    choice_num[:]=np.nan
    choice_num[left_choice] = 0
    choice_num[right_choice] = 1
    choice_num[middle_choice] = 2
    
    todel = np.isnan(choice_num)
    right_reward_conv = right_reward_conv[~todel]
    left_reward_conv = left_reward_conv[~todel]
    middle_reward_conv = middle_reward_conv[~todel]
    choice_num = choice_num[~todel]
    trialnum_differential = trialnum_differential[~todel]
   
    local_differential_income_right = right_reward_conv - left_reward_conv - middle_reward_conv
    local_differential_income_left = - right_reward_conv + left_reward_conv - middle_reward_conv
    local_differential_income_middle = - right_reward_conv - left_reward_conv + middle_reward_conv
    choice_local_differential_income = choice_num
    
    local_fractional_income_right = right_reward_conv/(right_reward_conv+left_reward_conv+middle_reward_conv)
    local_fractional_income_left = left_reward_conv/(right_reward_conv+left_reward_conv+middle_reward_conv)
    local_fractional_income_middle = middle_reward_conv/(right_reward_conv+left_reward_conv+middle_reward_conv)
    choice_local_fractional_income = choice_num
    
    todel = np.isnan(local_fractional_income_right)
    local_fractional_income_right = local_fractional_income_right[~todel]
    local_fractional_income_left = local_fractional_income_left[~todel]
    local_fractional_income_middle = local_fractional_income_middle[~todel]
    choice_local_fractional_income = choice_local_fractional_income[~todel]
    trialnum_fractional = trialnum_differential[~todel]
    #%
    dataout = dict()
    dataout['local_fractional_income_right'] = local_fractional_income_right
    dataout['local_fractional_income_left'] = local_fractional_income_left
    dataout['local_fractional_income_middle'] = local_fractional_income_middle
    dataout['choice_local_fractional_income'] = choice_local_fractional_income
    dataout['trialnum_fractional'] = trialnum_fractional.values
    dataout['local_differential_income_right'] = local_differential_income_right
    dataout['local_differential_income_left'] = local_differential_income_left
    dataout['local_differential_income_middle'] = local_differential_income_middle
    dataout['choice_local_differential_income'] = choice_local_differential_income
    dataout['trialnum_differential'] = trialnum_differential.values
    #%
    return dataout

def bin_psychometric_curve(local_income,choice_num,local_income_binnum,circular = False):
    #%
# =============================================================================
#     local_income = reward_ratio_arc
#     choice_num = choice_arc
#     if circular:
#         #%
#         bottoms = np.arange(np.pi*-1,np.pi, 2*np.pi/local_income_binnum)
#         tops = np.arange(2*np.pi/local_income_binnum-np.pi,np.pi+2*np.pi/local_income_binnum/10, 2*np.pi/local_income_binnum)
#         tops[tops > np.pi] = np.pi
#         bottoms[bottoms < np.pi*-1] = np.pi*-1
#     else:
#         #%%
# =============================================================================
    bottoms = np.arange(0,100, 100/local_income_binnum)
    tops = np.arange(100/local_income_binnum,100.005, 100/local_income_binnum)
    tops[tops > 100] = 100
    bottoms[bottoms < 0] = 0
    reward_ratio_mean = list()
    reward_ratio_sd = list()
    choice_ratio_mean = list()
    choice_ratio_sd = list()
    n = list()
    for bottom,top in zip(bottoms,tops):
        minval = np.percentile(local_income,bottom)
        maxval = np.percentile(local_income,top)
        if minval == maxval:
            idx = (local_income== minval)
        else:
            idx = (local_income>= minval) & (local_income < maxval)
        if circular:
            reward_ratio_mean.append(scipy.stats.circmean(local_income[idx],high = np.pi, low = np.pi*-1))
            reward_ratio_sd.append(scipy.stats.circstd(local_income[idx],high = np.pi, low = np.pi*-1))
        else:
            reward_ratio_mean.append(np.mean(local_income[idx]))
            reward_ratio_sd.append(np.std(local_income[idx]))
# =============================================================================
#         choice_ratio_mean.append(np.mean(choice_num[idx]))
#         choice_ratio_sd.append(np.std(choice_num[idx]))
# =============================================================================
        if circular:
            choice_ratio_mean.append(scipy.stats.circmean(choice_num[idx],high = np.pi, low = np.pi*-1))
            choice_ratio_sd.append(scipy.stats.circstd(choice_num[idx],high = np.pi, low = np.pi*-1))
            n.append(np.sum(idx))
        else:
            bootstrap = bs.bootstrap(choice_num[idx], stat_func=bs_stats.mean)
            choice_ratio_mean.append(bootstrap.value)
            choice_ratio_sd.append(bootstrap.error_width())
            n.append(np.sum(idx))
        #%
    return reward_ratio_mean, reward_ratio_sd, choice_ratio_mean, choice_ratio_sd, n


def logistic_regression_on_trial_history(key, fittype):
#%
    trials_back = logistic_regression_trials_back 
    first_session = logistic_regression_first_session
    max_bias = logistic_regression_max_bias
    label = list()
    data = list()
    if len((lab.WaterRestriction()&key).fetch('water_restriction_number'))>0:
        wrnumber = (lab.WaterRestriction()&key).fetch('water_restriction_number')[0]
        df_bias = pd.DataFrame(SessionBias()& key & 'session >' +str(first_session-1))
        if len(df_bias) > 0:
            df_bias['biasval'] = df_bias.loc[:,['session_bias_choice_right','session_bias_choice_left','session_bias_middle']].T.max()
            sessionsneeded = (df_bias.loc[df_bias['biasval']<=max_bias,'session']).values
            df_behaviortrial_all = None
            for session in sessionsneeded:
                df_behaviortrial_now = pd.DataFrame((experiment.BehaviorTrial()*experiment.SessionBlock()*SessionStats() & key & 'session =' +str(session)) & 'trial > session_pretraining_trial_num')
                if type(df_behaviortrial_all) != pd.DataFrame:
                    df_behaviortrial_all  = df_behaviortrial_now
                else:
                    df_behaviortrial_all = df_behaviortrial_all.append(df_behaviortrial_now)
            #print(key)
            #%
            if len(df_behaviortrial_all)>0:
                sessions = np.unique(df_behaviortrial_all['session'])
                for session in sessions:
                    if session >= first_session:
                        df_behaviortrial=df_behaviortrial_all[df_behaviortrial_all['session']==session]#TODO remove pretraining trials
                        idx = np.argsort(df_behaviortrial['trial'])
                        choices = df_behaviortrial['trial_choice'].values[idx]
                        choices_digitized = np.zeros(len(choices))
                        choices_digitized[choices=='right']=1
                        choices_digitized[choices=='left']=-1
                        outcomes = df_behaviortrial['outcome'].values[idx]
                        rewards_digitized = choices_digitized.copy()
                        rewards_digitized[outcomes=='miss']=0
                        non_rewards_digitized = choices_digitized.copy()
                        non_rewards_digitized[outcomes=='hit']=0
                        for trial in range(trials_back,len(rewards_digitized)):
                            if choices_digitized[trial] != 0:
                                label.append(choices_digitized[trial])
                                if fittype == 'RNRC':
                                    data.append(np.concatenate([rewards_digitized[trial-trials_back:trial],choices_digitized[trial-trials_back:trial],non_rewards_digitized[trial-trials_back:trial]]))
                                elif fittype == 'RC':
                                    data.append(np.concatenate([rewards_digitized[trial-trials_back:trial],choices_digitized[trial-trials_back:trial]]))
                                elif fittype == 'NRC':
                                    data.append(np.concatenate([non_rewards_digitized[trial-trials_back:trial],choices_digitized[trial-trials_back:trial]]))
                                elif fittype == 'RNR':
                                    data.append(np.concatenate([rewards_digitized[trial-trials_back:trial],non_rewards_digitized[trial-trials_back:trial]]))
                                elif fittype == 'R':
                                    data.append(rewards_digitized[trial-trials_back:trial])
                                elif fittype == 'NR':
                                    data.append(non_rewards_digitized[trial-trials_back:trial])
                                elif fittype == 'C':
                                    data.append(choices_digitized[trial-trials_back:trial])
                                else:
                                    print('unidentified fittype.. ERROR')
                                    
                label = np.array(label)
                data = np.matrix(data)
                if len(data) > 1:
                    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.15, random_state=0)
                    logisticRegr = LogisticRegression(solver = 'lbfgs')
                    logisticRegr.fit(x_train, y_train)
                    #predictions = logisticRegr.predict(x_test)
                    score = logisticRegr.score(x_test, y_test)        
                    coefficients = logisticRegr.coef_
                    coefficients = coefficients[0]
                    if fittype == 'RNRC':
                        coeff_rewards = coefficients[trials_back-1::-1]
                        coeff_choices = coefficients[-trials_back-1:trials_back-1:-1]
                        coeff_nonrewards = coefficients[-1:-trials_back-1:-1]
                        key['coefficients_rewards_subject'] = coeff_rewards
                        key['coefficients_choices_subject'] = coeff_choices
                        key['coefficients_nonrewards_subject'] = coeff_nonrewards
                    elif fittype == 'RC':
                        coeff_rewards = coefficients[trials_back-1::-1]
                        coeff_choices = coefficients[-1:trials_back-1:-1]
                        key['coefficients_rewards_subject'] = coeff_rewards
                        key['coefficients_choices_subject'] = coeff_choices
                    elif fittype == 'NRC':
                        coeff_nonrewards = coefficients[trials_back-1::-1]
                        coeff_choices = coefficients[-1:trials_back-1:-1]
                        key['coefficients_nonrewards_subject'] = coeff_nonrewards
                        key['coefficients_choices_subject'] = coeff_choices
                    elif fittype == 'RNR':
                        coeff_rewards = coefficients[trials_back-1::-1]
                        coeff_nonrewards = coefficients[-1:trials_back-1:-1]
                        key['coefficients_rewards_subject'] = coeff_rewards
                        key['coefficients_nonrewards_subject'] = coeff_nonrewards
                    elif fittype == 'R':
                        coeff_rewards = coefficients[::-1]
                        key['coefficients_rewards_subject'] = coeff_rewards
                    elif fittype == 'NR':
                        coeff_nonrewards = coefficients[::-1]
                        key['coefficients_nonrewards_subject'] = coeff_nonrewards
                    elif fittype == 'C':
                        coeff_choices = coefficients[::-1]
                        key['coefficients_choices_subject'] = coeff_choices
                    key['score_subject'] = score
                    #%
                    return key
                    print(wrnumber + ' coefficients fitted for ' + fittype)
                else:
                    return None
                    print('not enough data for' + wrnumber +' in '+ fittype)
#%%
def logistic_regression_on_trial_history_3lp(key, fittype):
#%%
    trials_back = logistic_regression_trials_back 
    first_session = logistic_regression_first_session
    max_bias = logistic_regression_max_bias
    if len((lab.WaterRestriction()&key).fetch('water_restriction_number'))>0:
        wrnumber = (lab.WaterRestriction()&key).fetch('water_restriction_number')[0]
        df_bias = pd.DataFrame(SessionBias()*SessionTrainingType() & key & 'session >' +str(first_session-1) & 'session_task_protocol = 101')
        if len(df_bias) > 0:
            df_bias['biasval'] = df_bias.loc[:,['session_bias_choice_right','session_bias_choice_left','session_bias_middle']].T.max()
            sessionsneeded = (df_bias.loc[df_bias['biasval']<=max_bias,'session']).values
            df_behaviortrial_all = None
            for session in sessionsneeded:
                df_behaviortrial_now = pd.DataFrame((experiment.BehaviorTrial()*experiment.SessionBlock()*SessionStats() & key & 'session =' +str(session) & 'trial > session_pretraining_trial_num'))
                if type(df_behaviortrial_all) != pd.DataFrame:
                    df_behaviortrial_all  = df_behaviortrial_now
                else:
                    df_behaviortrial_all = df_behaviortrial_all.append(df_behaviortrial_now)
            #print(key)
            #%
            if len(df_behaviortrial_all)>0:
                for direction in ['left','right','middle']:
                    label = list()
                    data = list()
                    sessions = np.unique(df_behaviortrial_all['session'])
                    for session in sessions:
                        if session >= first_session:
                            df_behaviortrial=df_behaviortrial_all[df_behaviortrial_all['session']==session]
                            idx = np.argsort(df_behaviortrial['trial'])
                            outcomes = df_behaviortrial['outcome'].values[idx]
                            choices = df_behaviortrial['trial_choice'].values[idx]
                            choices_digitized = np.zeros(len(choices))
                            choices_digitized[choices == direction]=1
                            choices_digitized[(choices != direction) & (outcomes != 'ignore')]=-1
                            rewards_digitized = choices_digitized.copy()
                            rewards_digitized[outcomes=='miss']=0
                            non_rewards_digitized = choices_digitized.copy()
                            non_rewards_digitized[outcomes=='hit']=0
                            for trial in range(trials_back,len(rewards_digitized)):
                                if choices_digitized[trial] != 0:
                                    label.append(choices_digitized[trial])
                                    if fittype == 'RNRC':
                                        data.append(np.concatenate([rewards_digitized[trial-trials_back:trial],choices_digitized[trial-trials_back:trial],non_rewards_digitized[trial-trials_back:trial]]))
                                    elif fittype == 'RC':
                                        data.append(np.concatenate([rewards_digitized[trial-trials_back:trial],choices_digitized[trial-trials_back:trial]]))
                                    elif fittype == 'NRC':
                                        data.append(np.concatenate([non_rewards_digitized[trial-trials_back:trial],choices_digitized[trial-trials_back:trial]]))
                                    elif fittype == 'RNR':
                                        data.append(np.concatenate([rewards_digitized[trial-trials_back:trial],non_rewards_digitized[trial-trials_back:trial]]))
                                    elif fittype == 'R':
                                        data.append(rewards_digitized[trial-trials_back:trial])
                                    elif fittype == 'NR':
                                        data.append(non_rewards_digitized[trial-trials_back:trial])
                                    elif fittype == 'C':
                                        data.append(choices_digitized[trial-trials_back:trial])
                                    else:
                                        print('unidentified fittype.. ERROR')
                                        
                    label = np.array(label)
                    data = np.matrix(data)
                    if len(data) > 1:
                        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.15, random_state=0)
                        logisticRegr = LogisticRegression(solver = 'lbfgs')
                        logisticRegr.fit(x_train, y_train)
                        #predictions = logisticRegr.predict(x_test)
                        score = logisticRegr.score(x_test, y_test)        
                        coefficients = logisticRegr.coef_
                        coefficients = coefficients[0]
                        if fittype == 'RNRC':
                            coeff_rewards = coefficients[trials_back-1::-1]
                            coeff_choices = coefficients[-trials_back-1:trials_back-1:-1]
                            coeff_nonrewards = coefficients[-1:-trials_back-1:-1]
                            key['coefficients_rewards_subject'+'_'+direction] = coeff_rewards
                            key['coefficients_choices_subject'+'_'+direction] = coeff_choices
                            key['coefficients_nonrewards_subject'+'_'+direction] = coeff_nonrewards
                        elif fittype == 'RC':
                            coeff_rewards = coefficients[trials_back-1::-1]
                            coeff_choices = coefficients[-1:trials_back-1:-1]
                            key['coefficients_rewards_subject'+'_'+direction] = coeff_rewards
                            key['coefficients_choices_subject'+'_'+direction] = coeff_choices
                        elif fittype == 'NRC':
                            coeff_nonrewards = coefficients[trials_back-1::-1]
                            coeff_choices = coefficients[-1:trials_back-1:-1]
                            key['coefficients_nonrewards_subject'+'_'+direction] = coeff_nonrewards
                            key['coefficients_choices_subject'+'_'+direction] = coeff_choices
                        elif fittype == 'RNR':
                            coeff_rewards = coefficients[trials_back-1::-1]
                            coeff_nonrewards = coefficients[-1:trials_back-1:-1]
                            key['coefficients_rewards_subject'+'_'+direction] = coeff_rewards
                            key['coefficients_nonrewards_subject'+'_'+direction] = coeff_nonrewards
                        elif fittype == 'R':
                            coeff_rewards = coefficients[::-1]
                            key['coefficients_rewards_subject'+'_'+direction] = coeff_rewards
                        elif fittype == 'NR':
                            coeff_nonrewards = coefficients[::-1]
                            key['coefficients_nonrewards_subject'+'_'+direction] = coeff_nonrewards
                        elif fittype == 'C':
                            coeff_choices = coefficients[::-1]
                            key['coefficients_choices_subject'+'_'+direction] = coeff_choices
                        key['score_subject'+'_'+direction] = score
                    else:
                        print('not enough data for' + wrnumber +' in '+ fittype)                    
                        return None
                        
                        #%
                print(wrnumber + ' coefficients fitted for ' + fittype)
                return key
            else:
                print('not enough data for' + wrnumber +' in '+ fittype)                    
                return None
                

#%
def logistic_regression_on_trial_history_convolved(key, fittype):
    #%
    filter_time_constants = np.arange(1,25,2)
    trials_back = logistic_regression_trials_back 
    first_session = logistic_regression_first_session
    max_bias = logistic_regression_max_bias
    label = list()
    data = list()
    if len((lab.WaterRestriction()&key).fetch('water_restriction_number'))>0:
        wrnumber = (lab.WaterRestriction()&key).fetch('water_restriction_number')[0]
        df_bias = pd.DataFrame(SessionBias()& key & 'session >' +str(first_session-1))
        if len(df_bias) > 0:
            df_bias['biasval'] = df_bias.loc[:,['session_bias_choice_right','session_bias_choice_left','session_bias_middle']].T.max()#np.abs(df_bias['session_bias_choice']*2 -1) # TODO FIX THIS!!!
            sessionsneeded = (df_bias.loc[df_bias['biasval']<=max_bias,'session']).values
            df_behaviortrial_all = None
            for session in sessionsneeded:
                df_behaviortrial_now = pd.DataFrame((experiment.BehaviorTrial() & key & 'session =' +str(session)))
                if type(df_behaviortrial_all) != pd.DataFrame:
                    df_behaviortrial_all  = df_behaviortrial_now
                else:
                    df_behaviortrial_all = df_behaviortrial_all.append(df_behaviortrial_now)   
            if len(df_behaviortrial_all)>0:
                sessions = np.unique(df_behaviortrial_all['session'])
                for session in sessions:
                    if session >= first_session:
                        df_behaviortrial=df_behaviortrial_all[df_behaviortrial_all['session']==session]
                        idx = np.argsort(df_behaviortrial['trial'])
                        choices = df_behaviortrial['trial_choice'].values[idx]
                        choices_digitized = np.zeros(len(choices))
                        choices_digitized[choices=='right']=1
                        choices_digitized[choices=='left']=-1
                        outcomes = df_behaviortrial['outcome'].values[idx]
                        rewards_digitized = choices_digitized.copy()
                        rewards_digitized[outcomes=='miss']=0
                        non_rewards_digitized = choices_digitized.copy()
                        non_rewards_digitized[outcomes=='hit']=0
                        choices_conv = np.zeros([len(filter_time_constants),len(choices_digitized)])
                        rewards_conv = np.zeros([len(filter_time_constants),len(choices_digitized)])
                        non_rewards_conv = np.zeros([len(filter_time_constants),len(choices_digitized)])
                        #print(key)
                        if len(rewards_digitized)>trials_back:
                            for filtidx,filter_tau in enumerate(filter_time_constants):
                                choices_conv[filtidx,:] = np.convolve(np.exp(-np.arange(0,trials_back)/filter_tau)[::-1],choices_digitized,mode = 'same')
                                rewards_conv[filtidx,:] = np.convolve(np.exp(-np.arange(0,trials_back)/filter_tau)[::-1],rewards_digitized,mode = 'same')
                                non_rewards_conv[filtidx,:] = np.convolve(np.exp(-np.arange(0,trials_back)/filter_tau)[::-1],non_rewards_digitized,mode = 'same')
                            for trial in range(trials_back,len(rewards_digitized)):
                                if choices_digitized[trial] != 0:
                                    label.append(choices_digitized[trial])
                                    if fittype == 'RNRC':
                                        data.append(np.concatenate([rewards_conv[:,trial],choices_conv[:,trial],non_rewards_conv[:,trial]]))
                                    elif fittype == 'RC':
                                        data.append(np.concatenate([rewards_conv[:,trial],choices_conv[:,trial]]))
                                    elif fittype == 'NRC':
                                        data.append(np.concatenate([non_rewards_conv[:,trial],choices_conv[:,trial]]))
                                    elif fittype == 'RNR':
                                        data.append(np.concatenate([rewards_conv[:,trial],non_rewards_conv[:,trial]]))
                                    elif fittype == 'R':
                                        data.append(rewards_conv[:,trial])                                    
                                    elif fittype == 'NR':
                                        data.append(non_rewards_conv[:,trial])                                    
                                    elif fittype == 'C':
                                        data.append(choices_conv[:,trial])
                                    else:
                                        print('unidentified fittype.. ERROR')
                label = np.array(label)
                data = np.matrix(data)
                if len(data) > 1:
                    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.15, random_state=0)
                    logisticRegr = LogisticRegression(solver = 'lbfgs')
                    logisticRegr.fit(x_train, y_train)
                    #predictions = logisticRegr.predict(x_test)
                    score = logisticRegr.score(x_test, y_test)        
                    coefficients = logisticRegr.coef_
                    coefficients = coefficients[0]
                    if fittype == 'RNRC':
                        coeff_rewards = coefficients[trials_back-1::]
                        coeff_choices = coefficients[-trials_back-1:trials_back-1:]
                        coeff_nonrewards = coefficients[-1:-trials_back-1:]
                        key['coefficients_rewards_subject'] = coeff_rewards
                        key['coefficients_choices_subject'] = coeff_choices
                        key['coefficients_nonrewards_subject'] = coeff_nonrewards
                    elif fittype == 'RC':
                        coeff_rewards = coefficients[trials_back-1::]
                        coeff_choices = coefficients[-1:trials_back-1:]
                        key['coefficients_rewards_subject'] = coeff_rewards
                        key['coefficients_choices_subject'] = coeff_choices
                    elif fittype == 'NRC':
                        coeff_nonrewards = coefficients[trials_back-1::]
                        coeff_choices = coefficients[-1:trials_back-1:]
                        key['coefficients_nonrewards_subject'] = coeff_nonrewards
                        key['coefficients_choices_subject'] = coeff_choices
                    elif fittype == 'RNR':
                        coeff_rewards = coefficients[trials_back-1::]
                        coeff_nonrewards = coefficients[-1:trials_back-1:]
                        key['coefficients_rewards_subject'] = coeff_rewards
                        key['coefficients_nonrewards_subject'] = coeff_nonrewards
                    elif fittype == 'R':
                        coeff_rewards = coefficients[::]
                        key['coefficients_rewards_subject'] = coeff_rewards
                    elif fittype == 'NR':
                        coeff_nonrewards = coefficients[::]
                        key['coefficients_nonrewards_subject'] = coeff_nonrewards
                    elif fittype == 'C':
                        coeff_choices = coefficients[::]
                        key['coefficients_choices_subject'] = coeff_choices
                    key['score_subject'] = score
                    key['filter_time_constants'] = filter_time_constants
                    #%
                    return key
                    print(wrnumber + ' coefficients fitted for ' + fittype)
                else:
                    return None
                    print('not enough data for' + wrnumber +' in '+ fittype)


                        
#%%
@schema
class TrialReactionTime(dj.Computed):
    definition = """
    -> experiment.BehaviorTrial
    ---
    reaction_time : decimal(8,4) # reaction time in seconds (first lick relative to go cue) [-1 in case of ignore trials]
    first_lick_time : decimal(8,4) # time of the first lick after GO cue from trial start in seconds [-1 in case of ignore trials]
    """
    def make(self, key):
        df_licks=pd.DataFrame((experiment.ActionEvent & key).fetch())
        df_gocue = pd.DataFrame((experiment.TrialEvent() & key).fetch())
        gocue_time = df_gocue['trial_event_time'][df_gocue['trial_event_type'] == 'go']
        lick_times = (df_licks['action_event_time'][df_licks['action_event_time'].values>gocue_time.values] - gocue_time.values).values
        key['reaction_time'] = -1
        key['first_lick_time'] = -1
        if len(lick_times) > 0:
            key['reaction_time'] = float(min(lick_times))  
            key['first_lick_time'] = float(min(lick_times))  + float(gocue_time.values)
        self.insert1(key,skip_duplicates=True)
        
@schema
class TrialLickBoutLenght(dj.Computed):
    definition = """
    -> experiment.BehaviorTrial
    ---
    lick_bout_length : decimal(8,4) # lick bout lenght in seconds
    """
    def make(self, key):
        maxlickinterval = .2
        df_lickrhythm = pd.DataFrame(((experiment.BehaviorTrial()*experiment.ActionEvent()) & key)*TrialReactionTime())
        
        if len(df_lickrhythm )>0 and df_lickrhythm['outcome'][0]== 'hit':
            df_lickrhythm['licktime'] = np.nan
            df_lickrhythm['licktime'] = df_lickrhythm['action_event_time']-df_lickrhythm['first_lick_time']
            df_lickrhythm['lickdirection'] = np.nan
            df_lickrhythm.loc[df_lickrhythm['action_event_type'] == 'left lick','lickdirection'] = 'left'
            df_lickrhythm.loc[df_lickrhythm['action_event_type'] == 'right lick','lickdirection'] = 'right'
            df_lickrhythm.loc[df_lickrhythm['action_event_type'] == 'middle lick','lickdirection'] = 'middle'
            df_lickrhythm['firs_licktime_on_the_other_side'] = np.nan
            df_lickrhythm['lickboutlength'] = np.nan

            firs_lick_on_the_other_side = float(np.min(df_lickrhythm.loc[(df_lickrhythm['lickdirection'] != df_lickrhythm['trial_choice']) & (df_lickrhythm['licktime'] > 0) ,'licktime']))
            if np.isnan(firs_lick_on_the_other_side):
                firs_lick_on_the_other_side = np.inf
            df_lickrhythm['firs_licktime_on_the_other_side'] = firs_lick_on_the_other_side
            lickbouttimes = df_lickrhythm.loc[(df_lickrhythm['lickdirection'] == df_lickrhythm['trial_choice']) & (df_lickrhythm['licktime'] < firs_lick_on_the_other_side) & (df_lickrhythm['licktime'] >= 0),'licktime']
            
            if len(lickbouttimes)>1 and any(lickbouttimes.diff().values>maxlickinterval):
                lickbouttimes  = lickbouttimes[:np.where(lickbouttimes.diff().values>maxlickinterval)[0][0]]
            lickboutlenghtnow = float(np.max(lickbouttimes))
            if np.isnan(lickboutlenghtnow):
                lickboutlenghtnow = 0
            #df_lickrhythm['lickboutlength'] = lickboutlenghtnow 
            #%%
        else:
            lickboutlenghtnow = 0
        key['lick_bout_length'] = lickboutlenghtnow
        self.insert1(key,skip_duplicates=True)
        
        
@schema
class SessionStats(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    session_trialnum : int #number of trials
    session_blocknum : int #number of blocks
    session_hits : int #number of hits
    session_misses : int #number of misses
    session_ignores : int #number of ignores
    session_ignore_trial_nums : longblob #number of ignores
    session_autowaters : int #number of autowaters
    session_length : decimal(10, 4) #length of the session in seconds
    session_pretraining_trial_num = null: int #number of pretraining trials
    session_1st_3_ignores = null : int #trialnum where the first three ignores happened in a row
    session_1st_2_ignores = null : int #trialnum where the first three ignores happened in a row
    session_1st_ignore = null : int #trialnum where the first ignore happened    
    """
    def make(self, key):
        #%%
        keytoadd = key
        keytoadd['session_trialnum'] = len(experiment.SessionTrial()&key)
        keytoadd['session_blocknum'] = len(experiment.SessionBlock()&key)
        keytoadd['session_hits'] = len(experiment.BehaviorTrial()&key&'outcome = "hit"')
        keytoadd['session_misses'] = len(experiment.BehaviorTrial()&key&'outcome = "miss"')
        keytoadd['session_ignores'] = len(experiment.BehaviorTrial()&key&'outcome = "ignore"')
        keytoadd['session_autowaters'] = len(experiment.TrialNote & key &'trial_note_type = "autowater"')
        if keytoadd['session_trialnum'] > 0:
            keytoadd['session_length'] = float(((experiment.SessionTrial() & key).fetch('trial_stop_time')).max())
        else:
            keytoadd['session_length'] = 0
        df_choices = pd.DataFrame((experiment.BehaviorTrial()*experiment.SessionBlock()) & key)
        if len(df_choices)>0:
            realtraining = (df_choices['p_reward_left']<1) & (df_choices['p_reward_right']<1) & ((df_choices['p_reward_middle']<1) | df_choices['p_reward_middle'].isnull())
            if not realtraining.values.any():
                keytoadd['session_pretraining_trial_num'] = keytoadd['session_trialnum']
               # print('all pretraining')
            else:
                keytoadd['session_pretraining_trial_num'] = realtraining.values.argmax()
                #print(str(realtraining.values.argmax())+' out of '+str(keytoadd['session_trialnum']))
            if (df_choices['outcome'][keytoadd['session_pretraining_trial_num']:] == 'ignore').values.any():
                keytoadd['session_1st_ignore'] = (df_choices['outcome'][keytoadd['session_pretraining_trial_num']:] == 'ignore').values.argmax()+keytoadd['session_pretraining_trial_num']+1
                if (np.convolve([1,1],(df_choices['outcome'][keytoadd['session_pretraining_trial_num']:] == 'ignore').values)==2).any():
                    keytoadd['session_1st_2_ignores'] = (np.convolve([1,1],(df_choices['outcome'][keytoadd['session_pretraining_trial_num']:] == 'ignore').values)==2).argmax() +keytoadd['session_pretraining_trial_num']+1
                if (np.convolve([1,1,1],(df_choices['outcome'][keytoadd['session_pretraining_trial_num']:] == 'ignore').values)==3).any():
                    keytoadd['session_1st_3_ignores'] = (np.convolve([1,1,1],(df_choices['outcome'][keytoadd['session_pretraining_trial_num']:] == 'ignore').values)==3).argmax() +keytoadd['session_pretraining_trial_num']+1
       #%%
        self.insert1(keytoadd,skip_duplicates=True)
        
@schema
class BlockStats(dj.Computed):
    definition = """
    -> experiment.SessionBlock
    ---
    block_trialnum : int #number of trials in block
    block_ignores : int #number of ignores
    block_reward_rate: decimal(8,4) # miss = 0, hit = 1
    """
    def make(self, key):
        keytoinsert = key
        keytoinsert['block_trialnum'] = len((experiment.BehaviorTrial() & key))
        keytoinsert['block_ignores'] = len((experiment.BehaviorTrial() & key & 'outcome = "ignore"'))
        keytoinsert['block_reward_rate'] = len((experiment.BehaviorTrial() & key & 'outcome = "hit"'))/keytoinsert['block_trialnum']
        #%%
        self.insert1(keytoinsert,skip_duplicates=True)
        
@schema
class SessionReactionTimeHistogram(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    reaction_time_bins : longblob # reaction time bin edges in seconds (first lick relative to go cue)
    reaction_time_values_all_trials  : longblob # trial numbers for each reaction time bin (ignore trials are not included))
    reaction_time_values_miss_trials  : longblob # trial numbers for each reaction time bin
    reaction_time_values_hit_trials  : longblob # trial numbers for each reaction time bin
    """
    def make(self, key):
        df_behaviortrial = pd.DataFrame(((experiment.BehaviorTrial() & key) * (experiment.SessionTrial() & key)  * (TrialReactionTime() & key)).fetch())
        reaction_times_all = np.array((df_behaviortrial['reaction_time'][df_behaviortrial['outcome']!='ignore']).values, dtype=np.float32)
        reaction_times_hit = np.array((df_behaviortrial['reaction_time'][df_behaviortrial['outcome']=='hit']).values, dtype=np.float32)
        reaction_times_miss = np.array((df_behaviortrial['reaction_time'][df_behaviortrial['outcome']=='miss']).values, dtype=np.float32)        
        vals_all,bins = np.histogram(reaction_times_all,100,(0,1))
        vals_hit,bins = np.histogram(reaction_times_hit,100,(0,1))
        vals_miss,bins = np.histogram(reaction_times_miss,100,(0,1))        
        key['reaction_time_bins'] = bins
        key['reaction_time_values_all_trials'] = vals_all
        key['reaction_time_values_miss_trials'] = vals_miss
        key['reaction_time_values_hit_trials'] = vals_hit
        self.insert1(key,skip_duplicates=True)
@schema
class SessionLickRhythmHistogram(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    lick_rhythm_bins : longblob # lick rhythm time bin edges in seconds (lick time relative to the first lick)
    lick_rhythm_values_all_trials  : longblob # trial numbers for each lick rhythm time bin (ignore trials are not included))
    lick_rhythm_values_miss_trials  : longblob # trial numbers for each lick rhythm time bin
    lick_rhythm_values_hit_trials  : longblob # trial numbers for each lick rhythm time bin
    """
    def make(self, key):
        df_licks=pd.DataFrame((experiment.ActionEvent() & key) * (experiment.BehaviorTrial() & key) * (TrialReactionTime() & key))
        if len(df_licks) > 0: # there might be empty sessions
            alltrials = df_licks['outcome']!='ignore'
            misstrials = df_licks['outcome']=='miss'
            hittrials = df_licks['outcome']=='hit'
            lick_times_from_first_lick_all = np.array( df_licks['action_event_time'][alltrials] - df_licks['first_lick_time'][alltrials] , dtype=np.float32)
            lick_times_from_first_lick_miss = np.array( df_licks['action_event_time'][misstrials] - df_licks['first_lick_time'][misstrials] , dtype=np.float32)
            lick_times_from_first_lick_hit = np.array( df_licks['action_event_time'][hittrials] - df_licks['first_lick_time'][hittrials] , dtype=np.float32)
            vals_all,bins = np.histogram(lick_times_from_first_lick_all,100,(0,1))
            vals_miss,bins = np.histogram(lick_times_from_first_lick_miss,100,(0,1))
            vals_hit,bins = np.histogram(lick_times_from_first_lick_hit,100,(0,1))
            key['lick_rhythm_bins'] = bins
            key['lick_rhythm_values_all_trials'] = vals_all
            key['lick_rhythm_values_miss_trials'] = vals_miss
            key['lick_rhythm_values_hit_trials'] = vals_hit
            self.insert1(key,skip_duplicates=True)

@schema
class SessionRuns(dj.Computed):
    definition = """
    # a run is a sequence of trials when the mouse chooses the same option
    -> experiment.Session
    run_num : int # number of choice block
    ---
    run_start : int # first trial #the switch itself
    run_end : int # last trial #one trial before the next choice
    run_choice : varchar(8) # left or right
    run_length : int # number of trials in this run
    run_hits : int # number of hit trials
    run_misses : int # number of miss trials
    run_consecutive_misses: int # number of consecutive misses before switch
    run_ignores : int # number of ignore trials
    """
    def make(self, key):     
        #%%
        #key = {'subject_id':453475,'session':1}
        df_choices = pd.DataFrame(experiment.BehaviorTrial()&key)
        if len(df_choices)>10:
            df_choices['run_choice'] = df_choices['trial_choice']
            ignores = np.where(df_choices['run_choice']=='none')[0]
            if len(ignores)>0:
                ignoreblock = np.diff(np.concatenate([[0],ignores]))>1
                ignores = ignores[ignoreblock.argmax():]
                ignoreblock = ignoreblock[ignoreblock.argmax():]
                while any(ignoreblock):
                    df_choices.loc[ignores[ignoreblock],'run_choice'] = df_choices.loc[ignores[ignoreblock]-1,'run_choice'].values
                    ignores = np.where(df_choices['run_choice']=='none')[0]
                    ignoreblock = np.diff(np.concatenate([[0],ignores]))>1
                    try:
                        ignores = ignores[ignoreblock.argmax():]
                        ignoreblock = ignoreblock[ignoreblock.argmax():]
                    except:
                        ignoreblock = []

            df_choices['run_choice_num'] = np.nan
            df_choices.loc[df_choices['run_choice'] == 'left','run_choice_num'] = 0
            df_choices.loc[df_choices['run_choice'] == 'right','run_choice_num'] = 1
            df_choices.loc[df_choices['run_choice'] == 'middle','run_choice_num'] = 2
            diffchoice = np.abs(np.diff(df_choices['run_choice_num']))
            diffchoice[np.isnan(diffchoice)] = 0
            switches = np.where(diffchoice>0)[0]
            if any(np.where(df_choices['run_choice']=='none')[0]):
                runstart = np.concatenate([[np.max(np.where(df_choices['run_choice']=='none')[0])+1],switches+1])
            else:
                runstart = np.concatenate([[0],switches+1])
            runend = np.concatenate([switches,[len(df_choices)-1]])
            columns = list(key.keys())
            columns.extend(['run_num','run_start','run_end','run_choice','run_length','run_hits','run_misses','run_consecutive_misses','run_ignores'])
            df_key = pd.DataFrame(data = np.zeros((len(runstart),len(columns))),columns = columns)
    
            ## this is where I generate and insert the dataframe
            for keynow in key.keys(): 
                df_key[keynow] = key[keynow]
            for run_num,(run_start,run_end) in enumerate(zip(runstart,runend)):
                df_key.loc[run_num,'run_num'] = run_num + 1 
                df_key.loc[run_num,'run_start'] = run_start +1 
                df_key.loc[run_num,'run_end'] = run_end + 1 
                try:
                    df_key.loc[run_num,'run_choice'] = df_choices['run_choice'][run_start]
                except:
                    print('error in sessionruns')
                    print(key)
                    df_key.loc[run_num,'run_choice'] = df_choices['run_choice'][run_start]
                df_key.loc[run_num,'run_length'] = run_end-run_start+1
                df_key.loc[run_num,'run_hits'] = sum(df_choices['outcome'][run_start:run_end+1]=='hit')
                df_key.loc[run_num,'run_misses'] = sum(df_choices['outcome'][run_start:run_end+1]=='miss')
                #df_key.loc[run_num,'run_consecutive_misses'] = sum(df_choices['outcome'][(df_choices['outcome'][run_start:run_end+1]=='miss').idxmax():run_end+1]=='miss')
                if sum(df_choices['outcome'][run_start:run_end+1]=='miss') == len(df_choices['outcome'][run_start:run_end+1]=='miss'):
                    df_key.loc[run_num,'run_consecutive_misses'] = sum(df_choices['outcome'][run_start:run_end+1]=='miss')
                else:
                    df_key.loc[run_num,'run_consecutive_misses'] = sum(df_choices['outcome'][(df_choices['outcome'][run_start:run_end+1]!='miss')[::-1].idxmax():run_end+1]=='miss')        
                
                df_key.loc[run_num,'run_ignores'] = sum(df_choices['outcome'][run_start:run_end+1]=='ignore')
                #%
            self.insert(df_key.to_records(index=False))
        #%%
@schema
class SessionTrainingType(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    session_task_protocol : tinyint # the number of the dominant task protocol in the session
    session_real_foraging : bool # True if it is real foraging, false in case of pretraining
    """
    def make(self, key):
        task_protocol,p_reward_left,p_reward_right,p_reward_middle = (experiment.BehaviorTrial() *experiment.SessionBlock() & key).fetch('task_protocol','p_reward_left','p_reward_right','p_reward_middle')
        if len(task_protocol)>0:  # in some sessions there is no behavior at all..
            key['session_task_protocol'] = np.median(task_protocol)
            p_reward_left = np.asarray(p_reward_left,'float')
            p_reward_right = np.asarray(p_reward_right,'float')
            p_reward_middle = np.asarray(p_reward_middle,'float')
            if any((p_reward_left<1) & (p_reward_left>0)) or any((p_reward_right<1) & (p_reward_right>0)) or any((p_reward_middle<1) & (p_reward_middle>0)):
                key['session_real_foraging'] =  True
            else:
                key['session_real_foraging'] =  False
            self.insert1(key,skip_duplicates=True)


       
@schema
class SessionBias(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    session_bias_choice_right : float # 0 = other , 1 = right
    session_bias_lick_right : float # 0 = other , 1 = right
    session_bias_choice_left : float # 0 = other , 1 = left
    session_bias_lick_left : float # 0 = other , 1 = left
    session_bias_choice_middle : float # 0 = other , 1 = middle
    session_bias_lick_middle : float # 0 = other , 1 = middle
    """
    def make(self, key):
        #%%
# =============================================================================
#         wr_name = 'FOR02'
#         session = 8
#         subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
#         key = {
#                 'subject_id':subject_id,
#                 'session':session
#                 } 
# =============================================================================
        #key = {'subject_id':453475,'session':1}
        choices = (experiment.BehaviorTrial()&key).fetch('trial_choice')
        #print(key)
        
        if len(choices)>10:
            choice_right = sum(choices=='right')
            choice_left = sum(choices=='left')
            choice_middle = sum(choices=='middle')
            licks = (experiment.ActionEvent()&key).fetch('action_event_type')
            lick_left = sum(licks =='left lick')
            lick_right = sum(licks =='right lick')
            lick_middle = sum(licks =='middle lick')
            choice_bias_right = choice_right/(choice_right+choice_left+choice_middle)
            choice_bias_left = choice_left/(choice_right+choice_left+choice_middle)
            choice_bias_middle = choice_middle/(choice_right+choice_left+choice_middle)
            if len(licks)>0:
                lick_bias_right = lick_right/(lick_right+lick_left+lick_middle)
                lick_bias_left = lick_left/(lick_right+lick_left+lick_middle)
                lick_bias_middle = lick_middle/(lick_right+lick_left+lick_middle)
            else:
                lick_bias_right = choice_bias_right
                lick_bias_left = choice_bias_left
                lick_bias_middle = choice_bias_middle
            key['session_bias_choice_right'] = choice_bias_right
            key['session_bias_lick_right'] = lick_bias_right
            key['session_bias_choice_left'] = choice_bias_left
            key['session_bias_lick_left'] = lick_bias_left
            key['session_bias_choice_middle'] = choice_bias_middle
            key['session_bias_lick_middle'] = lick_bias_middle
            self.insert1(key,skip_duplicates=True)
        #%%
@schema
class BlockRewardRatio(dj.Computed):
    definition = """
    -> experiment.SessionBlock
    ---
    block_reward_ratio : decimal(8,4) # miss = 0, hit = 1
    block_reward_ratio_first_tertile : decimal(8,4) # 
    block_reward_ratio_second_tertile : decimal(8,4) # 
    block_reward_ratio_third_tertile : decimal(8,4) # 
    block_length : smallint #
    block_reward_ratio_right : decimal(8,4) # other = 0, right = 1
    block_reward_ratio_first_tertile_right : decimal(8,4) # other = 0, right = 1
    block_reward_ratio_second_tertile_right : decimal(8,4) # other = 0, right = 1 
    block_reward_ratio_third_tertile_right : decimal(8,4) # other = 0, right = 1
    block_reward_ratio_left : decimal(8,4) # other = 0, right = 1
    block_reward_ratio_first_tertile_left : decimal(8,4) # other = 0, left = 1
    block_reward_ratio_second_tertile_left : decimal(8,4) # other = 0, left = 1 
    block_reward_ratio_third_tertile_left : decimal(8,4) # other = 0, left = 1
    block_reward_ratio_middle : decimal(8,4) # other = 0, right = 1
    block_reward_ratio_first_tertile_middle : decimal(8,4) # other = 0, middle = 1
    block_reward_ratio_second_tertile_middle : decimal(8,4) # other = 0, middle = 1 
    block_reward_ratio_third_tertile_middle : decimal(8,4) # other = 0, middle = 1
    block_reward_ratios_incremental_right : longblob
    block_reward_ratios_incremental_left : longblob
    block_reward_ratios_incremental_middle : longblob
    block_reward_ratios_incr_window : smallint
    block_reward_ratios_incr_step : smallint
    """    
    def make(self, key):
        #%%
        block_reward_ratio_window_starts = np.arange(block_reward_ratio_increment_window/2,block_reward_ratio_increment_max,block_reward_ratio_increment_step,dtype = int)-int(round(block_reward_ratio_increment_window/2))
        block_reward_ratio_window_ends = np.arange(block_reward_ratio_increment_window/2,block_reward_ratio_increment_max,block_reward_ratio_increment_step,dtype = int)+int(round(block_reward_ratio_increment_window/2))
        block_reward_ratios_incremental_r=np.ones(len(block_reward_ratio_window_ends))*np.nan
        block_reward_ratios_incremental_l=np.ones(len(block_reward_ratio_window_ends))*np.nan
        block_reward_ratios_incremental_m=np.ones(len(block_reward_ratio_window_ends))*np.nan
        #%
        #key = {'subject_id' : 453478, 'session' : 3, 'block':3}
        df_behaviortrial = pd.DataFrame((experiment.BehaviorTrial() & key))
        df_behaviortrial['reward']=0
        df_behaviortrial.loc[df_behaviortrial['outcome'] == 'hit' , 'reward'] = 1
        df_behaviortrial.loc[df_behaviortrial['outcome'] == 'miss' , 'reward'] = 0     
        df_behaviortrial['reward_L']=0
        df_behaviortrial['reward_R']=0
        df_behaviortrial['reward_M']=0
        df_behaviortrial.loc[(df_behaviortrial['trial_choice'] == 'left') & (df_behaviortrial['outcome'] == 'hit') ,'reward_L']=1
        df_behaviortrial.loc[(df_behaviortrial['trial_choice'] == 'right') & (df_behaviortrial['outcome'] == 'hit') ,'reward_R']=1
        df_behaviortrial.loc[(df_behaviortrial['trial_choice'] == 'middle') & (df_behaviortrial['outcome'] == 'hit') ,'reward_M']=1
        trialnum = len(df_behaviortrial)
        key['block_reward_ratio'] = -1
        key['block_reward_ratio_first_tertile'] = -1
        key['block_reward_ratio_second_tertile'] = -1
        key['block_reward_ratio_third_tertile'] = -1
        key['block_reward_ratio_right'] = -1
        key['block_reward_ratio_first_tertile_right'] = -1
        key['block_reward_ratio_second_tertile_right'] = -1
        key['block_reward_ratio_third_tertile_right'] = -1
        key['block_reward_ratio_left'] = -1
        key['block_reward_ratio_first_tertile_left'] = -1
        key['block_reward_ratio_second_tertile_left'] = -1
        key['block_reward_ratio_third_tertile_left'] = -1
        key['block_reward_ratio_middle'] = -1
        key['block_reward_ratio_first_tertile_middle'] = -1
        key['block_reward_ratio_second_tertile_middle'] = -1
        key['block_reward_ratio_third_tertile_middle'] = -1
        key['block_reward_ratios_incremental_right'] = block_reward_ratios_incremental_r
        key['block_reward_ratios_incremental_left'] = block_reward_ratios_incremental_l
        key['block_reward_ratios_incremental_middle'] = block_reward_ratios_incremental_m
        key['block_reward_ratios_incr_window'] = block_reward_ratio_increment_window 
        key['block_reward_ratios_incr_step'] =  block_reward_ratio_increment_step
        #trialnums = (BlockStats()&'subject_id = '+str(key['subject_id'])).fetch('block_trialnum')
        key['block_length'] = trialnum
        if trialnum >10:
            tertilelength = int(np.floor(trialnum /3))
            
            block_reward_ratio = df_behaviortrial.reward.mean()
            block_reward_ratio_first_tertile = df_behaviortrial.reward[:tertilelength].mean()
            block_reward_ratio_second_tertile = df_behaviortrial.reward[-tertilelength:].mean()
            block_reward_ratio_third_tertile = df_behaviortrial.reward[tertilelength:2*tertilelength].mean()
            
            
            if df_behaviortrial.reward.sum() == 0:# np.isnan(block_reward_ratio_differential):
                block_reward_ratio_right = -1
                block_reward_ratio_left = -1
                block_reward_ratio_middle = -1
            else:
                block_reward_ratio_right = df_behaviortrial.reward_R.sum()/df_behaviortrial.reward.sum()
                block_reward_ratio_left = df_behaviortrial.reward_L.sum()/df_behaviortrial.reward.sum()
                block_reward_ratio_middle = df_behaviortrial.reward_M.sum()/df_behaviortrial.reward.sum()
            
            if df_behaviortrial.reward[:tertilelength].sum() == 0: #np.isnan(block_reward_ratio_first_tertile_differential):
                block_reward_ratio_first_tertile_right = -1
                block_reward_ratio_first_tertile_left = -1
                block_reward_ratio_first_tertile_middle = -1
            else:
                block_reward_ratio_first_tertile_right = df_behaviortrial.reward_R[:tertilelength].sum()/df_behaviortrial.reward[:tertilelength].sum()
                block_reward_ratio_first_tertile_left = df_behaviortrial.reward_L[:tertilelength].sum()/df_behaviortrial.reward[:tertilelength].sum()
                block_reward_ratio_first_tertile_middle = df_behaviortrial.reward_M[:tertilelength].sum()/df_behaviortrial.reward[:tertilelength].sum()
                
            if df_behaviortrial.reward[tertilelength:2*tertilelength].sum() == 0: #np.isnan(block_reward_ratio_third_tertile_differential):
                block_reward_ratio_second_tertile_right = -1
                block_reward_ratio_second_tertile_left = -1
                block_reward_ratio_second_tertile_middle = -1
            else:
                block_reward_ratio_second_tertile_right= df_behaviortrial.reward_R[tertilelength:2*tertilelength].sum()/df_behaviortrial.reward[tertilelength:2*tertilelength].sum()
                block_reward_ratio_second_tertile_left = df_behaviortrial.reward_L[tertilelength:2*tertilelength].sum()/df_behaviortrial.reward[tertilelength:2*tertilelength].sum()
                block_reward_ratio_second_tertile_middle = df_behaviortrial.reward_M[tertilelength:2*tertilelength].sum()/df_behaviortrial.reward[tertilelength:2*tertilelength].sum()
            
            if df_behaviortrial.reward[-tertilelength:].sum() == 0: #np.isnan(block_reward_ratio_second_tertile_differential):
                block_reward_ratio_third_tertile_right = -1
                block_reward_ratio_third_tertile_left = -1
                block_reward_ratio_third_tertile_middle = -1
            else:
                block_reward_ratio_third_tertile_right  = df_behaviortrial.reward_R[-tertilelength:].sum()/df_behaviortrial.reward[-tertilelength:].sum()
                block_reward_ratio_third_tertile_left = df_behaviortrial.reward_L[-tertilelength:].sum()/df_behaviortrial.reward[-tertilelength:].sum()
                block_reward_ratio_third_tertile_middle = df_behaviortrial.reward_M[-tertilelength:].sum()/df_behaviortrial.reward[-tertilelength:].sum()
            
            
            
            key['block_reward_ratio'] = block_reward_ratio
            key['block_reward_ratio_first_tertile'] = block_reward_ratio_first_tertile
            key['block_reward_ratio_second_tertile'] = block_reward_ratio_second_tertile
            key['block_reward_ratio_third_tertile'] = block_reward_ratio_third_tertile
            key['block_reward_ratio_right'] = block_reward_ratio_right
            key['block_reward_ratio_first_tertile_right'] = block_reward_ratio_first_tertile_right
            key['block_reward_ratio_second_tertile_right'] = block_reward_ratio_second_tertile_right
            key['block_reward_ratio_third_tertile_right'] = block_reward_ratio_third_tertile_right
            
            key['block_reward_ratio_left'] = block_reward_ratio_left
            key['block_reward_ratio_first_tertile_left'] = block_reward_ratio_first_tertile_left
            key['block_reward_ratio_second_tertile_left'] = block_reward_ratio_second_tertile_left
            key['block_reward_ratio_third_tertile_left'] = block_reward_ratio_third_tertile_left
            
            key['block_reward_ratio_middle'] = block_reward_ratio_middle
            key['block_reward_ratio_first_tertile_middle'] = block_reward_ratio_first_tertile_middle
            key['block_reward_ratio_second_tertile_middle'] = block_reward_ratio_second_tertile_middle
            key['block_reward_ratio_third_tertile_middle'] = block_reward_ratio_third_tertile_middle
            
            for i,(t_start,t_end) in enumerate(zip(block_reward_ratio_window_starts,block_reward_ratio_window_ends)):
                if trialnum >= t_end and df_behaviortrial.reward[t_start:t_end].sum()>0:
                    block_reward_ratios_incremental_r[i] = df_behaviortrial.reward_R[t_start:t_end].sum()/df_behaviortrial.reward[t_start:t_end].sum()
                    block_reward_ratios_incremental_l[i] = df_behaviortrial.reward_L[t_start:t_end].sum()/df_behaviortrial.reward[t_start:t_end].sum()
                    block_reward_ratios_incremental_m[i] = df_behaviortrial.reward_M[t_start:t_end].sum()/df_behaviortrial.reward[t_start:t_end].sum()
            key['block_reward_ratios_incremental_right'] = block_reward_ratios_incremental_r
            key['block_reward_ratios_incremental_left'] = block_reward_ratios_incremental_l
            key['block_reward_ratios_incremental_middle'] = block_reward_ratios_incremental_m

        self.insert1(key,skip_duplicates=True)

@schema
class BlockChoiceRatio(dj.Computed):
    definition = """ # value between 0 and 1 for left and 1 right choices, averaged over the whole block or a fraction of the block
    -> experiment.SessionBlock
    ---
    block_choice_ratio_right = null : decimal(8,4) # 0 = rest, 1 = right
    block_choice_ratio_first_tertile_right = null: decimal(8,4) # 0 = rest, 1 = right
    block_choice_ratio_second_tertile_right = null : decimal(8,4) # 0 = rest, 1 = right
    block_choice_ratio_third_tertile_right = null : decimal(8,4) # 0 = rest, 1 = right
    block_choice_ratios_incremental_right = null: longblob
    
    block_choice_ratio_left = null: decimal(8,4) # 0 = rest, 1 = left
    block_choice_ratio_first_tertile_left = null: decimal(8,4) # 0 = rest, 1 = left
    block_choice_ratio_second_tertile_left = null: decimal(8,4) # 0 = rest, 1 = left
    block_choice_ratio_third_tertile_left = null: decimal(8,4) # 0 = rest, 1 = left
    block_choice_ratios_incremental_left = null: longblob
    
    block_choice_ratio_middle = null: decimal(8,4) # 0 = rest, 1 = middle
    block_choice_ratio_first_tertile_middle = null: decimal(8,4) # 0 = rest, 1 = middle
    block_choice_ratio_second_tertile_middle = null: decimal(8,4) # 0 = rest, 1 = middle
    block_choice_ratio_third_tertile_middle = null: decimal(8,4) # 0 = rest, 1 = middle
    block_choice_ratios_incremental_middle = null: longblob
    """    
    def make(self, key):
        #%%
       # warnings.filterwarnings("error")
        block_reward_ratio_window_starts = np.arange(block_reward_ratio_increment_window/2,block_reward_ratio_increment_max,block_reward_ratio_increment_step,dtype = int)-int(round(block_reward_ratio_increment_window/2))
        block_reward_ratio_window_ends = np.arange(block_reward_ratio_increment_window/2,block_reward_ratio_increment_max,block_reward_ratio_increment_step,dtype = int)+int(round(block_reward_ratio_increment_window/2))
        block_choice_ratios_incremental_right=np.ones(len(block_reward_ratio_window_ends))*np.nan
        block_choice_ratios_incremental_left=np.ones(len(block_reward_ratio_window_ends))*np.nan
        block_choice_ratios_incremental_middle=np.ones(len(block_reward_ratio_window_ends))*np.nan
        
        df_behaviortrial = pd.DataFrame((experiment.BehaviorTrial() & key))
        df_behaviortrial['choice_L']=0
        df_behaviortrial['choice_R']=0
        df_behaviortrial['choice_M']=0
        df_behaviortrial.loc[(df_behaviortrial['trial_choice'] == 'left'),'choice_L']=1
        df_behaviortrial.loc[(df_behaviortrial['trial_choice'] == 'right'),'choice_R']=1
        df_behaviortrial.loc[(df_behaviortrial['trial_choice'] == 'middle'),'choice_M']=1
        trialnum = len(df_behaviortrial)
# =============================================================================
#         key['block_choice_ratio_right'] = -1
#         key['block_choice_ratio_first_tertile_right'] = -1
#         key['block_choice_ratio_second_tertile_right'] = -1
#         key['block_choice_ratio_third_tertile_right'] = -1
#         key['block_choice_ratios_incremental_right']=block_choice_ratios_incremental_right
#         key['block_choice_ratio_left'] = -1
#         key['block_choice_ratio_first_tertile_left'] = -1
#         key['block_choice_ratio_second_tertile_left'] = -1
#         key['block_choice_ratio_third_tertile_left'] = -1
#         key['block_choice_ratios_incremental_left']=block_choice_ratios_incremental_left
#         key['block_choice_ratio_middle'] = -1
#         key['block_choice_ratio_first_tertile_middle'] = -1
#         key['block_choice_ratio_second_tertile_middle'] = -1
#         key['block_choice_ratio_third_tertile_middle'] = -1
#         key['block_choice_ratios_incremental_middle']=block_choice_ratios_incremental_middle
# =============================================================================
        if trialnum >15:
            tertilelength = int(np.floor(trialnum /3))
#%%
            if df_behaviortrial.choice_L.sum()+df_behaviortrial.choice_R.sum()+df_behaviortrial.choice_M.sum()>0:
                key['block_choice_ratio_right'] = df_behaviortrial.choice_R.sum()/(df_behaviortrial.choice_L.sum()+df_behaviortrial.choice_R.sum()+df_behaviortrial.choice_M.sum())
                key['block_choice_ratio_left'] = df_behaviortrial.choice_L.sum()/(df_behaviortrial.choice_L.sum()+df_behaviortrial.choice_R.sum()+df_behaviortrial.choice_M.sum())
                key['block_choice_ratio_middle'] = df_behaviortrial.choice_M.sum()/(df_behaviortrial.choice_L.sum()+df_behaviortrial.choice_R.sum()+df_behaviortrial.choice_M.sum())
                
                if (df_behaviortrial.choice_L[:tertilelength].sum()+df_behaviortrial.choice_R[:tertilelength].sum()+df_behaviortrial.choice_M[:tertilelength].sum())>0:
                    key['block_choice_ratio_first_tertile_right'] = df_behaviortrial.choice_R[:tertilelength].sum()/(df_behaviortrial.choice_L[:tertilelength].sum()+df_behaviortrial.choice_R[:tertilelength].sum()+df_behaviortrial.choice_M[:tertilelength].sum())
                    key['block_choice_ratio_first_tertile_left'] = df_behaviortrial.choice_L[:tertilelength].sum()/(df_behaviortrial.choice_L[:tertilelength].sum()+df_behaviortrial.choice_R[:tertilelength].sum()+df_behaviortrial.choice_M[:tertilelength].sum())
                    key['block_choice_ratio_first_tertile_middle'] = df_behaviortrial.choice_M[:tertilelength].sum()/(df_behaviortrial.choice_L[:tertilelength].sum()+df_behaviortrial.choice_R[:tertilelength].sum()+df_behaviortrial.choice_M[:tertilelength].sum())
                if (df_behaviortrial.choice_L[tertilelength:2*tertilelength].sum()+df_behaviortrial.choice_R[tertilelength:2*tertilelength].sum()+df_behaviortrial.choice_M[tertilelength:2*tertilelength].sum())>0:
                    key['block_choice_ratio_second_tertile_right'] = df_behaviortrial.choice_R[tertilelength:2*tertilelength].sum()/(df_behaviortrial.choice_L[tertilelength:2*tertilelength].sum()+df_behaviortrial.choice_R[tertilelength:2*tertilelength].sum()+df_behaviortrial.choice_M[tertilelength:2*tertilelength].sum())
                    key['block_choice_ratio_second_tertile_left'] = df_behaviortrial.choice_L[tertilelength:2*tertilelength].sum()/(df_behaviortrial.choice_L[tertilelength:2*tertilelength].sum()+df_behaviortrial.choice_R[tertilelength:2*tertilelength].sum()+df_behaviortrial.choice_M[tertilelength:2*tertilelength].sum())
                    key['block_choice_ratio_second_tertile_middle'] = df_behaviortrial.choice_M[tertilelength:2*tertilelength].sum()/(df_behaviortrial.choice_L[tertilelength:2*tertilelength].sum()+df_behaviortrial.choice_R[tertilelength:2*tertilelength].sum()+df_behaviortrial.choice_M[tertilelength:2*tertilelength].sum())
                if (df_behaviortrial.choice_L[-tertilelength:].sum()+df_behaviortrial.choice_R[-tertilelength:].sum()+df_behaviortrial.choice_M[-tertilelength:].sum())>0:
                    key['block_choice_ratio_third_tertile_right'] = df_behaviortrial.choice_R[-tertilelength:].sum()/(df_behaviortrial.choice_L[-tertilelength:].sum()+df_behaviortrial.choice_R[-tertilelength:].sum()+df_behaviortrial.choice_M[-tertilelength:].sum())
                    key['block_choice_ratio_third_tertile_left'] = df_behaviortrial.choice_L[-tertilelength:].sum()/(df_behaviortrial.choice_L[-tertilelength:].sum()+df_behaviortrial.choice_R[-tertilelength:].sum()+df_behaviortrial.choice_M[-tertilelength:].sum())
                    key['block_choice_ratio_third_tertile_middle'] = df_behaviortrial.choice_M[-tertilelength:].sum()/(df_behaviortrial.choice_L[-tertilelength:].sum()+df_behaviortrial.choice_R[-tertilelength:].sum()+df_behaviortrial.choice_M[-tertilelength:].sum())
            for i,(t_start,t_end) in enumerate(zip(block_reward_ratio_window_starts,block_reward_ratio_window_ends)):
                if trialnum >= t_end:
                    block_choice_ratios_incremental_right[i] = df_behaviortrial.choice_R[t_start:t_end].sum()/(df_behaviortrial.choice_L[t_start:t_end].sum()+df_behaviortrial.choice_R[t_start:t_end].sum()+df_behaviortrial.choice_M[t_start:t_end].sum())
                    block_choice_ratios_incremental_left[i] = df_behaviortrial.choice_L[t_start:t_end].sum()/(df_behaviortrial.choice_L[t_start:t_end].sum()+df_behaviortrial.choice_R[t_start:t_end].sum()+df_behaviortrial.choice_M[t_start:t_end].sum())
                    block_choice_ratios_incremental_middle[i] = df_behaviortrial.choice_M[t_start:t_end].sum()/(df_behaviortrial.choice_L[t_start:t_end].sum()+df_behaviortrial.choice_R[t_start:t_end].sum()+df_behaviortrial.choice_M[t_start:t_end].sum())
            key['block_choice_ratios_incremental_right'] = block_choice_ratios_incremental_right
            key['block_choice_ratios_incremental_left'] = block_choice_ratios_incremental_left
            key['block_choice_ratios_incremental_middle'] = block_choice_ratios_incremental_middle
            #%%
        try:
            self.insert1(key,skip_duplicates=True)
        except:
            print('error with blockchoce ratio: '+str(key['subject_id']))
            #print(key)     
     
@schema
class BlockAutoWaterCount(dj.Computed):
    definition = """
    -> experiment.SessionBlock
    ---
    block_autowater_count : smallint # number of autowater trials in block
    """
    def make(self, key):
        df_autowater = pd.DataFrame(experiment.TrialNote()*experiment.SessionBlock() & key)
        if len(df_autowater) == 0:
            block_autowater_count = 0
        else:
            block_autowater_count =(df_autowater['trial_note_type']=='autowater').sum()
        key['block_autowater_count'] = block_autowater_count
        self.insert1(key,skip_duplicates=True)

@schema
class SessionBlockSwitchChoices(dj.Computed): # TODO update to 3  lickports
    definition = """
    -> experiment.Session
    ---
    block_length_prev : longblob
    block_length_next : longblob
    choices_matrix : longblob # 0 = left, 1=right, 2=middle
    p_r_prev : longblob #
    p_l_prev : longblob #
    p_l_next : longblob # 
    p_r_next : longblob #
    p_m_prev : longblob #
    p_m_next : longblob # 
    p_l_change : longblob # 
    p_r_change : longblob #
    p_m_change : longblob #
    
    """    
    def make(self, key):
        minblocklength = 30
        prevblocklength = 30
        nextblocklength = 100
        df_behaviortrial = pd.DataFrame((experiment.BehaviorTrial() & key) * (experiment.SessionBlock() & key) * (BlockRewardRatio()&key))
        df_behaviortrial.fillna(value=pd.np.nan, inplace=True)
        if len(df_behaviortrial)>0:
            df_behaviortrial['trial_choice_plot'] = np.nan
            df_behaviortrial.loc[df_behaviortrial['trial_choice']=='left','trial_choice_plot'] = 0
            df_behaviortrial.loc[df_behaviortrial['trial_choice']=='right','trial_choice_plot'] = 1
            df_behaviortrial.loc[df_behaviortrial['trial_choice']=='middle','trial_choice_plot'] = 2
            blockchanges=np.where(np.diff(df_behaviortrial['block']))[0]
            p_change_L = list()
            p_change_R = list()
            p_change_M = list()
            p_L_prev = list()
            p_R_prev = list()
            p_M_prev = list()
            p_L_next = list()
            p_R_next = list()
            p_M_next = list()
            choices_matrix = list()
            block_length_prev = list()
            block_length_next = list()
            for idx in blockchanges:
                prev_blocknum = df_behaviortrial['block_length'][idx]
                next_blocknum = df_behaviortrial['block_length'][idx+1]
                prev_block_p_L = df_behaviortrial['p_reward_left'][idx]
                next_block_p_L = df_behaviortrial['p_reward_left'][idx+1]
                prev_block_p_R = df_behaviortrial['p_reward_right'][idx]
                next_block_p_R = df_behaviortrial['p_reward_right'][idx+1]
                prev_block_p_M = df_behaviortrial['p_reward_middle'][idx]
                next_block_p_M = df_behaviortrial['p_reward_middle'][idx+1]
                if prev_blocknum > minblocklength and next_blocknum > minblocklength:
                    block_length_prev.append(prev_blocknum)
                    block_length_next.append(next_blocknum)
                    p_L_prev.append(float(prev_block_p_L))
                    p_R_prev.append(float(prev_block_p_R))
                    p_M_prev.append(float(prev_block_p_M))
                    p_L_next.append(float(next_block_p_L))
                    p_R_next.append(float(next_block_p_R))
                    p_M_next.append(float(next_block_p_M))
                    p_change_L.append(float((next_block_p_L-prev_block_p_L)))
                    p_change_R.append(float(next_block_p_R-prev_block_p_R))
                    p_change_M.append(float(next_block_p_M-prev_block_p_M))
                    choices = np.array(df_behaviortrial['trial_choice_plot'][max([np.max([idx-prevblocklength+1,idx-prev_blocknum+1]),0]):idx+np.min([nextblocklength+1,next_blocknum+1])],dtype=np.float32)
                    if next_blocknum < nextblocklength:
                        ending = np.ones(nextblocklength-next_blocknum)*np.nan
                        choices = np.concatenate([choices,ending])
                    if prev_blocknum < prevblocklength:
                        preceding = np.ones(prevblocklength-prev_blocknum)*np.nan
                        choices = np.concatenate([preceding,choices])
                    choices_matrix.append(choices)
            choices_matrix = np.asmatrix(choices_matrix) 
            key['block_length_prev'] = block_length_prev
            key['block_length_next'] = block_length_next
            key['p_l_prev'] = p_L_prev
            key['p_r_prev'] = p_R_prev
            key['p_m_prev'] = p_M_prev
            key['p_l_next'] = p_L_next
            key['p_r_next'] = p_R_next
            key['p_m_next'] = p_M_next
            key['p_l_change'] = p_change_L
            key['p_r_change'] = p_change_R
            key['p_m_change'] = p_change_M
            key['choices_matrix'] = choices_matrix
            self.insert1(key,skip_duplicates=True)
    
    
    
@schema
class SessionFittedChoiceCoefficients(dj.Computed):    
    definition = """
    -> experiment.Session
    ---
    coefficients_rewards : longblob
    coefficients_choices  : longblob
    score :  decimal(8,4)
    """    
    def make(self, key):
        #%%
        #print(key)
        try:
            #%%
            df_behaviortrial = pd.DataFrame((experiment.BehaviorTrial() & key))
            if len(df_behaviortrial)>0:
                trials_back = logistic_regression_trials_back
                idx = np.argsort(df_behaviortrial['trial'])
                choices = df_behaviortrial['trial_choice'][idx].values
                choices_digitized = np.zeros(len(choices))
                choices_digitized[choices=='right']=1
                choices_digitized[choices=='left']=-1
                #choices_digitized[choices=='middle']=110
                outcomes = df_behaviortrial['outcome'][idx].values
                rewards_digitized = choices_digitized.copy()
                rewards_digitized[outcomes=='miss']=0
                label = list()
                data = list()
                for trial in range(trials_back,len(rewards_digitized)):
                    if choices_digitized[trial] != 0:
                        label.append(choices_digitized[trial])
                        data.append(np.concatenate([rewards_digitized[trial-trials_back:trial],choices_digitized[trial-trials_back:trial]]))
                label = np.array(label)
                if len(data)>0:
                    data = np.matrix(data)
                    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.15, random_state=0)
                    logisticRegr = LogisticRegression(solver = 'lbfgs')
                    logisticRegr.fit(x_train, y_train)
                    #predictions = logisticRegr.predict(x_test)
                    score = logisticRegr.score(x_test, y_test)        
                    coefficients = logisticRegr.coef_
                    coefficients = coefficients[0]
                    coeff_rewards = coefficients[trials_back-1::-1]
                    coeff_choices = coefficients[-1:trials_back-1:-1]
                    key['coefficients_rewards'] = coeff_rewards
                    key['coefficients_choices'] = coeff_choices
                    key['score'] = score
                    #%%
                    self.insert1(key,skip_duplicates=True)
                    #%%
        except:
            print('couldn\'t fit the logistic regression')
                #%%

# =============================================================================
#   #%%
#             df_behaviortrial = pd.DataFrame((experiment.BehaviorTrial() & key))
#             if len(df_behaviortrial)>0:
#                 trials_back = logistic_regression_trials_back
#                 idx = np.argsort(df_behaviortrial['trial'])
#                 
#                 
#                 
#                 choices = df_behaviortrial['trial_choice'][idx].values
#                 choices_digitized_right = np.zeros(len(choices))
#                 choices_digitized_right[choices=='right']=1
#                 choices_digitized_right[choices=='left']=-1
#                 choices_digitized_right[choices=='middle']=-1
#                 
#                 choices_digitized_left = np.zeros(len(choices))
#                 choices_digitized_left[choices=='right']=-1
#                 choices_digitized_left[choices=='left']=1
#                 choices_digitized_left[choices=='middle']=-1
#                 
#                 choices_digitized_middle = np.zeros(len(choices))
#                 choices_digitized_middle[choices=='right']=-1
#                 choices_digitized_middle[choices=='left']=-1
#                 choices_digitized_middle[choices=='middle']=1
# 
#                 label_all = np.zeros(len(choices))
#                 label_all[choices=='right']=1
#                 label_all[choices=='left']=3
#                 label_all[choices=='middle']=2
# 
# 
#                 outcomes = df_behaviortrial['outcome'][idx].values
#                 rewards_digitized_right = choices_digitized_right.copy()
#                 rewards_digitized_right[outcomes=='miss']=0
#                 rewards_digitized_left = choices_digitized_left.copy()
#                 rewards_digitized_left[outcomes=='miss']=0
#                 rewards_digitized_middle = choices_digitized_middle.copy()
#                 rewards_digitized_middle[outcomes=='miss']=0
#                 label = list()
#                 data = list()
#                 for trial in range(trials_back,len(rewards_digitized)):
#                     if choices_digitized_right[trial] != 0:
#                         label.append(label_all[trial])
#                         data.append(np.concatenate([rewards_digitized_left[trial-trials_back:trial],rewards_digitized_right[trial-trials_back:trial],rewards_digitized_middle[trial-trials_back:trial]]))
#                 label = np.array(label)
#                 if len(data)>0:
#                     data = np.matrix(data)
#                     x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.15, random_state=0)
#                     logisticRegr = LogisticRegression(solver = 'lbfgs')
#                     logisticRegr.fit(x_train, y_train)
#                     #predictions = logisticRegr.predict(x_test)
#                     score = logisticRegr.score(x_test, y_test)        
#                     coefficients = logisticRegr.coef_
#                     coefficients = coefficients[0]
#                     coeff_left_reward = coefficients[trials_back-1::-1]
#                     coeff_right_reward = coefficients[-trials_back-1:trials_back-1:-1]
#                     coeff_middle_reward = coefficients[-1:-trials_back-1:-1]
# # =============================================================================
# #                     key['coefficients_rewards'] = coeff_rewards
# #                     key['coefficients_choices'] = coeff_choices
# #                     key['score'] = score
# #                     #%%
# #                     self.insert1(key,skip_duplicates=True)
# # =============================================================================
#                     plt.plot(coeff_left_reward,'r-')
#                     plt.plot(coeff_right_reward,'b-')
#                     plt.plot(coeff_middle_reward,'g-')
# =============================================================================
#%%


                
@schema
class SubjectFittedChoiceCoefficientsRNRC(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_rewards_subject : longblob
    coefficients_nonrewards_subject  : longblob
    coefficients_choices_subject  : longblob
    score_subject :  decimal(8,4)
    """    
    def make(self, key):
        key  = logistic_regression_on_trial_history(key, 'RNRC')
        if key:
            self.insert1(key,skip_duplicates=True)    
        

@schema
class SubjectFittedChoiceCoefficientsRC(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_rewards_subject : longblob
    coefficients_choices_subject  : longblob
    score_subject :  decimal(8,4)
    """    
    def make(self, key):
        key  = logistic_regression_on_trial_history(key, 'RC')
        if key:
            self.insert1(key,skip_duplicates=True)

                        
                    
@schema
class SubjectFittedChoiceCoefficientsNRC(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_nonrewards_subject : longblob
    coefficients_choices_subject  : longblob
    score_subject :  decimal(8,4)
    """    
    def make(self, key):
        key  = logistic_regression_on_trial_history(key, 'NRC')
        if key:
            self.insert1(key,skip_duplicates=True)
 
@schema
class SubjectFittedChoiceCoefficientsRNR(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_rewards_subject : longblob
    coefficients_nonrewards_subject  : longblob
    score_subject :  decimal(8,4)
    """    
    def make(self, key):
        key  = logistic_regression_on_trial_history(key, 'RNR')
        if key:
            self.insert1(key,skip_duplicates=True)               

@schema
class SubjectFittedChoiceCoefficientsOnlyRewards(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_rewards_subject : longblob
    score_subject :  decimal(8,4)
    """    
    def make(self, key):
        key  = logistic_regression_on_trial_history(key, 'R')
        if key:
            self.insert1(key,skip_duplicates=True)
    
@schema
class SubjectFittedChoiceCoefficientsOnlyUnRewardeds(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_nonrewards_subject : longblob
    score_subject :  decimal(8,4)
    """    
    def make(self, key):
        key  = logistic_regression_on_trial_history(key, 'NR')
        if key:
            self.insert1(key,skip_duplicates=True)

@schema
class SubjectFittedChoiceCoefficientsOnlyChoices(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_choices_subject : longblob
    score_subject :  decimal(8,4)
    """    
    def make(self, key):
        key  = logistic_regression_on_trial_history(key, 'C')
        if key:
            self.insert1(key,skip_duplicates=True)



@schema
class SubjectFittedChoiceCoefficients3lpRNRC(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_rewards_subject_left : longblob
    coefficients_nonrewards_subject_left  : longblob
    coefficients_choices_subject_left  : longblob
    score_subject_left :  decimal(8,4)
    coefficients_rewards_subject_right : longblob
    coefficients_nonrewards_subject_right  : longblob
    coefficients_choices_subject_right  : longblob
    score_subject_right :  decimal(8,4)
    coefficients_rewards_subject_middle : longblob
    coefficients_nonrewards_subject_middle  : longblob
    coefficients_choices_subject_middle  : longblob
    score_subject_middle :  decimal(8,4)
    """    
    def make(self, key):
        key  = logistic_regression_on_trial_history_3lp(key, 'RNRC')
        if key and len(key.keys())>1:
            try:
                self.insert1(key,skip_duplicates=True)    
            except:
                print('problem with inserting coefficient')
                print(key)
                    
        

@schema
class SubjectFittedChoiceCoefficients3lpRC(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_rewards_subject_left : longblob
    coefficients_choices_subject_left  : longblob
    score_subject_left :  decimal(8,4)
    coefficients_rewards_subject_right : longblob
    coefficients_choices_subject_right : longblob
    score_subject_right :  decimal(8,4)
    coefficients_rewards_subject_middle : longblob
    coefficients_choices_subject_middle  : longblob
    score_subject_middle :  decimal(8,4)
    """    
    def make(self, key):
        key  = logistic_regression_on_trial_history_3lp(key, 'RC')
        if key and len(key.keys())>1:
            self.insert1(key,skip_duplicates=True)

                        
                    
@schema
class SubjectFittedChoiceCoefficients3lpNRC(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_nonrewards_subject_left : longblob
    coefficients_choices_subject_left  : longblob
    score_subject_left :  decimal(8,4)
    coefficients_nonrewards_subject_right : longblob
    coefficients_choices_subject_right  : longblob
    score_subject_right :  decimal(8,4)
    coefficients_nonrewards_subject_middle : longblob
    coefficients_choices_subject_middle  : longblob
    score_subject_middle :  decimal(8,4)
    """    
    def make(self, key):
        key  = logistic_regression_on_trial_history_3lp(key, 'NRC')
        if key and len(key.keys())>1:
            self.insert1(key,skip_duplicates=True)
 
@schema
class SubjectFittedChoiceCoefficients3lpRNR(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_rewards_subject_left : longblob
    coefficients_nonrewards_subject_left  : longblob
    score_subject_left :  decimal(8,4)
    coefficients_rewards_subject_right : longblob
    coefficients_nonrewards_subject_right : longblob
    score_subject_right :  decimal(8,4)
    coefficients_rewards_subject_middle : longblob
    coefficients_nonrewards_subject_middle  : longblob
    score_subject_middle :  decimal(8,4)
    """    
    def make(self, key):
        key  = logistic_regression_on_trial_history_3lp(key, 'RNR')
        if key and len(key.keys())>1:
            self.insert1(key,skip_duplicates=True)               

@schema
class SubjectFittedChoiceCoefficients3lpR(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_rewards_subject_left : longblob
    score_subject_left :  decimal(8,4)
    coefficients_rewards_subject_right : longblob
    score_subject_right :  decimal(8,4)
    coefficients_rewards_subject_middle : longblob
    score_subject_middle :  decimal(8,4)
    """    
    def make(self, key):
        key  = logistic_regression_on_trial_history_3lp(key, 'R')
        if key and len(key.keys())>1:
            self.insert1(key,skip_duplicates=True)
    
@schema
class SubjectFittedChoiceCoefficients3lpNR(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_nonrewards_subject_left : longblob
    score_subject_left :  decimal(8,4)
    coefficients_nonrewards_subject_right : longblob
    score_subject_right :  decimal(8,4)
    coefficients_nonrewards_subject_middle : longblob
    score_subject_middle :  decimal(8,4)
    """    
    def make(self, key):
        key  = logistic_regression_on_trial_history_3lp(key, 'NR')
        if key and len(key.keys())>1:
            self.insert1(key,skip_duplicates=True)

@schema
class SubjectFittedChoiceCoefficients3lpC(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_choices_subject_left : longblob
    score_subject_left :  decimal(8,4)
    coefficients_choices_subject_right : longblob
    score_subject_right :  decimal(8,4)
    coefficients_choices_subject_middle : longblob
    score_subject_middle :  decimal(8,4)
    """    
    def make(self, key):
        key  = logistic_regression_on_trial_history_3lp(key, 'C')
        if key and len(key.keys())>1:
            self.insert1(key,skip_duplicates=True)


 

@schema
class SubjectFittedChoiceCoefficientsVSTime(dj.Computed):    ## TODO: BIAS correction
    definition = """
    -> lab.Subject
    ---
    coefficients_rewards_subject : longblob
    score_subject :  decimal(8,4)
    """    
    def make(self, key):
        #print(key)
        
        timesteps_back = 90
        first_session = logistic_regression_first_session
        label = list()
        data = list()
        if len((lab.WaterRestriction()&key).fetch('water_restriction_number'))>0 and len(experiment.TrialEvent()&key)>0:
            #%%
            wrnumber = (lab.WaterRestriction()&key).fetch('water_restriction_number')[0]
            
            #%%
            if len((experiment.BehaviorTrial() & key & 'session >' +str(first_session-1)))>0:
                sessions = np.unique((experiment.BehaviorTrial() & key & 'session >' +str(first_session-1)).fetch('session'))
                for session in sessions:
                    if session >= first_session:
                        #%%
                        df_behaviortrial = pd.DataFrame((experiment.SessionTrial()*(experiment.TrialEvent()&'trial_event_type = "go"')*experiment.BehaviorTrial())&key&'session = '+str(session))
                        df_behaviortrial['choicetime'] = np.asarray(df_behaviortrial['trial_start_time'] + df_behaviortrial['trial_event_time'],dtype = float)
                        mintval = np.floor(df_behaviortrial['choicetime'].min())
                        maxtval = np.round(df_behaviortrial['choicetime'].max())+1
                        df_behaviortrial['choicetime'] = df_behaviortrial['choicetime'] - mintval
                        maxtval -= mintval
                        choices_digitized = np.zeros(int(maxtval))
                        choices_digitized[np.asarray(np.floor(df_behaviortrial.loc[df_behaviortrial['trial_choice']=='left','choicetime'].values),dtype = int)] = -1
                        choices_digitized[np.asarray(np.floor(df_behaviortrial.loc[df_behaviortrial['trial_choice']=='right','choicetime'].values),dtype = int)] = 1
                        rewards_digitized = choices_digitized.copy()
                        rewards_digitized[np.asarray(np.floor(df_behaviortrial.loc[df_behaviortrial['outcome']=='miss','choicetime'].values),dtype = int)]=0

                        for trial in range(timesteps_back,len(rewards_digitized)):
                            if choices_digitized[trial] != 0:
                                label.append(choices_digitized[trial])
                                data.append(rewards_digitized[trial-timesteps_back:trial])
                label = np.array(label)
                data = np.matrix(data)
                if len(data) > 1:
                    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.15, random_state=0)
                    logisticRegr = LogisticRegression(solver = 'lbfgs')
                    logisticRegr.fit(x_train, y_train)
                    #predictions = logisticRegr.predict(x_test)
                    score = logisticRegr.score(x_test, y_test)        
                    coefficients = logisticRegr.coef_
                    coefficients = coefficients[0]
                    coeff_rewards = coefficients[::-1]
                    key['coefficients_rewards_subject'] = coeff_rewards
                    key['score_subject'] = score
                    self.insert1(key,skip_duplicates=True)
                    print(wrnumber + ' coefficients fitted versus time')
                else:
                    print('not enough data for' + wrnumber)
            else:
                print('not enough data for ' + wrnumber)
# =============================================================================
#         else:
#             print('no WR number for this guy')
# =============================================================================
    

@schema
class SubjectFittedChoiceCoefficientsConvR(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_rewards_subject : longblob
    score_subject :  decimal(8,4)
    filter_time_constants : longblob
    
    """    
    def make(self, key):
        key  = logistic_regression_on_trial_history_convolved(key, 'R')
        if key:
            self.insert1(key,skip_duplicates=True)


            
@schema
class SessionPsychometricDataBoxCar(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    local_fractional_income_right : longblob # Value_right/(Value_all)
    local_fractional_income_left : longblob # Value_left/(Value_all)
    local_fractional_income_middle : longblob # Value_middle/(Value_all)
    choice_local_fractional_income : longblob # 0 = left, 1 = right, 2  = middle
    trialnum_local_fractional_income : longblob # trial number
    local_differential_income_right : longblob # Value_right - Value_rest
    local_differential_income_left : longblob # Value_left - Value_rest
    local_differential_income_middle : longblob # Value_middle - Value_rest
    choice_local_differential_income : longblob # 0 = left, 1 = right, 2  = middle
    trialnum_local_differential_income : longblob # trial number
    local_filter : longblob
    """  
    def make(self,key):
        #%%
        #key = {'subject_id': 452274, 'session': 5}
        warnings.filterwarnings("ignore", category=RuntimeWarning)        
        local_filter = np.ones(10)
        local_filter = local_filter/sum(local_filter)
        df_behaviortrial = pd.DataFrame(experiment.BehaviorTrial()&key)
        if len(df_behaviortrial)>1:
            #local_fractional_income, choice_local_fractional_income, trialnum_fractional, local_differential_income, choice_local_differential_income, trialnum_differential  = calculate_local_income(df_behaviortrial,local_filter)
            #print(key)
            data = calculate_local_income_three_ports(df_behaviortrial,local_filter)
            key['local_fractional_income_right'] = data['local_fractional_income_right']
            key['local_fractional_income_left'] = data['local_fractional_income_left']
            key['local_fractional_income_middle'] = data['local_fractional_income_middle']
            key['choice_local_fractional_income'] = data['choice_local_fractional_income']
            key['trialnum_local_fractional_income'] = data['trialnum_fractional']
            key['local_differential_income_right'] = data['local_differential_income_right']
            key['local_differential_income_left'] = data['local_differential_income_left']
            key['local_differential_income_middle'] = data['local_differential_income_middle']
            key['choice_local_differential_income']= data['choice_local_differential_income']
            key['trialnum_local_differential_income'] = data['trialnum_differential']
            key['local_filter'] = local_filter
            #%%
            self.insert1(key,skip_duplicates=True)

@schema
class SessionPsychometricDataFitted(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    local_fractional_income_right : longblob # Value_right/(Value_all)
    local_fractional_income_left : longblob # Value_left/(Value_all)
    local_fractional_income_middle : longblob # Value_middle/(Value_all)
    choice_local_fractional_income : longblob # 0 = left, 1 = right, 2  = middle
    trialnum_local_fractional_income : longblob # trial number
    local_differential_income_right : longblob # Value_right - Value_rest
    local_differential_income_left : longblob # Value_left - Value_rest
    local_differential_income_middle : longblob # Value_middle - Value_rest
    choice_local_differential_income : longblob # 0 = left, 1 = right, 2  = middle
    trialnum_local_differential_income : longblob # trial number
    local_filter : longblob
    """  
    def make(self,key):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        df_coeff = pd.DataFrame(SubjectFittedChoiceCoefficientsOnlyRewards()&key)
        if len(df_coeff)>0:
            local_filter = df_coeff['coefficients_rewards_subject'][0]#.mean()
            local_filter = local_filter/sum(local_filter)
            df_behaviortrial = pd.DataFrame(experiment.BehaviorTrial()&key)
            if len(df_behaviortrial)>1:
                #local_fractional_income, choice_local_fractional_income, trialnum_fractional, local_differential_income, choice_local_differential_income, trialnum_differential  = calculate_local_income(df_behaviortrial,local_filter)
                data = calculate_local_income_three_ports(df_behaviortrial,local_filter)
                key['local_fractional_income_right'] = data['local_fractional_income_right']
                key['local_fractional_income_left'] = data['local_fractional_income_left']
                key['local_fractional_income_middle'] = data['local_fractional_income_middle']
                key['choice_local_fractional_income'] = data['choice_local_fractional_income']
                key['trialnum_local_fractional_income'] = data['trialnum_fractional']
                key['local_differential_income_right'] = data['local_differential_income_right']
                key['local_differential_income_left'] = data['local_differential_income_left']
                key['local_differential_income_middle'] = data['local_differential_income_middle']
                key['choice_local_differential_income']= data['choice_local_differential_income']
                key['trialnum_local_differential_income'] = data['trialnum_differential']
                key['local_filter'] = local_filter
                self.insert1(key,skip_duplicates=True)
                
                
                

@schema    
class SubjectPsychometricCurveBoxCarFractional(dj.Computed):
    definition = """
    -> lab.Subject
    ---
    reward_ratio_mean  : longblob
    reward_ratio_sd  : longblob
    choice_ratio_mean  : longblob
    choice_ratio_sd  : longblob
    sigmoid_fit_mu : double
    sigmoid_fit_sigma : double 
    sigmoid_fit_exp_temperature : double
    sigmoid_fit_exp_bias : double 
    linear_fit_slope : double
    linear_fit_c : double
    trial_num : longblob
    local_filter : longblob
    """     
    def make(self,key):  
        #%
# =============================================================================
#         key = {'subject_id': 457495}
# =============================================================================
        minsession = 8
        reward_ratio_binnum = 10
        df_psychcurve = pd.DataFrame(SessionPsychometricDataBoxCar()&key & 'session > '+str(minsession-1))
        if len(df_psychcurve )>0:
            print(key)
            reward_ratio_combined = np.concatenate(df_psychcurve['local_fractional_income_right'].values)
            choice_num = np.concatenate(df_psychcurve['choice_local_fractional_income'].values)
            mu,sigma = curve_fit(norm.cdf, reward_ratio_combined, choice_num,p0=[0,1])[0]
            out = curve_fit(lambda t,a,b: 1/(1+np.exp(-a*(t-.5)+b)),  reward_ratio_combined,  choice_num)
            sigmoid_exp_temperature = out[0][0]
            sigmoid_exp_bias = out[0][1]
            out = curve_fit(lambda t,a,b: a*t+b,  reward_ratio_combined,  choice_num)
            slope = out[0][0]
            c = out[0][1]
            reward_ratio_mean, reward_ratio_sd, choice_ratio_mean, choice_ratio_sd, n = bin_psychometric_curve(reward_ratio_combined,choice_num,reward_ratio_binnum)
            
            key['reward_ratio_mean'] = reward_ratio_mean
            key['reward_ratio_sd'] = reward_ratio_sd
            key['choice_ratio_mean'] = choice_ratio_mean
            key['choice_ratio_sd'] = choice_ratio_sd
            key['sigmoid_fit_mu'] = mu
            key['sigmoid_fit_sigma'] = sigma     
            key['sigmoid_fit_exp_temperature'] = sigmoid_exp_temperature
            key['sigmoid_fit_exp_bias'] = sigmoid_exp_bias     
            key['linear_fit_slope'] = slope
            key['linear_fit_c'] = c
            key['trial_num'] = n
            key['local_filter'] = df_psychcurve['local_filter'][0]
            #%
            print(key)
            self.insert1(key,skip_duplicates=True)


@schema    
class SubjectPolarPsyCurveBoxCarFractional2lp(dj.Computed):
    definition = """
    -> lab.Subject
    ---
    reward_ratio_mean  : longblob
    reward_ratio_sd  : longblob
    choice_ratio_mean  : longblob
    choice_ratio_sd  : longblob
    trial_num : longblob
    local_filter : longblob
    """     
# =============================================================================
#     sigmoid_fit_mu : double
#     sigmoid_fit_sigma : double 
#     sigmoid_fit_exp_temperature : double
#     sigmoid_fit_exp_bias : double 
#     linear_fit_slope : double
#     linear_fit_c : double
# =============================================================================    
    def make(self,key):  
        #%
        #key = {'subject_id': 452273}
        minsession = 8
        reward_ratio_binnum = 10
        df_psychcurve = pd.DataFrame(SessionPsychometricDataBoxCar()*SessionTrainingType()&key & 'session > '+str(minsession-1)&'session_task_protocol = 100')
        if len(df_psychcurve )>0:
            print(key)
            #%
            vector_middle = [1,0]
            vector_right = [-1*np.cos(np.pi/3),np.sin(np.pi/3)]
            vector_left = [-1*np.cos(np.pi/3),-1*np.sin(np.pi/3)]
            reward_ratio_combined_right = np.concatenate(df_psychcurve['local_fractional_income_right'].values)
            reward_ratio_combined_left = np.concatenate(df_psychcurve['local_fractional_income_left'].values)
            reward_ratio_combined_middle = np.concatenate(df_psychcurve['local_fractional_income_middle'].values)
            reward_ratio_x_coords = reward_ratio_combined_right*vector_right[0] + reward_ratio_combined_middle*vector_middle[0] +reward_ratio_combined_left*vector_left[0]
            reward_ratio_y_coords = reward_ratio_combined_right*vector_right[1] + reward_ratio_combined_middle*vector_middle[1] +reward_ratio_combined_left*vector_left[1]
            reward_ratio_arc = np.arctan2(reward_ratio_y_coords,reward_ratio_x_coords)
            arc_left_right_middle = np.asarray(np.arctan2([vector_left[1],vector_right[1],vector_middle[1]],[vector_left[0],vector_right[0],vector_middle[0]]))
            choice_num = np.asarray(np.concatenate(df_psychcurve['choice_local_fractional_income'].values),int)
            choice_arc = arc_left_right_middle[choice_num]

            
            reward_ratio_mean, reward_ratio_sd, choice_ratio_mean, choice_ratio_sd, n = bin_psychometric_curve(reward_ratio_arc,choice_arc,reward_ratio_binnum,circular = True)
            #%
            key['reward_ratio_mean'] = reward_ratio_mean
            key['reward_ratio_sd'] = reward_ratio_sd
            key['choice_ratio_mean'] = choice_ratio_mean
            key['choice_ratio_sd'] = choice_ratio_sd
# =============================================================================
#             key['sigmoid_fit_mu'] = None
#             key['sigmoid_fit_sigma'] = None
#             key['sigmoid_fit_exp_temperature'] = None
#             key['sigmoid_fit_exp_bias'] = None
#             key['linear_fit_slope'] = None
#             key['linear_fit_c'] = None
# =============================================================================
            key['trial_num'] = n
            key['local_filter'] = df_psychcurve['local_filter'][0]
            #%
            self.insert1(key,skip_duplicates=True)

@schema    
class SubjectPolarPsyCurveBoxCarFractional3lp(dj.Computed):
    definition = """
    -> lab.Subject
    ---
    reward_ratio_mean  : longblob
    reward_ratio_sd  : longblob
    choice_ratio_mean  : longblob
    choice_ratio_sd  : longblob
    trial_num : longblob
    local_filter : longblob
    """     
# =============================================================================
#     sigmoid_fit_mu : double
#     sigmoid_fit_sigma : double 
#     sigmoid_fit_exp_temperature : double
#     sigmoid_fit_exp_bias : double 
#     linear_fit_slope : double
#     linear_fit_c : double
# =============================================================================    
    def make(self,key):  
        #%
        #key = {'subject_id': 452273}
        minsession = 8
        reward_ratio_binnum = 10
        df_psychcurve = pd.DataFrame(SessionPsychometricDataBoxCar()*SessionTrainingType()&key & 'session > '+str(minsession-1)&'session_task_protocol = 101')
        if len(df_psychcurve )>0:
            print(key)
            #%
            vector_middle = [1,0]
            vector_right = [-1*np.cos(np.pi/3),np.sin(np.pi/3)]
            vector_left = [-1*np.cos(np.pi/3),-1*np.sin(np.pi/3)]
            reward_ratio_combined_right = np.concatenate(df_psychcurve['local_fractional_income_right'].values)
            reward_ratio_combined_left = np.concatenate(df_psychcurve['local_fractional_income_left'].values)
            reward_ratio_combined_middle = np.concatenate(df_psychcurve['local_fractional_income_middle'].values)
            reward_ratio_x_coords = reward_ratio_combined_right*vector_right[0] + reward_ratio_combined_middle*vector_middle[0] +reward_ratio_combined_left*vector_left[0]
            reward_ratio_y_coords = reward_ratio_combined_right*vector_right[1] + reward_ratio_combined_middle*vector_middle[1] +reward_ratio_combined_left*vector_left[1]
            reward_ratio_arc = np.arctan2(reward_ratio_y_coords,reward_ratio_x_coords)
            arc_left_right_middle = np.asarray(np.arctan2([vector_left[1],vector_right[1],vector_middle[1]],[vector_left[0],vector_right[0],vector_middle[0]]))
            choice_num = np.asarray(np.concatenate(df_psychcurve['choice_local_fractional_income'].values),int)
            choice_arc = arc_left_right_middle[choice_num]

            
            reward_ratio_mean, reward_ratio_sd, choice_ratio_mean, choice_ratio_sd, n = bin_psychometric_curve(reward_ratio_arc,choice_arc,reward_ratio_binnum,circular = True)
            #%
            key['reward_ratio_mean'] = reward_ratio_mean
            key['reward_ratio_sd'] = reward_ratio_sd
            key['choice_ratio_mean'] = choice_ratio_mean
            key['choice_ratio_sd'] = choice_ratio_sd
# =============================================================================
#             key['sigmoid_fit_mu'] = None
#             key['sigmoid_fit_sigma'] = None
#             key['sigmoid_fit_exp_temperature'] = None
#             key['sigmoid_fit_exp_bias'] = None
#             key['linear_fit_slope'] = None
#             key['linear_fit_c'] = None
# =============================================================================
            key['trial_num'] = n
            key['local_filter'] = df_psychcurve['local_filter'][0]
            #%
            self.insert1(key,skip_duplicates=True)

@schema    
class SubjectPsychometricCurveBoxCarDifferential(dj.Computed):
    definition = """
    -> lab.Subject
    ---
    reward_ratio_mean  : longblob
    reward_ratio_sd  : longblob
    choice_ratio_mean  : longblob
    choice_ratio_sd  : longblob
    sigmoid_fit_mu : double
    sigmoid_fit_sigma : double 
    sigmoid_fit_exp_temperature : double
    sigmoid_fit_exp_bias : double 
    linear_fit_slope : double
    linear_fit_c : double
    trial_num : longblob
    local_filter : longblob
    """     
    def make(self,key): 
        #%
        minsession = 8
        reward_ratio_binnum = 10
        df_psychcurve = pd.DataFrame(SessionPsychometricDataBoxCar()&key & 'session > '+str(minsession-1))
        if len(df_psychcurve )>0:
            
            reward_ratio_combined = np.concatenate(df_psychcurve['local_differential_income_right'].values)
            choice_num = np.concatenate(df_psychcurve['choice_local_differential_income'].values)
            mu,sigma = curve_fit(norm.cdf, reward_ratio_combined, choice_num,p0=[0,1])[0]
            out = curve_fit(lambda t,a,b: 1/(1+np.exp(-a*(t)+b)),  reward_ratio_combined,  choice_num)
            sigmoid_exp_temperature = out[0][0]
            sigmoid_exp_bias = out[0][1]
            out = curve_fit(lambda t,a,b: a*t+b,  reward_ratio_combined,  choice_num)
            slope = out[0][0]
            c = out[0][1]
            reward_ratio_mean, reward_ratio_sd, choice_ratio_mean, choice_ratio_sd, n = bin_psychometric_curve(reward_ratio_combined,choice_num,reward_ratio_binnum)
            #%
            key['reward_ratio_mean'] = reward_ratio_mean
            key['reward_ratio_sd'] = reward_ratio_sd
            key['choice_ratio_mean'] = choice_ratio_mean
            key['choice_ratio_sd'] = choice_ratio_sd
            key['sigmoid_fit_mu'] = mu
            key['sigmoid_fit_sigma'] = sigma      
            key['sigmoid_fit_exp_temperature'] = sigmoid_exp_temperature
            key['sigmoid_fit_exp_bias'] = sigmoid_exp_bias 
            key['linear_fit_slope'] = slope
            key['linear_fit_c'] = c
            key['trial_num'] = n
            key['local_filter'] = df_psychcurve['local_filter'][0]
            #%
            self.insert1(key,skip_duplicates=True)
            
@schema    
class SubjectPolarPsyCurveBoxCarDifferential2lp(dj.Computed):
    definition = """
    -> lab.Subject
    ---
    reward_ratio_mean  : longblob
    reward_ratio_sd  : longblob
    choice_ratio_mean  : longblob
    choice_ratio_sd  : longblob
    trial_num : longblob
    local_filter : longblob
    """     
# =============================================================================
#     sigmoid_fit_mu : double
#     sigmoid_fit_sigma : double 
#     sigmoid_fit_exp_temperature : double
#     sigmoid_fit_exp_bias : double 
#     linear_fit_slope : double
#     linear_fit_c : double
# =============================================================================    
    def make(self,key):  
        #%
        #key = {'subject_id': 452273}
        minsession = 8
        reward_ratio_binnum = 10
        
        df_psychcurve = pd.DataFrame(SessionPsychometricDataBoxCar()*SessionTrainingType()&key & 'session > '+str(minsession-1)&'session_task_protocol = 100')
        if len(df_psychcurve )>0:
            print(key)
            #%
            vector_middle = [1,0]
            vector_right = [-1*np.cos(np.pi/3),np.sin(np.pi/3)]
            vector_left = [-1*np.cos(np.pi/3),-1*np.sin(np.pi/3)]
            reward_ratio_combined_right = np.concatenate(df_psychcurve['local_differential_income_right'].values)
            reward_ratio_combined_left = np.concatenate(df_psychcurve['local_differential_income_left'].values)
            reward_ratio_combined_middle = np.concatenate(df_psychcurve['local_differential_income_middle'].values)
            reward_ratio_x_coords = reward_ratio_combined_right*vector_right[0] + reward_ratio_combined_middle*vector_middle[0] +reward_ratio_combined_left*vector_left[0]
            reward_ratio_y_coords = reward_ratio_combined_right*vector_right[1] + reward_ratio_combined_middle*vector_middle[1] +reward_ratio_combined_left*vector_left[1]
            reward_ratio_arc = np.arctan2(reward_ratio_y_coords,reward_ratio_x_coords)
            arc_left_right_middle = np.asarray(np.arctan2([vector_left[1],vector_right[1],vector_middle[1]],[vector_left[0],vector_right[0],vector_middle[0]]))
            choice_num = np.asarray(np.concatenate(df_psychcurve['choice_local_differential_income'].values),int)
            choice_arc = arc_left_right_middle[choice_num]

            
            reward_ratio_mean, reward_ratio_sd, choice_ratio_mean, choice_ratio_sd, n = bin_psychometric_curve(reward_ratio_arc,choice_arc,reward_ratio_binnum,circular = True)
            #%
            key['reward_ratio_mean'] = reward_ratio_mean
            key['reward_ratio_sd'] = reward_ratio_sd
            key['choice_ratio_mean'] = choice_ratio_mean
            key['choice_ratio_sd'] = choice_ratio_sd
# =============================================================================
#             key['sigmoid_fit_mu'] = None
#             key['sigmoid_fit_sigma'] = None
#             key['sigmoid_fit_exp_temperature'] = None
#             key['sigmoid_fit_exp_bias'] = None
#             key['linear_fit_slope'] = None
#             key['linear_fit_c'] = None
# =============================================================================
            key['trial_num'] = n
            key['local_filter'] = df_psychcurve['local_filter'][0]
            #%
            self.insert1(key,skip_duplicates=True) 

           
@schema    
class SubjectPolarPsyCurveBoxCarDifferential3lp(dj.Computed):
    definition = """
    -> lab.Subject
    ---
    reward_ratio_mean  : longblob
    reward_ratio_sd  : longblob
    choice_ratio_mean  : longblob
    choice_ratio_sd  : longblob
    trial_num : longblob
    local_filter : longblob
    """     
# =============================================================================
#     sigmoid_fit_mu : double
#     sigmoid_fit_sigma : double 
#     sigmoid_fit_exp_temperature : double
#     sigmoid_fit_exp_bias : double 
#     linear_fit_slope : double
#     linear_fit_c : double
# =============================================================================    
    def make(self,key):  
        #%
        #key = {'subject_id': 452273}
        minsession = 8
        reward_ratio_binnum = 10
        
        df_psychcurve = pd.DataFrame(SessionPsychometricDataBoxCar()*SessionTrainingType()&key & 'session > '+str(minsession-1)&'session_task_protocol = 101')
        if len(df_psychcurve )>0:
            print(key)
            #%
            vector_middle = [1,0]
            vector_right = [-1*np.cos(np.pi/3),np.sin(np.pi/3)]
            vector_left = [-1*np.cos(np.pi/3),-1*np.sin(np.pi/3)]
            reward_ratio_combined_right = np.concatenate(df_psychcurve['local_differential_income_right'].values)
            reward_ratio_combined_left = np.concatenate(df_psychcurve['local_differential_income_left'].values)
            reward_ratio_combined_middle = np.concatenate(df_psychcurve['local_differential_income_middle'].values)
            reward_ratio_x_coords = reward_ratio_combined_right*vector_right[0] + reward_ratio_combined_middle*vector_middle[0] +reward_ratio_combined_left*vector_left[0]
            reward_ratio_y_coords = reward_ratio_combined_right*vector_right[1] + reward_ratio_combined_middle*vector_middle[1] +reward_ratio_combined_left*vector_left[1]
            reward_ratio_arc = np.arctan2(reward_ratio_y_coords,reward_ratio_x_coords)
            arc_left_right_middle = np.asarray(np.arctan2([vector_left[1],vector_right[1],vector_middle[1]],[vector_left[0],vector_right[0],vector_middle[0]]))
            choice_num = np.asarray(np.concatenate(df_psychcurve['choice_local_differential_income'].values),int)
            choice_arc = arc_left_right_middle[choice_num]

            
            reward_ratio_mean, reward_ratio_sd, choice_ratio_mean, choice_ratio_sd, n = bin_psychometric_curve(reward_ratio_arc,choice_arc,reward_ratio_binnum,circular = True)
            #%
            key['reward_ratio_mean'] = reward_ratio_mean
            key['reward_ratio_sd'] = reward_ratio_sd
            key['choice_ratio_mean'] = choice_ratio_mean
            key['choice_ratio_sd'] = choice_ratio_sd
# =============================================================================
#             key['sigmoid_fit_mu'] = None
#             key['sigmoid_fit_sigma'] = None
#             key['sigmoid_fit_exp_temperature'] = None
#             key['sigmoid_fit_exp_bias'] = None
#             key['linear_fit_slope'] = None
#             key['linear_fit_c'] = None
# =============================================================================
            key['trial_num'] = n
            key['local_filter'] = df_psychcurve['local_filter'][0]
            #%
            self.insert1(key,skip_duplicates=True)

            
            

@schema    
class SubjectPsychometricCurveFittedFractional(dj.Computed):
    definition = """
    -> lab.Subject
    ---
    reward_ratio_mean  : longblob
    reward_ratio_sd  : longblob
    choice_ratio_mean  : longblob
    choice_ratio_sd  : longblob
    sigmoid_fit_mu : double
    sigmoid_fit_sigma : double 
    sigmoid_fit_exp_temperature : double
    sigmoid_fit_exp_bias : double 
    linear_fit_slope : double
    linear_fit_c : double
    trial_num : longblob
    local_filter : longblob
    """     
    def make(self,key):  
        minsession = 8
        reward_ratio_binnum = 10
        df_psychcurve = pd.DataFrame(SessionPsychometricDataFitted()&key & 'session > '+str(minsession-1))
        if len(df_psychcurve )>0:
            #%
            reward_ratio_combined = np.concatenate(df_psychcurve['local_fractional_income_right'].values)
            choice_num = np.concatenate(df_psychcurve['choice_local_fractional_income'].values)
            mu,sigma = curve_fit(norm.cdf, reward_ratio_combined, choice_num,p0=[0,1])[0]
            out = curve_fit(lambda t,a,b: 1/(1+np.exp(-a*(t-.5)+b)),  reward_ratio_combined,  choice_num)
            sigmoid_exp_temperature = out[0][0]
            sigmoid_exp_bias = out[0][1]
            out = curve_fit(lambda t,a,b: a*t+b,  reward_ratio_combined,  choice_num)
            slope = out[0][0]
            c = out[0][1]
            reward_ratio_mean, reward_ratio_sd, choice_ratio_mean, choice_ratio_sd, n = bin_psychometric_curve(reward_ratio_combined,choice_num,reward_ratio_binnum)
            
            key['reward_ratio_mean'] = reward_ratio_mean
            key['reward_ratio_sd'] = reward_ratio_sd
            key['choice_ratio_mean'] = choice_ratio_mean
            key['choice_ratio_sd'] = choice_ratio_sd
            key['sigmoid_fit_mu'] = mu
            key['sigmoid_fit_sigma'] = sigma
            key['sigmoid_fit_exp_temperature'] = sigmoid_exp_temperature
            key['sigmoid_fit_exp_bias'] = sigmoid_exp_bias 
            key['linear_fit_slope'] = slope
            key['linear_fit_c'] = c
            key['trial_num'] = n
            key['local_filter'] = df_psychcurve['local_filter'][0]
            self.insert1(key,skip_duplicates=True)
    
@schema    
class SubjectPsychometricCurveFittedDifferential(dj.Computed):
    definition = """
    -> lab.Subject
    ---
    reward_ratio_mean  : longblob
    reward_ratio_sd  : longblob
    choice_ratio_mean  : longblob
    choice_ratio_sd  : longblob
    sigmoid_fit_mu : double
    sigmoid_fit_sigma : double
    sigmoid_fit_exp_temperature : double
    sigmoid_fit_exp_bias : double 
    linear_fit_slope : double
    linear_fit_c : double
    trial_num : longblob
    local_filter : longblob
    """     
    def make(self,key):  
        minsession = 8
        reward_ratio_binnum = 10
        df_psychcurve = pd.DataFrame(SessionPsychometricDataFitted()&key & 'session > '+str(minsession-1))
        if len(df_psychcurve )>0:
            #%
            reward_ratio_combined = np.concatenate(df_psychcurve['local_differential_income_right'].values)
            choice_num = np.concatenate(df_psychcurve['choice_local_differential_income'].values)   
            
            mu,sigma = curve_fit(norm.cdf, reward_ratio_combined, choice_num,p0=[0,1])[0]
            out = curve_fit(lambda t,a,b: 1/(1+np.exp(-a*(t)+b)),  reward_ratio_combined,  choice_num)
            sigmoid_exp_temperature = out[0][0]
            sigmoid_exp_bias = out[0][1]
            out = curve_fit(lambda t,a,b: a*t+b,  reward_ratio_combined,  choice_num)
            slope = out[0][0]
            c = out[0][1]
# =============================================================================
#             x = np.arange(-3,3,.1)
#             y = norm.cdf(x, mu1, sigma1)
#             plt.plot(x,y)
#             
# =============================================================================
            reward_ratio_mean, reward_ratio_sd, choice_ratio_mean, choice_ratio_sd, n = bin_psychometric_curve(reward_ratio_combined,choice_num,reward_ratio_binnum)
            
            key['reward_ratio_mean'] = reward_ratio_mean
            key['reward_ratio_sd'] = reward_ratio_sd
            key['choice_ratio_mean'] = choice_ratio_mean
            key['choice_ratio_sd'] = choice_ratio_sd
            key['sigmoid_fit_mu'] = mu
            key['sigmoid_fit_sigma'] = sigma
            key['sigmoid_fit_exp_temperature'] = sigmoid_exp_temperature
            key['sigmoid_fit_exp_bias'] = sigmoid_exp_bias 
            key['linear_fit_slope'] = slope
            key['linear_fit_c'] = c
            key['trial_num'] = n
            key['local_filter'] = df_psychcurve['local_filter'][0]
            self.insert1(key,skip_duplicates=True)    
            
@schema    
class SessionPerformance(dj.Computed):     # works only for left-right ... needs to be upgraded
    definition = """
    -> experiment.Session
    ---
    performance_boxcar_fractional  : double
    performance_boxcar_differential  : double
    performance_fitted_fractional  : double
    performance_fitted_differential  : double
    """       
    def make(self,key):  
        #%
# =============================================================================
#         #key = {'subject_id':453475,'session':21}
# =============================================================================
        df_choices = pd.DataFrame(SessionPsychometricDataBoxCar()&key)
        #%
        if len(df_choices)>10:
            #%
            df_psycurve_fractional = pd.DataFrame(SubjectPsychometricCurveBoxCarFractional()&key)
            df_psycurve_differential = pd.DataFrame(SubjectPsychometricCurveBoxCarDifferential()&key)
            #%
            local_income = np.asarray(df_choices['local_fractional_income_right'][0].tolist())
            choice = np.asarray(df_choices['choice_local_fractional_income'][0].tolist())
            mu = df_psycurve_fractional['sigmoid_fit_mu'][0]
            sigma = df_psycurve_fractional['sigmoid_fit_sigma'][0]
            slope = df_psycurve_fractional['linear_fit_slope'][0]
            c = df_psycurve_fractional['linear_fit_c'][0]
            parameters = dict()
            parameters = {'fit_type':'linear','slope':slope,'c':c}
            key['performance_boxcar_fractional'] =  calculate_average_likelihood(local_income,choice,parameters)
                
            local_income = np.asarray(df_choices['local_differential_income_right'][0].tolist())
            choice = np.asarray(df_choices['choice_local_differential_income'][0].tolist())
            mu = df_psycurve_differential['sigmoid_fit_mu'][0]
            sigma = df_psycurve_differential['sigmoid_fit_sigma'][0]
            slope = df_psycurve_differential['linear_fit_slope'][0]
            c = df_psycurve_differential['linear_fit_c'][0]
            parameters = {'fit_type':'sigmoid','mu':mu,'sigma':sigma}
            key['performance_boxcar_differential'] =  calculate_average_likelihood(local_income,choice,parameters)
            
            
            #%%
            df_choices = pd.DataFrame(SessionPsychometricDataFitted()&key)
            df_psycurve_fractional = pd.DataFrame(SubjectPsychometricCurveFittedFractional()&key)
            df_psycurve_differential = pd.DataFrame(SubjectPsychometricCurveFittedDifferential()&key)
            
            local_income = np.asarray(df_choices['local_fractional_income_right'][0].tolist())
            choice = np.asarray(df_choices['choice_local_fractional_income'][0].tolist())
            mu = df_psycurve_fractional['sigmoid_fit_mu'][0]
            sigma = df_psycurve_fractional['sigmoid_fit_sigma'][0]
            slope = df_psycurve_fractional['linear_fit_slope'][0]
            c = df_psycurve_fractional['linear_fit_c'][0]
            parameters = {'fit_type':'linear','slope':slope,'c':c}
            key['performance_fitted_fractional'] =  calculate_average_likelihood(local_income,choice,parameters)
            #%%
            local_income = np.asarray(df_choices['local_differential_income_right'][0].tolist())
            choice = np.asarray(df_choices['choice_local_differential_income'][0].tolist())
            mu = df_psycurve_differential['sigmoid_fit_mu'][0]
            sigma = df_psycurve_differential['sigmoid_fit_sigma'][0]
            slope = df_psycurve_differential['linear_fit_slope'][0]
            c = df_psycurve_differential['linear_fit_c'][0]
            parameters = {'fit_type':'sigmoid','mu':mu,'sigma':sigma}
            key['performance_fitted_differential'] =  calculate_average_likelihood(local_income,choice,parameters)
            if key['performance_fitted_differential']!= None and key['performance_fitted_fractional']!= None and key['performance_boxcar_fractional']!= None and key['performance_boxcar_differential']!= None :
                self.insert1(key,skip_duplicates=True)
            else:
                print('couldn''t calculate performance for the following:')
                print(key)
    
