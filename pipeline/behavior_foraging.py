import datajoint as dj
import pipeline.lab as lab
import pipeline.experiment as experiment
from pipeline.pipeline_tools import get_schema_name
schema = dj.schema(get_schema_name('behavior_foraging'),locals())
import numpy as np
import pandas as pd
import math
dj.config["enable_python_native_blobs"] = True
#%%
block_reward_ratio_increment_step = 10
block_reward_ratio_increment_window = 20
block_reward_ratio_increment_max = 200
bootstrapnum = 100

def draw_bs_pairs_linreg(x, y, size=1): 
    """Perform pairs bootstrap for linear regression."""#from serhan aya

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(shape=size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds)) # sampling the indices (1d array requirement)
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return bs_slope_reps, bs_intercept_reps

#%%
@schema
class TrialReactionTime(dj.Computed):
    definition = """
    -> experiment.BehaviorTrial
    ---
    reaction_time = null : decimal(8,4) # reaction time in seconds (first lick relative to go cue) [-1 in case of ignore trials]
    """
    def make(self, key):
        df_licks=pd.DataFrame((experiment.ActionEvent & key).fetch())
        df_gocue = pd.DataFrame((experiment.TrialEvent() & key).fetch())
        gocue_time = df_gocue['trial_event_time'][df_gocue['trial_event_type'] == 'go']
        lick_times = (df_licks['action_event_time'][df_licks['action_event_time'].values>gocue_time.values] - gocue_time.values).values
        if len(lick_times) > 0:
            key['reaction_time'] = float(min(lick_times))
        self.insert1(key,skip_duplicates=True)

@schema # TODO remove bias check?
class BlockStats(dj.Computed):
    definition = """
    -> experiment.SessionBlock
    ---
    block_trial_num : int # number of trials in block
    block_ignore_num : int # number of ignores
    block_reward_rate = null: decimal(8,4) # hits / (hits + misses)
    """
    def make(self, key):
        keytoinsert = key
        keytoinsert['block_trial_num'] = len((experiment.BehaviorTrial() & key))
        keytoinsert['block_ignore_num'] = len((experiment.BehaviorTrial() & key & 'outcome = "ignore"'))
        try:
            keytoinsert['block_reward_rate'] = len((experiment.BehaviorTrial() & key & 'outcome = "hit"')) / (len((experiment.BehaviorTrial() & key & 'outcome = "miss"')) + len((experiment.BehaviorTrial() & key & 'outcome = "hit"')))
        except:
            pass
        self.insert1(keytoinsert,skip_duplicates=True)
 
    
@schema #remove bias check trials from statistics # 03/25/20 NW added nobiascheck terms for hit, miss and ignore trial num
class SessionStats(dj.Computed):
    definition = """
    -> experiment.Session
    ---

    session_total_trial_num : int #number of trials
    session_block_num : int #number of blocks, including bias check
    session_block_num_nobiascheck : int # number of blocks, no bias check
    session_hit_num : int #number of hits
    session_hit_num_nobiascheck : int # number of hits without bias check
    session_miss_num : int #number of misses
    session_miss_num_nobiascheck: int # number of misses without bias check
    session_ignore_num : int #number of ignores
    session_ignore_num_nobiascheck : int #number of ignores without bias check
    session_ignore_trial_nums = null : longblob #trial number of ignore trials
    session_autowater_num : int #number of trials with autowaters
    session_length : decimal(10, 4) #length of the session in seconds
    session_bias_check_trial_num = null: int #number of bias check trials
    session_bias_check_trial_idx = null: longblob # index of bias check trials
    session_1st_3_ignores = null : int #trialnum where the first three ignores happened in a row
    session_1st_2_ignores = null : int #trialnum where the first three ignores happened in a row
    session_1st_ignore = null : int #trialnum where the first ignore happened  
    session_biascheck_block = null : longblob # the index of bias check blocks
    """
    def make(self, key):
#%%
        #key = {'subject_id' : 467913, 'session' : 20}
        keytoadd = key
        #print(key)
        keytoadd['session_total_trial_num'] = len(experiment.SessionTrial()&key)
        keytoadd['session_block_num'] = len(experiment.SessionBlock()&key)
        keytoadd['session_biascheck_block'] = np.array([0])
        keytoadd['session_biascheck_block'] = np.nan
        keytoadd['session_bias_check_trial_idx'] = np.array([0])
        keytoadd['session_bias_check_trial_idx'] = np.nan
        keytoadd['session_hit_num'] = len(experiment.BehaviorTrial()&key&'outcome = "hit"')
        keytoadd['session_miss_num'] = len(experiment.BehaviorTrial()&key&'outcome = "miss"')
        keytoadd['session_ignore_num'] = len(experiment.BehaviorTrial()&key&'outcome = "ignore"')
        keytoadd['session_autowater_num'] = len(experiment.TrialNote & key &'trial_note_type = "autowater"')
        if keytoadd['session_total_trial_num'] > 0:
            keytoadd['session_length'] = float(((experiment.SessionTrial() & key).fetch('trial_stop_time')).max())
        else:
            keytoadd['session_length'] = 0
        df_choices = pd.DataFrame((experiment.BehaviorTrial()*experiment.SessionBlock()) & key)
        if len(df_choices)>0:
            realtraining = (df_choices['p_reward_left']<1) & (df_choices['p_reward_right']<1) & ((df_choices['p_reward_middle']<1) | df_choices['p_reward_middle'].isnull())
            realtraining1 = np.array(realtraining)
            realtraining2 = np.multiply(realtraining1,1)
            if not realtraining.values.any():
                keytoadd['session_bias_check_trial_num'] = 0
               # print('all pretraining')
            else:
                keytoadd['session_bias_check_trial_num'] = realtraining.values.argmax()
                keytoadd['session_bias_check_trial_idx'] = np.array([x+1 for x, y in enumerate(realtraining2) if y == 0])
                #print(str(realtraining.values.argmax())+' out of '+str(keytoadd['session_trialnum']))
            if (df_choices['outcome'][keytoadd['session_bias_check_trial_num']:] == 'ignore').values.any():
                keytoadd['session_1st_ignore'] = (df_choices['outcome'][keytoadd['session_bias_check_trial_num']:] == 'ignore').values.argmax()+keytoadd['session_bias_check_trial_num']+1
                if (np.convolve([1,1],(df_choices['outcome'][keytoadd['session_bias_check_trial_num']:] == 'ignore').values)==2).any():
                    keytoadd['session_1st_2_ignores'] = (np.convolve([1,1],(df_choices['outcome'][keytoadd['session_bias_check_trial_num']:] == 'ignore').values)==2).argmax() +keytoadd['session_bias_check_trial_num']+1
                if (np.convolve([1,1,1],(df_choices['outcome'][keytoadd['session_bias_check_trial_num']:] == 'ignore').values)==3).any():
                    keytoadd['session_1st_3_ignores'] = (np.convolve([1,1,1],(df_choices['outcome'][keytoadd['session_bias_check_trial_num']:] == 'ignore').values)==3).argmax() +keytoadd['session_bias_check_trial_num']+1
            # get the hit, miss and ignore without bias check 03/25/20 NW
            keytoadd['session_hit_num_nobiascheck'] = len(df_choices['outcome'][realtraining] == 'hit')
            keytoadd['session_miss_num_nobiascheck'] = len(df_choices['outcome'][realtraining] == 'miss')
            keytoadd['session_ignore_num_nobiascheck'] = len(df_choices['outcome'][realtraining] == 'ignore')                    
            # get the block num without bias check 03/25/20 NW
            p_reward_left,p_reward_right,p_reward_middle = (experiment.SessionBlock() & key).fetch('p_reward_left','p_reward_right','p_reward_middle')
            p_reward_left = p_reward_left.astype(float)
            p_reward_right = p_reward_right.astype(float)
            p_reward_middle = p_reward_middle.astype(float)
            for i in range(keytoadd['session_block_num_nobiascheck']):
                if (p_reward_left[i]==1) or (p_reward_right[i]==1) or (p_reward_middle[i]==1):
                    keytoadd['session_block_num_nobiascheck'] = keytoadd['session_block_num_nobiascheck']-1
                    keytoadd['session_biascheck_block'] = np.append(keytoadd['session_biascheck_block'],i+1)
            try:
                if len(keytoadd['session_biascheck_block'])>1 and math.isnan(keytoadd['session_biascheck_block'][0]):
                    keytoadd['session_biascheck_block'] = np.delete(keytoadd['session_biascheck_block'],0)
            except:
                pass
        self.insert1(keytoadd,skip_duplicates=True)
  #%%

@schema # TODO do we need bias check here?
class SessionRuns(dj.Computed):
    definition = """
    # a run is a sequence of trials when the mouse chooses the same option
    -> experiment.Session
    run_num : int # number of choice block
    ---
    run_start : int # first trial #the switch itself
    run_end : int # last trial #one trial before the next choice
    run_choice : varchar(8) # left or right or middle
    run_length : int # number of trials in this run
    run_hits : int # number of hit trials
    run_misses : int # number of miss trials
    run_consecutive_misses: int # number of consecutive misses before switch
    run_ignores : int # number of ignore trials
    """
    def make(self, key):     
        #%
        #key = {'subject_id':453475,'session':10}
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
            
            
@schema
class SessionTaskProtocol(dj.Computed):
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
class BlockRewardRatioNoBiasCheck(dj.Computed): # without bias check
    definition = """
    -> experiment.SessionBlock
    ---
    block_reward_ratio = null: decimal(8,4) # miss = 0, hit = 1
    block_reward_ratio_first_tertile = null: decimal(8,4) # 
    block_reward_ratio_second_tertile = null: decimal(8,4) # 
    block_reward_ratio_third_tertile = null: decimal(8,4) # 
    block_length = null: smallint #
    block_reward_ratio_right = null: decimal(8,4) # other = 0, right = 1
    block_reward_ratio_first_tertile_right = null: decimal(8,4) # other = 0, right = 1
    block_reward_ratio_second_tertile_right = null: decimal(8,4) # other = 0, right = 1 
    block_reward_ratio_third_tertile_right = null: decimal(8,4) # other = 0, right = 1
    block_reward_ratio_left = null: decimal(8,4) # other = 0, right = 1
    block_reward_ratio_first_tertile_left = null: decimal(8,4) # other = 0, left = 1
    block_reward_ratio_second_tertile_left = null: decimal(8,4) # other = 0, left = 1 
    block_reward_ratio_third_tertile_left = null: decimal(8,4) # other = 0, left = 1
    block_reward_ratio_middle = null: decimal(8,4) # other = 0, right = 1
    block_reward_ratio_first_tertile_middle = null: decimal(8,4) # other = 0, middle = 1
    block_reward_ratio_second_tertile_middle = null: decimal(8,4) # other = 0, middle = 1 
    block_reward_ratio_third_tertile_middle = null: decimal(8,4) # other = 0, middle = 1
    block_reward_ratios_incremental_right = null: longblob
    block_reward_ratios_incremental_left = null: longblob
    block_reward_ratios_incremental_middle = null: longblob
    block_reward_ratios_incr_window = null: smallint
    block_reward_ratios_incr_step = null: smallint
    """    
    def make(self, key):
        #%%
        block_reward_ratio_window_starts = np.arange(block_reward_ratio_increment_window/2,block_reward_ratio_increment_max,block_reward_ratio_increment_step,dtype = int)-int(round(block_reward_ratio_increment_window/2))
        block_reward_ratio_window_ends = np.arange(block_reward_ratio_increment_window/2,block_reward_ratio_increment_max,block_reward_ratio_increment_step,dtype = int)+int(round(block_reward_ratio_increment_window/2))
        block_reward_ratios_incremental_r=np.ones(len(block_reward_ratio_window_ends))*np.nan
        block_reward_ratios_incremental_l=np.ones(len(block_reward_ratio_window_ends))*np.nan
        block_reward_ratios_incremental_m=np.ones(len(block_reward_ratio_window_ends))*np.nan
        #%
        #key = {'subject_id' : 453478, 'session' : 3, 'block':1}
        # To skip bias check trial 04/02/20 NW
        bias_check_block = pd.DataFrame(SessionStats() & key)
        bias_check_block = bias_check_block['session_biascheck_block']      
        
        df_behaviortrial = pd.DataFrame((experiment.BehaviorTrial() & key)) 
        if any(df_behaviortrial['block'][0] == int(y) for y in bias_check_block):  #trialnum >10:
            pass
        
        else:     
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
                
            tertilelength = int(np.floor(trialnum /3))            
            block_reward_ratio = df_behaviortrial.reward.mean()
            block_reward_ratio_first_tertile = df_behaviortrial.reward[:tertilelength].mean()
            block_reward_ratio_second_tertile = df_behaviortrial.reward[tertilelength:2*tertilelength].mean()
            block_reward_ratio_third_tertile = df_behaviortrial.reward[-tertilelength:].mean()
            
            
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
class BlockChoiceRatioNoBiasCheck(dj.Computed): # without bias check
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
        bias_check_block = pd.DataFrame(SessionStats() & key)
        bias_check_block = bias_check_block['session_biascheck_block']  
        
        # To skip bias check trial 04/02/20 NW
        if any(df_behaviortrial['block'] == int(y) for y in bias_check_block):#trialnum >15:
            pass
        else:        
            df_behaviortrial['choice_L']=0
            df_behaviortrial['choice_R']=0
            df_behaviortrial['choice_M']=0
            df_behaviortrial.loc[(df_behaviortrial['trial_choice'] == 'left'),'choice_L']=1
            df_behaviortrial.loc[(df_behaviortrial['trial_choice'] == 'right'),'choice_R']=1
            df_behaviortrial.loc[(df_behaviortrial['trial_choice'] == 'middle'),'choice_M']=1
            trialnum = len(df_behaviortrial)

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
            print('error with blockchoice ratio: '+str(key['subject_id']))
            #print(key)     

@schema
class SessionMatchBias(dj.Computed): # bias check removed, 
    definition = """
    -> experiment.Session
    ---
    match_idx_r = null: decimal(8,4) # slope of log ratio R from whole blocks
    match_idx_r_first_tertile = null: decimal(8,4) # from the first tertile blocks
    match_idx_r_second_tertile = null: decimal(8,4) # from the second tertile blocks
    match_idx_r_third_tertile = null: decimal(8,4) # from the third tertile blocks
    match_idx_l = null: decimal(8,4) # slope of log ratio L from whole blocks
    match_idx_l_first_tertile = null: decimal(8,4) # from the first tertile blocks
    match_idx_l_second_tertile = null: decimal(8,4) # from the second tertile blocks
    match_idx_l_third_tertile = null: decimal(8,4) # from the third tertile blocks 
    match_idx_m = null: decimal(8,4) # slope of log ratio M from whole blocks
    match_idx_m_first_tertile = null: decimal(8,4) # from the first tertile blocks
    match_idx_m_second_tertile = null: decimal(8,4) # from the second tertile blocks
    match_idx_m_third_tertile = null: decimal(8,4) # from the third tertile blocks
    bias_r = null: decimal(8,4) # intercept of log ratio R from whole blocks
    bias_r_first_tertile = null: decimal(8,4) # from the first tertile blocks
    bias_r_second_tertile = null: decimal(8,4) # from the second tertile blocks
    bias_r_third_tertile = null: decimal(8,4) # from the third tertile blocks
    bias_l = null: decimal(8,4) # intercept of log ratio R from whole blocks
    bias_l_first_tertile = null: decimal(8,4) # from the first tertile blocks
    bias_l_second_tertile = null: decimal(8,4) # from the second tertile blocks
    bias_l_third_tertile = null: decimal(8,4) # from the third tertile blocks
    bias_m = null: decimal(8,4) # intercept of log ratio R from whole blocks
    bias_m_first_tertile = null: decimal(8,4) # from the first tertile blocks
    bias_m_second_tertile = null: decimal(8,4) # from the second tertile blocks
    bias_m_third_tertile = null: decimal(8,4) # from the third tertile blocks    
    """
    def make(self, key):
        #key = {'subject_id' : 465022, 'session' : 30}       
       
        bias_check_block = pd.DataFrame(SessionStats() & key)
        bias_check_block = bias_check_block['session_biascheck_block']
        df_block = pd.DataFrame((experiment.SessionBlock() & key))
        df_behaviortrial = pd.DataFrame((experiment.BehaviorTrial() & key))
        
        block_R_RewardRatio  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_R_RewardRatio_first  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_R_RewardRatio_second  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_R_RewardRatio_third  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_L_RewardRatio  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_L_RewardRatio_first  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_L_RewardRatio_second  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_L_RewardRatio_third  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_M_RewardRatio  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_M_RewardRatio_first  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_M_RewardRatio_second  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_M_RewardRatio_third  = np.ones(bias_check_block['session_block_num'])*np.nan
        
        block_R_ChoiceRatio  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_R_ChoiceRatio_first  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_R_ChoiceRatio_second  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_R_ChoiceRatio_third  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_L_ChoiceRatio  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_L_ChoiceRatio_first  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_L_ChoiceRatio_second  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_L_ChoiceRatio_third  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_M_ChoiceRatio  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_M_ChoiceRatio_first  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_M_ChoiceRatio_second  = np.ones(bias_check_block['session_block_num'])*np.nan
        block_M_ChoiceRatio_third  = np.ones(bias_check_block['session_block_num'])*np.nan
        
        df_block = pd.DataFrame((experiment.SessionBlock() & key))
        for x in range(len(df_block['block'])):
            if any(x == int(y) for y in bias_check_block):
                pass
            else:
                BlockRewardRatio = pd.DataFrame(BlockRewardRatioNoBiasCheck() & key & 'block ='+str(x))
                BlockChoiceRatio = pd.DataFrame(BlockChoiceRatioNoBiasCheck() & key & 'block ='+str(x))

                if df_behaviortrial['task_protocol'][0] == 100: # if it's 2lp task, then only calculate R log ratio
                    block_R_RewardRatio[x] = np.log2(BlockRewardRatio['block_reward_ratio_right'][x]/BlockRewardRatio['block_reward_ratio_left'][x])
                    block_R_RewardRatio_first[x] = np.log2(BlockRewardRatio['block_reward_ratio_first_tertile_right'][x]/BlockRewardRatio['block_reward_ratio_first_tertile_left'][x])
                    block_R_RewardRatio_second[x] = np.log2(BlockRewardRatio['block_reward_ratio_second_tertile_right'][x]/BlockRewardRatio['block_reward_ratio_second_tertile_left'][x])
                    block_R_RewardRatio_third[x] = np.log2(BlockRewardRatio['block_reward_ratio_third_tertile_right'][x]/BlockRewardRatio['block_reward_ratio_third_tertile_left'][x])
                    
                    block_R_ChoiceRatio[x] = np.log2(BlockChoiceRatio['block_choice_ratio_right'][x]/BlockChoiceRatio['block_choice_ratio_left'][x])
                    block_R_ChoiceRatio_first[x] = np.log2(BlockChoiceRatio['block_choice_ratio_first_tertile_right'][x]/BlockChoiceRatio['block_choice_ratio_first_tertile_left'][x])
                    block_R_ChoiceRatio_second[x] = np.log2(BlockChoiceRatio['block_choice_ratio_second_tertile'][x]/BlockChoiceRatio['block_choice_ratio_second_tertile_left'][x])
                    block_R_ChoiceRatio_third[x] = np.log2(BlockChoiceRatio['block_choice_ratio_third_tertile'][x]/BlockChoiceRatio['block_choice_ratio_third_tertile_left'][x])
                    
                elif df_behaviortrial['task_protocol'][0] == 101: # if ti's 3lp task, then calculate R, L and M
                    block_R_RewardRatio[x] = np.log2(BlockRewardRatio['block_reward_ratio_right'][x]/(BlockRewardRatio['block_reward_ratio_left'][x]+BlockRewardRatio['block_reward_ratio_middle'][x]))
                    block_R_RewardRatio_first[x] = np.log2(BlockRewardRatio['block_reward_ratio_first_tertile_right'][x]/(BlockRewardRatio['block_reward_ratio_first_tertile_left'][x]+BlockRewardRatio['block_reward_ratio_first_tertile_middle'][x]))
                    block_R_RewardRatio_second[x] = np.log2(BlockRewardRatio['block_reward_ratio_second_tertile_right'][x]/(BlockRewardRatio['block_reward_ratio_second_tertile_left'][x]+BlockRewardRatio['block_reward_ratio_second_tertile_middle'][x]))
                    block_R_RewardRatio_third[x] = np.log2(BlockRewardRatio['block_reward_ratio_third_tertile_right'][x]/(BlockRewardRatio['block_reward_ratio_third_tertile_left'][x]+BlockRewardRatio['block_reward_ratio_third_tertile_middle'][x]))
        
                    block_R_ChoiceRatio[x] = np.log2(BlockChoiceRatio['block_choice_ratio_right'][x]/(BlockChoiceRatio['block_choice_ratio_left'][x]+BlockChoiceRatio['block_choice_ratio_middle'][x]))
                    block_R_ChoiceRatio_first[x] = np.log2(BlockChoiceRatio['block_choice_ratio_first_tertile_right'][x]/(BlockChoiceRatio['block_choice_ratio_first_tertile_left'][x]+BlockChoiceRatio['block_choice_ratio_first_tertile_middle'][x]))
                    block_R_ChoiceRatio_second[x] = np.log2(BlockChoiceRatio['block_choice_ratio_second_tertile_right'][x]/(BlockChoiceRatio['block_choice_ratio_second_tertile_left'][x]+BlockChoiceRatio['block_choice_ratio_second_tertile_middle'][x]))
                    block_R_ChoiceRatio_third[x] = np.log2(BlockChoiceRatio['block_choice_ratio_third_tertile_right'][x]/(BlockChoiceRatio['block_choice_ratio_third_tertile_left'][x]+BlockChoiceRatio['block_choice_ratio_third_tertile_middle'][x]))
          
                    block_L_RewardRatio[x] = np.log2(BlockRewardRatio['block_reward_ratio_left'][x]/(BlockRewardRatio['block_reward_ratio_right'][x]+BlockRewardRatio['block_reward_ratio_middle'][x]))
                    block_L_RewardRatio_first[x] = np.log2(BlockRewardRatio['block_reward_ratio_first_tertile_left'][x]/(BlockRewardRatio['block_reward_ratio_first_tertile_right'][x]+BlockRewardRatio['block_reward_ratio_first_tertile_middle'][x]))
                    block_L_RewardRatio_second[x] = np.log2(BlockRewardRatio['block_reward_ratio_second_tertile_left'][x]/(BlockRewardRatio['block_reward_ratio_second_tertile_right'][x]+BlockRewardRatio['block_reward_ratio_second_tertile_middle'][x]))
                    block_L_RewardRatio_third[x] = np.log2(BlockRewardRatio['block_reward_ratio_third_tertile_left'][x]/(BlockRewardRatio['block_reward_ratio_third_tertile_right'][x]+BlockRewardRatio['block_reward_ratio_third_tertile_middle'][x]))
        
                    block_L_ChoiceRatio[x] = np.log2(BlockChoiceRatio['block_choice_ratio_left'][x]/(BlockChoiceRatio['block_choice_ratio_right'][x]+BlockChoiceRatio['block_choice_ratio_middle'][x]))
                    block_L_ChoiceRatio_first[x] = np.log2(BlockChoiceRatio['block_choice_ratio_first_tertile_left'][x]/(BlockChoiceRatio['block_choice_ratio_first_tertile_right'][x]+BlockChoiceRatio['block_choice_ratio_first_tertile_middle'][x]))
                    block_L_ChoiceRatio_second[x] = np.log2(BlockChoiceRatio['block_choice_ratio_second_tertile_left'][x]/(BlockChoiceRatio['block_choice_ratio_second_tertile_right'][x]+BlockChoiceRatio['block_choice_ratio_second_tertile_middle'][x]))
                    block_L_ChoiceRatio_third[x] = np.log2(BlockChoiceRatio['block_choice_ratio_third_tertile_left'][x]/(BlockChoiceRatio['block_choice_ratio_third_tertile_right'][x]+BlockChoiceRatio['block_choice_ratio_third_tertile_middle'][x]))
          
                    block_M_RewardRatio[x] = np.log2(BlockRewardRatio['block_reward_ratio_middle'][x]/(BlockRewardRatio['block_reward_ratio_right'][x]+BlockRewardRatio['block_reward_ratio_left'][x]))
                    block_M_RewardRatio_first[x] = np.log2(BlockRewardRatio['block_reward_ratio_first_tertile_middle'][x]/(BlockRewardRatio['block_reward_ratio_first_tertile_right'][x]+BlockRewardRatio['block_reward_ratio_first_tertile_left'][x]))
                    block_M_RewardRatio_second[x] = np.log2(BlockRewardRatio['block_reward_ratio_second_tertile_middle'][x]/(BlockRewardRatio['block_reward_ratio_second_tertile_right'][x]+BlockRewardRatio['block_reward_ratio_second_tertile_left'][x]))
                    block_M_RewardRatio_third[x] = np.log2(BlockRewardRatio['block_reward_ratio_third_tertile_middle'][x]/(BlockRewardRatio['block_reward_ratio_third_tertile_right'][x]+BlockRewardRatio['block_reward_ratio_third_tertile_left'][x]))
        
                    block_M_ChoiceRatio[x] = np.log2(BlockChoiceRatio['block_choice_ratio_middle'][x]/(BlockChoiceRatio['block_choice_ratio_right'][x]+BlockChoiceRatio['block_choice_ratio_left'][x]))
                    block_M_ChoiceRatio_first[x] = np.log2(BlockChoiceRatio['block_choice_ratio_first_tertile_middle'][x]/(BlockChoiceRatio['block_choice_ratio_first_tertile_right'][x]+BlockChoiceRatio['block_choice_ratio_first_tertile_left'][x]))
                    block_M_ChoiceRatio_second[x] = np.log2(BlockChoiceRatio['block_choice_ratio_second_tertile_middle'][x]/(BlockChoiceRatio['block_choice_ratio_second_tertile_right'][x]+BlockChoiceRatio['block_choice_ratio_second_tertile_left'][x]))
                    block_M_ChoiceRatio_third[x] = np.log2(BlockChoiceRatio['block_choice_ratio_third_tertile_middle'][x]/(BlockChoiceRatio['block_choice_ratio_third_tertile_right'][x]+BlockChoiceRatio['block_choice_ratio_third_tertile_left'][x]))
        
        block_R_RewardRatio = block_R_RewardRatio[~np.isnan(block_R_RewardRatio)]
        block_L_RewardRatio = block_L_RewardRatio[~np.isnan(block_L_RewardRatio)]
        block_M_RewardRatio = block_M_RewardRatio[~np.isnan(block_M_RewardRatio)]
        block_R_RewardRatio_first = block_R_RewardRatio_first[~np.isnan(block_R_RewardRatio_first)]
        block_L_RewardRatio_first = block_L_RewardRatio_first[~np.isnan(block_L_RewardRatio_first)]
        block_M_RewardRatio_first = block_M_RewardRatio_first[~np.isnan(block_M_RewardRatio_first)]       
        block_R_RewardRatio_second = block_R_RewardRatio_second[~np.isnan(block_R_RewardRatio_second)]
        block_L_RewardRatio_second = block_L_RewardRatio_second[~np.isnan(block_L_RewardRatio_second)]
        block_M_RewardRatio_second = block_M_RewardRatio_second[~np.isnan(block_M_RewardRatio_second)]       
        block_R_RewardRatio_third = block_R_RewardRatio_third[~np.isnan(block_R_RewardRatio_third)]
        block_L_RewardRatio_third = block_L_RewardRatio_third[~np.isnan(block_L_RewardRatio_third)]
        block_M_RewardRatio_third = block_M_RewardRatio_third[~np.isnan(block_M_RewardRatio_third)]    
        
        block_R_ChoiceRatio = block_R_ChoiceRatio[~np.isnan(block_R_ChoiceRatio)]
        block_L_ChoiceRatio = block_L_ChoiceRatio[~np.isnan(block_L_ChoiceRatio)]
        block_M_ChoiceRatio = block_M_ChoiceRatio[~np.isnan(block_M_ChoiceRatio)]
        block_R_ChoiceRatio_first = block_R_ChoiceRatio_first[~np.isnan(block_R_ChoiceRatio_first)]
        block_L_ChoiceRatio_first = block_L_ChoiceRatio_first[~np.isnan(block_L_ChoiceRatio_first)]
        block_M_ChoiceRatio_first = block_M_ChoiceRatio_first[~np.isnan(block_M_ChoiceRatio_first)]       
        block_R_ChoiceRatio_second = block_R_ChoiceRatio_second[~np.isnan(block_R_ChoiceRatio_second)]
        block_L_ChoiceRatio_second = block_L_ChoiceRatio_second[~np.isnan(block_L_ChoiceRatio_second)]
        block_M_ChoiceRatio_second = block_M_ChoiceRatio_second[~np.isnan(block_M_ChoiceRatio_second)]       
        block_R_ChoiceRatio_third = block_R_ChoiceRatio_third[~np.isnan(block_R_ChoiceRatio_third)]
        block_L_ChoiceRatio_third = block_L_ChoiceRatio_third[~np.isnan(block_L_ChoiceRatio_third)]
        block_M_ChoiceRatio_third = block_M_ChoiceRatio_third[~np.isnan(block_M_ChoiceRatio_third)]    
        
        key['match_idx_r'], key['bias_r'] = draw_bs_pairs_linreg(block_R_RewardRatio, block_R_ChoiceRatio, size=bootstrapnum)
        key['match_idx_l'], key['bias_l'] = draw_bs_pairs_linreg(block_L_RewardRatio, block_L_ChoiceRatio, size=bootstrapnum)
        key['match_idx_m'], key['bias_m'] = draw_bs_pairs_linreg(block_M_RewardRatio, block_M_ChoiceRatio, size=bootstrapnum)
        key['match_idx_r_first_tertile'], key['bias_r_first_tertile'] = draw_bs_pairs_linreg(block_R_RewardRatio_first, block_R_ChoiceRatio_first, size=bootstrapnum)
        key['match_idx_l_first_tertile'], key['bias_l_first_tertile'] = draw_bs_pairs_linreg(block_L_RewardRatio_first, block_L_ChoiceRatio_first, size=bootstrapnum)
        key['match_idx_m_first_tertile'], key['bias_m_first_tertile'] = draw_bs_pairs_linreg(block_M_RewardRatio_first, block_M_ChoiceRatio_first, size=bootstrapnum)
        key['match_idx_r_second_tertile'], key['bias_r_second_tertile'] = draw_bs_pairs_linreg(block_R_RewardRatio_second, block_R_ChoiceRatio_second, size=bootstrapnum)
        key['match_idx_l_second_tertile'], key['bias_l_second_tertile'] = draw_bs_pairs_linreg(block_L_RewardRatio_second, block_L_ChoiceRatio_second, size=bootstrapnum)
        key['match_idx_m_second_tertile'], key['bias_m_second_tertile'] = draw_bs_pairs_linreg(block_M_RewardRatio_second, block_M_ChoiceRatio_second, size=bootstrapnum)
        key['match_idx_r_third_tertile'], key['bias_r_third_tertile'] = draw_bs_pairs_linreg(block_R_RewardRatio_third, block_R_ChoiceRatio_third, size=bootstrapnum)
        key['match_idx_l_third_tertile'], key['bias_l_third_tertile'] = draw_bs_pairs_linreg(block_L_RewardRatio_third, block_L_ChoiceRatio_third, size=bootstrapnum)
        key['match_idx_m_third_tertile'], key['bias_m_third_tertile'] = draw_bs_pairs_linreg(block_M_RewardRatio_third, block_M_ChoiceRatio_third, size=bootstrapnum)
                      
        self.insert1(key,skip_duplicates=True)     
        
        
        
@schema
class BlockEfficiency(dj.Computed): # bias check excluded
    definition = """
    -> experiment.SessionBlock
    ---
    block_effi_one_p_reward =  null: decimal(8,4) # denominator = max of the reward assigned probability (no baiting)
    block_effi_one_p_reward_first_tertile =  null: decimal(8,4) # first tertile
    block_effi_one_p_reward_second_tertile =  null: decimal(8,4) # second tertile
    block_effi_one_p_reward_third_tertile =  null: decimal(8,4) # third tertile
    block_effi_sum_p_reward =  null: decimal(8,4) # denominator = sum of the reward assigned probability (no baiting)
    block_effi_sum_p_reward_first_tertile =  null: decimal(8,4) # first tertile
    block_effi_sum_p_reward_second_tertile =  null: decimal(8,4) # second tertile
    block_effi_sum_p_reward_third_tertile =  null: decimal(8,4) # third tertile
    block_effi_one_a_reward =  null: decimal(8,4) # denominator = max of the reward assigned probability + baiting)
    block_effi_one_a_reward_first_tertile =  null: decimal(8,4) # first tertile
    block_effi_one_a_reward_second_tertile =  null: decimal(8,4) # second tertile
    block_effi_one_a_reward_third_tertile =  null: decimal(8,4) # third tertile
    block_effi_sum_a_reward =  null: decimal(8,4) # denominator = sum of the reward assigned probability + baiting)
    block_effi_sum_a_reward_first_tertile =  null: decimal(8,4) # first tertile
    block_effi_sum_a_reward_second_tertile =  null: decimal(8,4) # second tertile
    block_effi_sum_a_reward_third_tertile =  null: decimal(8,4) # third tertile
    """
    def make(self, key):
        
        # key = {'subject_id':465022,'session':30, 'block': 10}
        keytoinsert = key
        bias_check_block = pd.DataFrame(SessionStats() & keytoinsert)
        bias_check_block = bias_check_block['session_biascheck_block']  
        if any(keytoinsert['block'] == int(y) for y in bias_check_block):
            pass
        else:
            p_reward_left,p_reward_right,p_reward_middle = (experiment.SessionBlock() & keytoinsert).fetch('p_reward_left','p_reward_right','p_reward_middle')
            p_reward_left = p_reward_left.astype(float)
            p_reward_right = p_reward_right.astype(float)
            p_reward_middle = p_reward_middle.astype(float)
            max_prob_reward = np.nanmax([p_reward_left,p_reward_right,p_reward_middle]) 
            sum_prob_reward = np.nansum([p_reward_left,p_reward_right,p_reward_middle])
            
            reward_available = pd.DataFrame((experiment.BehaviorTrial()*experiment.SessionBlock()* experiment.TrialAvailableReward() & keytoinsert))
            reward_available_left = reward_available['trial_available_reward_left']
            reward_available_right = reward_available['trial_available_reward_right']
            reward_available_middle = reward_available['trial_available_reward_middle']
            
            tertilelength = int(np.floor(len(reward_available_left) /3)) 
            if reward_available_middle[0]:
                max_reward_available = reward_available[["trial_available_reward_left", "trial_available_reward_right", "trial_available_reward_middle"]].max(axis=1)
                max_reward_available_first = max_reward_available[:tertilelength]
                max_reward_available_second = max_reward_available[tertilelength:2*tertilelength]
                max_reward_available_third = max_reward_available[-tertilelength:]
                
                sum_reward_available = reward_available_left + reward_available_right + reward_available_middle
                sum_reward_available_first = sum_reward_available[:tertilelength]
                sum_reward_available_second = sum_reward_available[tertilelength:2*tertilelength]
                sum_reward_available_third = sum_reward_available[-tertilelength:]
            else:
                max_reward_available = reward_available[["trial_available_reward_left", "trial_available_reward_right"]].max(axis=1)
                max_reward_available_first = max_reward_available[:tertilelength]
                max_reward_available_second = max_reward_available[tertilelength:2*tertilelength]
                max_reward_available_third = max_reward_available[-tertilelength:]
                
                sum_reward_available = reward_available_left + reward_available_right
                sum_reward_available_first = sum_reward_available[:tertilelength]
                sum_reward_available_second = sum_reward_available[tertilelength:2*tertilelength]
                sum_reward_available_third = sum_reward_available[-tertilelength:]            
            
            BlockRewardRatio = pd.DataFrame(BlockRewardRatioNoBiasCheck & key)

            keytoinsert['block_effi_one_p_reward'] = BlockRewardRatio['block_reward_ratio']/max_prob_reward
            keytoinsert['block_effi_one_p_reward_first_tertile'] = BlockRewardRatio['block_reward_ratio_first_tertile']/max_prob_reward
            keytoinsert['block_effi_one_p_reward_second_tertile'] = BlockRewardRatio['block_reward_ratio_second_tertile']/max_prob_reward
            keytoinsert['block_effi_one_p_reward_third_tertile'] = BlockRewardRatio['block_reward_ratio_third_tertile']/max_prob_reward
            
            keytoinsert['block_effi_sum_p_reward'] = BlockRewardRatio['block_reward_ratio']/sum_prob_reward
            keytoinsert['block_effi_sum_p_reward_first_tertile'] = BlockRewardRatio['block_reward_ratio_first_tertile']/sum_prob_reward
            keytoinsert['block_effi_sum_p_reward_second_tertile'] = BlockRewardRatio['block_reward_ratio_second_tertile']/sum_prob_reward
            keytoinsert['block_effi_sum_p_reward_third_tertile'] = BlockRewardRatio['block_reward_ratio_third_tertile']/sum_prob_reward
            
            keytoinsert['block_effi_one_a_reward'] = BlockRewardRatio['block_reward_ratio']/max_reward_available.mean()
            keytoinsert['block_effi_one_a_reward_first_tertile'] = BlockRewardRatio['block_reward_ratio_first_tertile']/max_reward_available_first.mean()
            keytoinsert['block_effi_one_a_reward_second_tertile'] = BlockRewardRatio['block_reward_ratio_second_tertile']/max_reward_available_second.mean()
            keytoinsert['block_effi_one_a_reward_third_tertile'] = BlockRewardRatio['block_reward_ratio_third_tertile']/max_reward_available_third.mean()
            
            keytoinsert['block_effi_sum_a_reward'] = BlockRewardRatio['block_reward_ratio']/sum_reward_available.mean()
            keytoinsert['block_effi_sum_a_reward_first_tertile'] = BlockRewardRatio['block_reward_ratio_first_tertile']/sum_reward_available_first.mean()
            keytoinsert['block_effi_sum_a_reward_second_tertile'] = BlockRewardRatio['block_reward_ratio_second_tertile']/sum_reward_available_second.mean()
            keytoinsert['block_effi_sum_a_reward_third_tertile'] = BlockRewardRatio['block_reward_ratio_third_tertile']/sum_reward_available_third.mean()
                        
        self.insert1(keytoinsert,skip_duplicates=True)


# something about bias?
# reward rates for each block?
# choice rates for each block?