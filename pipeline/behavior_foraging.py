import datajoint as dj
import pipeline.lab as lab
import pipeline.experiment as experiment
from pipeline.pipeline_tools import get_schema_name
schema = dj.schema(get_schema_name('behavior_foraging'),locals())

import numpy as np
import pandas as pd
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
        self.insert1(keytoinsert,skip_duplicates=True)
 
    
    
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
        self.insert1(keytoadd,skip_duplicates=True)
  

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
        #%
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
            self.insert(df_key.to_records(index=False))
            
            
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


# something about bias?
# reward rates for each block?
# choice rates for each block?