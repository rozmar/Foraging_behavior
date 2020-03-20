import datajoint as dj

# =============================================================================
# import numpy as np
# 
import pipeline.lab as lab#, ccf
from pipeline.pipeline_tools import get_schema_name
#from . import get_schema_name
# 
schema = dj.schema(get_schema_name('experiment'),locals())
#schema = dj.schema('rozmar_foraging_experiment',locals())
# =============================================================================
#schema = dj.schema('rozmar_tutorial', locals())

@schema
class BrainLocation(dj.Manual):
    definition = """
    brain_location_name: varchar(32)  # unique name of this brain location (could be hash of the non-primary attr)
    ---
    -> lab.BrainArea
    -> lab.Hemisphere
    -> lab.SkullReference
    """

@schema
class Session(dj.Manual):
    definition = """
    -> lab.Subject
    session : smallint 		# session number
    ---
    session_date  : date
    session_time : time
    -> lab.Person
    -> lab.Rig
    """


@schema
class Task(dj.Lookup):
    definition = """
    # Type of tasks
    task            : varchar(12)                  # task type
    ----
    task_description : varchar(4000)
    """
    contents = [
         ('audio delay', 'auditory delayed response task (2AFC)'),
         ('audio mem', 'auditory working memory task'),
         ('s1 stim', 'S1 photostimulation task (2AFC)'),
         ('foraging', 'foraging task based on Bari-Cohen 2019'),
         ('foraging 3lp','foraging task based on Bari-Cohen 2019 with variable delay period')
         ]


@schema
class TaskProtocol(dj.Lookup):
    definition = """
    # SessionType
    -> Task
    task_protocol : tinyint # task protocol
    ---
    task_protocol_description : varchar(4000)
    """
    contents = [
         ('audio delay', 1, 'high tone vs. low tone'),
         ('s1 stim', 2, 'mini-distractors'),
         ('s1 stim', 3, 'full distractors, with 2 distractors (at different times) on some of the left trials'),
         ('s1 stim', 4, 'full distractors'),
         ('s1 stim', 5, 'mini-distractors, with different levels of the mini-stim during sample period'),
         ('s1 stim', 6, 'full distractors; same as protocol 4 but with a no-chirp trial-type'),
         ('s1 stim', 7, 'mini-distractors and full distractors (only at late delay)'),
         ('s1 stim', 8, 'mini-distractors and full distractors (only at late delay), with different levels of the mini-stim and the full-stim during sample                 period'),
         ('s1 stim', 9, 'mini-distractors and full distractors (only at late delay), with different levels of the mini-stim and the full-stim during sample period'),
         ('foraging', 100, 'moving lickports, delay period, early lick punishment, sound GO cue then free choice'),
         ('foraging 3lp', 101, 'moving lickports, delay period, early lick punishment, sound GO cue then free choice from three lickports'),
         ]



@schema
class Photostim(dj.Manual):
    definition = """
    -> Session
    photo_stim :  smallint 
    ---
    -> lab.PhotostimDevice
    -> BrainLocation
    ml_location=null: float # um from ref ; right is positive; based on manipulator coordinates/reconstructed track
    ap_location=null: float # um from ref; anterior is positive; based on manipulator coordinates/reconstructed track
    dv_location=null: float # um from dura; ventral is positive; based on manipulator coordinates/reconstructed track
    ml_angle=null: float # Angle between the manipulator/reconstructed track and the Medio-Lateral axis. A tilt towards the right hemishpere is positive.
    ap_angle=null: float # Angle between the manipulator/reconstructed track and the Anterior-Posterior axis. An anterior tilt is positive.
    duration=null:  decimal(8,4)   # (s)
    waveform=null:  longblob       # normalized to maximal power. The value of the maximal power is specified for each PhotostimTrialEvent individually
    """
#not used
# =============================================================================
#     class Profile(dj.Part):
#         # NOT USED CURRENT
#         definition = """
#         -> master
#         (profile_x, profile_y, profile_z) -> ccf.CCF(ccf_x, ccf_y, ccf_z)
#         ---
#         intensity_timecourse   :  longblob  # (mW/mm^2)
#         """
# 
#     # contents = [{
#     #     'photostim_device': 'OBIS470',
#     #     'photo_stim': 0,  # TODO: correct? whatmeens?
#     #     'duration': 0.5,
#     #     # FIXME/TODO: .3s of 40hz sin + .2s rampdown @ 100kHz. int32??
#     #     'waveform': np.zeros(int((0.3+0.2)*100000), np.int32)
#     # }]
# =============================================================================

@schema
class SessionBlock(dj.Imported):
    definition = """
    -> Session
    block : smallint 		# block number
    ---
    block_uid : int  # unique across sessions/animals
    block_start_time : decimal(10, 4)  # (s) relative to session beginning
    p_reward_left = null: decimal(8, 4)  # reward probability on the left waterport
    p_reward_right = null : decimal(8, 4)  # reward probability on the right waterport
    p_reward_middle = null : decimal(8, 4)  # reward probability on the middle waterport
    """

@schema
class SessionTrial(dj.Imported):
    definition = """
    -> Session
    trial : smallint 		# trial number
    ---
    trial_uid : int  # unique across sessions
    trial_start_time : decimal(10, 4)  # (s) relative to session beginning 
    trial_stop_time : decimal(10, 4)  # (s) relative to session beginning 
    """

@schema 
class TrialNoteType(dj.Lookup):
    definition = """
    trial_note_type : varchar(20)
    """
    contents = zip(('autolearn', 'protocol #', 'bad', 'bitcode','autowater','random_seed_start'))


@schema
class TrialNote(dj.Imported):
    definition = """
    -> SessionTrial
    -> TrialNoteType
    ---
    trial_note  : varchar(255) 
    """

@schema
class TrainingType(dj.Lookup):
    definition = """
    # Mouse training
    training_type : varchar(100) # mouse training
    ---
    training_type_description : varchar(2000) # description
    """
    contents = [
         ('regular', ''),
         ('regular + distractor', 'mice were first trained on the regular S1 photostimulation task  without distractors, then the training continued in the presence of distractors'),
         ('regular or regular + distractor', 'includes both training options')
         ]


@schema
class SessionTraining(dj.Manual):
    definition = """
    -> Session
    -> TrainingType
    """


@schema
class SessionTask(dj.Manual):
    definition = """
    -> Session
    -> TaskProtocol
    """


@schema
class SessionComment(dj.Manual):
    definition = """
    -> Session
    session_comment : varchar(767)
    """

@schema
class SessionDetails(dj.Manual):
    definition = """
    -> Session
    session_weight : decimal(8,4) # weight of the mouse at the beginning of the session
    session_water_earned : decimal(8,4) # water earned by the mouse during the session
    session_water_extra : decimal(8,4) # extra water provided after the session
    """

# NOT used
# =============================================================================
# @schema
# class Period(dj.Lookup):
#     definition = """
#     period: varchar(12)
#     ---
#     period_start: float  # (s) start of this period relative to GO CUE
#     period_end: float    # (s) end of this period relative to GO CUE
#     """
# 
#     contents = [('sample', -2.4, -1.2),
#                 ('delay', -1.2, 0.0),
#                 ('response', 0.0, 1.2)]
# 
# =============================================================================

# ---- behavioral trials ----

@schema
class TrialInstruction(dj.Lookup):
    definition = """
    # Instruction to mouse 
    trial_instruction  : varchar(8) 
    """
    contents = zip(('left', 'right'))

@schema
class Choice(dj.Lookup):
    definition = """
    # Choice of the mouse (if there is no instruction)
    trial_choice  : varchar(8) 
    """
    contents = zip(('left', 'right','middle','none'))

@schema
class Outcome(dj.Lookup):
    definition = """
    outcome : varchar(32)
    """
    contents = zip(('hit', 'miss', 'ignore'))


@schema
class EarlyLick(dj.Lookup):
    definition = """
    early_lick  :  varchar(32)
    ---
    early_lick_description : varchar(4000)
    """
    contents = [
        ('early', 'early lick during sample and/or delay'),
        ('early, presample only', 'early lick in the presample period, after the onset of the scheduled wave but before the sample period'),
        ('no early', '')]

@schema
class WaterValveData(dj.Imported):
    definition = """
    -> SessionTrial
    ----
    water_valve_lateral_pos = null: int # position value of the motor
    water_valve_rostrocaudal_pos = null: int # position value of the motor
    water_valve_dorsoventral_pos = null: int # position value of the motor
    water_valve_time_left = null: decimal(5,4) # seconds of valve open time
    water_valve_time_right = null: decimal(5,4) # seconds of valve open time
    water_valve_time_middle = null: decimal(5,4) # seconds of valve open time
    """

@schema
class BehaviorTrial(dj.Imported):
    definition = """
    -> SessionTrial
    ----
    -> [nullable] SessionBlock
    -> TaskProtocol
    -> [nullable] TrialInstruction
    -> [nullable] Choice
    -> EarlyLick
    -> Outcome
    """


@schema
class TrialEventType(dj.Lookup):
    definition = """
    trial_event_type  : varchar(12)  
    """
    contents = zip(('delay', 'go', 'sample', 'presample', 'trialend'))


@schema
class TrialEvent(dj.Imported):
    definition = """
    -> BehaviorTrial 
    trial_event_id: smallint
    ---
    -> TrialEventType
    trial_event_time : decimal(8, 4)   # (s) from trial start, not session start
    duration : decimal(8,4)  #  (s)  
    """


@schema
class ActionEventType(dj.Lookup):
    definition = """
    action_event_type : varchar(32)
    ----
    action_event_description : varchar(1000)
    """
    contents =[  
       ('left lick', ''), 
       ('right lick', ''),
       ('middle lick', ''),
       ]


@schema
class ActionEvent(dj.Imported):
    definition = """
    -> BehaviorTrial
    action_event_id: smallint
    ---
    -> ActionEventType
    action_event_time : decimal(8,4)  # (s) from trial start
    """

# ---- Photostim trials ----

@schema
class PhotostimTrial(dj.Imported):
    definition = """
    -> SessionTrial
    """


@schema
class PhotostimEvent(dj.Imported):
    definition = """
    -> PhotostimTrial
    photostim_event_id: smallint
    ---
    -> Photostim
    photostim_event_time : decimal(8,3)   # (s) from trial start
    power : decimal(8,3)   # Maximal power (mW)
    """


@schema
class PassivePhotostimTrial(dj.Computed):
    definition = """
    -> SessionTrial
    """
    key_source = PhotostimTrial() - BehaviorTrial()

    def make(self, key):
        self.insert1(key)

