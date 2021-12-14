import datajoint as dj
from pipeline import pipeline_tools, lab, experiment, behavior_foraging
import foraging_populate
dj.conn()
#%% populate tables
foraging_populate.populatemytables(paralel = True, cores = 8)
