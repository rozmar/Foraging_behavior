import datajoint as dj
from pipeline import pipeline_tools, lab, experiment, behavioranal_obsolete
import foraging_populate
dj.conn()
#%% populate tables
foraging_populate.populatemytables(paralel = True, cores = 9)