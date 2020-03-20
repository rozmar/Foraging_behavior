import datajoint as dj
def get_schema_name(name):
    if name == 'lab':
        return 'map_v1_'+name
    else:
        return 'group_shared_foraging-'+name
