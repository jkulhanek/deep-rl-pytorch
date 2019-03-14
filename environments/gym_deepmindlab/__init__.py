LEVELS = ['lt_chasm', 'lt_hallway_slope', 'lt_horseshoe_color', 'lt_space_bounce_hard', \
'nav_maze_random_goal_01','nav_maze_random_goal_02', 'nav_maze_random_goal_03', 'nav_maze_static_01', \
'nav_maze_static_02', 'seekavoid_arena_01', 'stairway_to_melon']

def _to_pascal(text):
    return ''.join(map(lambda x: x.capitalize(), text.split('_')))
    
MAP = { _to_pascal(l):l for l in LEVELS }