from gym import register, make
from environments.gym_deepmindlab import MAP

for key, l in MAP.items():
    register(
        id='DeepmindLab%s-v0' % key ,
        entry_point='environments.gym_deepmindlab.env:DeepmindLabEnv',
        kwargs = dict(scene = l)
    )

