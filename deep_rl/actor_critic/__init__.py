from .ppo import PPO  # noqa: F401
from .a2c import PAAC  # noqa: F401
from .unreal import build_unreal  # noqa: F401
from .a3c import A3C  # noqa: F401
A2C = PAAC  # noqa: F401
Unreal = build_unreal(PAAC, name='Unreal')  # noqa: F401
UnrealA3C = build_unreal(A3C)  # noqa: F401
