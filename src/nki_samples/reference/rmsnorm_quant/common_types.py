from enum import Enum, IntEnum

class NormType(Enum):
  NO_NORM = 0
  RMS_NORM = 1
  LAYER_NORM = 2
  RMS_NORM_SKIP_GAMMA = 3