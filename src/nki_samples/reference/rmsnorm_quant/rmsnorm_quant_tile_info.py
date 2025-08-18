from dataclasses import dataclass

import neuronxcc.nki.language as nl
from ..util.tile_info import TiledDimInfo


#
#
# Primary tuple that holds all tile information needed for the kernel
#
@dataclass(frozen=True)
class RMSNormQuantTileInfo:
  # Tile information for the outer dimensions
  outer_dim_tile: TiledDimInfo
  # Tile information for the processed (normalized/quantized) dimension
  proc_dim_tile: TiledDimInfo

  # Factory methods
  # ONLY CONSTRUCT THIS USING THE FACTORY METHODS BELOW

  @classmethod
  def build(cls, processing_shape: tuple[int, int]) -> 'RMSNormQuantTileInfo':
    # We tile the outer dim into partition dimension sized tiles.  We don't do any subtiling
    # because that actually hurts performance.  We can keep the resources busy simply by
    # working on a tile at a time of size [pmax, PD] where PD is the processing dimension.
    outer_dim_tile = TiledDimInfo.build(processing_shape[0], nl.tile_size.pmax)
    # We tile the processing dimension when applying RMS norm.  Specifically, when
    # multiplying by the gamma because we use the PE to broadcast the gamma vector across
    # the partition dimension.  By tiling in this dimension plus using the PE, we can do some
    # computation overlap with the PE and other engines.  Since we are using the PE, the
    # tile size is limited to the max moving free dimension size.
    proc_dim_tile = TiledDimInfo.build(processing_shape[1], nl.tile_size.gemm_moving_fmax)

    return RMSNormQuantTileInfo(outer_dim_tile, proc_dim_tile)
