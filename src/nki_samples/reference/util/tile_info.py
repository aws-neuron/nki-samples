from dataclasses import dataclass
from typing import Optional, Union
from neuronxcc.nki.compiler.backends.neuron.indexing import TileIndex
from .kernel_helpers import get_ceil_quotient


#
# Basic tiled dimension info
#
@dataclass(frozen=True)
class TiledDimInfo:
  """
  Private
  """

  @classmethod
  def _build(cls, tiled_dim_size: int, tile_size: int, subtile_info: 'TiledDimInfo' = None) -> 'TiledDimInfo':
    tile_count = get_ceil_quotient(tiled_dim_size, tile_size)
    return TiledDimInfo(tiled_dim_size, tile_size, tile_count, subtile_info)

  @classmethod
  def _get_mask(cls, tile_indices: TileIndex, *, min_inclusive: int = None, max_exclusive: int = None) -> bool:
    assert min_inclusive is not None or max_exclusive is not None, "At least one bound must be supplied."
    # Set up the mask based on the supplied bounds
    if min_inclusive is not None and max_exclusive is not None:
      return (tile_indices >= min_inclusive) & (tile_indices < max_exclusive)
    elif min_inclusive is not None:
      return tile_indices >= min_inclusive
    else:
      return tile_indices < max_exclusive

  """
  Public
  """
  # Size of the dimension being tiled
  tiled_dim_size: int
  # The size of each tile
  tile_size: int
  # The number of tiles needed to cover the dimension being tiled
  tile_count: int
  # Subtile information (if there is any)
  subtile_dim_info: 'Optional[TiledDimInfo]' = None

  # Factory methods
  # ONLY CONSTRUCT THIS USING THE FACTORY METHODS BELOW

  # Build a tiled version
  @classmethod
  def build(cls, tiled_dim_size: int, tile_size: int) -> 'TiledDimInfo':
    return cls._build(tiled_dim_size, tile_size)

  # Build a subtiled version
  @classmethod
  def build_with_subtiling(cls, tiled_dim_size: int, tile_size: int, subtile_size: int) -> 'TiledDimInfo':
    subtiled_dim_info = cls.build(tile_size, subtile_size)
    return cls._build(tiled_dim_size, tile_size, subtiled_dim_info)

  def is_subtiled(self) -> bool:
    return self.subtile_dim_info is not None

  # Calculate indices for the tile given a tile number and offset
  # NOTE: This ignores any subtiling if there is any
  def get_tile_indices(self, tile_num: Union[int, TileIndex], tile_offset: Union[int, TileIndex]) -> Union[int, TileIndex]:
    return tile_num * self.tile_size + tile_offset

  # Same idea as the above method but also factor in subtiles
  def get_subtile_indices(
    self, tile_num: Union[int, TileIndex], subtile_num: Union[int, TileIndex], subtile_offset: Union[int, TileIndex]
  ) -> Union[int, TileIndex]:
    assert self.is_subtiled(), "Error: This tiled dimension has no subtiles"
    return tile_num * self.tile_size + subtile_num * self.subtile_dim_info.tile_size + subtile_offset

  # Get a mask for staying within the bounds of a tiled dimension
  def get_tile_mask(
    self,
    tile_num: Union[int, TileIndex],
    tile_offset: Union[int, TileIndex],
    *,
    min_inclusive: int = None,
    max_exclusive: int = None,
  ) -> bool:
    # Set up the mask based on the supplied bounds
    tile_indices = self.get_tile_indices(tile_num, tile_offset)
    return self._get_mask(tile_indices, min_inclusive=min_inclusive, max_exclusive=max_exclusive)

  # Get a mask for staying within the bounds of a subtiled dimension
  def get_subtile_mask(
    self,
    tile_num: Union[int, TileIndex],
    subtile_num: Union[int, TileIndex],
    subtile_offset: Union[int, TileIndex],
    *,
    min_inclusive: int = None,
    max_exclusive: int = None,
  ) -> bool:
    # Set up the mask based on the supplied bounds
    tile_indices = self.get_subtile_indices(tile_num, subtile_num, subtile_offset)
    return self._get_mask(tile_indices, min_inclusive=min_inclusive, max_exclusive=max_exclusive)
