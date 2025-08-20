"""
Environment flag utilities for NKI kernels.
"""
import os

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}

def env_flag(name: str, default: bool = False) -> bool:
    """
    Parse an environment variable as a boolean flag.
    
    Args:
        name: Environment variable name
        default: Default value if env var is not set
        
    Returns:
        True if env var is set to a truthy value, False otherwise
    """
    value = os.getenv(name)
    if value is None:
        return default
    v = value.strip().lower()
    if v in _TRUE_VALUES:
        return True
    if v in _FALSE_VALUES:
        return False
    return default  # fallback for unexpected strings

def use_dma_transpose_default() -> bool:
    """
    Global default for enabling DMA-based transpose on TRN2.
    Opt-in, disabled by default.
    
    Returns:
        True if NKI_USE_DMA_TRANSPOSE environment variable is set to a truthy value
    """
    return env_flag("NKI_USE_DMA_TRANSPOSE", False)
