"""Sphinx configuration."""

import datetime
import os
import shutil

import sys
from pathlib import Path

sys.path.insert(0, str(Path('..', 'src/').resolve()))

def _insert_doc(decorated_nki_func):
    decorated_nki_func.__doc__ = decorated_nki_func.func.__doc__
    decorated_nki_func.__name = decorated_nki_func.func.__name__

import nki_samples.reference.attention as attn
_insert_doc(attn.flash_fwd)
_insert_doc(attn.flash_attn_bwd)
_insert_doc(attn.fused_self_attn_for_SD_small_head_size)

import nki_samples.reference.vision as vision
_insert_doc(vision.select_and_scatter_kernel)
_insert_doc(vision.resize_nearest_fixed_dma_kernel)

import nki_samples.reference.allocated_attention as alloc_attn
_insert_doc(alloc_attn.allocated_fused_self_attn_for_SD_small_head_size)

import nki_samples.reference.allocated_fused_linear as alloc_fl
_insert_doc(alloc_fl.allocated_fused_rms_norm_qkv)

import nki_samples.reference.rmsnorm_quant.rmsnorm_quant as rmsnorm_quant
_insert_doc(rmsnorm_quant.rmsnorm_quant_kernel)

def run_apidoc(app):
    """Generate doc stubs using sphinx-apidoc."""
    module_dir = os.path.join(app.srcdir, "../src/")
    output_dir = os.path.join(app.srcdir, "_apidoc")
    excludes = []

    # Ensure that any stale apidoc files are cleaned up first.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cmd = [
        "--separate",
        "--module-first",
        "--doc-project=API Reference",
        "-o",
        output_dir,
        module_dir,
    ]
    cmd.extend(excludes)

    try:
        from sphinx.ext import apidoc  # Sphinx >= 1.7

        apidoc.main(cmd)
    except ImportError:
        from sphinx import apidoc  # Sphinx < 1.7

        cmd.insert(0, apidoc.__file__)
        apidoc.main(cmd)


def setup(app):
    """Register our sphinx-apidoc hook."""
    app.connect("builder-inited", run_apidoc)


# Sphinx configuration below.
project = 'nki_samples'
version = '1.x'
release = 'mainline'
copyright = "{}, Amazon.com".format(datetime.datetime.now().year)

extensions = [
    "sphinx.ext.autodoc",
    'sphinx.ext.autosummary',
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary

html_theme = "sphinxdoc"

source_suffix = ".rst"
master_doc = "index"

autoclass_content = "class"
autodoc_member_order = "bysource"
default_role = "py:obj"

htmlhelp_basename = "{}doc".format(project)

napoleon_use_rtype = False
