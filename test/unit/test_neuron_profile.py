from neuronxcc.nki import benchmark
from neuronxcc.nki import profile
import neuronxcc.nki.language as nl
import numpy as np
import pytest
import os
import shutil
import tempfile


WORKING_DIRECTORY = tempfile.mkdtemp()
SAVE_NEFF_NAME = "cus_file123.neff"
SAVE_TRACE_NAME = "profile-custom.ntff"
NUM_EXECS = 20
PROFILE_NTH = 10  
JSON_REPORTS = "json_reports"

@profile(working_directory=WORKING_DIRECTORY, save_neff_name=SAVE_NEFF_NAME, overwrite=False , save_trace_name=SAVE_TRACE_NAME, num_execs=NUM_EXECS, profile_nth=PROFILE_NTH)
def nki_tensor_tensor_add(a_tensor, b_tensor):
  c_output = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
 
  a = nl.load(a_tensor)
  b = nl.load(b_tensor)

  c_tile = a + b

  nl.store(c_output, value=c_tile)

  return c_output

class TestNeuronProfile:
    def _get_ntff_path(self, trace_val):
        """
        Prepares ntff file name based on execution trace number
        """
        if trace_val == 1:
            return os.path.join(WORKING_DIRECTORY, f"{os.path.splitext(os.path.basename(SAVE_TRACE_NAME))[0]}.ntff")
        else:
            return os.path.join(WORKING_DIRECTORY, f"{os.path.splitext(os.path.basename(SAVE_TRACE_NAME))[0]}_exec_{trace_val}.ntff")

    @pytest.fixture
    def traces(self):
        ret = []
        if NUM_EXECS < PROFILE_NTH:
            ret.append(self._get_ntff_path(PROFILE_NTH))
        else:
            curr = PROFILE_NTH
            while curr <= NUM_EXECS:
                ret.append(self._get_ntff_path(curr))
                curr += PROFILE_NTH
        return ret
    
    @pytest.fixture
    def num_reports(self):
        if NUM_EXECS < PROFILE_NTH:
            return 1
        else:
            return NUM_EXECS // PROFILE_NTH

    def test_output_artifacts_created(self, traces, num_reports):
        # delete artifact directory, only testing non-overwrite functionality
        if os.path.exists(WORKING_DIRECTORY):
            shutil.rmtree(WORKING_DIRECTORY)

        # creates dummy input to invoke profile kernel
        a = np.zeros([128, 1024]).astype(np.float16)
        b = np.random.random_sample([128, 1024]).astype(np.float16)

        output_nki = nki_tensor_tensor_add(a, b)

        # now asserting artifacts are correctly created     
        assert os.path.exists(os.path.join(WORKING_DIRECTORY, SAVE_NEFF_NAME)) # neff
        
        for trace in traces:
            assert os.path.exists(trace) # trace
        
        # json reports
        report_dir = os.path.join(WORKING_DIRECTORY, JSON_REPORTS)

        assert os.path.exists(report_dir) # actually exists
        assert len(os.listdir(report_dir)) == num_reports # report all iterations queried

        # post condition cleanup
        if os.path.exists(WORKING_DIRECTORY):
            shutil.rmtree(WORKING_DIRECTORY)

