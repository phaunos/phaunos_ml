import pytest
import numpy as np

from phaunos_ml.utils.feature_utils import seq2frames


class TestFeaturesUtils:

    def test_seq2frames(self):

        # Create random array (format CHT)
        C = 3
        H = 64
        T = 5000
        seq = np.random.rand(C, H, T)

        # Create frames of lenght 82 and hop length 19 (format NCHW)
        frame_len = 82
        frame_hop_len = 17

        frames = seq2frames(seq, frame_len, frame_hop_len, center=False)
        assert frames.shape[0] == 5000 // frame_hop_len + 1
        assert np.array_equal(
            frames[10],
            seq[:,:,10*frame_hop_len:10*frame_hop_len+frame_len])
        
        frames = seq2frames(seq, frame_len, frame_hop_len, center=True)
        assert frames.shape[0] == 5000 // frame_hop_len + 1
        assert np.array_equal(
            frames[10],
            seq[:,:,10*frame_hop_len-frame_len//2:10*frame_hop_len-frame_len//2+frame_len])
