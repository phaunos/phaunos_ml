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
        frames = seq2frames(seq, frame_len, frame_hop_len)
        n_frames = frames.shape[0]
        n_padding = (n_frames - 1) * frame_hop_len + frame_len - seq.shape[-1]

        assert np.array_equal(
            frames[0],
            seq[:,:,:frame_len])
        assert np.array_equal(
            frames[10],
            seq[:,:,10*frame_hop_len:10*frame_hop_len+frame_len])
        assert np.array_equal(
            frames[n_frames-1,:,:,-n_padding-10:-n_padding],
            seq[:,:,-10:])
