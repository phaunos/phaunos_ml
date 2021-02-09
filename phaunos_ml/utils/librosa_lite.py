"""Shameless copy of librosa 0.8 mel spectrogram to not have to install
all librosa dependencies"""
import warnings
import numpy as np
from numpy import fft
from numpy.lib.stride_tricks import as_strided


MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10


class ParameterError(Exception):
    pass


def frame(x, frame_length, hop_length, axis=-1):
    """Slice a data array into (overlapping) frames.
    This implementation uses low-level stride manipulation to avoid
    making a copy of the data.  The resulting frame representation
    is a new view of the same input data.
    However, if the input data is not contiguous in memory, a warning
    will be issued and the output will be a full copy, rather than
    a view of the input data.
    For example, a one-dimensional input ``x = [0, 1, 2, 3, 4, 5, 6]``
    can be framed with frame length 3 and hop length 2 in two ways.
    The first (``axis=-1``), results in the array ``x_frames``::
        [[0, 2, 4],
         [1, 3, 5],
         [2, 4, 6]]
    where each column ``x_frames[:, i]`` contains a contiguous slice of
    the input ``x[i * hop_length : i * hop_length + frame_length]``.
    The second way (``axis=0``) results in the array ``x_frames``::
        [[0, 1, 2],
         [2, 3, 4],
         [4, 5, 6]]
    where each row ``x_frames[i]`` contains a contiguous slice of the input.
    This generalizes to higher dimensional inputs, as shown in the examples below.
    In general, the framing operation increments by 1 the number of dimensions,
    adding a new "frame axis" either to the end of the array (``axis=-1``)
    or the beginning of the array (``axis=0``).
    Parameters
    ----------
    x : np.ndarray
        Array to frame
    frame_length : int > 0 [scalar]
        Length of the frame
    hop_length : int > 0 [scalar]
        Number of steps to advance between frames
    axis : 0 or -1
        The axis along which to frame.
        If ``axis=-1`` (the default), then ``x`` is framed along its last dimension.
        ``x`` must be "F-contiguous" in this case.
        If ``axis=0``, then ``x`` is framed along its first dimension.
        ``x`` must be "C-contiguous" in this case.
    Returns
    -------
    x_frames : np.ndarray [shape=(..., frame_length, N_FRAMES) or (N_FRAMES, frame_length, ...)]
        A framed view of ``x``, for example with ``axis=-1`` (framing on the last dimension)::
            x_frames[..., j] == x[..., j * hop_length : j * hop_length + frame_length]
        If ``axis=0`` (framing on the first dimension), then::
            x_frames[j] = x[j * hop_length : j * hop_length + frame_length]
    Raises
    ------
    ParameterError
        If ``x`` is not an `np.ndarray`.
        If ``x.shape[axis] < frame_length``, there is not enough data to fill one frame.
        If ``hop_length < 1``, frames cannot advance.
        If ``axis`` is not 0 or -1.  Framing is only supported along the first or last axis.
    See Also
    --------
    numpy.asfortranarray : Convert data to F-contiguous representation
    numpy.ascontiguousarray : Convert data to C-contiguous representation
    numpy.ndarray.flags : information about the memory layout of a numpy `ndarray`.
    """

    if not isinstance(x, np.ndarray):
        raise ParameterError(
            "Input must be of type numpy.ndarray, " "given type(x)={}".format(type(x))
        )

    if x.shape[axis] < frame_length:
        raise ParameterError(
            "Input is too short (n={:d})"
            " for frame_length={:d}".format(x.shape[axis], frame_length)
        )

    if hop_length < 1:
        raise ParameterError("Invalid hop_length: {:d}".format(hop_length))

    if axis == -1 and not x.flags["F_CONTIGUOUS"]:
        warnings.warn(
            "librosa.util.frame called with axis={} "
            "on a non-contiguous input. This will result in a copy.".format(axis)
        )
        x = np.asfortranarray(x)
    elif axis == 0 and not x.flags["C_CONTIGUOUS"]:
        warnings.warn(
            "librosa.util.frame called with axis={} "
            "on a non-contiguous input. This will result in a copy.".format(axis)
        )
        x = np.ascontiguousarray(x)

    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    strides = np.asarray(x.strides)

    new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize

    if axis == -1:
        shape = list(x.shape)[:-1] + [frame_length, n_frames]
        strides = list(strides) + [hop_length * new_stride]

    elif axis == 0:
        shape = [n_frames, frame_length] + list(x.shape)[1:]
        strides = [hop_length * new_stride] + list(strides)

    else:
        raise ParameterError("Frame axis={} must be either 0 or -1".format(axis))

    return as_strided(x, shape=shape, strides=strides)


def dtype_r2c(d, default=np.complex64):
    """Find the complex numpy dtype corresponding to a real dtype.
    This is used to maintain numerical precision and memory footprint
    when constructing complex arrays from real-valued data
    (e.g. in a Fourier transform).
    A `float32` (single-precision) type maps to `complex64`,
    while a `float64` (double-precision) maps to `complex128`.
    Parameters
    ----------
    d : np.dtype
        The real-valued dtype to convert to complex.
        If ``d`` is a complex type already, it will be returned.
    default : np.dtype, optional
        The default complex target type, if ``d`` does not match a
        known dtype
    Returns
    -------
    d_c : np.dtype
        The complex dtype
    See Also
    --------
    dtype_c2r
    numpy.dtype

    """
    mapping = {
        np.dtype(np.float32): np.complex64,
        np.dtype(np.float64): np.complex128,
        np.dtype(np.float): np.complex,
    }

    # If we're given a complex type already, return it
    dt = np.dtype(d)
    if dt.kind == "c":
        return dt

    # Otherwise, try to map the dtype.
    # If no match is found, return the default.
    return np.dtype(mapping.get(dt, default))


def pad_center(data, size, axis=-1, **kwargs):
    """Pad an array to a target length along a target axis.
    This differs from `np.pad` by centering the data prior to padding,
    analogous to `str.center`
    Parameters
    ----------
    data : np.ndarray
        Vector to be padded and centered
    size : int >= len(data) [scalar]
        Length to pad ``data``
    axis : int
        Axis along which to pad and center the data
    kwargs : additional keyword arguments
      arguments passed to `np.pad`
    Returns
    -------
    data_padded : np.ndarray
        ``data`` centered and padded to length ``size`` along the
        specified axis
    Raises
    ------
    ParameterError
        If ``size < data.shape[axis]``
    """

    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ParameterError(
            ("Target size ({:d}) must be " "at least input size ({:d})").format(size, n)
        )

    return np.pad(data, lengths, **kwargs)


def mel_to_hz(mels, htk=False):
    """Convert mel bin numbers to frequencies
    Parameters
    ----------
    mels          : np.ndarray [shape=(n,)], float
        mel bins to convert
    htk           : bool
        use HTK formula instead of Slaney
    Returns
    -------
    frequencies   : np.ndarray [shape=(n,)]
        input mels in Hz
    See Also
    --------
    hz_to_mel
    """

    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels
    Parameters
    ----------
    frequencies   : number or np.ndarray [shape=(n,)] , float
        scalar or array of frequencies
    htk           : bool
        use HTK formula instead of Slaney
    Returns
    -------
    mels        : number or np.ndarray [shape=(n,)]
        input frequencies in Mels
    See Also
    --------
    mel_to_hz
    """

    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    """Compute an array of acoustic frequencies tuned to the mel scale.
    Parameters
    ----------
    n_mels    : int > 0 [scalar]
        Number of mel bins.
    fmin      : float >= 0 [scalar]
        Minimum frequency (Hz).
    fmax      : float >= 0 [scalar]
        Maximum frequency (Hz).
    htk       : bool
        If True, use HTK formula to convert Hz to mel.
        Otherwise (False), use Slaney's Auditory Toolbox.
    Returns
    -------
    bin_frequencies : ndarray [shape=(n_mels,)]
        Vector of ``n_mels`` frequencies in Hz which are uniformly spaced on the Mel
        axis.
    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)


def fft_frequencies(sr=22050, n_fft=2048):
    """Alternative implementation of `np.fft.fftfreq`
    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate
    n_fft : int > 0 [scalar]
        FFT window size
    Returns
    -------
    freqs : np.ndarray [shape=(1 + n_fft/2,)]
        Frequencies ``(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)``
    """

    return np.linspace(0, float(sr) / 2, int(1 + n_fft // 2), endpoint=True)


def mel(
    sr,
    n_fft,
    n_mels=128,
    fmin=0.0,
    fmax=None,
    htk=False,
    norm="slaney",
    dtype=np.float32,
):
    """Create a Mel filter-bank.
    This produces a linear transformation matrix to project 
    FFT bins onto Mel-frequency bins.
    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal
    n_fft     : int > 0 [scalar]
        number of FFT components
    n_mels    : int > 0 [scalar]
        number of Mel bands to generate
    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use ``fmax = sr / 2.0``
    htk       : bool [scalar]
        use HTK formula instead of Slaney
    norm : {None, 'slaney', or number} [scalar]
        If 'slaney', divide the triangular mel weights by the width of the mel band
        (area normalization).
        If numeric, use `librosa.util.normalize` to normalize each filter by to unit l_p norm.
        See `librosa.util.normalize` for a full description of supported norm values
        (including `+-np.inf`).
        Otherwise, leave all the triangles aiming for a peak value of 1.0
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.
    Returns
    -------
    M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix
    See also
    --------
    librosa.util.normalize
    Notes
    -----
    This function caches at level 10.
    """

    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
    else:
        pass
        #weights = util.normalize(weights, norm=norm, axis=-1)

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn(
            "Empty filters detected in mel frequency basis. "
            "Some channels will produce empty responses. "
            "Try increasing your sampling rate (and fmax) or "
            "reducing n_mels."
        )

    return weights


def melspectrogram(
    y=None,
    sr=22050,
    S=None,
    n_fft=2048,
    hop_length=512,
    win_length=None,
    center=True,
    pad_mode="reflect",
    power=2.0,
    **kwargs,
):
    """Compute a mel-scaled spectrogram.
    If a spectrogram input ``S`` is provided, then it is mapped directly onto
    the mel basis by ``mel_f.dot(S)``.
    If a time-series input ``y, sr`` is provided, then its magnitude spectrogram
    ``S`` is first computed, and then mapped onto the mel scale by
    ``mel_f.dot(S**power)``.
    By default, ``power=2`` operates on a power spectrum.
    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time-series
    sr : number > 0 [scalar]
        sampling rate of ``y``
    S : np.ndarray [shape=(d, t)]
        spectrogram
    n_fft : int > 0 [scalar]
        length of the FFT window
    hop_length : int > 0 [scalar]
        number of samples between successive frames.
        See `librosa.stft`
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match ``n_fft``.
        If unspecified, defaults to ``win_length = n_fft``.
    center : boolean
        - If `True`, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If `False`, then frame ``t`` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses reflection padding.
    power : float > 0 [scalar]
        Exponent for the magnitude melspectrogram.
        e.g., 1 for energy, 2 for power, etc.
    kwargs : additional keyword arguments
        Mel filter bank parameters.
        See `librosa.filters.mel` for details.
    Returns
    -------
    S : np.ndarray [shape=(n_mels, t)]
        Mel spectrogram
    """

    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        win_length=win_length,
        center=center,
        pad_mode=pad_mode,
    )

    # Build a Mel filter
    mel_basis = mel(sr, n_fft, **kwargs)

    return np.dot(mel_basis, S)


def _spectrogram(
    y=None,
    S=None,
    n_fft=2048,
    hop_length=512,
    power=1,
    win_length=None,
    center=True,
    pad_mode="reflect",
):
    """Helper function to retrieve a magnitude spectrogram.
    This is primarily used in feature extraction functions that can operate on
    either audio time-series or spectrogram input.
    Parameters
    ----------
    y : None or np.ndarray [ndim=1]
        If provided, an audio time series
    S : None or np.ndarray
        Spectrogram input, optional
    n_fft : int > 0
        STFT window size
    hop_length : int > 0
        STFT hop length
    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by ``window``.
        The window will be of length ``win_length`` and then padded
        with zeros to match ``n_fft``.
        If unspecified, defaults to ``win_length = n_fft``.
    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If ``False``, then frame ``t`` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses reflection padding.
    Returns
    -------
    S_out : np.ndarray [dtype=np.float32]
        - If ``S`` is provided as input, then ``S_out == S``
        - Else, ``S_out = |stft(y, ...)|**power``
    n_fft : int > 0
        - If ``S`` is provided, then ``n_fft`` is inferred from ``S``
        - Else, copied from input
    """

    if S is not None:
        # Infer n_fft from spectrogram shape
        n_fft = 2 * (S.shape[0] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        S = (
            np.abs(
                stft(
                    y,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    center=center,
                    pad_mode=pad_mode,
                )
            )
            ** power
        )

    return S, n_fft


def stft(
    y,
    n_fft=2048,
    hop_length=None,
    win_length=None,
    center=True,
    dtype=None,
    pad_mode="reflect",
):
    """Short-time Fourier transform (STFT).
    The STFT represents a signal in the time-frequency domain by
    computing discrete Fourier transforms (DFT) over short overlapping
    windows.
    This function returns a complex-valued matrix D such that
    - ``np.abs(D[f, t])`` is the magnitude of frequency bin ``f``
      at frame ``t``, and
    - ``np.angle(D[f, t])`` is the phase of frequency bin ``f``
      at frame ``t``.
    The integers ``t`` and ``f`` can be converted to physical units by means
    of the utility functions `frames_to_sample` and `fft_frequencies`.
    Parameters
    ----------
    y : np.ndarray [shape=(n,)], real-valued
        input signal
    n_fft : int > 0 [scalar]
        length of the windowed signal after padding with zeros.
        The number of rows in the STFT matrix ``D`` is ``(1 + n_fft/2)``.
        The default value, ``n_fft=2048`` samples, corresponds to a physical
        duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the
        default sample rate in librosa. This value is well adapted for music
        signals. However, in speech processing, the recommended value is 512,
        corresponding to 23 milliseconds at a sample rate of 22050 Hz.
        In any case, we recommend setting ``n_fft`` to a power of two for
        optimizing the speed of the fast Fourier transform (FFT) algorithm.
    hop_length : int > 0 [scalar]
        number of audio samples between adjacent STFT columns.
        Smaller values increase the number of columns in ``D`` without
        affecting the frequency resolution of the STFT.
        If unspecified, defaults to ``win_length // 4`` (see below).
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by ``window`` of length ``win_length``
        and then padded with zeros to match ``n_fft``.
        Smaller values improve the temporal resolution of the STFT (i.e. the
        ability to discriminate impulses that are closely spaced in time)
        at the expense of frequency resolution (i.e. the ability to discriminate
        pure tones that are closely spaced in frequency). This effect is known
        as the time-frequency localization trade-off and needs to be adjusted
        according to the properties of the input signal ``y``.
        If unspecified, defaults to ``win_length = n_fft``.
    center : boolean
        If ``True``, the signal ``y`` is padded so that frame
        ``D[:, t]`` is centered at ``y[t * hop_length]``.
        If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.
        Defaults to ``True``,  which simplifies the alignment of ``D`` onto a
        time grid by means of `librosa.frames_to_samples`.
        Note, however, that ``center`` must be set to `False` when analyzing
        signals with `librosa.stream`.
        .. see also:: `librosa.stream`
    dtype : np.dtype, optional
        Complex numeric type for ``D``.  Default is inferred to match the
        precision of the input signal.
    pad_mode : string or function
        If ``center=True``, this argument is passed to `np.pad` for padding
        the edges of the signal ``y``. By default (``pad_mode="reflect"``),
        ``y`` is padded on both sides with its own reflection, mirrored around
        its first and last sample respectively.
        If ``center=False``,  this argument is ignored.
        .. see also:: `numpy.pad`
    Returns
    -------
    D : np.ndarray [shape=(1 + n_fft/2, n_frames), dtype=dtype]
        Complex-valued matrix of short-term Fourier transform
        coefficients.
    See Also
    --------
    istft : Inverse STFT
    reassigned_spectrogram : Time-frequency reassigned spectrogram
    Notes
    -----
    This function caches at level 20.
    """

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    #fft_window = get_window(window, win_length, fftbins=True)
    fft_window = np.hanning(win_length)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
#    util.valid_audio(y)

    # Pad the time series so that frames are centered
    if center:
        if n_fft > y.shape[-1]:
            warnings.warn(
                "n_fft={} is too small for input signal of length={}".format(
                    n_fft, y.shape[-1]
                )
            )

        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    elif n_fft > y.shape[-1]:
        raise ParameterError(
            "n_fft={} is too small for input signal of length={}".format(
                n_fft, y.shape[-1]
            )
        )

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    if dtype is None:
        dtype = dtype_r2c(y.dtype)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty(
        (int(1 + n_fft // 2), y_frames.shape[1]), dtype=dtype, order="F"
    )

#    fft = get_fftlib()

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t] = fft.rfft(
            fft_window * y_frames[:, bl_s:bl_t], axis=0
        )
    return stft_matrix
