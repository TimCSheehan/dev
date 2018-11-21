def apply_filter(input_sig, input_filter):
    """Apply a filter to an input timeseries (using freq domain multiplication)

    Args:
        input_sig (float): timeseries to be filtered
        input_filter (float): filter to apply to ißnput_sig

    Returns:
        filt_sig (float array): filtered signal 

    """
    
    import numpy as np


    # fft our signal
    fft_sig = np.fft.rfft(input_sig)

    # need to zero pad to make the filter the same length as the signal
    X = len(input_sig)
    Y = len(input_filter)

    # zero pad in the time domain
    if Y<X:
        input_filter = np.hstack((input_filter, np.zeros(X-Y)))

    # fft the filter
    fft_filt = np.fft.rfft(input_filter)

    # multiply in freq domain, then ifft to go back into the time domain
    return np.fft.irfft(fft_sig*fft_filt)