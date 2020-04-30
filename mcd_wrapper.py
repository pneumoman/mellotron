from mcd import dtw, metrics
# ref_data and syn_data are a numpy arrays of shape [n_frames, n_mel]

def _calculate_mcd(ref_data, syn_data):
    try:
        dist, path = dtw.dtw(ref_data, syn_data, metrics.eucCepDist)
        # the denominator is smaller and the avg mcd is larger if repeating or skipping occurs.
        # d = ref_data.shape[0] - np.abs(ref_data.shape[0] - syn_data.shape[0])
        d = ref_data.shape[0]
        return dist / d
    except IndexError:
        return -1.0
