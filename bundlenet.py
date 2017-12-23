"""

Functions for bundlenet: a convolutional neural network 
for segmentation of human brain connectomes


"""

def read_sl(fname):
    """ 
    Reads streamlins from file.
    """
    tgram = load_trk(fname)
    sl = list(dtu.move_streamlines(tgram.streamlines, 
                                   np.eye(4), tgram.affine))
    return sl


def reduce_sl(sl, dilation_iter=5, size=100):
    """ 
    Reduces a 3D streamline to a binarized 100 x 100 image.
    
    
    """
    vol = np.zeros(t1_img.shape, dtype=bool)
    sl = np.round(sl).astype(int).T
    vol[sl[0], sl[1], sl[2]] = 1
    # emphasize it a bit:
    vol = binary_dilation(vol, iterations=dilation_iter)
    vol = resize(vol, (size, size, size))
    projected = np.concatenate([np.max(vol, dim) for dim in range(len(vol.shape))])
    projected = resize(projected, (size, size, 1))
    return projected