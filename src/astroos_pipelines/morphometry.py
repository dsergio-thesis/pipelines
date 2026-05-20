
from typing import Tuple, Optional
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, rotate
from skimage import measure, filters, morphology
from skimage.measure import label, regionprops
from math import log10

def sigma_clip_background(image: np.ndarray, mask: Optional[np.ndarray] = None,
                          sigma: float = 3.0, iters: int = 5) -> Tuple[float, float]:
    """
    Background mean and std using iterative sigma-clipping.

    Parameters
    ----------
    image : 2D array
        Input image.
    mask : 2D boolean array, optional
        True for pixels to consider (if None, use all pixels).
    sigma : float
        Clipping threshold (in sigma).
    iters : int
        Number of sigma-clipping iterations.

    Returns
    -------
    (bkg_mean, bkg_std)
    """

    if mask is None:
        data = image.ravel()
    else:
        data = image[mask].ravel()

    # remove NaNs
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0, 0.0

    for _ in range(iters):
        m = np.mean(data)
        s = np.std(data)
        good = (data > m - sigma * s) & (data < m + sigma * s)
        if good.sum() == data.size:
            break
        data = data[good]
        if data.size == 0:
            break
    
    
    if data.size == 0:
        return 0.0, 0.0
    
    return float(np.nanmean(data)), float(np.nanstd(data))


def simple_segmentation(
        image: np.ndarray, 
        nsigma: float = 2.5,
        min_area: int = 20,
        m_label: str = None,
    ) -> np.ndarray:
    """
    Source segmentation to produce a galaxy mask.

    Steps:
    - Estimate background via sigma clipping
    - Threshold at mean + nsigma * std
    - Keep largest connected component and remove small objects

    Parameters
    ----------
    image : 2D array
        Input image.
    nsigma : float
        Detection threshold above background.
    min_area : int
        Minimum area (pixels) to keep objects.

    Returns
    -------
    mask : 2D boolean array
        True where source is detected.
    """

    # print(f"simple segmentation on {m_label}")
    # print("simple_segmentation called", image.shape)
    bkg_mean, bkg_std = sigma_clip_background(image)

    # print(f"bkg_mean, bkg_std: {(bkg_mean, bkg_std)}")

    thresh = bkg_mean + nsigma * bkg_std
    bw = image > thresh
    # bw = morphology.remove_small_objects(ar=bw, min_size=min_area, connectivity=1)
    bw = morphology.remove_small_objects(ar=bw, max_size=min_area, connectivity=1)

    # Label and pick largest component
    lbl = label(bw)
    if lbl.max() == 0:
        # fallback: use Otsu threshold
        t = filters.threshold_otsu(image)
        bw = image > t
        bw = morphology.remove_small_objects(bw, max_size=min_area)
        lbl = label(bw)
        if lbl.max() == 0:
            return np.zeros_like(image, dtype=bool)
    props = regionprops(lbl)
    # choose label of largest area
    areas = [p.area for p in props]
    largest_label = props[np.argmax(areas)].label
    mask = lbl == largest_label
    return mask


# --- Photometric helpers ---


def centroid_flux(image: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """
    Flux-weighted (first moments) centroid of pixels inside mask.

    Returns (y_centroid, x_centroid) in image coordinates (row, col).
    """
    y, x = np.nonzero(mask)
    # print(f"x: {x}")
    # print(f"image.shape: {image.shape}")
    vals = image[y, x].astype(float)
    tot = vals.sum()
    if tot == 0:
        return float(np.mean(y)), float(np.mean(x))
    yc = (y * vals).sum() / tot
    xc = (x * vals).sum() / tot
    return float(yc), float(xc)


def growth_curve_radii(image: np.ndarray, mask: np.ndarray,
                       center: Tuple[float, float],
                       fractions=(0.2, 0.5, 0.8)) -> dict:
    """
    Compute radii that contain given fractions of the total galaxy flux.
    Radii are measured in pixels from center.

    Parameters
    ----------
    image : 2D array
    mask : 2D boolean array
    center : (yc, xc)
    fractions : iterable of floats (0<f<1)

    Returns
    -------
    dict mapping fraction -> radius_in_pixels
    """
    yc, xc = center
    yy, xx = np.indices(image.shape)
    rr = np.hypot(xx - xc, yy - yc)
    pix_yx = np.nonzero(mask)
    pix_r = rr[pix_yx]
    pix_flux = image[pix_yx].astype(float)
    # sort by radius
    order = np.argsort(pix_r)
    r_sorted = pix_r[order]
    f_sorted = pix_flux[order]
    cumsum = np.cumsum(f_sorted)
    tot = cumsum[-1] if cumsum.size > 0 else 0.0
    radii = {}
    if tot <= 0:
        for f in fractions:
            radii[f] = 0.0
        return radii
    for f in fractions:
        target = f * tot
        idx = np.searchsorted(cumsum, target)
        if idx >= len(r_sorted):
            radii[f] = float(r_sorted[-1])
        else:
            radii[f] = float(r_sorted[idx])
    return radii


# --- Morphometric measurements ---


def concentration_c(image: np.ndarray, mask: np.ndarray, center: Tuple[float, float], cmin=0.0, cmax=5.0) -> float:
    """
    Concentration index C = 5 * log10(r80 / r20)
    Where rX are radii containing X% of the total flux (measured inside mask).

    Returns
    -------
    C (float)
    """
    radii = growth_curve_radii(image, mask, center, fractions=(0.2, 0.8))
    r20 = radii.get(0.2, 0.0)
    r80 = radii.get(0.8, 0.0)
    if r20 <= 0 or r80 <= 0 or r80 <= r20:
        return 0.0
    
    C_raw = 5.0 * np.log10(r80 / r20)
    C_norm = (C_raw - cmin) / (cmax - cmin)
    return np.clip(C_norm, 0.0, 1.0)



def asymmetry(image: np.ndarray, mask: np.ndarray,
              center: Tuple[float, float],
              bg_image: Optional[np.ndarray] = None) -> float:
    """
    Compute asymmetry A as normalized 180-degree rotation residual:

      A = (sum abs(I - I_rot)) / (2 * sum abs(I))

    Centered on the provided center (yc, xc). The factor 2 accounts for the rotated image
    being double-counted. This is the standard Conselice-like asymmetry calculation.
    If bg_image is provided, it will be used to estimate and subtract background asymmetry.

    Returns
    -------
    A (float)
    """
    yc, xc = center
    # crop a minimal bounding box around mask for speed
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0.0
    minr, maxr = ys.min(), ys.max()
    minc, maxc = xs.min(), xs.max()
    pad = 5
    slr = slice(max(minr - pad, 0), min(image.shape[0], maxr + pad + 1))
    slc = slice(max(minc - pad, 0), min(image.shape[1], maxc + pad + 1))
    sub = image[slr, slc]
    sub_mask = mask[slr, slc]
    # recenter coordinates inside subimage
    yc_sub = yc - slr.start
    xc_sub = xc - slc.start

    # rotate subimage by 180 deg around center with scipy.rotate (order=1 interpolation)
    # To rotate around arbitrary center, shift center to image center, rotate, shift back.
    cy, cx = np.array(sub.shape) / 2.0
    # shift to center-of-rotation
    shift_y = cy - yc_sub
    shift_x = cx - xc_sub
    shifted = ndi.shift(sub, shift=(shift_y, shift_x), order=1, mode='constant', cval=0.0)
    rotated = rotate(shifted, 180.0, reshape=False, order=1, mode='constant', cval=0.0)
    unshifted_rot = ndi.shift(rotated, shift=(-shift_y, -shift_x), order=1, mode='constant', cval=0.0)

    # Use mask to compute sums
    galaxy_flux = np.abs(sub * sub_mask).sum()
    if galaxy_flux == 0:
        return 0.0

    diff = np.abs((sub - unshifted_rot) * sub_mask).sum()
    A = diff / (2.0 * galaxy_flux)

    # background correction (optional)
    if bg_image is not None:
        # compute asymmetry on background stamp (same size)
        bg_sub = bg_image[slr, slc]
        shifted_b = ndi.shift(bg_sub, shift=(shift_y, shift_x), order=1, mode='constant', cval=0.0)
        rot_b = rotate(shifted_b, 180.0, reshape=False, order=1, mode='constant', cval=0.0)
        unrot_b = ndi.shift(rot_b, shift=(-shift_y, -shift_x), order=1, mode='constant', cval=0.0)
        bflux = np.abs(bg_sub * sub_mask).sum() + 1e-12
        bdiff = np.abs((bg_sub - unrot_b) * sub_mask).sum()
        Ab = bdiff / (2.0 * bflux)
        A = max(0.0, A - Ab)
    return float(A)


def smoothness(image: np.ndarray, mask: np.ndarray, center: Tuple[float, float],
               smoothing_scale: Optional[float] = None) -> float:
    """
    Smoothness (clumpiness) S computed as:

      S = sum abs(I - I_smooth) / sum abs(I)

    where I_smooth is the image convolved with a gaussian kernel. Following typical
    definitions, the gaussian sigma can be chosen as a fraction of the galaxy radius.
    If smoothing_scale is None, we estimate characteristic radius as r50 (half-light radius)
    and use sigma = 0.25 * r50.

    Returns
    -------
    S (float)
    """
    radii = growth_curve_radii(image, mask, center, fractions=(0.5,))
    r50 = radii.get(0.5, 0.0)
    if smoothing_scale is None:
        sigma = max(1.0, 0.25 * r50)
    else:
        sigma = float(smoothing_scale)

    # gaussian smoothing (use scipy gaussian_filter)
    sm = gaussian_filter(image, sigma=sigma, mode='reflect')
    # compute sum over mask
    num = np.abs((image - sm) * mask).sum()
    den = np.abs(image * mask).sum()
    if den == 0:
        return 0.0
    return float(num / den)


def entropy_measure(image: np.ndarray, mask: np.ndarray, nbins: int = 64) -> float:
    """
    Shannon entropy of the pixel intensity distribution inside mask.

    H = - sum p_i * log2(p_i)

    where p_i are probabilities from histogram of pixel intensities (normalized).
    """
    pix = image[mask].ravel()
    if pix.size == 0:
        return 0.0
    # shift to positive
    pmin = pix.min()
    if not np.isfinite(pmin):
        pix = pix[np.isfinite(pix)]
        if pix.size == 0:
            return 0.0
        pmin = pix.min()
    if pmin < 0:
        pix = pix - pmin
    hist, _ = np.histogram(pix, bins=nbins, range=(pix.min(), pix.max() if pix.max()>pix.min() else pix.min()+1e-6))
    prob = hist.astype(float) / max(1, hist.sum())
    prob = prob[prob > 0]
    H = -np.sum(prob * np.log2(prob))
    H_max = np.log2(nbins)
    H = H / H_max if H_max > 0 else 0.0
    return float(H)


def spirality_proxy(image: np.ndarray, mask: np.ndarray, center: Tuple[float, float]) -> float:
    """
    spirality (σ_psi) based on gradients:

    - compute image gradients (dy, dx) and gradient angle psi = atan2(dy, dx)
    - for each pixel inside mask compute radial angle theta_rad = atan2(y - yc, x - xc)
    - compute angle difference delta = wrap_angle(psi - theta_rad)
    - measure dispersion of delta (e.g., standard deviation of sin(delta) or circular std)
    - larger dispersion -> more non-radial structure (spirals, arms)

    Returns
    -------
    sigma_psi : float
        A non-negative scalar; relative measure (units: radians)
    """
    yy, xx = np.indices(image.shape)
    yc, xc = center
    # compute gradient
    dy, dx = np.gradient(image.astype(float))
    psi = np.arctan2(dy, dx)  # gradient orientation
    # radial angle from center to pixel
    theta = np.arctan2(yy - yc, xx - xc)
    # compute difference wrapped to [-pi, pi]
    delta = psi - theta
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    # consider only mask pixels
    dmask = delta[mask]
    if dmask.size == 0:
        return 0.0
    # use circular dispersion measure: sqrt( -2 ln R ), where R = resultant length / N
    sin_sum = np.sum(np.sin(dmask))
    cos_sum = np.sum(np.cos(dmask))
    R = np.hypot(sin_sum, cos_sum) / dmask.size
    # prevent domain errors
    R = np.clip(R, 1e-12, 1.0)
    circ_dispersion = np.sqrt(-2.0 * np.log(R))
    # also return mean abs dev for interpretability
    mad = np.mean(np.abs(np.sin(dmask)))  # in [0,1]
    # combine metrics to one proxy (circ_dispersion in radians, mad dimensionless)
    # scale mad (0..1) to radians by multiplying by pi/2 for intuitive scale, then average
    sigma_psi = 0.5 * circ_dispersion + 0.5 * (mad * np.pi / 2.0)
    return float(sigma_psi)


def measure_morfometry(image: np.ndarray, mask: Optional[np.ndarray] = None,
                       do_segmentation: bool = True) -> dict:
    """
    Compute a set of morphometric descriptors for a single galaxy stamp.

    Parameters
    ----------
    image : 2D float array
        Galaxy stamp. Should be background-subtracted or contain background (the function does a clipping).
    mask : 2D bool array, optional
        Binary mask selecting galaxy pixels (True = galaxy). If None and do_segmentation=True,
        a simple segmentation is performed.
    do_segmentation : bool
        Whether to attempt segmentation when mask is None.

    Returns
    -------
    results : dict
        Dictionary with keys:
          - 'mask' : boolean mask used
          - 'centroid' : (yc, xc)
          - 'C' : concentration index
          - 'A' : asymmetry
          - 'S' : smoothness
          - 'H' : entropy
          - 'sigma_psi' : spirality proxy
          - 'radii' : dict with r20, r50, r80 in pixels (as available)
    """
    image = image.squeeze()

    img = np.array(image, copy=False)
    if mask is None:
        if do_segmentation:
            mask = simple_segmentation(img)
            
        else:
            raise ValueError("mask is required if do_segmentation is False")
   
    mask = mask.squeeze()
    
    # print(f"mask.shape: {mask.shape}")
    yc, xc = centroid_flux(img, mask)
    radii = growth_curve_radii(img, mask, (yc, xc), fractions=(0.2, 0.5, 0.8))
    # compute metrics
    C = concentration_c(img, mask, (yc, xc))
    A = asymmetry(img, mask, (yc, xc))
    S = smoothness(img, mask, (yc, xc))
    H = entropy_measure(img, mask)
    sigma_psi = spirality_proxy(img, mask, (yc, xc))

    res = {
        'mask': mask,
        'centroid': (yc, xc),
        'C': C,
        'A': A,
        'S': S,
        'H': H,
        'sigma_psi': sigma_psi,
        'radii': {'r20': radii.get(0.2, 0.0),
                  'r50': radii.get(0.5, 0.0),
                  'r80': radii.get(0.8, 0.0)}
    }
    return res

def synthetic_spiral(shape=(128, 128), center=None, noise=0.02):
    """
    Example synthetic galaxy
    """

    y, x = np.indices(shape)
    if center is None:
        cy, cx = np.array(shape) / 2
    else:
        cy, cx = center
    r = np.hypot(x - cx, y - cy)
    # radial profile
    img = np.exp(-r / 15.0)
    # add spiral modulation
    theta = np.arctan2(y - cy, x - cx)
    spir = 1.0 + 0.4 * np.sin(4 * theta + r / 6.0)
    img *= spir
    img += noise * np.random.randn(*shape)
    img = np.clip(img, 0, None)
    return img

def morfomytry(image: np.ndarray) -> dict:

    import matplotlib.pyplot as plt
    from skimage import data, util

    # stamp = synthetic_spiral()
    stamp = image.squeeze()

    mask = simple_segmentation(stamp, nsigma=0, min_area=50)
    res = measure_morfometry(stamp, mask=mask, do_segmentation=True)
    # print("Results:", {k: v for k, v in res.items() if k not in ('mask',)})

    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 3, 1)
    # plt.title("Original")
    # plt.imshow(np.flipud(stamp), origin='lower')
    # plt.subplot(1, 3, 2)
    # plt.title("Mask")
    # plt.imshow(np.flipud(stamp), origin='lower')
    # plt.contour(np.flipud(res['mask']), colors='red', linewidths=3.0)
    # plt.subplot(1, 3, 3)
    # plt.title("Masked")
    # plt.imshow(np.flipud(stamp * res['mask']), origin='lower')
    # plt.colorbar()
    # plt.show()

    # mirror mask vertically
    
    output_mask = np.flipud(res['mask'])
    return res, output_mask
