import numpy as np

def channels_to_rgb8bit(img_uint16: np.ndarray) -> np.ndarray:
    a = img_uint16.astype(np.float32)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"Expected HWC with >=3 channels, got {a.shape}")
    rgb = a[..., [1, 2, 0]]  # TRITC, CY5, DAPI
    if a.shape[-1] > 3:
        rgb = rgb + a[..., 3:4]  # FITC
    rgb = np.clip(rgb, 0, 65535.0)
    return (rgb // 256.0).astype(np.uint8)

def _scale_to_u8_percentile(x: np.ndarray, pct: float = 99.5) -> np.ndarray:
    """
    Scale a single-channel image to uint8 using a high percentile for contrast.
    """
    x = np.asarray(x)
    if x.size == 0:
        return np.zeros_like(x, dtype=np.uint8)
    xf = x.astype(np.float32, copy=False)
    hi = float(np.percentile(xf, pct))
    if not np.isfinite(hi) or hi <= 0:
        hi = float(xf.max()) if xf.max() > 0 else 1.0
    y = np.clip(xf / hi * 255.0, 0.0, 255.0).astype(np.uint8)
    return y

def _gray_to_rgb(gray_u8: np.ndarray) -> np.ndarray:
    """
    Convert HxW uint8 grayscale to HxWx3 uint8.
    """
    g = gray_u8
    if g.ndim != 2:
        g = np.squeeze(g)
    return np.repeat(g[..., None], 3, axis=2)

def _downsample_nn_hwc(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    rr = (np.linspace(0, h - 1, out_h)).astype(np.int32)
    cc = (np.linspace(0, w - 1, out_w)).astype(np.int32)
    return img[rr][:, cc, :]

def make_channel_montage(img_hwc: np.ndarray) -> np.ndarray:
    """
    Build a montage image (uint8 RGB) from a raw HxWxC image (often uint16):
      Left: composite RGB (channels_to_rgb8bit)
      Right: 2x2 grid of channels (DAPI, TRITC, FITC, CY5) grayscale

    Channel mapping assumed:
      0=DAPI, 1=TRITC, 2=CY5, 3=FITC
    """
    img = np.asarray(img_hwc)
    if img.ndim != 3 or img.shape[2] < 1:
        raise ValueError(f"Expected HxWxC image, got {img.shape}")

    h, w, c = img.shape
    comp = channels_to_rgb8bit(img)  # HxWx3 uint8

    # Extract channels safely (if missing -> blank)
    def get_ch(ch_idx: int) -> np.ndarray:
        if ch_idx < c:
            g = _scale_to_u8_percentile(img[..., ch_idx])
            return _gray_to_rgb(g)
        return np.zeros((h, w, 3), dtype=np.uint8)

    dapi = get_ch(0)
    tritc = get_ch(1)
    fitc = get_ch(3)
    cy5 = get_ch(2)

    # Build right panel as 2x2 of half-res images
    h2 = max(1, h // 2)
    w2 = max(1, w // 2)

    dapi_s = _downsample_nn(dapi, h2, w2)
    tritc_s = _downsample_nn(tritc, h2, w2)
    fitc_s = _downsample_nn(fitc, h2, w2)
    cy5_s = _downsample_nn(cy5, h2, w2)

    right = np.zeros((h, w, 3), dtype=np.uint8)
    right[0:h2, 0:w2, :] = dapi_s
    right[0:h2, w2:w, :] = tritc_s
    right[h2:h, 0:w2, :] = fitc_s
    right[h2:h, w2:w, :] = cy5_s

    # Final montage: (h, 2w, 3)
    out = np.zeros((h, 2 * w, 3), dtype=np.uint8)
    out[:, 0:w, :] = comp
    out[:, w:2 * w, :] = right
    return out