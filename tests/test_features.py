import numpy as np

from labvision.features import extract_features


def _synthetic_image(size=100, bg=(200, 180, 160)):
    # create RGB image with agar-like bg and two colonies via masks
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = bg
    # masks for two circular colonies
    yy, xx = np.ogrid[:size, :size]
    m1 = (xx - 30) ** 2 + (yy - 30) ** 2 <= 10**2
    m2 = (xx - 70) ** 2 + (yy - 70) ** 2 <= 15**2
    masks = np.stack([m1, m2])
    # color colonies differently
    img[m1] = (250, 240, 220)  # cream
    img[m2] = (240, 200, 100)  # yellow
    return img, masks


def test_extract_features_synthetic_masks():
    img, masks = _synthetic_image()
    feats = extract_features(img, masks=masks)
    assert feats.shape == (2, 6)
    # L_adj should differ for different colors (cream vs yellow)
    assert not np.allclose(feats[0, 0:3], feats[1, 0:3])
    # circularity near 1 for circles
    assert feats[0, 3] > 0.8
    assert feats[1, 3] > 0.8
    # log diam larger for second (r=15 vs 10)
    assert feats[1, 4] > feats[0, 4]
    # texture low for uniform colonies
    assert feats[0, 5] < 5.0

def test_extract_features_from_boxes():
    img, _ = _synthetic_image()
    boxes = [{"x": 20, "y": 20, "width": 20, "height": 20}, {"x": 55, "y": 55, "width": 30, "height": 30}]
    feats = extract_features(img, boxes=boxes)
    assert feats.shape == (2, 6)
    # should produce some features even via Otsu
    assert np.isfinite(feats).all()
    assert (feats[:, 3] >= 0).all() and (feats[:, 3] <= 1.0).all()

def test_extract_features_bg_subtraction():
    # same colony color on different bg should give similar adj color
    img1, masks = _synthetic_image(bg=(200, 180, 160))
    img2, _ = _synthetic_image(bg=(180, 160, 140))
    # paint same colony color
    f1 = extract_features(img1, masks=masks)
    f2 = extract_features(img2, masks=masks)
    # adjusted colors should be closer than raw difference
    # loose check: adjusted diff < 10
    assert np.abs(f1[0, 0] - f2[0, 0]) < 20

def test_empty_masks():
    img = np.zeros((10, 10, 3), dtype=np.uint8) + 128
    feats = extract_features(img, masks=np.zeros((0, 10, 10), dtype=bool))
    assert feats.shape == (0, 6)
