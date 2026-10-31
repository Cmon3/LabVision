import numpy as np

from labvision.morphotypes import cluster_plate, purity_flag, standardize


def test_single_colony_one_type():
    feats = np.array([[0, 0, 0, 0.9, 3.5, 2.0]])
    labels = cluster_plate(feats, threshold=1.0)
    assert list(labels) == [0]
    assert purity_flag(1).startswith("Single")

def test_identical_features_one_type():
    feats = np.zeros((10, 6))
    feats[:, 3] = 0.9
    feats[:, 4] = 3.5
    labels = cluster_plate(feats, threshold=1.0)
    # all identical should be one cluster (std fallback)
    assert len(set(labels)) == 1

def test_two_distinct_groups():
    # two groups differing in L color by ~2 std devs
    np.random.seed(0)
    g1 = np.zeros((10, 6)) + np.array([0, 0, 0, 0.9, 3.5, 2.0])
    g1[:, 0:3] += np.random.randn(10, 3) * 0.2
    g2 = np.zeros((10, 6)) + np.array([30, 5, 20, 0.9, 3.5, 2.0])
    g2[:, 0:3] += np.random.randn(10, 3) * 0.2
    feats = np.vstack([g1, g2])
    labels = cluster_plate(feats, threshold=2.0, color_weight=1.5)
    # should split into 2
    assert len(set(labels)) == 2
    # first 10 same label, last 10 same label but different
    assert len(set(labels[:10])) == 1
    assert len(set(labels[10:])) == 1
    assert labels[0] != labels[10]

def test_small_cluster_merge():
    # 10 colonies: 9 identical, 1 outlier far but < thresh for small merge (max(3,1)=3)
    feats = np.zeros((10, 6))
    feats[:, 3] = 0.9
    feats[:, 4] = 3.5
    feats[9, 0] = 100  # far outlier
    labels = cluster_plate(feats, threshold=2.0)
    # outlier cluster size 1 (<3) should be merged -> 1 type
    assert len(set(labels)) == 1
    assert purity_flag(1).startswith("Single")
    assert "possible mixed" in purity_flag(2)

def test_standardize_fixed_scales():
    feats = np.array([[103, 0, 20, 0.5, 3.5, 11]])
    std = standardize(feats)
    # should be near zero for mean
    assert np.allclose(std[0, 0], 0, atol=0.1)
    # check not per-plate: same input always same output
    std2 = standardize(feats)
    assert np.allclose(std, std2)

def test_empty():
    feats = np.zeros((0, 6))
    labels = cluster_plate(feats)
    assert len(labels) == 0
