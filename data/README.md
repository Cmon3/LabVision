# Data

Datasets are **not committed** to this repo. Download them yourself and place them here.

Used (public / synthetic only):

- AGAR (bacterial colonies on plates)
- BBBC005 (synthetic cells, in- and out-of-focus)

Before using any dataset, check its license and record it here:

| Dataset | License | Cite |
|---------|---------|------|
| AGAR (higher- and lower-resolution representative sample) | CC BY-NC 2.0 (non-commercial) | Majchrowska et al. 2021, *AGAR: annotated dataset for automatic detection of colonies*, arXiv:2108.01234 |
| BBBC005 v1 (synthetic cells) | CC0 | Broad Bioimage Benchmark Collection |

Expected layout (the app's example plates and the benchmarks read these paths):

```
data/agar_demo/AGAR_representative/lower-resolution/*.jpg
data/agar_demo/AGAR_representative/higher-resolution/{bright,dark,vague}/*.jpg
data/bbbc005/BBBC005_v1_images/ , data/bbbc005/BBBC005_v1_ground_truth/
```

Do not send private, patient or unpublished lab images to any API.
