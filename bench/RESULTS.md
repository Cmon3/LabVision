# Benchmark run_bench.py

Prompt BBBC005: `cell`, AGAR: `bacterial colony`

## BBBC005 F1 (n=200)

MAE 12.14 | mean % err 34.1%

| image | true | pred | err | %err |
|---|---:|---:|---:|---:|
| SIMCEPImages_A11_C44_F1_s22_w1.TIF | 44 | 41 | 3 | 6.8 |
| SIMCEPImages_A08_C31_F1_s04_w1.TIF | 31 | 31 | 0 | 0.0 |
| SIMCEPImages_A02_C5_F1_s24_w1.TIF | 5 | 0 | 5 | 100.0 |
| SIMCEPImages_A09_C35_F1_s15_w2.TIF | 35 | 28 | 7 | 20.0 |
| SIMCEPImages_A14_C57_F1_s11_w2.TIF | 57 | 43 | 14 | 24.6 |
| SIMCEPImages_A20_C83_F1_s15_w2.TIF | 83 | 54 | 29 | 34.9 |
| SIMCEPImages_A01_C1_F1_s14_w2.TIF | 1 | 0 | 1 | 100.0 |
| SIMCEPImages_A03_C10_F1_s19_w2.TIF | 10 | 7 | 3 | 30.0 |
| SIMCEPImages_A15_C61_F1_s12_w1.TIF | 61 | 61 | 0 | 0.0 |
| SIMCEPImages_A08_C31_F1_s24_w1.TIF | 31 | 29 | 2 | 6.5 |
| SIMCEPImages_A07_C27_F1_s22_w1.TIF | 27 | 27 | 0 | 0.0 |
| SIMCEPImages_A24_C100_F1_s22_w1.TIF | 100 | 97 | 3 | 3.0 |
| SIMCEPImages_A21_C87_F1_s12_w2.TIF | 87 | 51 | 36 | 41.4 |
| SIMCEPImages_A16_C66_F1_s14_w2.TIF | 66 | 44 | 22 | 33.3 |
| SIMCEPImages_A12_C48_F1_s04_w1.TIF | 48 | 48 | 0 | 0.0 |
| SIMCEPImages_A02_C5_F1_s24_w2.TIF | 5 | 0 | 5 | 100.0 |
| SIMCEPImages_A19_C78_F1_s05_w1.TIF | 78 | 77 | 1 | 1.3 |
| SIMCEPImages_A17_C70_F1_s21_w2.TIF | 70 | 43 | 27 | 38.6 |
| SIMCEPImages_A02_C5_F1_s10_w2.TIF | 5 | 0 | 5 | 100.0 |
| SIMCEPImages_A21_C87_F1_s06_w1.TIF | 87 | 84 | 3 | 3.4 |
| ... 180 more | | | | |

## AGAR (n=40)

MAE 46.38 | mean % err 99.4%

| image | true | pred | err | %err |
|---|---:|---:|---:|---:|
| 1399.jpg | -1 | 1 | 2 | - |
| 349.jpg | 99 | 1 | 98 | 99.0 |
| 356.jpg | 4 | 0 | 4 | 100.0 |
| 510.jpg | 69 | 0 | 69 | 100.0 |
| 518.jpg | 298 | 0 | 298 | 100.0 |
| 525.jpg | 83 | 1 | 82 | 98.8 |
| 734.jpg | 54 | 1 | 53 | 98.1 |
| 735.jpg | 74 | 1 | 73 | 98.6 |
| 736.jpg | 28 | 0 | 28 | 100.0 |
| 971.jpg | 130 | 0 | 130 | 100.0 |
| 4826.jpg | 11 | 0 | 11 | 100.0 |
| 4904.jpg | 22 | 1 | 21 | 95.5 |
| 5206.jpg | 101 | 0 | 101 | 100.0 |
| 5207.jpg | 6 | 0 | 6 | 100.0 |
| 5212.jpg | 42 | 0 | 42 | 100.0 |
| 5270.jpg | 35 | 1 | 34 | 97.1 |
| 5271.jpg | 43 | 0 | 43 | 100.0 |
| 5308.jpg | 10 | 0 | 10 | 100.0 |
| 5312.jpg | 25 | 1 | 24 | 96.0 |
| 8386.jpg | 24 | 0 | 24 | 100.0 |
| ... 20 more | | | | |

*Counts from SAM 3.1 `sam-3.1` via `segment` + `measure`; cached by image hash.*