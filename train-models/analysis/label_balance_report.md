# Phân phối aspect và sentiment của bộ dữ liệu sữa rửa mặt

Thống kê từng dòng CSV một lần, không nhân theo `duplicate_count`. Các tập: train 8.882, validation 1.904, test 1.904. Dữ liệu và code training được giữ nguyên.

Nhãn gốc: đếm mỗi cặp aspect–sentiment khác nhau một lần trong một review. Một review có thể đóng góp nhiều cặp, kể cả nhiều sentiment của cùng aspect. Tổng cặp nhãn có thể lớn hơn số review.

Target model: tái hiện cách `ABSADataset` hiện tại tạo một sentiment cho mỗi aspect, lấy sentiment xuất hiện sau cùng. `None` là lớp nội bộ khi aspect không được đề cập; không phải nhãn mới được gán vào CSV và không được tính là neutral.

## Tổng quan ba tập

| Tập | Review | positive gốc | neutral gốc | negative gốc | Câu có nhiều sentiment cùng aspect | Cặp không giữ trong target |
|---|---:|---:|---:|---:|---:|---:|
| train | 8.882 | 20.890 | 3.106 | 3.777 | 629 | 672 |
| val | 1.904 | 4.577 | 666 | 792 | 114 | 121 |
| test | 1.904 | 4.399 | 661 | 815 | 131 | 133 |

## train: nhãn gốc trong CSV

| Aspect | positive | neutral | negative | Tổng cặp | Review đề cập |
|---|---:|---:|---:|---:|---:|
| CleansingEffectiveness | 2.477 | 328 | 107 | 2.912 | 2.889 |
| HydrationSkinComfort | 2.267 | 16 | 268 | 2.551 | 2.522 |
| SkinCompatibilitySafety | 2.619 | 546 | 375 | 3.540 | 3.458 |
| TextureFoaming | 705 | 498 | 112 | 1.315 | 1.268 |
| Fragrance | 1.523 | 518 | 263 | 2.304 | 2.277 |
| IngredientsFormulation | 494 | 158 | 19 | 671 | 668 |
| Packaging | 1.964 | 110 | 904 | 2.978 | 2.875 |
| PriceValue | 1.467 | 48 | 151 | 1.666 | 1.645 |
| ShopService | 1.195 | 31 | 683 | 1.909 | 1.888 |
| Delivery | 2.039 | 21 | 218 | 2.278 | 2.259 |
| General | 4.140 | 832 | 677 | 5.649 | 5.352 |

## train: target thực tế của model

| Aspect | positive | neutral | negative | None nội bộ |
|---|---:|---:|---:|---:|
| CleansingEffectiveness | 2.459 | 326 | 104 | 5.993 |
| HydrationSkinComfort | 2.248 | 15 | 259 | 6.360 |
| SkinCompatibilitySafety | 2.561 | 538 | 359 | 5.424 |
| TextureFoaming | 682 | 478 | 108 | 7.614 |
| Fragrance | 1.506 | 512 | 259 | 6.605 |
| IngredientsFormulation | 491 | 158 | 19 | 8.214 |
| Packaging | 1.876 | 107 | 892 | 6.007 |
| PriceValue | 1.455 | 48 | 142 | 7.237 |
| ShopService | 1.175 | 31 | 682 | 6.994 |
| Delivery | 2.027 | 21 | 211 | 6.623 |
| General | 3.924 | 798 | 630 | 3.530 |
| **Tổng** | **20.404** | **3.032** | **3.665** | **70.601** |

## val: nhãn gốc trong CSV

| Aspect | positive | neutral | negative | Tổng cặp | Review đề cập |
|---|---:|---:|---:|---:|---:|
| CleansingEffectiveness | 553 | 77 | 22 | 652 | 646 |
| HydrationSkinComfort | 503 | 9 | 61 | 573 | 568 |
| SkinCompatibilitySafety | 592 | 125 | 72 | 789 | 773 |
| TextureFoaming | 163 | 109 | 25 | 297 | 287 |
| Fragrance | 366 | 95 | 52 | 513 | 509 |
| IngredientsFormulation | 101 | 50 | 8 | 159 | 155 |
| Packaging | 421 | 13 | 202 | 636 | 620 |
| PriceValue | 318 | 8 | 33 | 359 | 355 |
| ShopService | 256 | 5 | 138 | 399 | 397 |
| Delivery | 444 | 3 | 39 | 486 | 486 |
| General | 860 | 172 | 140 | 1.172 | 1.118 |

## val: target thực tế của model

| Aspect | positive | neutral | negative | None nội bộ |
|---|---:|---:|---:|---:|
| CleansingEffectiveness | 548 | 77 | 21 | 1.258 |
| HydrationSkinComfort | 500 | 9 | 59 | 1.336 |
| SkinCompatibilitySafety | 581 | 124 | 68 | 1.131 |
| TextureFoaming | 159 | 103 | 25 | 1.617 |
| Fragrance | 365 | 94 | 50 | 1.395 |
| IngredientsFormulation | 98 | 49 | 8 | 1.749 |
| Packaging | 406 | 13 | 201 | 1.284 |
| PriceValue | 315 | 8 | 32 | 1.549 |
| ShopService | 254 | 5 | 138 | 1.507 |
| Delivery | 444 | 3 | 39 | 1.418 |
| General | 827 | 164 | 127 | 786 |
| **Tổng** | **4.497** | **649** | **768** | **15.030** |

## test: nhãn gốc trong CSV

| Aspect | positive | neutral | negative | Tổng cặp | Review đề cập |
|---|---:|---:|---:|---:|---:|
| CleansingEffectiveness | 520 | 71 | 15 | 606 | 603 |
| HydrationSkinComfort | 485 | 5 | 47 | 537 | 534 |
| SkinCompatibilitySafety | 541 | 117 | 69 | 727 | 711 |
| TextureFoaming | 163 | 116 | 28 | 307 | 299 |
| Fragrance | 292 | 95 | 59 | 446 | 439 |
| IngredientsFormulation | 100 | 30 | 5 | 135 | 135 |
| Packaging | 431 | 31 | 199 | 661 | 644 |
| PriceValue | 274 | 12 | 34 | 320 | 317 |
| ShopService | 265 | 7 | 166 | 438 | 427 |
| Delivery | 446 | 3 | 45 | 494 | 490 |
| General | 882 | 174 | 148 | 1.204 | 1.143 |

## test: target thực tế của model

| Aspect | positive | neutral | negative | None nội bộ |
|---|---:|---:|---:|---:|
| CleansingEffectiveness | 519 | 71 | 13 | 1.301 |
| HydrationSkinComfort | 483 | 5 | 46 | 1.370 |
| SkinCompatibilitySafety | 530 | 115 | 66 | 1.193 |
| TextureFoaming | 161 | 111 | 27 | 1.605 |
| Fragrance | 286 | 94 | 59 | 1.465 |
| IngredientsFormulation | 100 | 30 | 5 | 1.769 |
| Packaging | 415 | 31 | 198 | 1.260 |
| PriceValue | 274 | 12 | 31 | 1.587 |
| ShopService | 254 | 7 | 166 | 1.477 |
| Delivery | 442 | 3 | 45 | 1.414 |
| General | 835 | 168 | 140 | 761 |
| **Tổng** | **4.299** | **647** | **796** | **15.202** |

## Nhận định từ số liệu hiện tại

- Train có 27.773 cặp nhãn gốc: positive 75.22%, neutral 11.18%, negative 13.60%.
- Cặp hiếm: HydrationSkinComfort#neutral có 16 mẫu train, IngredientsFormulation#negative có 19, Delivery#neutral có 21. Delivery#neutral chỉ có 3 mẫu validation và 3 mẫu test.
- Aspect cũng lệch: General xuất hiện trong 5.352 review train; IngredientsFormulation xuất hiện trong 668 review.
- Trong 97.702 vị trí target train (8.882 review × 11 aspect), None nội bộ có 70.601 vị trí, chiếm 72.26%.
- Code hiện chọn SCENARIOS = ['baseline'], dùng CrossEntropyLoss không có class weights. Nhánh class_weights và focal_loss tồn tại nhưng chưa được bật.
- Tỷ lệ sentiment tổng thể của ba tập khá gần nhau; các cặp hiếm vẫn có ít mẫu ở validation/test. Phép chia hiện tại là chia ngẫu nhiên theo nhóm nội dung, chưa phân tầng đa nhãn.
- Mất cân bằng và việc gộp nhiều sentiment thành một target là hai vấn đề riêng. Cả hai đều hiện hữu; thống kê này không tự thay đổi cách gán nhãn hay huấn luyện.

## Nguồn và xác minh

- train: [train_cleanser.csv](../data/train_cleanser.csv), SHA-256 `b0d8382252f02b7d4fc5de37648cb944827774e1521e5f79c1a19e71a6490edd`.
- val: [val_cleanser.csv](../data/val_cleanser.csv), SHA-256 `81b3eb447721441281ca9dd6926d089d7598071a1e8149ade5df34cc9dae8827`.
- test: [test_cleanser.csv](../data/test_cleanser.csv), SHA-256 `c73c37713d51beacacf90a2b06c1fa8ec2bd818b01cdda51b2fd0c965b26abca`.
- [Code training](../TrainPhoBert_BiLSTM_Beauty.py); [biểu đồ tập train](train_label_balance.png).

