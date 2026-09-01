# Tapşırıq Bölgüsü — 5 Üzv (yenilənmiş icra planı)

**Layihə:** Early Warning of Hydrate Formation in Offshore Well Service Lines
(3W Dataset 2.0.0, Event 9) · DLE-AI-202, Cohort I 2026 · Track 2

> Bu sənəd `team_responsibilities_all_members.md`-ni **əvəz etmir** — onu 5 nəfərə
> yenidən paylayır və qalan vaxta uyğunlaşdırır. Elmi məzmun, data kontraktı
> (§0.1), model kontraktı (§0.2), nəticə kontraktı (§0.3) və
> `DL_Project_Statement_Hydrate3W.docx`-un **Addendum**-u dəyişmir — onlar hələ də
> mübahisəli hallarda son həqiqətdir.

---

## 0. Vəziyyət — niyə bölgü dəyişir

| | |
|---|---|
| Bu gün | **1 Sentyabr 2026** |
| Təhvil | **7 Sentyabr 2026, 23:59** (Bakı) → **6 gün** |
| Orijinal qrafik | "31 Avqusta qədər tam CV sweep, 1–6 Sentyabr yalnız yazı" |
| Reallıq | `results.csv` boşdur, pipeline heç bir dəfə uçdan-uca işləməyib, 30 funksiya stub-dır |

**Nəticə:** yazı həftəsi artıq başlayıb, amma yazılacaq nəticə yoxdur. Ona görə
scope rəsmi olaraq kəsilir:

### Bu buraxılışa daxildir (məcburi)
- **Result 1:** XGBoost + TCN + GRU, eyni false-alarm nöqtəsində
  (`target_far = 1/100 h`), hər biri `real_only` və `real_plus_sim` şərtində
- `n_splits=5`, **`n_repeats=1`** (3 yerinə — compute 3 dəfə azalır, mean ± std
  hələ də hesabatlana bilir, fold sayı Table 1-də şəffaf göstərilir)
- 4 figure + head-to-head cədvəl, hamısı `run_all.sh`-dan

### Bu buraxılışa daxil DEYİL
- **SSL masked-reconstruction pretraining** (`ssl_pretrain.py`, `finetune.py`,
  `ReconstructionHead`) — komanda sənədinin §0.6-sı bunu Result 1 tamamlanana
  qədər qadağan edir. **Heç kim bu fayllara toxunmur.** Discussion-da bir
  cümlə ilə "vaxt çərçivəsində cəhd edilmədi" yazılır — brief bunu açıq şəkildə
  qəbul edir.
- Stretch Transformer (§12.4)
- GRU yalnız TCN işlədikdən sonra; vaxt çatmazsa Discussion-da qeyd olunur

---

## 1. Bölgü — kim nəyə sahibdir

Orijinal plan 4 nəfərlik idi və **Member 1 (data) bütün layihəni bloklayırdı**
(9 stub, hər kəs onu gözləyir). 5-ci nəfər məhz oraya gedir: data modulu
**M1 (window/cache)** və **M2 (splits/keyfiyyət)** kimi ikiyə bölünür.

| Üzv | Sahə | Sahib olduğu fayllar | Açıq stub sayı |
|---|---|---|---|
| **M1** | Data core — *pipeline bloklayıcısı* | `src/data/windowing.py`, `src/data/build_cache.py` | ✅ **0 — bitdi** (1 Sen) |
| **M2** | Splits + data keyfiyyəti | `src/data/splits.py`, `availability.py`, `stats.py`, `download.py` | 6 |
| **M3** | Baseline (XGBoost) | `src/baselines/*` | 4 |
| **M4** | Deep models | `src/models/train_loop.py`, `losses.py`, `heads.py`, `profile.py` | 4 (SSL istisna) |
| **M5** | Eval + repro + hesabat | `src/eval/*`, `run_all.sh`, `report/`, `presentation/` | 7 |

Sahiblik **qapı deyil, məsuliyyətdir**: başqasının modulunda real səhv görürsənsə
(səhv sabit, vahid uyğunsuzluğu) düzəlt və ya sahibi ilə birlikdə işlə — öz
modulunda onun ətrafından dolanma (§0.5).

---

## 2. M1 — Data core ✅ BİTDİ (1 Sen)

**Missiya:** xam 3W Parquet → kontraktı tam ödəyən `.npz` cache. Bütün digər
dörd nəfər sənin çıxışını gözləyir, ona görə bu **Gün 1-in ən yüksək prioriteti**dir.

> **Vəziyyət:** bütün stublar yazıldı və test edildi (44 test `windowing` üzrə,
> cəmi 61). `DATA_FINDINGS.md` ölçmələrinə görə üç dizayn dəyişikliyi tətbiq
> olundu: decimation 30× (pəncərə 30 dəqiqə), `failure_time` = transient onset,
> NaN etiketli pəncərələr atılır. `build_cache` real data üzərində işə salınıb.
> M1-in qalan işi: `availability.py` və `stats.py` (M2-yə keçdi, aşağı bax).

**Açıq stublar (yazıldı):**
- [`windowing.py:117`](src/data/windowing.py#L117) `VariableSelector.fit`
- [`windowing.py:123`](src/data/windowing.py#L123) `VariableSelector.transform`
- [`windowing.py:130`](src/data/windowing.py#L130) `mask_missing`
- [`windowing.py:152`](src/data/windowing.py#L152) `WindowBuilder.build_windows`
- [`build_cache.py:5`](src/data/build_cache.py#L5) `build_cache`

**İş bəndləri:**

1. **`mask_missing(X, fill)`** → `(X_filled, missing_mask)`. `mask==1` yalnız
   dəyər həm mövcud, həm də *frozen olmayan* halda. `ffill` maksimum boşluq
   uzunluğu ilə məhdud olsun, sonra qalanına train fold ortalaması.

2. **`VariableSelector`** — per-variable missing fraction > `max_missing_frac`
   (default 0.5) olan kanalları at. **Yalnız training instanslarında fit et**
   (DL2.1) — bu leakage qaydasıdır, rubrikada 20% "Experimental rigour"
   bəndindədir. `frozen_run_seconds=60`: ≥60 eyni ardıcıl oxunuş → NaN.

3. **`build_windows`** — burada **iki tələ var:**
   - ⚠️ **Ox konvensiyası:** funksiyanın docstring-i `(n_windows, window_size,
     n_channels)` deyir, amma bağlayıcı kontrakt (Addendum **A.2**) və bütün
     downstream (`features.transform`, `TCN.forward`, cache) **channels-first
     `[N, C, W]`** gözləyir. **Channels-first qaytar** və docstring-i düzəlt.
     Bu, səssiz shape səhvinin bir nömrəli mənbəyidir.
   - Etiketləməni **yenidən yazma** — mövcud `label_window(...,
     rule="most_severe")`-i çağır (o test edilib, 12/12).
   - `min_valid_frac`-dan aşağı pəncərələri at (DL2.4).
   - `window_end_time` **məcburidir** — lead time bundan hesablanır (DL2.3).

4. **`build_cache`** — `build_windows` yalnız 4 array qaytarır, **kontrakt isə
   8 + 2 tələb edir.** Cache faylı bunları da yazmalıdır:
   `X, mask, y, group, inst_id, t_end, is_sim` + instans səviyyəsində
   **`failure_time`** (blokaj başlama vaxtı, blokaj yoxdursa NaN) və
   **`normal_hours`**. Bu ikisi olmadan M5 headline metrikləri hesablaya bilmir.
   Format `make_fake_data.py`-ın çıxışı ilə **bit-bit uyğun** olsun.

5. **DL2.1b sənədə düşməli faktı:** real Event-9 instanslarında transient
   fazanın həqiqi davametmə paylanmasını ölç. `window_size=60s` **yoxlanılmamış
   defaultdur**. Əgər real transientlər çox uzundursa, `window_size`-ı artır —
   **və M4-ə xəbər ver** (TCN-in receptive field-i 61-dir, yalnız 1 addım
   ehtiyatla).

**Bitmə tərifi:** `python -m src.data.build_cache` fake data üzərində 40 `.npz`
yazır, hər biri kontrakt shape/dtype-larına uyğun; sonra real data üzərində
işləyir. **Gün 1 sonu (fake), Gün 2 (real).**

---

## 3. M2 — Splits və data keyfiyyəti

**Missiya:** leakage-dən qorunmanı təmin et və hesabatın Dataset bölməsinin
bütün rəqəmlərini istehsal et.

**Açıq stublar:**
- [`splits.py:36`](src/data/splits.py#L36) `GroupedKFoldSplitter.split`
- [`splits.py:56`](src/data/splits.py#L56) `fold_report`
- [`availability.py:7`](src/data/availability.py#L7) `variable_availability_table`
- [`stats.py:6`](src/data/stats.py#L6) `transient_duration_histogram`
- [`stats.py:10`](src/data/stats.py#L10) `annotated_trace_figure`

**İş bəndləri:**

1. **`split()` — nested, üç yollu.** İki mərhələ:
   - Xarici: `StratifiedGroupKFold` (`groups=well_id`) → train_wells / test_wells
   - Daxili: **yalnız train_wells içində** quyu səviyyəsində `val_frac=0.2`
     ayrılması → val_idx
   - **Simulyasiya pseudo-quyuları yalnız `train_idx`-də ola bilər** (DL3.2) —
     onları xarici split-in namizəd hovuzundan *əvvəlcədən* çıxar.
   - `n_repeats` sklearn-də native yoxdur: `random_state + repeat_idx` ilə dövr.
     **Bu buraxılışda `n_repeats=1`.**

2. **`fold_report()` — hesabatın Table 1-i və ilk işlədilməli şeydir** (DL3.3).
   Hər fold üçün: `n_train_wells, n_val_wells, n_test_wells,
   n_val_positive_events, n_test_positive_events`.
   **Əgər hansısa fold-da `n_val_positive_events == 0` çıxırsa** — `val_frac`-ı
   genişlət və ya fold sayını azalt. Sıfır pozitiv nümunə üzərində threshold
   tuning etmək mənasızdır.

3. **W1.3 — iki rəqəm ki hesabat onlar olmadan yazıla bilməz:**
   - 57 real Event-9 instansının arxasında **nə qədər fərqli quyu** var (DL1.2).
     Bu rəqəm statistik olaraq nə qədər fold-un mənalı olduğunu təyin edir.
   - `is_monotonic_severity()`-ni **hər real instansda** işlət və keçən faiz
     bildir (Addendum A.5). Hamısı `True`-dursa, `label_rule` seçimi heç bir
     nəticəni dəyişmir — bunu yaz. Hətta biri `False`-dursa, bu **hesabatlanası
     data-keyfiyyət tapıntısıdır**.
   - **Median/min/max timestep sayı** per instance → §2.2 floor-justification
     paraqrafındaki `[insert actual median timestep count]` placeholder-i bu
     rəqəmlə doldurulur. Placeholder qalarsa avtomatik bal itkisidir.

4. **`variable_availability_table`** — per-kanal missing/frozen faiz cədvəli.
   M1 dondurulmuş dəyişən siyahısını buradan alır (W1.5).

5. **`stats.py` iki figure:** transient davametmə histogramı və bir instansın
   annotasiyalı trace-i (Normal → Transient → Established zonaları rənglənmiş).

**Bitmə tərifi:** `results/instance_inventory.csv` + `fold_report()` DataFrame-i
diskdə, 2 figure `figures/`-də. **Gün 2 sonu.**

---

## 4. M3 — Baseline (XGBoost)

**Missiya:** ədalətli, güclü, "strawman olmayan" baseline. Track 2 baseline
tələbini götürmür — bu modul dərin modellər qədər ciddi qiymətləndirilir.
Sən **birinci uçdan-uca nəticəni** verirsən (XGBoost CPU-da işləyir, GPU
gözləmir).

**Açıq stublar:**
- [`features.py:21`](src/baselines/features.py#L21) `RollingFeatureExtractor.transform`
- [`tune.py:6`](src/baselines/tune.py#L6) `search`
- [`calibrate.py:6`](src/baselines/calibrate.py#L6) `fit_calibrator`
- [`importance.py:6`](src/baselines/importance.py#L6) `summarize_importances`

`XGBoostBaseline` **artıq hazırdır** ([xgb_model.py](src/baselines/xgb_model.py)) —
`compute_sample_weight()` ilə, `scale_pos_weight` olmadan (A.9.3: o parametr
multiclass-da heç nə etmir).

**İş bəndləri:**

1. **`transform` — multi-timescale, mask-aware (Addendum A.7-də *məcburi* edilib).**
   3 miqyas (`scales=[1.0, 0.5, 0.25]` — pəncərənin sonundan), 6 stat
   (`mean, std, min, max, slope, last_diff`). Kritik detallar:
   - Statlar **yalnız `mask==1` olan dəyərlər üzərində** hesablanır
   - `slope` = zaman oxuna qarşı least-squares fit
   - Sonda **per-kanal presence fraction** (`mask.mean(axis=-1)`) ayrı feature
     bloku kimi əlavə olunur — XGBoost "pəncərənin nə qədəri həqiqətən var"
     məlumatını birinci dərəcəli feature kimi görür
   - Giriş **channels-first** `(N, C, W)`

2. **`search`** — budget-matched hyperparameter axtarışı, `groups_train` ilə
   qruplaşdırılmış CV. **Neçə konfiqurasiya sınadığını qeyd et** — dərin
   modellərlə eyni tuning büdcəsi olmalıdır, əks halda müqayisə ədalətsizdir.

3. **`fit_calibrator`** — isotonic və ya Platt, **yalnız validation fold-da** fit.
   ECE (M5) və reliability diagram üçün öncə/sonra ehtimalları saxla.

4. **`summarize_importances`** — fold başına, sonra ümumiləşdir: hansı xam
   kanallar dominant? Bunu 3W deskriptorundakı fiziki hidrat imzası ilə
   müqayisə et — Discussion-un ən güclü paraqrafı buradan çıxır.

5. **Result 1 qaçışları:** eyni pipeline **iki dəfə** — `condition=real_only` və
   `condition=real_plus_sim`. Hər ikisi `results/results.csv`-ə əlavə olunur.
   Sim instanslar heç vaxt val/test-də deyil (M2 bunu təmin edir).

**Bitmə tərifi:** `results.csv`-də XGBoost sətirləri, hər iki şərt üçün.
**Gün 3.**

---

## 5. M4 — Deep models

**Missiya:** TCN (əsas) və GRU (ikincil) öyrədilə bilən vəziyyətə gətir.
`TCN` və `GRUClassifier` **artıq yazılıb və doğrudur** — çatışmayan hissə
`Trainer`-dır, yəni **hazırda dərin modellər heç öyrədilə bilmir.** Bu, sənin
bir nömrəli işidir.

**Açıq stublar:**
- [`train_loop.py:46`](src/models/train_loop.py#L46) `Trainer.fit`
- [`train_loop.py:60`](src/models/train_loop.py#L60) `Trainer.predict_proba`
- [`train_loop.py:65`](src/models/train_loop.py#L65) `Trainer.load_best_checkpoint`
- [`profile.py:5`](src/models/profile.py#L5) `profile_model`

**Toxunulmaz:** `ssl_pretrain.py`, `finetune.py`, `heads.py:15
ReconstructionHead` — §0.6-ya görə Result 1 bitənə qədər qadağandır.

**İş bəndləri:**

1. **`Trainer` = adapter.** `TCN.fit()` yoxdur və olmamalıdır (§0.2). Trainer
   PyTorch modelini M5-in gördüyü `fit`/`predict_proba` formasına sarır.
   - Early stopping **validation PR-AUC**-da, `patience=10` — **accuracy-də
     yox** (DL7.1: 57 transient vs 594 normal, accuracy mənasızdır)
   - `val_loader` **yalnız** M2-nin `split()`-inin verdiyi `val_idx`-dən qurulur
     (DL7.0). Təsadüfi sətir-səviyyəli split quyu-səviyyəli leakage nəzarətini
     səssizcə məhv edir.
   - Hər epoch checkpoint (DL7.2), `amp=True` (DL7.3)
   - **Seed** `base_seed + fold_index` qaydası ilə: numpy, torch, CUDA,
     DataLoader worker-lər (DL7.4)
   - Metrikləri **diskə** yaz, yalnız stdout-a deyil

2. **Kodda assertion (bir dəfəlik əl yoxlaması deyil):**
   ```python
   assert tcn.receptive_field() >= window_size
   ```
   Defaultlarla receptive field **61**, `window_size` **60** — yalnız 1 addım
   ehtiyat. M1 `window_size`-ı artırsa, bu assertion olmadan TCN səssizcə
   pəncərənin bir hissəsini görməyi dayandırır (DL5.1).

3. **Class imbalance:** `losses.weighted_cross_entropy` və ya `FocalLoss` —
   XGBoost-un `compute_sample_weight` ilə eyni məntiq, ki müqayisə arxitekturanı
   izolə etsin, preprocessing fərqini yox (DL6.2).

4. **GRU `bidirectional=False`** — dəyişmə. Bidirectional GRU pəncərə içində
   gələcəyə baxır, bu early-warning üçün ədalətsizdir (DL6.1).

5. **`profile_model`** — parametr sayı, VRAM high-water mark, epoch vaxtı.
   Hesabatda "compute büdcəsinə dürüst şəkildə sığır" iddiası bu rəqəmlərlə
   dəstəklənir (DL5.4).

6. **Result 1 qaçışları:** TCN × {real_only, real_plus_sim}, sonra GRU × ikisi.

**Bitmə tərifi:** `results.csv`-də TCN sətirləri. **Gün 4.** GRU: Gün 4–5.

---

## 6. M5 — Eval, reproduksiya, hesabat

**Missiya:** rubrikanın 35%-i (Paper 15% + Presentation 8% + Track-specific
metrics 15%-in çoxu) bilavasitə sənin çıxışındır. Sən nəticələri gözləmirsən —
random array-lər üzərində indi yazırsan (W4.1).

**Açıq stublar:**
- [`thresholds.py:19`](src/eval/thresholds.py#L19) `select_threshold`
- [`thresholds.py:39`](src/eval/thresholds.py#L39) `select_threshold_curve`
- [`metrics.py:66`](src/eval/metrics.py#L66) `expected_calibration_error`
- [`aggregate.py:35`](src/eval/aggregate.py#L35) `to_latex_table`
- [`plots.py`](src/eval/plots.py) — **4 figure-un hamısı**

`alarm.py` **hazırdır və trailing (causal) smoothing ilə test edilib** (A.9.1) —
oradaki `_trailing_mean`-i mərkəzləşdirilmiş `np.convolve(..., mode="same")` ilə
əvəz etmək layihənin ən ciddi leakage bug-ıdır. `tests/test_alarm.py` bunu
qoruyur.

**İş bəndləri:**

1. **`select_threshold`** — `np.linspace(0, 1, 200)` üzərində sweep; hər namizəd
   üçün **yalnız Normal validation instanslarında** `alarm_times()` onset-lərini
   say, `val_normal_hours`-a böl, `target_far = 1/100 h`-ə **aşmadan** ən yaxın
   olanı seç. Seçilmiş dəyəri fold başına log et — fold-lar arasında güclü
   dəyişirsə, bu qeyri-sabitlik özü hesabatlanası tapıntıdır (DL8.1).

2. **`select_threshold_curve`** — lead-time vs false-alarm-rate əyrisi. W4.3
   bunu "ən informativ figure" adlandırır: 100 saatlıq nöqtə headline olsa da,
   500 və digər büdcələr bir figure-da görünür (A.6).

3. **Dondurma qaydası:** `threshold`, `smooth_window`, `min_duration` — üçü də
   **yalnız validation**-da seçilir, sonra o fold-un test quyularına
   **dəyişdirilmədən** tətbiq olunur. Test rəqəmini gördükdən sonra hər hansı
   dəyişiklik brief-də avtomatik bal itkisidir (S3 qaydası).

4. **4 figure** (hamısı `run_all.sh`-dan, dekorativ vizual yoxdur):
   - `plot_annotated_trace` — real instans + alarm nöqtəsi
   - `plot_lead_time_vs_false_alarm_rate` — headline trade-off
   - `plot_per_well_lead_time_box` — fold-lar arası yayılma
   - `plot_reliability_diagram` — kalibrasiya öncə/sonra (M3-dən)

5. **`run_all.sh`** — hazırda yalnız `inventory`-yə qoşulub. Tam zəncir:
   windowing → cache → grouped-CV sweep → aggregate → figures.
   Çıxış: `results/fold_metrics.csv`, `results/summary.csv`, `figures/*.png`.
   **Təmiz checkout-dan bir əmrlə hər rəqəm və figure yenidən yaranmalıdır.**

6. **IEEE hesabat + 10–12 slayd + `contribution_report.pdf`**, `v1.0-final` tag.
   Abstract-da tag-lənmiş repo linki. Heç bir rəqəm əllə yazılmır.

**Bitmə tərifi:** figures Gün 5, hesabat qaralaması Gün 6, təhvil Gün 6 axşamı
(7-si ehtiyat gündür, son gün deyil).

---

## 7. Gündəlik qrafik

| Gün | M1 (data core) | M2 (splits/keyfiyyət) | M3 (baseline) | M4 (deep) | M5 (eval/paper) |
|---|---|---|---|---|---|
| **1 Sen** | venv + `pytest` yaşıl; `mask_missing`, `VariableSelector` | inventory `--verify` real data-da; well count | `transform` (fake data ilə) | `Trainer.fit` skeleti | `select_threshold` (random array-lərlə) |
| **2 Sen** | `build_windows` (channels-first!), `build_cache` fake-də | `split()` + `fold_report` → **Table 1** | tuning + calibration | `Trainer` bitir, TCN fake-də öyrənir | `threshold_curve`, ECE |
| **3 Sen** | real cache hazır → **S2 elan** | availability + 2 figure | **XGBoost × 2 şərt → results.csv** | TCN real data-da | 4 figure |
| **4 Sen** | cache regen dəstəyi, DL2.1b hesabatı | Dataset bölməsi mətni | importances → Discussion | **TCN × 2 şərt → results.csv**; GRU başlayır | `run_all.sh` tam qoşulur |
| **5 Sen** | — | — | — | **GRU × 2 şərt** | **S3 FREEZE** → test fold-lar bir dəfə işlədilir; cədvəllər/figurelar |
| **6 Sen** | hamı: öz paper bölməsi + defense hazırlığı | | | | hesabat + slaydlar + contribution report → **təhvil** |
| **7 Sen** | ehtiyat günü (son gün kimi planlamayın) | | | | |

---

## 8. Git iş axını

```bash
git clone <repo-url> && cd team-hydrate3w
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
pytest tests/ -v                                    # 29 test yaşıl olmalı
python src/data/make_fake_data.py --out data/fake/ --n_instances 40
```

- Branch adları: `m1/windowing`, `m2/splits`, `m3/features`, `m4/trainer`,
  `m5/thresholds` — feature başına bir branch, kiçik commit-lər
- **`main`-ə force-push yoxdur**
- **Hər kəs birinci gündən, hər gün commit edir.** İnstruktor Git tarixini
  contribution report ilə tutuşdurur; son saatlarda tökülən kod töhfə sayılmır.
  Commit sayı yarışı yoxdur — nə etdiyini aydın göstərən mənalı commit-lər.
- `data/`, `results/`, `checkpoints/`, `*.pt` gitignore-dadır (A.9.2 düzəlişi
  yoxlanılıb: `download_data.sh` və `.gitkeep` izlənir).

---

## 9. Sync nöqtələri

| | Nə vaxt | Nə olur |
|---|---|---|
| **S1** | 1 Sen axşamı | Hər beş nəfər fake data üzərində bir şeyi uçdan-uca işlədib; kontrakt donub |
| **S2** | Real cache hazır olanda (2–3 Sen) | M1 elan edir, M2 real quyu sayını bildirir; hamı data yolunu dəyişir |
| **S3** | **5 Sen — FREEZE** | Bütün tuning validation-da bitib. Bundan sonra hyperparameter dəyişikliyi **yoxdur**. Yalnız bundan sonra M5 test fold-larını **bir dəfə** işlədir |

---

## 10. Qırmızı xətlər (avtomatik bal itkisi)

1. Heç bir simulyasiya instansı val/test fold-a düşmür
2. `VariableSelector`, scaler, feature extractor — **yalnız train fold**-da fit
3. Threshold / smoothing / min_duration — **yalnız validation**, test üçün donmuş
4. Alarm smoothing **trailing (causal)**, mərkəzləşdirilmiş deyil
5. GRU unidirectional
6. Accuracy heç vaxt headline metrik deyil
7. False alarm **saat** başına, pəncərə sayı başına deyil
8. `run_all.sh` təmiz checkout-dan işləyir; heç bir rəqəm əllə yazılmır
9. §2.2 floor-justification paraqrafında placeholder qalmır
10. `[N, C, W]` channels-first — hər yerdə

---

## 11. Müdafiə — hər beş nəfərin cavablandırmalı olduğu suallar

Modulundan asılı olmayaraq:

1. Lead time necə tərif olunur və hansı iki timestamp-dan hesablanır?
2. Niyə accuracy headline metrik deyil?
3. Nə üçün grouped CV — adi k-fold nəyi pozardı?
4. `positive_score = P(Transient) + P(Established)` — niyə yalnız
   `P(Transient)` deyil?
5. Simulyasiya datası nəticələri necə dəyişdi, və niyə headline rəqəm
   `real_only`-dir?
