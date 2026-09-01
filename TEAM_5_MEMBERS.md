# Tapşırıq Bölgüsü — 5 Üzv (bərabərləşdirilmiş, 2 Sentyabr)

**Layihə:** Early Warning of Hydrate Formation in Offshore Well Service Lines
(3W Dataset 2.0.0, Event 9) · DLE-AI-202, Cohort I 2026 · Track 2

> Bu sənəd `team_responsibilities_all_members.md`-ni **əvəz etmir** — onu 5 nəfərə
> yenidən paylayır. Elmi məzmun, data kontraktı (§0.1), model kontraktı (§0.2),
> nəticə kontraktı (§0.3) və `DL_Project_Statement_Hydrate3W.docx`-un
> **Addendum**-u dəyişmir.
>
> **Əvvəlki versiyadan fərq:** iş yükü ölçülüb və bərabərləşdirilib. Köhnə
> bölgüdə M5-in üstündə 7 stub + bütün hesabat + bütün slaydlar + bütün
> reproduksiya vardı (~25 saat), M1-in kodu isə bitmişdi (0 saat). İndi hər beş
> nəfərdə **~15 saat** var — hesablama §1-dədir.

---

## 0. Vəziyyət

| | |
|---|---|
| Bu gün | **2 Sentyabr 2026** |
| Təhvil | **7 Sentyabr 2026, 23:59** (Bakı) → **5 gün** |
| Repo | https://github.com/AMammedova/team-hydrate3w |
| Testlər | **61 keçir** |
| Real data | yükləndi (3.6 GB), analiz edildi → `DATA_FINDINGS.md` |
| Cache | `data/cache/` qurulur (801 instans) |

### Artıq bitmiş işlər (yenidən görülməməli)

- `src/contract.py`, `inventory.py`, `make_fake_data.py`
- `windowing.py` — **tamamı** (labeling, frozen detection, kauzal `mask_missing`,
  per-instance normalizasiya, `WindowBuilder`, onset/normal-hours helper-ləri)
- `build_cache.py` — tamamı, sidecar + `--channels` override ilə
- `xgb_model.py`, `tcn.py`, `gru.py`, `alarm.py`, `metrics.py` (qismən)
- `tools/scan_*.py`, `tools/channel_availability.py` — ölçmə skriptləri

### Scope-dan çıxarılanlar (dəyişməz)

- **SSL pretraining** (`ssl_pretrain.py`, `finetune.py`, `ReconstructionHead`) —
  komanda sənədinin §0.6-sı Result 1 bitənə qədər qadağan edir. Heç kim
  toxunmur. Discussion-da bir cümlə ilə "vaxt çərçivəsində cəhd edilmədi".
- Stretch Transformer (§12.4)
- `n_repeats=3` → **1**

### Git qaydası (bir dəfəlik istisna)

Tarix bir dəfə təmizləndi (commit mesajlarındaki `Co-Authored-By` trailer-ləri
silindi). Bunun üçün `git push --force-with-lease` lazımdır və **bu yeganə
icazəli force-push-dur**. Əgər kimsə repo-nu bundan əvvəl clone edibsə, yenidən
clone etməlidir. Bundan sonra `main`-ə force-push **yoxdur**.

Brief-in akademik dürüstlük tələbi commit trailer-i ilə deyil, hesabatın
**Tools & Acknowledgements** bölməsi ilə ödənilir (`README.md`-də yer ayrılıb) —
orada istifadə olunan AI köməyi açıq yazılmalıdır. Bu bölmə **M5**-dədir.

---

## 1. İş yükü hesabı — niyə bərabərdir

Qalan iş üç növdür: **kod stubları (24 ədəd)**, **eksperiment qaçışları (6)** və
**yazı işi (8 IEEE bölməsi + 10–12 slayd + contribution report)**. Köhnə bölgü
yalnız kodu paylayırdı; yazı işi bir nəfərin üstündə qalırdı. İndi üçü də paylanır.

| Üzv | Sahə | Kod | Eksperiment | Yazı | **Cəmi** |
|---|---|---|---|---|---|
| **M1** | Data keyfiyyəti & dataset hekayəsi | 6.5 s | 3 s | 4.5 s | **~14 s** |
| **M2** | Splits & fold dizaynı | 9 s | — | 5.5 s | **~15 s** |
| **M3** | Baseline (XGBoost) | 9.5 s | 1.5 s | 4 s | **~15 s** |
| **M4** | Dərin modellər | 9.5 s | 2.5 s | 3 s | **~15 s** |
| **M5** | Eval & reproduksiya | 10 s | — | 5 s | **~15 s** |

**Bərabərləşdirmə üçün edilən üç dəyişiklik:**

1. **Figure-lar analizin sahibinə verildi.** `plots.py`-ın 4 figure-u əvvəl
   tamamilə M5-də idi. İndi: `plot_annotated_trace` → **M1** (onsuz da
   `stats.py`-də annotasiyalı trace çəkir, ikiqat iş idi),
   `plot_reliability_diagram` → **M3** (kalibrasiyanın sahibi), qalan ikisi
   (lead-time vs false-alarm, per-well box) → **M5**.
2. **Hesabat bölmələri beşə paylandı** — hər kəs öz modulunun bölməsini yazır,
   bu həm ədalətlidir, həm də müdafiədə hər kəsin öz bölməsini izah etməsini
   təmin edir.
3. **Slaydlar sahibinə görə** — hər üzv öz 1–3 slaydını hazırlayır, M5 yalnız
   yığır və formatlayır.

---

## 2. M1 — Data keyfiyyəti və dataset hekayəsi

**Missiya:** datanın nə olduğunu ölçülmüş rəqəmlərlə göstərmək. `windowing.py`
və `build_cache.py` bitdi; qalan iş data-keyfiyyət analizi, dataset figure-ları
və həssaslıq qaçışlarıdır.

**Kod (≈6.5 saat):**
- [`availability.py:7`](src/data/availability.py#L7) `variable_availability_table` —
  `tools/channel_availability.py`-dəki məntiqi düzgün modula köçür + test (3 s).
  **Bu, layihənin ən vacib data cədvəlidir** — §8-dəki quyu×kanal seçimi bundan çıxır.
- [`stats.py:6`](src/data/stats.py#L6) `transient_duration_histogram` (14 hadisə) (1.5 s)
- [`stats.py:10`](src/data/stats.py#L10) `annotated_trace_figure` — Normal→Transient→
  Established zonaları rənglənmiş bir real instans (1.5 s)
- [`plots.py:13`](src/eval/plots.py#L13) `plot_annotated_trace` — yuxarıdakının
  üstünə alarm nöqtəsini əlavə edən variant; eyni kodu paylaşın (0.5 s)
- [`download.py:4`](src/data/download.py#L4) — `download_data.sh`-a nazik sarğı (0.5 s)

**Eksperiment (≈3 saat):** cache həssaslıq qaçışları —
`--label-rule` sweep (most_severe / final_timestep / majority),
`--channels` iki arm (2-kanal vs 7-kanal, §8),
`--nan-label-policy drop` vs `normal` — hər biri üçün atılan pəncərə sayını qeyd et.

**Yazı (≈4.5 saat):** Paper **Introduction** + **Dataset** bölmələri.
Dataset bölməsinə mütləq düşməli rəqəmlər:
median 32,135 timestep (floor justification, placeholder-i doldur),
`is_monotonic_severity` 57/57, missingness/frozen statistikası,
quyu×kanal trade-off cədvəli. Slaydlar 1–3 (motivasiya + dataset).

---

## 3. M2 — Splits və fold dizaynı

**Missiya:** leakage-dən qorunma və "mean ± std mənalıdırmı?" sualına dürüst cavab.
**Bu, indi kritik yolun başıdır** — M3, M4, M5 fold-ları gözləyir.

**Kod (≈9 saat):**
- [`splits.py:36`](src/data/splits.py#L36) `split()` — **iki müstəqil grouped split**
  (5 s). Sadə `StratifiedGroupKFold` işləməyəcək: pozitiv quyular (7) və normal
  quyular (9) tamamilə ayrıdır (`DATA_FINDINGS.md` §2), ona görə:
  - pozitiv quyular üzərində bir grouped split
  - normal quyular üzərində **saatlara görə balanslaşdırılmış** ayrı split
    (hər test fold-unda ≥300 h olsun — WELL-00002 tək 1220 h, WELL-00007 isə 6 h)
  - fold-lar cütləşdirilir; simulyasiya pseudo-quyuları **yalnız `train_idx`**
  - nested val: yalnız o fold-un train quyularından
- [`splits.py:56`](src/data/splits.py#L56) `fold_report()` (2 s) — **`test_normal_hours`
  sütunu əlavə et** (spesifikasiyada yoxdur, amma false-alarm məxrəcidir).
  `n_val_positive_events == 0` olan fold varsa dayan və fold sayını azalt.
- Testlər: quyu heç vaxt iki tərəfdə olmasın, sim heç vaxt val/test-də olmasın (2 s)
- `cache_config.json`-dan kanal siyahısının fold-lar arası dəyişməzliyini
  yoxla — bu, `build_cache`-in yeganə leakage qeydini rəqəmlə bağlayır (1 s dolayısı)

**Qərar:** 7 pozitiv quyu var → `n_splits=5` çox incədir. **3 fold** (test-də
2–3 pozitiv quyu) və ya 7 quyu üzərində **leave-one-well-out** (7 fold, hər
test = 1 quyu). `fold_report()` çıxışına baxıb seç və hesabatda əsaslandır.

**Yazı (≈5.5 saat):** Paper **Experimental Setup** (Table 1 = fold report) +
**Discussion & Limitations** (57 real instansdan yalnız 14-ü pozitiv, 3-ü blokaj;
9 normal quyu; fold sayının statistik mənası). Slayd 4 (fold dizaynı).
**`contribution_report.pdf`**-i yığ (hamı imzalayır).

---

## 4. M3 — Baseline (XGBoost)

**Missiya:** ədalətli, güclü baseline. Track 2 baseline tələbini götürmür —
bu modul dərin modellər qədər ciddi qiymətləndirilir. **Birinci uçdan-uca
nəticə səndən gəlir** (XGBoost CPU-da işləyir, GPU gözləmir).

**Kod (≈9.5 saat):**
- [`features.py:21`](src/baselines/features.py#L21) `transform` (4 s) — 3 miqyas
  (`[1.0, 0.5, 0.25]`, pəncərənin sonundan) × 6 stat, **yalnız `mask==1`
  dəyərlər üzərində**, sonda per-kanal presence fraction ayrı blok kimi.
  Giriş channels-first `(N, C, W)`.
- [`tune.py:6`](src/baselines/tune.py#L6) `search` (2 s) — grouped CV, **neçə
  konfiqurasiya sınadığını qeyd et** (M4 ilə eyni tuning büdcəsi olmalıdır)
- [`calibrate.py:6`](src/baselines/calibrate.py#L6) (2 s) — isotonic/Platt,
  **yalnız validation fold-da** fit; öncə/sonra ehtimalları saxla
- [`importance.py:6`](src/baselines/importance.py#L6) (1 s)
- [`plots.py:29`](src/eval/plots.py#L29) `plot_reliability_diagram` (0.5 s) —
  kalibrasiya sənin olduğu üçün figure də sənindir

**Eksperiment (≈1.5 saat):** XGBoost × `{real_only, real_plus_sim}` →
`results/results.csv`.

**Yazı (≈4 saat):** Paper **Method (baseline/features)** + **Results (XGBoost)**.
Feature importance-ı 3W deskriptorundakı fiziki hidrat imzası ilə müqayisə et —
Discussion-un ən güclü paraqrafı buradan çıxır. Slayd: baseline + importance.

---

## 5. M4 — Dərin modellər

**Missiya:** TCN (əsas) və GRU (ikincil) öyrədilə bilən vəziyyətə gətir.
`TCN` və `GRUClassifier` **yazılıb və doğrudur** — çatışmayan `Trainer`-dır,
yəni hazırda dərin modellər **heç öyrədilə bilmir**. Bir nömrəli işin budur.

**Toxunulmaz:** `ssl_pretrain.py`, `finetune.py`, `ReconstructionHead`.

**Kod (≈9.5 saat):**
- [`train_loop.py:46`](src/models/train_loop.py#L46) `fit` (4 s) — early stopping
  **validation PR-AUC**-da (accuracy-də yox: 57 vs 594), `patience=10`,
  hər epoch checkpoint, `amp=True`, seed `base_seed + fold_index` (numpy, torch,
  CUDA, DataLoader worker-lər), metriklər **diskə**
- [`train_loop.py:60`](src/models/train_loop.py#L60) `predict_proba` (1 s)
- [`train_loop.py:65`](src/models/train_loop.py#L65) `load_best_checkpoint` (0.5 s)
- Kodda assertion: `assert tcn.receptive_field() >= window_size` (0.5 s).
  Receptive field 61, `window_size` 60 — yalnız 1 addım ehtiyat.
- [`profile.py:5`](src/models/profile.py#L5) — parametr sayı, VRAM peak, epoch vaxtı (1 s)
- Testlər: `Trainer` fake data üzərində, `val_loader` yalnız `val_idx`-dən (2 s)

**⚠️ Ətraf mühit qeydi:** bu maşında `torch 2.3.1+cpu` quraşdırılıb — CUDA yoxdur.
CPU-da debug olar, tam CV sweep olmaz. Paylaşılan A100 maşınında CUDA torch
quraşdır və VRAM peak-i oradan hesabatla.

**Eksperiment (≈2.5 saat):** TCN × 2 şərt, sonra GRU × 2 şərt.
GRU `bidirectional=False` — dəyişmə.

**Yazı (≈3 saat):** Paper **Related Work** (TCN/GRU sənaye time-series üçün) +
**Method (arxitekturalar, receptive field, parametr sayı)** + **Results (dərin
modellər)**. Slayd: arxitektura diaqramı (1–2).

---

## 6. M5 — Eval və reproduksiya

**Missiya:** rubrikanın baş metriklərini kodla təsbit etmək və hər rəqəmin
`run_all.sh`-dan çıxmasını təmin etmək. Nəticələri gözləmirsən — random
array-lər üzərində indi yazırsan (W4.1).

`alarm.py` **hazırdır və trailing (kauzal) smoothing ilə test edilib** (A.9.1).
Oradaki `_trailing_mean`-i mərkəzləşdirilmiş `np.convolve(..., mode="same")` ilə
əvəz etmək layihənin ən ciddi leakage bug-ıdır; `tests/test_alarm.py` qoruyur.

**Kod (≈10 saat):**
- [`thresholds.py:19`](src/eval/thresholds.py#L19) `select_threshold` (2.5 s) —
  `np.linspace(0,1,200)` sweep, **yalnız Normal validation instanslarında**
  `alarm_times()` onset-lərini say, `val_normal_hours`-a böl,
  `target_far = 1/100 h`-i **aşmadan** ən yaxınını seç. Seçilmiş dəyəri fold
  başına log et — fold-lar arası güclü dəyişirsə, bu özü hesabatlanası tapıntıdır.
- [`thresholds.py:39`](src/eval/thresholds.py#L39) `select_threshold_curve` (1.5 s) —
  W4.3-ün "ən informativ figure"-u
- [`metrics.py:66`](src/eval/metrics.py#L66) `expected_calibration_error` (1 s)
- [`aggregate.py:35`](src/eval/aggregate.py#L35) `to_latex_table` (1 s)
- [`plots.py:20`](src/eval/plots.py#L20) `plot_lead_time_vs_false_alarm_rate` (1.5 s)
- [`plots.py:25`](src/eval/plots.py#L25) `plot_per_well_lead_time_box` (1 s)
- [`run_all.sh`](run_all.sh) tam zəncir (1.5 s): inventory → cache → CV sweep →
  aggregate → figures. Çıxış: `results/fold_metrics.csv`, `results/summary.csv`,
  `figures/*.png`. **Təmiz checkout-dan bir əmr.**

**Dondurma qaydası:** `threshold`, `smooth_window`, `min_duration` — üçü də
yalnız validation-da seçilir, sonra o fold-un test quyularına **dəyişdirilmədən**
tətbiq olunur. Test rəqəmini gördükdən sonra dəyişiklik = avtomatik bal itkisi.

**Yazı (≈5 saat):** Paper **Abstract** (tag-lənmiş repo linki ilə),
**Method (evaluation: lead time / event recall / FAR tərifləri)**,
**Results (head-to-head cədvəl)**, **Conclusion**,
**Tools & Acknowledgements** (AI köməyinin açıqlanması burada).
Slaydları yığ və formatla (10–12), `v1.0-final` tag.

---

## 7. Gündəlik qrafik

| Gün | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|
| **2 Sen** | availability.py | **splits.py** (kritik yol) | features.py | Trainer.fit | select_threshold |
| **3 Sen** | stats.py + 2 figure | **fold_report → Table 1** | tune + calibrate | Trainer bitir, TCN fake-də | curve + ECE |
| **4 Sen** | həssaslıq qaçışları | fold qərarı + Dataset mətni | **XGB × 2 → results.csv** | TCN real data-da | 3 figure |
| **5 Sen** | Introduction + Dataset | Exp. Setup + Discussion | Results (XGB) | **TCN×2, GRU×2 → results.csv** | **run_all.sh tam** |
| **6 Sen** | slaydlar 1–3 | contribution report | slayd | slayd | **S3 FREEZE** → test fold-lar bir dəfə; cədvəllər; hesabat; **təhvil** |
| **7 Sen** | ehtiyat günü (son gün kimi planlamayın) | | | | |

---

## 8. Kanal dəsti — qərar verildi (`DATA_FINDINGS.md` §8–§9)

Hər quyu fərqli kanalları ölçür, üstəlik bəzi instanslarda sensor bütün
qeydiyyat boyu donmuşdur. Kanal dəstini **quyu əhatəsinə** görə seçmək səhv
nəticə verirdi; düzgün kriteriya **hadisə əhatəsidir**
(`tools/channel_tradeoff.py`, instans səviyyəsində ölçür):

| Kanal dəsti | Kanal | Transient hadisə | Blokaj | Pozitiv quyu | Normal saat |
|---|---|---|---|---|---|
| **`P-MON-CKP, P-JUS-CKGL, T-TPT, T-JUS-CKP, P-ANULAR`** | **5** | **14/14** | **3/3** | **7/7** | **1,467 h** |
| `P-MON-CKP` tək | 1 | 14/14 | 3/3 | 7/7 | 1,422 h |
| `P-TPT, T-TPT` | 2 | 9/14 | 2/3 | 4/7 | 2,137 h |

**Əsas arm: 5 kanal** (yuxarıdakı). Cache bu dəstlə qurulur. 1,467 saat
100 saatlıq yalan-həyəcan büdcəsindən 14 dəfə çoxdur — kalibrləmə üçün
kifayətdir, hadisə itkisi isə bərpa olunmazdır.

**İkinci arm (M1-in həssaslıq qaçışı):** `P-TPT, T-TPT` — daha çox normal saat,
daha az hadisə. Discussion paraqrafı.

⚠️ Üç instansda (`WELL-00040_20181013160242`, `WELL-00041_20181013160201`,
`WELL-00014_20170214190000` — sonuncusu 3 blokaj hadisəsindən biridir)
`P-TPT`/`T-TPT` tamamilə donmuşdur. M2 fold qurarkən bunu bilməlidir: hadisə
sayı kanal dəstindən asılıdır.

---

## 9. Qırmızı xətlər (avtomatik bal itkisi)

1. Heç bir simulyasiya instansı val/test fold-a düşmür
2. `VariableSelector`, scaler, feature extractor — **yalnız train fold**-da fit
3. Threshold / smoothing / min_duration — **yalnız validation**, test üçün donmuş
4. Alarm smoothing **trailing (kauzal)**; `mask_missing` **yalnız ffill**
5. GRU unidirectional
6. Accuracy heç vaxt headline metrik deyil
7. False alarm **saat** başına, pəncərə sayı başına deyil
8. `run_all.sh` təmiz checkout-dan işləyir; heç bir rəqəm əllə yazılmır
9. §2.2 floor-justification paraqrafında placeholder qalmır (rəqəm: **32,135**)
10. `[N, C, W]` channels-first — hər yerdə

---

## 10. Sync nöqtələri

| | Nə vaxt | Nə olur |
|---|---|---|
| **S1** | ✅ 1 Sen | Kontrakt donub, repo GitHub-da, 61 test yaşıl, fake + real cache var |
| **S2** | 3 Sen | M2 `fold_report()`-u paylaşır; hamı fold strukturunu qəbul edir |
| **S3** | **6 Sen — FREEZE** | Bütün tuning validation-da bitib. Bundan sonra hyperparameter dəyişikliyi **yoxdur**. Yalnız bundan sonra M5 test fold-larını **bir dəfə** işlədir |

---

## 11. Müdafiə — hər beş nəfərin cavablandırmalı olduğu suallar

1. Lead time necə tərif olunur, hansı iki timestamp-dan hesablanır, və niyə
   blokaj yerinə transient onset seçildi?
2. Niyə accuracy headline metrik deyil?
3. Nə üçün grouped CV — adi k-fold nəyi pozardı?
4. `positive_score = P(Transient) + P(Established)` — niyə yalnız `P(Transient)` deyil?
5. Simulyasiya datası nəticələri necə dəyişdi, və niyə headline rəqəm `real_only`-dir?
6. Hidrat quyuları ilə normal quyular kəsişmir — bu niyə problemdir və nə etdik?
