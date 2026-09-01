# Real Data Tapıntıları — 1 Sentyabr 2026

**Mənbə:** `petrobras/3W` (3W Dataset 2.0.0), `data/3W/dataset/`, 3.6 GB
**Necə alınıb:** `src/data/inventory.py --verify` + parquet `class` sütununun
birbaşa oxunması. Hər rəqəm yenidən istehsal oluna bilər.

> ⚠️ **Bu sənəd layihənin headline metrikini yenidən müzakirəyə açır.**
> `DL_Project_Statement_Hydrate3W.docx` §11 (DL8.3) median lead time-ı
> *"(established-blockage onset) − (first alarm)"* kimi tərif edir. Real data-da
> **cəmi 3 instansda blokaj var** — bu tərif real data üzərində praktiki olaraq
> hesablana bilmir. Detal aşağıda, §3-də.

---

## 1. Təsdiqlənən rəqəmlər (sənədlə uyğun gələnlər)

| | Sənəddə | Realda | |
|---|---|---|---|
| Real Event-9 instans | 57 | **57** | ✅ |
| Simulyasiya Event-9 instans | 150 | **150** | ✅ |
| Normal Operation instans | 594 | **594** | ✅ |
| Fayl adı konvensiyası | qeyri-müəyyən | `WELL-000NN_<ts>.parquet` / `SIMULATED_000NN.parquet` | ✅ parser düzgün işləyir |

`inventory.py`-ın `_infer_source_and_well` parseri bu buraxılışda **düzgündür** —
dəyişiklik tələb olunmur (docstring-in tələb etdiyi yoxlama edildi).

### §2.2 floor-justification üçün lazım olan rəqəmlər (placeholder-i doldur)

| | median | min | max | ≥10k timestep |
|---|---|---|---|---|
| Real Event-9 (57) | **32,135** | 7,144 | 192,581 | 55/57 |
| Simulated (150) | 45,599 | 31,199 | 63,600 | 150/150 |
| Normal (594) | 21,474 | 10,610 | 143,231 | 594/594 |

Hesabata düşəcək cümlə: *"each 3W instance is a full-duration 1 Hz multivariate
recording, **32,135** timesteps at the median"*. Dürüst əlavə: 57 real
instansdan **2-si** fərdi olaraq 10k-dan aşağıdır (min 7,144) — aqreqat oxunuşda
(§2.2-nin qeyd etdiyi alternativ şərh) bu, məsələ deyil, amma hesabatda
göstərilməlidir.

### Addendum A.5 sualı həll olundu

`is_monotonic_severity()` → **57/57 True**. Yəni heç bir real instansda severity
geri qayıtmır. A.5-in proqnozu təsdiqləndi: `most_severe` və `final_timestep`
bu event üçün **hər pəncərə üçün riyazi olaraq eynidir**, `label_rule` seçimi
heç bir nəticəni dəyişmir. Bunu hesabatda bir cümlə ilə yaz — pulsuz
metodoloji xal.

---

## 2. Gözlənilməyən tapıntı №1 — quyular tamamilə ayrıdır

```
Hidrat (Event-9) quyuları : 15  →  WELL-00010, 14, 15, 16, 20, 33..42
Normal Operation quyuları :  9  →  WELL-00001..00008, 00019
KƏSİŞMƏ                   :  ∅  (heç bir ümumi quyu yoxdur)
```

**Nəticələr:**

1. **Sinif quyu kimliyindən mükəmməl proqnozlaşdırıla bilir.** Model "bu hansı
   quyudur" sualını həll edərək etiketi tapa bilər — sensor offsetləri, baza
   təzyiqləri, kalibrasiya fərqləri bunun üçün kifayətdir. Bu, validation-da
   əla, real həyatda mənasız nəticə deməkdir.
   → **W1.8-dəki "normalisation trap" indi opsional deyil, məcburidir:**
   normalizasiya **per-well** aparılmalıdır (hər quyunun öz statistikası ilə),
   qlobal deyil. Əks halda layihənin bütün nəticələri şübhəlidir.

2. **Grouped CV strukturu dəyişməlidir.** `StratifiedGroupKFold`-u 24 quyunun
   birləşməsi üzərində işlətmək fold-ları nəzarətsiz edir: bir fold-a 5 normal
   quyu, digərinə 1 düşə bilər. Doğru quruluş — **iki müstəqil split**:
   pozitiv quyular üzərində bir split, normal quyular üzərində ayrı split,
   sonra fold-lar cütləşdirilir.

3. `fold_report()`-a **`test_normal_hours` sütunu əlavə olunmalıdır** (hazırda
   spesifikasiyada yoxdur). False alarm/saat metrikinin məxrəcidir; onsuz bir
   fold-un threshold kalibrləməyə yetəri qədər normal saatı olub-olmadığını
   bilmirik.

### Normal saatlar necə paylanıb — A.6-nın narahatlığı təsdiqləndi

```
WELL-00002: 1220.4 h     WELL-00001:  555.6 h     WELL-00006:  674.8 h
WELL-00005:  353.1 h     WELL-00008:  337.0 h     WELL-00003:  154.8 h
WELL-00019:   39.8 h     WELL-00004:   35.8 h     WELL-00007:    6.0 h
                                              CƏM: 3,377.3 h
```

Addendum A.6 `target_far = 1/100 h`-i seçərkən *"real Normal-hours per fold are
limited once split by well"* deyirdi. Reallıq daha kəskindir: **cəmi 9 normal
quyu var**, və saatlar son dərəcə qeyri-bərabərdir. 5 fold-a bölsək, test
tərəfinə WELL-00007 (6 h) + WELL-00019 (39.8 h) düşən bir fold **45.8 saat**
alır — 100 saatda 1 yalan həyəcanı ölçmək üçün bu **kifayət deyil**.

→ **Tövsiyə:** normal quyular fold-lara saatlara görə balanslaşdırılmış şəkildə
paylansın (hər test fold-unda ≥300 h), 3–4 fold-la.

---

## 3. Gözlənilməyən tapıntı №2 — pozitiv hadisələr çox azdır

Real Event-9 qovluğundakı 57 instansın `class` sütununun faktiki tərkibi:

| Etiket kodu | Müşahidə sayı | Neçə instansda var |
|---|---|---|
| `0` (normal) | 2,216,977 | 57/57 |
| `109` (**transient** — hidrat başlayır) | 195,443 | **14/57** |
| `9` (**established blockage**) | 17,752 | **3/57** |
| `NaN` (etiketlənməmiş) | ~205,000 | 57/57 (median %11.2, max %50.4) |

**43 instansda nə transient, nə də blokaj var** — yalnız normal + etiketlənməmiş
müşahidələr. Onlar praktiki olaraq "hidrata meylli quyulardan normal
qeydiyyatlar"dır, pozitiv nümunə deyil.

### Pozitiv hadisələr quyulara görə

```
Transient (109) olan:  14 instans  →  cəmi 7 quyu
   WELL-00042: 5    WELL-00014: 3    WELL-00037: 2
   WELL-00016: 1    WELL-00020: 1    WELL-00040: 1    WELL-00041: 1

Blokaj (9) olan     :   3 instans  →  cəmi 3 quyu
   WELL-00014, WELL-00016, WELL-00042
```

### 🔴 Bunun ən ciddi nəticəsi: headline lead time metriki hesablana bilmir

DL8.3: `median_lead_time = (established-blockage onset) − (first reliable flag)`.
`failure_time` yalnız kod `9` olan instanslarda mövcuddur → **3 instans, 3 quyu**.

Grouped CV-də bu o deməkdir ki, əksər fold-ların test tərəfində **sıfır**
`failure_time` olacaq. Yəni layihənin abstract-a çıxacaq baş rəqəmi real data
üzərində n=3 ilə hesablanır — və çox fold-da ümumiyyətlə hesablanmır.

**Simulyasiya instansları isə tam əksinə:** 150/150-də hər üç faza var
(0 → 109 → 9, 3.19M established müşahidə). Yəni simulyasiya datası real datanın
**ən nadir hissəsini sistematik olaraq həddindən artıq təmsil edir.**
Bu, `real_only` vs `real_plus_sim` müqayisəsini (Result 1) daha maraqlı edir —
indi fərqin *niyə* yarandığına dair konkret, ölçülmüş izahımız var.

---

## 4. Gözlənilməyən tapıntı №3 — `window_size=60s` iki tərtib səhvdir

DL2.1b məhz bunun yoxlanmasını tələb edirdi. Real transient fazalarının
davametməsi (14 hadisə):

```
median = 12,332 s  (~3.4 saat)      q25 =  4,648 s      q75 = 16,730 s
min    =    182 s  (~3 dəqiqə)      max = 49,161 s (~13.7 saat)
60 saniyədən qısa olan: 0/14
```

1 Hz-də 60 saniyəlik pəncərə tipik transient fazanın **%0.5-ni** görür. Bu
pəncərə ilə model faktiki olaraq anlıq şəkil üzərində qərar verir.

**Tövsiyə — TCN arxitekturasını dəyişmədən problemi həll edən variant:**
siqnalı **1 Hz → 1/30 Hz-ə decimate et** (30 saniyəlik ortalama), pəncərəni
**60 sample** saxla. Nəticə: hər pəncərə **30 dəqiqəni** əhatə edir, `W=60`
qalır, TCN-in receptive field-i (61) hələ də tam pəncərəni örtür, `stride=5`
sample = 2.5 dəqiqə. Heç bir model kodu dəyişmir, yalnız `build_windows`-a
decimation əlavə olunur.

Alternativ (bahalı): `window_size`-ı 1800-ə qaldır və TCN-ə əlavə dilated
bloklar ver — receptive field 61 → ≥1800 olmalıdır, bu isə modeli və yaddaş
tələbini xeyli böyüdür.

---

## 5. Gözlənilməyən tapıntı №4 — etiketlərdə NaN var, spesifikasiyada yoxdur

**57/57 instansda `class` sütununda NaN var** (median %11.2, maksimum %50.4).
3W konvensiyasında bu "annotator bu aralığı təsnif etməyib" deməkdir — nə
normal, nə hadisə.

Spesifikasiya bunu heç yerdə müzakirə etmir. `label_window()` xam kodları
gözləyir və NaN-la nə edəcəyi tərif olunmayıb.

**Qərar tələb olunur (M1 + M4):** etiket aralığında NaN olan pəncərələr
atılsınmı, yoxsa NaN normal sayılsınmı? Tövsiyə: **atılsın** (ehtiyatlı seçim)
və neçə pəncərənin atıldığı hesabatlansın. NaN-ı normal saymaq false alarm
məxrəcini süni şəkildə şişirdir.

---

## 6. Nə dəyişməlidir — konkret siyahı

| # | Nə | Kim | Prioritet |
|---|---|---|---|
| 1 | Headline metrikin yenidən tərifi (§3 — komanda qərarı lazımdır) | hamı | 🔴 bu gün |
| 2 | `build_windows`-a decimation (1/30 Hz) + `window_size` yenidən | M1 | 🔴 |
| 3 | NaN etiket siyasəti + atılan pəncərə sayının hesabatlanması | M1 + M5 | 🔴 |
| 4 | Per-well normalizasiya (quyular ayrı olduğu üçün məcburi) | M1 | 🔴 |
| 5 | Split: pozitiv və normal quyular üçün **iki müstəqil** grouped split | M2 | 🔴 |
| 6 | `fold_report()`-a `test_normal_hours` sütunu | M2 | 🟡 |
| 7 | `n_splits` 5 → **3** (7 pozitiv quyu var, 5 fold çox incədir) | M2 | 🟡 |
| 8 | §2.2 floor placeholder → **32,135** | M2/M5 | 🟢 hazır |
| 9 | A.5 nəticəsi hesabata: monotonluq 57/57 | M2/M5 | 🟢 hazır |
| 10 | Discussion: sim data blokaj fazasını sistematik şişirdir | M5 | 🟢 |

---

## 7. Headline metrik üçün üç variant

**Variant A — transient onset-ə görə lead time (tövsiyə olunan).**
`failure_time` blokaj başlanğıcı yerinə **annotasiya edilmiş transient
başlanğıcı** kimi tərif olunur. 14 hadisədə mövcuddur (3 yerinə). Suala
çevrilir: *"annotator hadisəni işarələməzdən nə qədər əvvəl siqnal veririk?"* —
bu, hələ də tam qanuni early-warning sualıdır və bütün 7 quyunu istifadə edir.
Blokaja görə lead time ikinci dərəcəli metrik kimi, **n=3 açıq yazılmaqla**
verilir.

**Variant B — lead time-ı headline-dan çıxar.**
Baş rəqəmlər: **event recall @ matched FAR** + **PR-AUC** (14 hadisə üzərində).
Lead time yalnız limitasiya bölməsində. Ən ehtiyatlı, ən az maraqlı.

**Variant C — lead time simulyasiya instanslarında ölçülür.**
Statement bunu praktiki olaraq qadağan edir (*"simulated data is a training-time
augmentation, never a test-time substitute"*). Yalnız **aydın etiketlənmiş
əlavə analiz** kimi məqbuldur, headline kimi yox.

Hansı variant seçilirsə, `src/eval/alarm.py` və `metrics.py`-a toxunmadan
işləyir — dəyişən yalnız `build_cache`-in `failure_time`-ı necə doldurmasıdır.
Yəni qərar ucuzdur, amma **indi** verilməlidir.

**Verilmiş qərar (1 Sen):** Variant A. `build_cache` `failure_time`-ı transient
onset kimi yazır, `blockage_time`-ı ayrıca saxlayır.

---

## 8. Gözlənilməyən tapıntı №5 — hər quyu fərqli kanalları ölçür

Bu, ilk cache build-i işlətdikdən sonra çıxdı: **801 instansdan 289-u sıfır
istifadəyə yararlı pəncərə verdi.** Bütün WELL-00002 (209 instans) və
WELL-00008 (57 instans) — yəni 9 normal quyudan 2-si və 3,646 normal saatdan
**1,472-si** — tamamilə itdi.

Səbəb kod bug-ı deyil. `max_missing_frac=0.5` ilə bütün instansların birləşməsi
üzərində fit edilən kanal siyahısı **cəmi 5 kanal** saxladı
(`P-JUS-CKGL, P-MON-CKP, P-TPT, T-JUS-CKP, T-TPT`) — çünki 27 kanaldan 9-u
dataset boyu **%100 missing**-dir (`ESTADO-*` klapan vəziyyətləri, `P-JUS-BS`,
`PT-P`, `QBS`, `P-MON-CKGL`, `P-MON-SDV-P`), bir neçəsi isə həddi bir tük fərqlə
keçmir (`QGL` 0.569, `P-PDG` 0.584, `P-ANULAR` 0.656 — halbuki `P-PDG` daimi
quyudibi manometrdir, fiziki olaraq ən maraqlı kanallardan biri).

Amma əsas problem başqadır: **missingness qlobal deyil, quyuya xasdır.**

```
WELL-00002-də mövcud olan kanallar : P-ANULAR, P-PDG, P-TPT, T-TPT
WELL-00008-də mövcud olan kanallar : P-JUS-CKGL
```

Qlobal 5 kanaldan WELL-00002-də 3-ü, WELL-00008-də 4-ü heç yoxdur → pəncərənin
mask ortalaması `min_valid_frac=0.5`-dən aşağı düşür → bütün pəncərələr atılır.

### Ölçülmüş trade-off (`tools/channel_availability.py`)

Kanal dəsti = "normal quyuların ən azı K-sında mövcud olan kanallar":

| Dəst | Kanal | Pozitiv quyu (7-dən) | Normal quyu (9-dan) | Normal saat |
|---|---|---|---|---|
| K≥8 → `P-TPT, T-TPT` | **2** | **7/7** | **8/9** | **3,040 h** |
| K≥7 | 7 | 6/7 | 6/9 | 1,467 h |
| K≥6 | 13 | 5/7 | 2/9 | 1,230 h |
| K≥5 | 17 | 2/7 | 2/9 | 1,230 h |
| K≥1 (hamısı) | 22 | 0/7 | 0/9 | 0 h |

Yəni **kanal sayı ilə quyu əhatəsi bir-birinə ziddir** və 22 kanalın hamısını
tələb etmək **heç bir quyu** buraxmır.

### Qərar

**Əsas arm: 2 kanal (`P-TPT`, `T-TPT`)** + mask kanalları. Bütün 7 pozitiv quyu
və 3,040 normal saat qalır. Bu iki kanal TPT-nin (temperatur/təzyiq
transduseri) təzyiq və temperaturudur — hidratın fiziki imzasının məhz olduğu
yer, yəni bu, məcburiyyətdən doğan bəraət deyil, fiziki olaraq düzgün cütdür.
100 saatda 1 yalan həyəcan büdcəsini kalibrləmək üçün normal saat lazımdır,
ona görə 3,040 h vs 1,467 h fərqi həlledicidir.

**İkinci arm: 7 kanal** (K≥7), həssaslıq analizi kimi — *"daha çox kanal, daha
az quyu: hansı qazanır?"* Hesabatda güclü Discussion paraqrafı, itki deyil.

`build_cache` artıq `--channels` seçimini qəbul edir, ona görə hər iki arm
kod dəyişmədən qurulur:

```bash
python -m src.data.build_cache --root data/3W/dataset --out data/cache \
    --channels P-TPT,T-TPT
python -m src.data.build_cache --root data/3W/dataset --out data/cache_7ch \
    --channels ESTADO-PXO,ESTADO-W2,ESTADO-XO,P-MON-CKP,P-PDG,P-TPT,T-TPT
```

### Hesabata mütləq düşməli

Bu tapıntı **§2.3-ün missingness müzakirəsini əvəz etmir, kəskinləşdirir**:
statement dataset boyu missingness-dən danışır, amma real problem *quyular
arasında instrumentasiya fərqidir*. Bu, "unseen well" ümumiləşdirməsinin niyə
çətin olduğuna dair ayrıca, dürüst bir izahdır və Limitations bölməsinə düşür.
