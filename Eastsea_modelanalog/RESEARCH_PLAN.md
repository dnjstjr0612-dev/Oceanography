# 연구 계획 — 동해(EJS) 여름 SSTA 분석 예측: Source-Index Analog

작성일: 2026-06-02
선행 연구: `../nw-pacific-temp-prediction/` (NW Pacific 연안 SST 계절 예측, CESM-HR model analog)
기반 논문: Joh et al. (2024), *East/Japan Sea summer SST predictability from prior-winter atmosphere–ocean coupling*

---

## 0. 한 줄 요약과 핵심 가설

> **동해 여름 SST는 정말로 다계절(multi-season) 예측성이 있는가, 아니면 기존 analog이 예측성의 source를 잘못 잡은 것인가?**

이 질문을 검증 가능한 형태로 좁히면 다음과 같다.

- **귀무가설 H0 (동해는 진짜 낮은 예측성):** 이전 SST-패턴 analog이 동해에서 실패한 것은 동해가 근본적으로 낮은 persistence·낮은 예측성을 갖기 때문이다. 어떤 source index를 추가해도 baseline(persistence, SST-패턴 analog)을 유의하게 넘지 못한다.
- **대립가설 H1 (source를 잘못 봤다):** 동해의 여름 예측성은 **현재 SST 패턴**이 아니라 **전 겨울 원격 강제(Barents 고기압 → ECS 동풍 → 북향 Ekman/해양 이류)**에 들어 있다. SST 패턴이 아니라 **이 source 상태**로 analog을 고르면 baseline을 유의하게 넘는 skill이 회복된다.

**이 연구의 칼날은 두 analog의 직접 대비다:**

| | 매칭 기준 | 묻는 질문 |
|---|---|---|
| **D0 (baseline)** | 초기 SST 패턴 유사도 (Amaya RMSD) | "지금 SST가 비슷한 해" — 선행 연구가 동해에서 실패한 방식 |
| **D1 (source-index)** | 전 겨울 source 상태(BHI, ECSW) 유사도 | "여름 동해 SST를 만들 수 있는 **이전 겨울 대기–해양 상태**가 비슷한 해" |

→ **D1이 D0와 persistence를 유의하게 이긴다면 H1 채택**(source를 잘못 봤던 것), **못 이기면 H0 채택**(동해는 진짜 낮은 예측성). 둘 다 출판 가치가 있는 결론이다.

---

## 1. 배경

### 1-1. 선행 연구가 남긴 출발점
`nw-pacific-temp-prediction`의 CESM-HR model-analog 프레임워크는 알래스카·북동태평양 연안에서 NPMM/PDOe source를 추가해 baseline 대비 skill 향상을 보였다. 그러나 **동해(East/Japan Sea)는 명확한 실패 사례**였다 — `2026_spring_poster/make_east_sea_failure_fig.py`는 동해의 낮은 persistence·baseline ACC를 객관적 증거로 제시한다. 단, 그 실패는 **SST 패턴 유사도 기반 analog**의 실패였다.

### 1-2. Joh et al. (2024)의 주장
- 동해 여름 SST는 **이전 겨울 대기–해양 결합**과 연결될 수 있다.
- SPEAR seasonal prediction system에서 여름~늦가을 동해 SST event에 skill이 있으며, 특히 **겨울 초기화 long-lead forecast**에서 추가 skill이 나타난다.
- 물리 메커니즘: 겨울 Barents Sea 관련 대기 상태 → 동아시아/동중국해의 지속적 surface wind anomaly → northward Ekman heat transport anomaly → 여름 동해 SST.

→ 즉 Joh et al.은 "동해는 예측 불가"가 아니라 "예측성의 source가 현재 SST가 아닌 **원격 겨울 강제**에 있다"고 본다. 본 연구는 이 주장을 analog 틀에서 독립적으로 검증한다.

### 1-3. 인과 사슬 (가설된 pathway)
```
겨울 Barents Sea Z500 anomaly (BHI)
        │
        ▼
동아시아 / 동중국해 wind anomaly (ECS easterly, ECSW)
        │
        ▼
Northward Ekman heat transport / 해양 이류 (EKT)
        │
        ▼
대한해협 / Tsushima 유입 경로 (OHC, SSH)
        │
        ▼
여름(JAS) 동해 SSTA  ← target
```
각 화살표는 검증 대상이다. **Phase 0에서 화살표가 끊겨 있으면 analog 자체가 성립하지 않는다.**

---

## 2. 도메인·타깃·변수 정의

### 2-1. 도메인
- **Target 동해(EJS):** 35°N–45°N, 127°E–142°E
- **ECS easterly 영역:** 25°N–35°N, 120°E–130°E
- **Barents Sea Z500 영역:** 65°N–80°N, 20°E–60°E

### 2-2. 타깃
- `Y_y = JAS(7–9월) 평균 EJS area-mean SSTA`, 각 연도 y에 대해 산출.
- **반드시 선형 detrend 적용** — 동해는 강한 온난화 추세가 있어, detrend하지 않으면 "warm event"가 추세와 혼동되고 analog skill이 추세 매칭으로 거짓 부풀려진다. (선행 연구도 detrend 사용)

우선순위(사전 등록된 primary → secondary):
1. **JAS 동해 area-mean SSTA** ← **primary metric**
2. JAS 동해 warm-event 확률 (event skill)
3. JAS 동해 EOF1 PC
4. gridpoint pattern ACC

### 2-3. Predictors (모두 z-score 표준화 후 거리 계산에 투입)

| 기호 | 정의 | 계절 | 자료 |
|---|---|---|---|
| **BHI** | Barents Sea High index = NDJ 평균 Z500′ (65–80°N, 20–60°E) | NDJ (y-1년 11,12월 + y년 1월) | ERA5 |
| **ECSW** | East China Sea easterly = −u10′ DJF 평균 (ECS 영역). 동풍이 u10<0이므로 부호 뒤집음 | DJF | ERA5 |
| **EKT** | Ekman transport proxy. taux = ρ_air·C_D·\|U\|·u, M = −taux/(ρ_w·f). M>0이면 northward | DJF (또는 NDJ) | ERA5 |
| **OHC** | 0–100 m 적분 열용량 = ρ·c_p·∫T′(z)dz. 영역: ECS / 대한해협 / 남부 동해 | 겨울~봄 | GLORYS12 (optional) |
| **SSH** | sea surface height anomaly, 대한해협·남부 동해 | 겨울~봄 | GLORYS12 (optional) |

---

## 3. 거리 지표 (analog 후보 정렬 기준)

모든 predictor는 z-score 표준화. K개 최근접 후보의 타깃 평균을 예측값으로 사용.

```
D0 (baseline, SST 패턴)   : RMSD[ SSTA_EJS,init(y),  SSTA_EJS,init(i) ]
                            init = NDJ / DJF / MAM → JAS EJS SSTA

D1 (source-index)         : sqrt( ([BHI_y−BHI_i]² + [ECSW_y−ECSW_i]²) / 2 )

D2 (+Ekman, 물리 강화)     : sqrt( ([BHI]² + [ECSW]² + [EKT]²) / 3 )
                            ※ EKT는 ECSW와 같은 바람에서 파생 → 공선성 주의(§6)

D3 (season-gated)         : D1 을 같은 source 계절 조건 안에서만 적용
                            NDJ→JAS, DJF→JAS, MAM→JAS 각각 분리

D4 (EOF-filtered target)  : 타깃을 area-mean 대신 EJS JAS SSTA의 PC1로 교체 (확장)

D5 (SVD-mode)             : X=NDJ wind-speed anomaly, Y=JAS EJS SSTA 의 SVD1
                            wind expansion coefficient A 사용
                            sqrt( ([A_y−A_i]² + [BHI]² + [ECSW]²) / 3 )
```

**해석상 핵심:** D1은 "지금 SST가 비슷한 해"가 아니라 "여름 동해 SST를 만들 수 있는 **이전 겨울 source 상태가 비슷한 해**"를 찾는다. 이 점이 D0와의 근본적 차이다.

---

## 4. 분석 단계 — Decision Gate가 있는 단계적 진행

> 비싼 자료(GLORYS, CESM-HR)로 가기 전에, 싼 자료(OISST+ERA5)로 **"신호가 있긴 한가"**를 먼저 판정한다. 각 Phase 끝에 진행/중단 게이트가 있다.

### Phase 0 — 신호 진단 (가장 싸고 가장 결정적) 🟢 여기부터 시작
**자료:** OISST(월별) + ERA5(월별)만.
**할 일:**
1. BHI, ECSW, target JAS EJS SSTA 시계열 구축 (모두 detrend·표준화).
2. **Lagged 상관/회귀:** BHI(NDJ), ECSW(DJF) → JAS EJS SSTA 상관계수와 회귀 패턴 맵.
3. **Predictor 공선성 행렬:** BHI vs ECSW vs EKT 상관 (§6 — 셋이 같은 모드면 D1·D2가 사실상 1개 변수).
4. 인과 사슬의 각 화살표(BHI→ECSW, ECSW→여름SST) 개별 검증.

**Decision Gate 0:** 전 겨울 source → 여름 SSTA의 lagged 상관이 통계적으로 유의하지 않으면(부트스트랩/Fisher-Z CI), analog의 전제가 깨진 것 → **H0 쪽 근거를 정리하고 Phase 1은 음성 결과 확인용으로만 진행.**

### Phase 1 — 관측 leave-one-out analog (여전히 싸다)
**자료:** OISST + ERA5.
**할 일:**
1. **D0(SST 패턴), D1(source-index), D3(season-gated)** 각각으로 leave-one-out analog 예측.
2. Baseline 3종과 비교: ① climatology, ② init-season SSTA persistence, ③ D0(SST-패턴 analog, = 선행 연구가 동해에서 쓴 방식).
3. K(analog 개수) 민감도: 사전에 한 규칙으로 고정(예: K=고정값 또는 RMSE 최소화 LOO 내부 선택)하고 나머지는 보조.

**Decision Gate 1 (이 연구의 핵심 판정):**
- **D1 > {persistence, D0}** 가 유의하면 → **H1: source를 잘못 봤던 것.** 동해는 SST가 아니라 원격 겨울 강제로 예측 가능.
- 차이가 없으면 → **H0: 동해는 진짜 낮은 예측성.** (선행 연구 결론 강화)

### Phase 2 — 물리 경로 검증 (중간 비용)
**자료:** + GLORYS12.
**할 일:**
1. **D2(+Ekman), D4(+OHC/SSH)** 추가 — oceanic pathway를 더 직접 건드림.
2. 북향 Ekman transport·대한해협 유입 anomaly가 겨울→여름 지속되는지 GLORYS로 확인.
3. OHC(0–100m, ECS/대한해협/남부 동해) 선행성 검증.

### Phase 3 — CESM-HR model analog (가장 비쌈)
**자료:** CESM-HR (SST 필수, 가능하면 u10/v10/SLP/Z500/SSH/ocean T).
순서는 **반드시** 아래대로:
1. **Model-internal test:** CESM-HR 내부에서 leave-one-year-out. *"CESM-HR 자체에서 Barents/ECS source가 JAS EJS SSTA를 예측하는가?"* — 모델 안에서조차 안 되면 관측에서 기대할 수 없다. **이게 먼저다.**
2. **Observation-to-model analog:** X_OBS,y → X_CESM,i → Y_CESM,i.
   - **모델과 관측을 각각 따로 표준화**해야 한다 (model bias 때문). raw 값끼리 비교하면 "가까운 analog"이 물리적 근접이 아니라 단순 bias matching이 된다.
   - `z_OBS = (X_OBS − μ_OBS)/σ_OBS`, `z_CESM = (X_CESM − μ_CESM)/σ_CESM`.

---

## 5. 평가 지표

**연속(continuous):**
- `ACC = corr(Ŷ, Y)` ← primary
- `RMSE skill = 1 − RMSE_exp / RMSE_baseline`

**이벤트(event):**
- Brier Skill Score
- ROC-AUC
- Hit Rate, False Alarm Rate

모든 skill에 **부트스트랩 95% CI**를 붙인다 (선행 연구 관례, 작은 N 때문에 필수).

---

## 6. 통계적 위험요소 (사전에 명시 — 무시하면 거짓 양성)

1. **작은 표본:** 관측 JAS 타깃은 ~43개(OISST 1982–2025). LOO analog의 K-최근접은 ~42개 후보에서 고름. → CI 필수, 단일 ACC 수치 과신 금지. CESM-HR 대형 library가 필요한 근본 이유.
2. **추세 오염:** detrend 필수(§2-2). 안 하면 warm-event skill이 추세 매칭으로 부풀려짐.
3. **Predictor 공선성:** BHI·ECSW·EKT가 같은 대규모 모드의 표현이면 D1·D2의 "정보량"이 중복. Phase 0에서 상관행렬로 먼저 확인. EKT는 ECSW와 동일 바람 파생이라 특히 위험.
4. **다중 비교(garden of forking paths):** D0–D5 × 4 타깃 × 3 계절 × K값 = 많은 조합. **primary metric(JAS area-mean SSTA ACC, D1, 고정 K)을 사전 등록**하고 나머지는 보조·탐색으로 명시.
5. **정직한 baseline:** source-index analog은 climatology뿐 아니라 **persistence와 D0(SST-패턴 analog)까지** 이겨야 "원격 source가 진짜"라고 주장 가능.

---

## 7. 확장 (Phase 1–2 결과가 좋으면)
- **확장 1: EOF-filtered target** — area-mean 대신 EJS JAS SSTA PC1을 타깃으로. basin-scale warming mode와 frontal/eddy noise 분리.
- **확장 2: SVD-mode analog (D5)** — Joh et al. 방식에 더 가깝게, NDJ wind-speed anomaly와 JAS EJS SSTA의 SVD1 expansion coefficient 사용.
- **확장 3: GLORYS OHC/SSH 거리항 추가** — oceanic pathway 직접 건드리기.
- **확장 4: CESM-HR model analog** — u10/Z500/SSH/OHC까지 확장. 단 순서는 §4 Phase 3 고정.

---

## 8. 권장 시작점 (easy-first)
**Phase 0 → Phase 1을 OISST(월별) + ERA5(월별)만으로 먼저 끝낸다.** 두 자료 모두 표준 격자·소용량이고 CDS/PSL에서 바로 받을 수 있다. 여기서 Decision Gate 0·1의 답(H0인지 H1인지)이 사실상 결정된다. GLORYS12·CESM-HR는 이 답이 긍정적일 때만 투입한다.

선행 연구의 재사용 가능 코드: `../nw-pacific-temp-prediction/repo/notebooks_v2/ma_core.py`
(`amaya_rmsd_matrix`, `linear_detrend`, `temporal_acc_series`, `select_top_analogs`, `compute_oisst_ssta` 등 — D0·detrend·ACC·analog 선택 로직을 그대로 가져다 쓸 수 있다.)

---

## 9. (2026-06-03 갱신) 결과 요약과 두 입장으로의 분기

관측·모델 분석을 거쳐(노트북 01–09, `FINDINGS.md`) 큰 그림이 섰다:
**관측 = Barents 원격 강제(여름 reemergence), CESM-HR = 북태평양 경유(국소 persistence형).**
관측 Barents 신호는 북태평양 mode를 통제해도 살아남는다(partial corr +0.55) → 독립적 진짜 source.

여기서 연구는 질문이 다른 **두 입장**으로 갈린다. (배타적 아님 — 잇는 서사가 가장 강함)

### 입장 1 — Model-analog 분석가 ("동해를 어떻게 예측하나")
모델은 *주어진 도구*. model-analog은 **모델의 source를 물려받으므로**, obs 이론(Barents)이 아니라
**모델이 실제로 가진 source(ECS + 북태평양)**로 거리함수를 설계한다.

- **거리함수 비교 실험:**
  - `D1 = D(ECS)`  (하류 경로는 obs·model 동부호로 살아있음)
  - `D2 = D(ECS) + D(NP)`  (모델 진단 source 추가)
  - (Barents는 모델 거리에서 제외 — 모델 내부에서 EJS와 무관하므로 부적합)
- **필수 교차검증(out-of-sample):** obs에서 NP→EJS는 +0.20으로 강하지 않다.
  따라서 **모델에서 고른 NP source box가 관측에서도 여름 EJS SST를 예측하는지** 먼저 확인.
  안 되면 NP 기반 analog은 모델 내부 구조만 exploit할 뿐 obs 예측엔 무용.
- 지표는 `area_mean_acc`(격자점 ACC의 area-mean; 단일지수 ACC보다 매끈).
- **deliverable:** 동해 analog 예보 시스템 + 정직한 skill 평가. (Z3 불필요.)

### 입장 2 — Model 개발자 ("왜 안 맞나")
모델을 *검증 대상*으로. 관측 Barents 신호가 북태평양 통제 후에도 살아남으므로:

- **핵심 진술:** *CESM-HR piControl analog library는 관측에서 보이는 modern Barents-related
  EJS summer SST teleconnection을 결여한다.* (Leg: Barents→ECS 0.04 vs obs 0.28에서 끊김)
- **물어볼 질문:** Barents–EJS teleconnection은
  **(a) stationary internal mode**인가, **(b) 최근 수십 년에 강화된 forced teleconnection**인가?
  - forced 가설 검증(값쌈, 우선): **ERA5 장기간(1940–)으로 obs에서 Barents→EJS가 최근 수십 년에
    강해졌는지**(early vs late, sliding window). 강해졌다면 현대 Arctic(해빙 감소·WACE) 의존 시사.
  - 추가: transient(역사+미래) run에서 시기별로 teleconnection이 나타나는지 (단 d651030은 저해상도).
- **중위도 teleconnection bias 가설:** 모델이 **Barents blocking → East Asia/ECS로 이어지는
  Rossby wave train / stationary wave response**를 충분히 만들지 못할 수 있다.
  - 검증: 모델에서 Barents 고기압 composite에 대한 Z500/wave-activity 반응 진단.
    (PSL≠Z500 우려 시 모델 **Z3(500hPa)** 부분 다운로드 — `download_cesm_z3.py`.)
- **deliverable:** "CESM-HR이 관측 Barents→동해 teleconnection을 결여 + 그 이유(forced vs bias)"
  + "model-analog은 모델의 source를 물려받는다"는 방법론적 경고.

### 두 입장의 결정 차이
| | 입장 1 (analog) | 입장 2 (개발자) |
|---|---|---|
| Barents predictor | 모델 거리에서 제외 | 왜 약한지 *진단 대상* |
| 북태평양 source | 주력 사용 | 모델의 실제 source로 진단 |
| Z3 다운로드 | 불필요 | (bias 파고들 때만) |
| 다음 데이터 | 없음(보유분 충분) | ERA5 1940– / transient / Z3 |

### 즉시 착수 (입장 2의 forced 질문 — 가장 값싸고 신선)
**ERA5 1940–2024로 obs Barents→EJS의 시기 의존성 확인.** SST는 OISST(1981–)가 짧으므로
장기 SST(ERSSTv5 또는 HadISST)로 EJS 여름 SST를 확장. early/late·sliding-window 상관 → forced 여부.

### (2026-06-03) forced 테스트 결과 — bias 가설로 기움
ERSSTv5 + ERA5(1941–2023), sliding 25yr 상관:
- **Barents→동해: 거의 stationary** (early 1941–82 +0.33 / late 1983–2023 +0.37; 창 평균 ~0.3, 단조강화 없음).
  → "현대 Arctic forced" 가설 약함. **stationary internal mode**에 가까움.
- **ECS 동풍→동해: 1980s 전후 0→~0.5 급변**(하류 결합 최근 강화; 별개 현상).
- caveat: ERSST 2° 동해엔 거칠고 1940s 관측 희박(early 신뢰도↓).
- **함의:** 모델의 Barents→동해 부재는 1850 control의 Arctic 부재보다 **모델 teleconnection bias**
  (Barents 블로킹→동아시아 Rossby wave train/stationary wave 결여)일 가능성이 큼.
  → 다음: 모델 Z500(Z3) composite로 wave-train 반응 진단 (obs vs model).

### (2026-06-03) 입장 2 일반화 — CESM-HR만의 bias냐, HR coupled 전반이냐
관측 Barents→동해는 진짜·독립·stationary인데 CESM-HR은 wave-train bias로 결여. **이게 CESM-HR 특이 문제인지,
고해상도 coupled model 전반의 source-transfer 문제인지** CMIP6 HighResMIP로 검정.

- **후보(GC zarr에 control-1950/hist-1950 + 대기변수 보유):** HadGEM3-GC31-HM, HadGEM3-GC31-MM,
  ECMWF-IFS-HR, GFDL-CM4C192, CNRM-CM6-1-HR. (EC-Earth3P-HR·MPI-XR은 GC에 대기변수 없음.)
- **핵심 통찰:** Barents→동아시아 teleconnection은 **순수 대기** 현상 → SST 불필요.
  `psl/zg500`(Barents)+`uas`(ECS)만으로 leg 측정. (GC엔 tos·해양변수 0개라 SST는 ESGF tos 필요 → 후보만 추후.)
- **데이터 접근:** Pangeo GC zarr 영역 lazy. 단 이 위치선 ~130KB/s로 느림(박스/변수당 ~11분) → 다십년 전체로
  백그라운드 screening(`scripts/screen_highresmip_atm.py` → `figures/highresmip_atm_screening.csv`).
- **측정/판정:** Barents→ECS = corr(BHI_NDJ, ECSW_DJF). 다른 HR 모델도 ≈0 → **HR coupled 전반의 bias**(일반성↑,
  강한 결론). 일부 양수 → CESM-HR 특이/모델 의존. (psl≈Z500은 obs 0.80 검증; 모델에서도 확인 예정.)
- **통과(teleconnection 있음) 모델만** ESGF tos로 SST leg(ECS→EJS, Barents→EJS, NP→EJS, JAS reemergence) 정밀 검증.

---

## 10. (2026-06-04) 종료 상태 — 정지점

> 위 계획(Phase 0–3, 두 입장)은 **실행 완료**. 상세 결과·정지점 지도는 `FINDINGS.md`(§8 현황 지도)·`README.md`.

- **Phase 0/1 (관측):** Decision Gate 0·1 통과 → **H1 채택**(source를 잘못 봤던 것). D1(source) ≫ D0(SST패턴)·persistence.
- **Phase 3 (CESM-HR):** model-internal + obs-to-model 완료. 모델 source=북태평양(obs 전이 X), 유일 transferable=ECS(+0.168 유의).
- **Phase 2 (GLORYS OHC) — 실행 후 막다른 길로 종결(§7-5):** 위 §2-3·§3·확장 3의 OHC/SSH 경로를 GLORYS12로 검증.
  국소 OHC 기억(domain)·유입 OHC 해양 source(D4) **둘 다 비유의** → 동해 여름 SST엔 해양 예측자 없음(봄 장벽이 SST·OHC 동일).
  ⚠️ 따라서 §2-3 표의 "OHC/SSH (optional)"은 *검토 대상이 아니라 검증 후 배제됨*.
- **입장 2 일반화:** HighResMIP 6모델 Barents→ECS leg **0/6 유의**(전부 obs +0.28 아래) → HR coupled 전반 경향(suggestive).
- **산출물:** 통합 노트북 3개 + `paper/fig1–4.png`(논문용, 캡션 `paper/FIGURES.md`) + `scripts/make_paper_figs.py`.
- **남은 열린 실(아이디어 생기면):** ESGF zg500 wave-train composite(입장 2 정밀 확정) / 다음-겨울 reemergence / HR 다모델 SST leg / 글쓰기.
