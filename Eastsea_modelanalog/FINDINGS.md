# 동해(EJS) 여름 SST 예측성 — 관측 vs CESM-HR model-analog: 종합 결과

작성: 2026-06-03 · 코드: `notebooks/01`–`09`, `scripts/` · 계획: `RESEARCH_PLAN.md`

---

## 0. 한 줄 결론

> 동해 여름 SST는 **관측·모델 모두 예측 가능하지만 그 source가 다르다.**
> **관측 = 전 겨울 원격 Barents 강제**(winter→summer reemergence), **CESM-HR = 북태평양 대기 패턴**(국소 persistence형).
> 따라서 **CESM 기반 model-analog은 동해 예측성의 source를 (관측의 Barents가 아니라) 북태평양으로 잡으며**,
> 모델 library는 관측의 여름 reemergence를 재현하지 못한다. — 이것이 최초 질문
> *"analog이 동해 예측성의 source를 잘못 잡는가?"* 에 대한 답이다.

---

## 1. 질문과 가설

선행 `nw-pacific-temp-prediction`에서 동해는 **SST-패턴 analog의 실패 사례**였다. Joh et al. (2024)는
동해 여름 SST가 전 겨울 Barents 고기압 → 동중국해 동풍 → 북향 Ekman/이류 경로로 예측될 수 있다고 본다.

- **H0:** 동해는 진짜 낮은 예측성. **H1:** 예측성 source가 현재 SST가 아니라 전 겨울 원격 강제.
- 검증: **D1(source-index analog)** vs **D0(SST-패턴 analog)**.

---

## 2. 자료

| 자료 | 변수 | 용도 |
|---|---|---|
| OISST v2.1 (일별→월별 EJS box) | SST | target 동해 SSTA |
| ERA5 월별 (20–142°E) | Z500, msl, u10, v10 | Barents BHI, ECS ECSW |
| ERA5 월별 (북태평양 150–240°E) | msl | obs 북태평양 source |
| CESM-HR B1850 control (d651029) | SST(POP), PSL/TAUX/TAUY(cam.h0) | model-internal + library |

도메인: EJS 35–45°N/127–142°E, ECS 25–35°N/120–130°E, Barents 65–80°N/20–60°E, NP 25–45°N/185–225°E.

**주의(부호·대체):** CAM `TAUX`는 바람 u와 부호 반대(편서풍→TAUX<0) → 모델 ECSW=+TAUX. 모델엔 Z500 tseries가
없어 Barents를 **PSL**로 대체(관측 검증: corr(Z500,SLP)=0.80, BHI_SLP~동해=+0.41 → 대용 타당). → §6 caveat.

---

## 3. 방법

- **인덱스:** BHI=NDJ Barents Z500/PSL anomaly, ECSW=DJF ECS 동풍, NP=NDJ 북태평양 SLP. 모두 detrend·z-score.
- **타깃:** JAS 동해 area-mean SSTA (detrend; 온난화·control drift 제거).
- **analog 거리:** D1=source-index 유클리드, D0=초기 SST 패턴 **Amaya σ-RMSD**(obs·lib 각각 σ 표준화).
- **obs-to-model (진짜 model-analog):** CESM EJS SST를 OISST 격자로 **regrid**(Delaunay barycentric) →
  obs 겨울(DJF) 초기상태를 CESM 모든 겨울과 비교 → **top-17** analog → 그 analog의 **lead L개월 뒤 SST 평균**으로 예측.
- **skill 지표:** ACC. 권장 = **`area_mean_acc`**(격자점별 ACC의 area-mean; `index_acc` 단일지수보다 매끈, n작을때 노이즈↓).
  RMSE skill, event는 BSS·ROC-AUC.

---

## 4. 결과 — 관측 (notebooks 01–04, 08)

| 단계 | 핵심 결과 |
|---|---|
| **01 신호** | BHI(NDJ) r=0.52, ECSW(DJF) r=0.45 → JAS 동해, 둘 다 95% CI 0 배제. BHI⟂ECSW(0.28), 다중회귀 R²=0.37. EKT≈ECSW(0.96) 중복. |
| **02 LOO analog** | **D1 ACC +0.42 ≫ D0 −0.23**(SST패턴 analog), persistence 음/약. ACC(D1)−ACC(D0)=+0.67 CI[+0.22,+1.10] 유의 → **H1**. |
| **03 event** | warm-event AUC: **D1 0.84** vs D0 0.30(반skill). BSS +0.25. AUC차 +0.55 유의. |
| **04 robust+경로** | BHI+ECSW 결합 0.42 최고. 신호는 **겨울 집중**(MAM엔 소멸→persistence 잔재 아님). JAS 동해 EOF1=69%, PC1≈area-mean(r=1.00). 경로 composite 그림. |
| **08 lead 곡선** | source-analog ACC: 봄 음수 → **Aug–Oct +0.3~0.4 스파이크.** persistence: 초봄 0.49 → 여름 음수 붕괴(spring barrier). = **winter→summer reemergence.** |

**Barents는 북태평양과 독립적인 진짜 신호 (partial correlation, n=28):**

| | 값 |
|---|---|
| corr(Barents Z500, 동해) | **+0.52** |
| corr(North Pacific, 동해) | +0.20 |
| corr(Barents, North Pacific) | **−0.08** (거의 무관) |
| **partial corr(Barents, 동해 \| North Pacific 통제)** | **+0.55** |
| partial corr(North Pacific, 동해 \| Barents 통제) | +0.28 |
| partial corr(ECS wind, 동해 \| North Pacific 통제) | +0.41 |

→ 북태평양을 통제해도 Barents→동해가 **그대로 살아남는다**(+0.55). 즉 관측의 Barents 신호는
북태평양 순환의 proxy가 **아니라 독립적인 source**다. (모델은 이 신호를 결여 → §6 불일치가 더 선명해짐.)

**관측 결론:** 동해 여름 예측성의 source는 전 겨울 원격 강제(Barents+ECS)이며, **여름에 특정해서** 발현된다.
Barents는 북태평양과 독립적인 진짜 신호다. robust·물리설명됨.

---

## 5. 결과 — CESM-HR 모델 (notebooks 05–07, 09)

| 단계 | 핵심 결과 |
|---|---|
| **05 model-internal** | D1 source ACC 0.30, **D0 0.36·persistence 0.39로 D1과 비슷**(obs와 다름). BHI~Y=−0.10, ECSW~Y=+0.31(부호교정). drift 민감도 통과. |
| **06 teleconnection 진단** | Leg 분해: **Barents→ECS 바람 0.04(끊김)**, ECS바람→동해 +0.31(살아있음). 모델 실제 source=**북태평양**(NDJ PSL +0.44, 최강 +0.52@30N/150°W). |
| **07 북태평양 source** | D1_NP 0.40, **D1_NP+ECSW 0.42, +Barents 0.43** > D1_Barents 0.30(차 +0.11 CI[+0.01,+0.22] 유의). 단 NP도 persist/D0는 유의하게 못넘음(중복정보). |
| **09 진짜 obs-to-model** | CESM library로 관측예측: baseline(WIDE-local)=persistence형, **08의 여름 스파이크 재현 X**. +NP는 Jul(lead5)만 약간 살림. |

**모델 결론:** CESM-HR 1850 control에서 동해 여름 예측성의 source는 **북태평양 대기 패턴**(Aleutian Low/NPGO 계열)이며,
**Barents 경로는 끊겨 있다**(상류 Barents→ECS 링크 부재). 모델 동해는 국소 persistence형이라 obs의 여름 reemergence가 없다.

---

## 6. 큰 그림

| | 관측 | CESM-HR |
|---|---|---|
| 동해 여름 예측 source | **원격 Barents 겨울 강제** | **북태평양 대기 패턴** |
| 국소 persistence/SST | 무력(여름 음수) | 유효(persist≈source) |
| lead 곡선 | **여름(Aug–Oct) 스파이크** | 매끈한 감소(스파이크 없음) |
| 최적 analog | D1 0.42 ≫ D0 −0.23 | D1_NP 0.42 ≈ persist 0.39 |

**해석:** Barents→동아시아 teleconnection은 **현대 Arctic(해빙 감소)에 의존**할 가능성이 크고(WACE 메커니즘),
1850 perpetual control엔 약하거나 없다. 그래서 CESM 기반 model-analog은 동해 예측성을 북태평양으로 (잘못) 귀속한다.

---

## 7. Caveat (한계)

1. **소표본:** 관측 n=27~28 → 단일지수 ACC 노이즈 큼. → `area_mean_acc` 지표 + 부트스트랩 CI 권장.
2. **in-sample 선택:** 북태평양 박스를 동해 상관으로 골라 ACC 0.40–0.43은 낙관적 상한.
3. **PSL vs Z500 (미해결):** 모델 Barents를 PSL로 쟀다. 관측에선 SLP가 Z500의 좋은 대용이나, **모델에서 그 영역이
   baroclinic하면 PSL이 Z500 신호를 놓칠 수 있음.** → 모델 Z500(=Z3, ~150GB)로 직접 검증 필요(미수행). §8.
4. **1850 control vs 현대:** forced(현대 Arctic) 가설은 transient(d651030, 단 저해상도) 또는 ERA5 장기간으로 별도 검증 가능.

---

## 7-2. 입장 2 심화 — forced냐 bias냐 (2026-06-03)

관측 Barents 신호는 북태평양 통제 후에도 살아남으므로(partial +0.55, §4), "모델이 관측 teleconnection을
결여한다"가 성립. *왜* 결여하나? — 두 가설을 검증.

**forced 테스트 (ERSSTv5+ERA5 1941–2023, sliding 25yr):** Barents→동해는 **거의 stationary**
(early 0.33 / late 0.37, 단조강화 없음) → "현대 Arctic forced" 약함. ECS동풍→동해는 1980s 전후 0→0.5 급변.
→ Barents leg는 stationary internal mode에 가까움 → 모델 결여는 **bias 가능성** 시사.
(caveat: ERSST 2° 거칠고 1940s 관측 희박.)

**wave-train 진단 (모델 Z3 500hPa vs ERA5, 고−저 Barents 겨울 NDJ Z500 composite):**

| Z500 응답(고−저) | Barents | ECS(동아시아) | EJS |
|---|---|---|---|
| 관측 | +110 m | **+9.1 m** | −12.7 m |
| 모델 | +42 m | **−0.1 m** | −3.5 m |

**표면 바람 응답까지 (사슬 완결, ECS 박스 고−저 Barents):**

| | Z500′ | 표면 동서바람′ (동풍<0) |
|---|---|---|
| 관측 | +9.1 m | **u10 = −0.40 m/s (동풍 발생)** |
| 모델 | −0.1 m | eastward-stress ≈ −0.002 N/m² (**≈0, 응답 없음**) |

→ 관측은 Barents 고기압 겨울에 Z500 wave가 **동아시아까지 도달**(+9m)하고 **ECS 동풍(−0.40 m/s)까지 생긴다**.
**모델은 둘 다 ~0** — wave train도, 그 결과인 ECS 동풍도 없다. (Barents 진폭 42 vs 110 정규화해도 동일.)
→ **모델이 Barents→동아시아 Rossby wave train/stationary wave response를 못 만든다(=중위도 teleconnection bias).**
이것이 모델 Barents→ECS-바람 leg=0.04(§5)의 메커니즘적 원인이며, 동해 예측성이 북태평양으로 넘어간 이유.
그림: `fig_forced_teleconnection_sliding.png`, `fig_wavetrain_composite.png`, **`fig_wavetrain_full.png`(2×2 완결).**

**입장 2 결론:** CESM-HR piControl은 관측의 (stationary·진짜) Barents→동해 teleconnection을 **모델 bias로 결여**한다
— Barents 블로킹에서 동아시아로 이어지는 상층 wave train과 그 표면 바람 응답을 만들지 못하기 때문.

## 7-3. 입장 1 — model-analog으로 동해를 예측한다면 (2026-06-03)

**source 전이성 (obs↔model):** ECS 동풍만 양쪽 동부호로 전이(obs +0.45/model +0.31).
Barents는 모델 결여(obs +0.52/model −0.10), 북태평양은 모델 전용(obs +0.20/+0.05 약함, model +0.44).
obs NP 검증 그림 `fig_np_obs_validation.png`: 모델 NP box 위치는 관측에서 비유의.

**obs-to-model analog source 비교 (area_mean_acc, JAS=lead5–7 평균):**

| 실험 | JAS skill |
|---|---|
| WIDE-local(baseline) | +0.01 |
| **+ECS 동풍** | **+0.17** (최고) |
| +NP | +0.13 |
| +ECS+NP | +0.17 (NP 추가효과 ≈0) |
| persistence | +0.09 |

→ **유일하게 쓸모 있는 transferable source = ECS 동풍.** +ECS가 late-summer(Aug–Sep) skill을 살림(lead7 0.24).
NP는 obs로 전이 안 돼 무익. 그림 `fig_obs2model_sources.png`.

**거리함수 D1–D4 테스트 (순수 source 거리, JAS area_mean_acc, n=27):**

| | skill | vs D1 (CI) |
|---|---|---|
| D1=ECS | +0.135 | — |
| D2=ECS+Barents | +0.088 | −0.05 [−0.13,+0.05] 불유의 |
| D3=ECS+NP | +0.178 | +0.04 [−0.16,+0.25] 불유의 |
| D4=ECS+NP+Barents | +0.107 | −0.03 불유의 |

→ **관측 최강 predictor(Barents +0.52)를 넣어도 유의한 도움 없음(점추정 살짝 마이너스).** 모델 library에
Barents→동해 관계가 없어 analog 선택에 노이즈만 더함. NP도 유의 개선 없음(전이X). = **model-analog은 모델이
가진 source(ECS)만 쓸 수 있다**의 시연. (n=27이라 차이는 0과 구별 안 됨 → "유의하게 돕지 않는다".) 그림 `fig_distance_compare.png`.

**fig4 형식 source gain (obs-to-model, domain=WIDE-local 대비, JAS area_mean_acc, 부트스트랩):**

| source gain | JAS | 95% CI | |
|---|---|---|---|
| **+ECS** | +0.168 | **[+0.02,+0.31]** | **유의** |
| +NP | +0.117 | [−0.12,+0.36] | 비유의 |
| +NP+ECS | +0.155 | [−0.04,+0.36] | 비유의 |

→ 진짜 model-analog(obs-to-model)에서 **ECS 동풍만 유의한 source gain**. NP(모델 내부 source +0.44)는 점추정 +0.12지만
CI가 0 포함 = **obs로 신뢰성 전이 안 됨**(부트스트랩 확정). 선행 Alaska fig4에서 NPMM이 하던 역할을 동해선 ECS가 함.
그림 `fig_obs2model_fig4.png` (상단 ACC by lead, 하단 source gain ΔACC + CI). 지표=ACC(상관), RMSE 아님.
**참고(spring barrier):** domain/persistence가 여름에 ~0로 무너지는 건 물리적 봄 예측성 장벽(겨울 SST 기억이 여름 재성층으로 소실).
SST 기반으론 못 막음 → source(ECS)로 우회하거나 subsurface OHC(겨울 열용량은 여름 혼합층 아래 보존) 사용 필요(Phase 2 GLORYS).

**입장 1 결론:** 동해 model-analog의 최선은 **WIDE-local + ECS 동풍**, 여름 skill은 modest(area_mean_acc ~0.2).
천장이 낮은 **근본 이유 = 입장 2의 wave-train bias** (가장 강한 obs source Barents가 모델에 없고, 모델 source NP는 obs로 전이 안 됨).
지표는 `index_acc`(단일지수)보다 매끈한 **`area_mean_acc`**(격자점 ACC의 area-mean) 사용.

## 7-4. 입장 2 일반화 (완료, 2026-06-03) — CESM-HR만의 bias냐, HR coupled 전반이냐

관측 Barents→동해는 진짜·독립·stationary인데 CESM-HR은 wave-train bias로 결여 → **CMIP6 HighResMIP 다모델로
일반성 검정.** 핵심: Barents→동아시아 teleconnection은 순수 대기 현상이라 **SST 없이** `psl/zg500`(Barents)+`uas`(ECS)로
leg 측정 가능 (GC zarr엔 해양변수 0개라 SST는 추후 ESGF tos).

**측정:** Barents→ECS leg = corr(BHI_NDJ, ECSW_DJF=−uas), 각 모델 control-1950/hist-1950 다십년 전체.
`scripts/screen_highresmip_atm.py` → `figures/highresmip_atm_screening.csv`. 종합: `make_highresmip_summary.py` → `fig_highresmip_summary.png`.

| 모델 | n | Barents→ECS r | p (양측) | 판정 |
|---|---|---|---|---|
| **관측 (OISST/ERA5)** | 28 | **+0.28** | 0.149 | (기준; n작아 형식적 비유의) |
| CESM-HR | 269 | +0.04 | 0.51 | 비유의 |
| HadGEM3-GC31-HM | 100 | +0.07 | 0.52 | 비유의 |
| HadGEM3-GC31-MM | 100 | +0.14 | 0.17 | 비유의 |
| ECMWF-IFS-HR | 99 | +0.19 | 0.065 | 비유의 |
| GFDL-CM4C192 | 99 | +0.15 | 0.14 | 비유의 |
| CNRM-CM6-1-HR | 64 | −0.13 | 0.32 | 비유의 |

**결론: 6개 HR coupled 모델 전부 Barents→ECS leg가 약하고 비유의(범위 −0.13~+0.19, 0/6 유의).** 모두 관측 leg(+0.28)
아래. → Barents→동아시아 teleconnection 약화는 **CESM-HR 특이가 아니라 HR coupled 전반의 경향**으로 보임 (입장 2 일반화 지지).

**정직한 한계 (강한 단정 회피):**
1. 관측 leg(+0.28)도 n=28에서 형식적 유의는 아님(p=0.149) → "모델이 *유의한* leg를 못 만든다"고 강하게 말하기 어려움.
   단 관측의 본체 증거는 이 single-leg가 아니라 Barents→EJS 직접(+0.52)·partial(+0.55 |NP통제)·wave-train composite(§7-2).
2. single-leg 상관은 wave-train composite보다 noisy. 모델별 Z500 wave-train composite가 더 결정적이나 **GC zarr 속도(~130KB/s)로 차단**.
   → 본 screening은 *시사적(suggestive) 일반화*이지 정밀 증명은 아님. ESGF에서 모델별 zg500 받아 composite하면 확정 가능.
3. n·기간 불균일(64~269), CNRM은 hist-1950(transient) → 엄밀 비교는 control-1950로 통일 필요.

## 7-5. (b) OHC 아표층-기억 경로 — 게이트 실패로 종료 (2026-06-03)

**동기:** model-analog의 domain(SST 패턴)·persistence가 봄 장벽으로 여름에 ACC≈0으로 무너진다.
"여름까지 안 무너지는 domain"으로 OHC(아표층 열저장)를 시도 — 겨울 신호가 여름 혼합층 아래
보존되면 OHC가 SST보다 오래 기억할 수 있다는 가설. 프로젝트 원칙(싼 자료 게이트 먼저)대로
**obs(GLORYS12) 게이트**로 판정 후 모델(CESM 3D TEMP, 거대) 투자 여부 결정.

**자료:** GLORYS12V1 월별 thetao, EJS 박스 0–300m, 1993–2020 (`data/GLORYS/glorys_thetao_ejs_0-300m_1992-2020.nc`, 559MB).

**변종 ①: OHC를 domain(국소 기억)으로** — DJF predictor → 각 달 EJS SST 상관 (`analyze_ohc_gate.py`, `fig_ohc_gate.png`)

| predictor (DJF) | → JAS SST | vs SST persistence |
|---|---|---|
| SST(표층) | +0.059 | (기준) |
| OHC 0–100m | +0.053 | Δ −0.007, 95%CI[−0.04,+0.03] **비유의** |
| OHC 0–200m | +0.075 | Δ +0.018, 95%CI[−0.09,+0.14] **비유의** |

→ 세 곡선이 거의 포개져 같이 붕괴(Mar~0.6 → Aug 음수). **국소 아표층에도 겨울 기억 저장 안 됨.**

**변종 ②: 유입 OHC를 해양 source로 (D4, 이류 경로)** — 남부 EJS/대한해협(34.5–38N,128.5–133E)
OHC → EJS JAS SST (`analyze_ohc_inflow_gate.py`)

| predictor | r (95%CI) | EJS persistence 통제 partial |
|---|---|---|
| 유입 OHC0-200m DJF | −0.049 [−0.38,+0.31] **비유의** | −0.085 |
| 유입 OHC0-200m MAM | −0.085 [−0.50,+0.37] **비유의** | −0.112 |
| 유입 SST DJF (비교) | −0.030 [−0.39,+0.37] 비유의 | — |

→ 따뜻한 쓰시마 유입 OHC도 여름 동해 SST를 예측 못함. (유입 OHC DJF vs ECS easterly r=+0.18.)

**결론(둘 다 실패):** 동해 여름 SST엔 **해양 예측자가 표층·아표층·국소·유입 어디에도 없다.**
봄 장벽은 SST·OHC에 동일 작용. → **CESM-HR 3D TEMP 다운로드 불필요(게이트 차단).**
이는 메인 스토리를 강화: 여름 동해 SST의 유일한 예측성 경로는 **원격 대기 ECS source(+0.168 유의)**뿐.

## 7-6. 겨울 순환 family 귀속 — Barents는 노드인가 모드의 proxy인가 (nb4, 2026-06-04)

**동기:** Barents 블로킹은 그 자체로 **AO ↔ Ural/Barents blocking ↔ Siberian High ↔ EAWM ↔ ECS 동풍**으로
이어지는 겨울 동아시아 순환 사슬의 한 마디다. "Barents 단독 box"로 못 박은 게 적절한가, 아니면 동해 여름 SST
예측성은 더 넓은 **겨울 순환 모드** 전체에 들어 있고 Barents는 그 한 창(window)일 뿐인가? — *예측 개선*이 아니라
*귀속(attribution)* 질문으로 검정(공선성·n=27·ECS 병목 때문에 family를 distance에 쌓는 예측개선은 §7-3에서 이미 null).

**자료:** OISST(JAS EJS, n=27, 1994–2020) + ERA5 단기(z500/u10/v10/msl) + CPC AO index.
predictors(모두 detrend·z): AO(CPC), SHI=SLP(40–60N,80–120E; Gong&Ho), EAWM=−v10(25–40N,115–130E),
BHI=Z500 Barents, ECSW=−u10 ECS. `scripts/analyze_circulation_family.py`, `fig_circulation_family.png`.

**① 공선성 — 한 모드다:** 최대 |비대각 상관|=0.82(SHI–EAWM), BHI–SHI=+0.55, **BHI–AO=−0.61**. PCA PC1=46%
(loadings SHI+0.60·EAWM+0.52·BHI+0.52·ECSW+0.25·AO−0.19). → AO/SHI/EAWM/BHI는 **별개 source 4개가 아니라
한 겨울 동아시아 순환 모드**(AO 음→Siberian High·Barents 고기압 강화). ECS 동풍만 반쯤 독립(타 지수와 |r|<0.27).

**② 단일 신호(95% Fisher CI):** ECSW만 유의(+0.49 [+0.13,+0.73]). AO −0.30, BHI(DJF) +0.25, SHI +0.21,
EAWM +0.19 — 모두 CI 0 포함. (BHI 신호는 NDJ +0.48에 집중, DJF서 +0.25로 약화.)

**③·④ 노드 vs 모드는 계절 의존(칼날):**

| 계절 | BHI~Y 단순 | r(BHI,Y \| AO,SHI,EAWM) | 판정 |
|---|---|---|---|
| NDJ(초겨울) | +0.48 | **+0.35** [−0.07,+0.66] | Barents 노드 잔존 |
| DJF(한겨울) | +0.25 | **−0.04** [−0.44,+0.37] | family에 흡수(proxy) |

→ **초겨울 Barents=활성 노드, 한겨울=AO/SH/EAWM 모드의 한 창.** PC1(+0.35)은 BHI 단독(+0.25)보다 약간 크나
`r(PC1,Y|BHI)`는 NDJ +0.19/DJF +0.26로 유의 추가정보 없음. n=27이라 단정 불가, 단 **"Barents 단독"보다 "순환 모드"가
더 정확한 묘사**임을 시사. (§4의 "Barents⟂북태평양"과 모순 아님 — NP는 다른 태평양 모드, AO/SH/EAWM은 Barents가
*속한* 같은 대륙-북극 모드.)

**⑤ mediation 반전 (새 실마리):** ECS 동풍 통제 시 SHI/EAWM/BHI는 약해지나(부분 매개) **AO는 오히려 강해진다
(−0.30 → −0.43)**. → **AO엔 ECS 동풍과 무관한 별도 경로**가 있다(음의 AO → 따뜻한 여름 동해). "모든 게 ECS 동풍
경유"라는 기존 매개 그림을 부분 보완.

**⑥ 예측은 그대로:** family 더해도 ECS 단독(LOO +0.39)을 *유의하게* 못 넘음 — AO+ECSW +0.46(ΔvsECSW +0.07
CI[−0.13,+0.28]), full(5) +0.47(Δ +0.08 [−0.16,+0.33]), 모두 0 포함. 다중회귀 R²=0.50/adj-R²=0.38, VIF≤4.5.
→ **예보 천장은 여전히 ECS 동풍**(§7-3과 일치).

**nb4 결론:** Barents를 고립 source로 못 박지 말고 **AO–Siberian High–EAWM 겨울 순환 모드의 한 창**으로 보는 게
더 정확하다(특히 한겨울). *예측*엔 ECS 동풍이 여전히 병목이고 family의 추가 효용은 n=27에서 노이즈와 구별 안 됨.
가장 추적할 가치 있는 건 **AO의 ECS-독립 경로**.

---

## 8. 연구 현황 지도 — 정지점 (2026-06-03)

> 길 잃지 않게: **무엇이 확립됐고 / 무엇을 검증 후 버렸고(다시 하지 말 것) / 어디서 재개할지.**

### ✅ 확립된 것 (story 완결)
- **관측:** 동해 여름 SST 예측성 source = 전 겨울 **Barents 원격 강제**(→wave train→ECS 동풍→북향 Ekman).
  진짜·독립(NP 통제 partial +0.55)·stationary. D1(source) LOO +0.42 ≫ D0(SST패턴) −0.23. 여름 reemergence 스파이크.
- **모델(CESM-HR):** Barents→동아시아 **wave-train bias로 결여**(Z500 응답 ≈0) → 동해 예측성을 **북태평양**으로 귀속(단 obs 전이 X).
- **model-analog(obs-to-model):** CESM library가 쓸 수 있는 transferable source는 **ECS 동풍 하나뿐**.
  fig4 source gain: **+ECS +0.168 [+0.02,+0.31] 유의**, +NP 비유의. 천장이 낮은 근본이유 = 모델 wave-train bias.
- **입장 2 일반화:** HighResMIP 6모델 전부 Barents→ECS leg 비유의(0/6) → HR coupled 전반 경향(§7-4, suggestive).
- **귀속(nb4, §7-6):** AO/SHI/EAWM/BHI는 **한 겨울 순환 모드**(공선 ≤0.82, PC1 46%). Barents는 그 모드의 한 창
  (초겨울 노드, 한겨울 proxy). family 더해도 예측은 ECS 단독 못 넘음 — "Barents 단독"보다 "순환 모드"가 정확한 묘사.

### ❌ 검증 후 배제된 막다른 길 (재시도 불필요)
- **국소 OHC 기억 domain** (§7-5 변종①): GLORYS OHC0-200m도 SST처럼 여름 붕괴. Δ +0.018 비유의. → CESM 3D TEMP 불필요.
- **유입 OHC 해양 source** (§7-5 변종②): 쓰시마 유입 OHC→여름 동해 SST r≈−0.05 비유의. 해양 이류 경로 없음.
- **북태평양 source의 obs 전이:** 모델 최강 source(NP +0.44)는 obs로 전이 안 됨(fig4 +NP 비유의, partial 검증).
- **관측 Barents를 model-analog에 투입:** CESM-Barents가 EJS 무관이라 의미 없음(distance_compare에서 +Barents 유의 도움 X 확인).

### 🔓 열린 실 / 아이디어 씨앗 (재개 지점)
1. **입장 2 확정:** ESGF에서 HR 모델별 `zg500` 받아 wave-train composite (GC 속도 차단 우회) → "suggestive→정밀 증명".
2. **다음-겨울 reemergence:** OHC가 *같은 여름*엔 무익했지만 *다음 겨울* 재부상은 미검(고전 reemergence는 winter-to-winter). 별 연구주제.
3. **HR 모델 SST leg:** screening 통과 무관하게, ESGF tos로 ECS→EJS·NP→EJS leg를 다모델 비교(해양측 일반화).
4. **AO의 ECS-독립 경로 (nb4 §7-6 신규):** AO는 ECS 동풍 통제 후에도 동해 여름 SST와 −0.43 → ECS-Ekman과 다른
   제2 경로가 있음. 고−저 AO 겨울 composite로 **직접 가열(국소 surface heat flux) vs 국소 Ekman**을 분리 검증.
   (단일 partial 상관은 n=27 suppression 의심 → §7-4 교훈대로 composite가 더 robust.)
5. **글쓰기:** 위 ✅만으로도 완결된 스토리 — README/FINDINGS 최종화 후 노트/포스터.

→ **현재는 (5) 정지점.** 새 아이디어 생기면 1~4 중 택해 재개.

---

## 9. 산출물 지도

| 노트북 | 내용 | 그림 |
|---|---|---|
| 01 | 관측 신호 진단 | — |
| 02 | 관측 D1 vs D0 LOO analog | — |
| 03 | 관측 event skill | — |
| 04 | 관측 robust + 경로 composite | `phase1c_*` |
| 05 | 모델 model-internal | — |
| 06 | 모델 teleconnection 진단 | `phase3b_source_maps` |
| 07 | 모델 북태평양 source | `phase3c_npac_skill` |
| 08 | lead별 ACC(내부 LOO) | `fig_lead_acc_seasonal` |
| 09 | **진짜 obs-to-model analog** | `fig_obs2model_lead_acc` |
| **04(통합)** | **겨울 순환 family 귀속 (nb4, §7-6)** | `fig_circulation_family` |

분석 스크립트: `analyze_circulation_family.py`(공선성·PCA·양방향 partial·mediation·LOO), 노트북 빌더 `_build_nb4.py`.

전처리: `preprocess_cesm_indices/fields/regrid.py`, `download_era5*/cesm_hr_atm.py`.
캐시: `data/processed/cesm_*.{csv,nc}`.

### 논문용 그림 (`paper/`, plain style·영문 라벨, 캡션 `paper/FIGURES.md`)
재생성: `python scripts/make_paper_figs.py`

| 그림 | 내용 | 근거 §·스크립트 |
|---|---|---|
| `paper/fig1.png` | lead별 ACC: source analog vs persistence, (a)관측 (b)모델 | §4·§5 (lead-acc) |
| `paper/fig2.png` | 고-Barents 합성 Z500+표면바람 obs/model 2×2 (wave-train bias) | §7-2 (`analyze_wavetrain_full.py`) |
| `paper/fig3.png` | obs-to-model: ACC + source gain(ΔACC, 95%CI) | §7-3 (`analyze_obs2model_fig4.py`) |
| `paper/fig4.png` | HighResMIP 6모델 Barents→ECS leg | §7-4 (`make_highresmip_summary.py`) |

### 막다른 길 진단 그림 (본문 §7-5)
`figures/fig_ohc_gate.png`(국소 OHC), `analyze_ohc_inflow_gate.py`(유입 OHC, 표만) — 둘 다 비유의.
