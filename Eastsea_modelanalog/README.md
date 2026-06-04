# 동해(EJS) 여름 SST 예측성 — 관측 vs CESM-HR model-analog

> **한 줄 결론:** 동해 여름 SST는 관측·모델 모두 예측 가능하지만 **source가 다르다.**
> 관측 = 전 겨울 **Barents 원격 강제**(Rossby wave train → ECS 동풍 → 동해), CESM-HR = **북태평양** 경유.
> 모델은 Barents→동아시아 wave train을 **bias로 결여**하고, 그 결과 model-analog은 동해 예측성의 source를
> 북태평양으로 (잘못) 귀속한다. → "analog이 동해 source를 잘못 잡는가?"에 대한 메커니즘까지 닿은 답.

전체 결과는 **[FINDINGS.md](FINDINGS.md)**, 연구 설계·두 입장은 **[RESEARCH_PLAN.md](RESEARCH_PLAN.md)**.

---

## 현재 상태 — 정지점 (2026-06-03)

스토리는 **완결**(아래). 새 아이디어가 생기면 재개. 자세한 지도는 [FINDINGS §8](FINDINGS.md).

- ✅ **확립:** 관측=Barents 원격 / 모델=북태평양 / model-analog 유일 transferable source=**ECS(+0.168 유의)** / HighResMIP 6모델도 Barents→ECS 약함(입장 2 일반화).
- ❌ **막다른 길(재시도 X):** 국소 OHC 기억, 유입 OHC 해양 source, NP의 obs 전이, Barents를 model-analog에 투입 — 모두 검증 후 비유의.
- 🔓 **재개 후보:** ESGF `zg500`로 HR 모델 wave-train composite(입장 2 정밀 확정) / 다음-겨울 reemergence / HR 다모델 SST leg / 글쓰기.

---

## 스토리 한눈에

```
관측:  Barents 고기압(진짜·독립·stationary)
        → [Rossby wave train: Z500 +9m] → 동아시아 → ECS 동풍(−0.40 m/s)
        → 북향 Ekman → 여름 동해 SST  (여름 reemergence: lead 곡선 Aug–Oct 스파이크)

모델:  Barents → [wave train 못 만듦 = 중위도 teleconnection BIAS] → ✗ 동아시아 반응 없음
        → 동해 예측성은 북태평양(Aleutian/NPGO)에서 (단 이건 obs로 전이 안 됨)

model-analog: CESM library로 obs 동해 예측 → 쓸 source는 ECS 동풍 하나뿐
              (Barents=모델 결여, NP=obs 전이X) → 최선 WIDE+ECS, 여름 skill modest(~0.2)
              ★ 천장이 낮은 근본 이유 = 모델의 wave-train bias
```

**입장 1(예보자)** 과 **입장 2(개발자)** 가 만나는 지점: model-analog의 동해 천장이 낮은 게 모델 bias 때문.

---

## 통합 노트북 (재현 순서)

> 모두 `scripts/ejs_common.py`(공유 헬퍼)를 import하고 `data/processed/` 캐시를 읽는다.

| 노트북 | 내용 | 핵심 그림 |
|---|---|---|
| **[notebooks/01_observations.ipynb](notebooks/01_observations.ipynb)** | 관측: 신호(BHI·ECSW)→D1 vs D0 LOO→event→robust→lead 곡선→partial corr→forced 테스트 | `fig_lead_acc_seasonal`, `fig_forced_teleconnection_sliding` |
| **[notebooks/02_model_diagnosis.ipynb](notebooks/02_model_diagnosis.ipynb)** | 모델: model-internal→teleconnection 진단→북태평양 source→wave-train bias | `fig_wavetrain_full`, `phase3b_source_maps` |
| **[notebooks/03_model_analog.ipynb](notebooks/03_model_analog.ipynb)** | model-analog: NP obs 검증→obs-to-model source 비교(WIDE/ECS/NP) | `fig_obs2model_sources`, `fig_np_obs_validation` |

상세 기록(개발 과정)은 `notebooks/archive/` (구 01–09).

---

## 데이터 준비 (scripts/)

**다운로드** (geo_env): `download_era5.py`(월별 box), `download_era5_npac.py`(북태평양 msl),
`download_era5_longterm.py`(1940–2024), `download_cesm_hr_atm.py`(PSL/TAUX/TAUY),
`download_cesm_z3.py`(Z3 500hPa, 부분), `download_glorys_ejs.py`(GLORYS12 thetao 0–300m, OHC 게이트용),
ERSSTv5는 PSL HTTP. (CESM SST는 선행 프로젝트 보유.)

**전처리** → `data/processed/` 캐시:
`aggregate_oisst_monthly.py`(OISST 일별→월별 EJS),
`preprocess_cesm_indices.py`(PSL_bar·TAUX_ecs·EJS SST box),
`preprocess_cesm_fields.py`(광역 PSL/TAUX 장),
`preprocess_cesm_regrid.py`(CESM EJS SST→OISST 격자),
`preprocess_cesm_z500_field.py`(Z3 500hPa NDJ 장).

**분석 스크립트**(노트북이 흡수하지 못한 진단):
`analyze_forced_teleconnection.py`, `analyze_wavetrain_full.py`, `analyze_np_obs_validation.py`,
`analyze_obs2model_sources.py`, `analyze_obs2model_fig4.py`(source gain CI), `analyze_cesm_z500_barents.py`,
`analyze_ohc_gate.py`·`analyze_ohc_inflow_gate.py`(OHC 막다른 길 게이트),
`screen_highresmip_atm.py`·`make_highresmip_summary.py`(입장 2 일반화).

---

## 환경

- 인터프리터: **geo_env** (`C:\Users\dnjst\miniconda3\envs\geo_env\python.exe`).
  cdsapi·xarray·netCDF4·scipy·dask·nbformat·Malgun Gothic 폰트 포함.
- ⚠️ VS Code Code Runner 기본 `python`은 geo_env가 아님 → "Python: Select Interpreter"로 geo_env 고정.
- 노트북 실행: `python -m jupyter nbconvert --to notebook --execute --inplace <nb>` (PYTHONIOENCODING=utf-8).

---

## 핵심 숫자 (요약)

| | 관측 | CESM-HR |
|---|---|---|
| Barents → 동해 | +0.52 (NP 통제해도 +0.55) | −0.10 |
| ECS 동풍 → 동해 | +0.45 | +0.31 (전이됨) |
| 북태평양 → 동해 | +0.20 (약) | +0.44 (모델 전용) |
| LOO analog | D1 +0.42 ≫ D0 −0.23 | D1_NP 0.42 ≈ persist 0.39 |
| 고-Barents Z500 응답(ECS) | +9 m, u10 −0.40 m/s | ≈0, 바람 ≈0 |
| model-analog 동해(obs) | — | WIDE+ECS 여름 ~0.2 (NP 무익) |
