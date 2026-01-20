# 해양 데이터 분석 가이드북
## MITgcm 연구 준비를 위한 Python 기초 실습

**목표**: "MITgcm 출력이 더 이상 무섭지 않은 상태"

**대상 영역**: 쿠로시오 확장역 (115°E-160°E, 20°N-50°N)

---

## 목차

1. [환경 설정](#1-환경-설정)
2. [NetCDF 데이터 불러오기](#2-netcdf-데이터-불러오기)
3. [특정 영역 자르기 (Subsetting)](#3-특정-영역-자르기-subsetting)
4. [시간 평균 / 계절 평균](#4-시간-평균--계절-평균)
5. [Anomaly 계산](#5-anomaly-계산)
6. [Hovmöller Diagram](#6-hovmöller-diagram)
7. [Lat-Lon Map Plotting](#7-lat-lon-map-plotting)
8. [MLD-SST 상관관계 분석](#8-mld-sst-상관관계-분석)
9. [종합 연습 과제](#9-종합-연습-과제)

---

## 1. 환경 설정

### 1.1 필수 패키지 설치

```bash
# conda 환경 생성 (권장)
conda create -n ocean python=3.11
conda activate ocean

# 핵심 패키지
conda install -c conda-forge xarray netcdf4 dask
conda install -c conda-forge matplotlib cartopy
conda install -c conda-forge scipy numpy pandas

# 데이터 다운로드용
conda install -c conda-forge pooch requests
```

### 1.2 기본 import 템플릿

모든 분석의 시작점이 되는 코드입니다. 이 블록을 항상 먼저 실행하세요.

```python
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 (필요시)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
```

---

## 2. NetCDF 데이터 불러오기

### 2.1 NetCDF란?

NetCDF(Network Common Data Form)는 기후/해양 데이터의 표준 포맷입니다. 
- **다차원 배열** 저장에 최적화 (시간, 위도, 경도, 깊이)
- **메타데이터** 포함 (단위, 설명, 좌표계 정보)
- **자기 기술적(self-describing)**: 파일 자체에 구조 정보가 담겨 있음

### 2.2 OISST 데이터 불러오기

OISST(Optimum Interpolation Sea Surface Temperature)는 NOAA에서 제공하는 일별 전지구 SST 데이터입니다.

```python
# OISST 데이터 다운로드 (예시: 2023년 1월)
# 실제 URL은 NOAA ERDDAP이나 THREDDS에서 확인
url = "https://www.ncei.noaa.gov/thredds/dodsC/OisstBase/NetCDF/V2.1/AVHRR/202301/oisst-avhrr-v02r01.20230101.nc"

# xarray로 열기 (OPeNDAP 프로토콜 지원)
ds = xr.open_dataset(url)

# 데이터 구조 확인 - 가장 먼저 해야 할 일!
print(ds)
```

**출력 예시:**
```
<xarray.Dataset>
Dimensions:  (time: 1, lat: 720, lon: 1440, zlev: 1)
Coordinates:
  * time     (time) datetime64[ns] 2023-01-01T12:00:00
  * lat      (lat) float32 -89.88 -89.62 ... 89.62 89.88
  * lon      (lon) float32 0.125 0.375 ... 359.6 359.9
  * zlev     (zlev) float32 0.0
Data variables:
    sst      (time, zlev, lat, lon) float32 ...
    anom     (time, zlev, lat, lon) float32 ...
    err      (time, zlev, lat, lon) float32 ...
    ice      (time, zlev, lat, lon) float32 ...
```

### 2.3 로컬 파일에서 불러오기

```python
# 단일 파일
ds = xr.open_dataset('/path/to/data.nc')

# 여러 파일을 한 번에 (시계열 데이터)
# 파일명 패턴: oisst_202301.nc, oisst_202302.nc, ...
ds = xr.open_mfdataset('/path/to/oisst_*.nc', combine='by_coords')
```

### 2.4 데이터 구조 파악하기

**반드시 확인해야 할 것들:**

```python
# 1. 차원(Dimensions) 확인
print("차원:", ds.dims)

# 2. 좌표(Coordinates) 확인
print("\n위도 범위:", ds.lat.values.min(), "~", ds.lat.values.max())
print("경도 범위:", ds.lon.values.min(), "~", ds.lon.values.max())
print("시간 범위:", ds.time.values[0], "~", ds.time.values[-1])

# 3. 변수(Variables) 확인
print("\n변수 목록:", list(ds.data_vars))

# 4. 특정 변수의 속성(Attributes) 확인
print("\nSST 단위:", ds.sst.attrs.get('units', '단위 정보 없음'))
print("SST 설명:", ds.sst.attrs.get('long_name', '설명 없음'))

# 5. 결측값 확인
print("\n결측값 개수:", ds.sst.isnull().sum().values)
```

### 2.5 HYCOM 데이터 불러오기

HYCOM(HYbrid Coordinate Ocean Model)은 3차원 해양 재분석 자료입니다. MLD, 염분, 해류 등 다양한 변수를 포함합니다.

```python
# HYCOM OPeNDAP 접근 (예시)
url_hycom = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
ds_hycom = xr.open_dataset(url_hycom)

# HYCOM은 변수가 많으므로 필요한 것만 선택
ds_hycom = xr.open_dataset(url_hycom, drop_variables=['salinity', 'water_u', 'water_v'])
```

### 2.6 데이터 접근 방식 비교

```python
# xarray의 강점: 차원 이름으로 접근

# numpy 방식 (인덱스 기반) - 헷갈리기 쉬움
sst_numpy = ds.sst.values[0, 0, 100, 200]  # time=0, zlev=0, lat=100번째, lon=200번째

# xarray 방식 (이름 기반) - 명확함
sst_xarray = ds.sst.sel(time='2023-01-01', lat=35, lon=140, method='nearest')

# 범위 선택
sst_region = ds.sst.sel(lat=slice(20, 50), lon=slice(115, 160))
```

---

## 3. 특정 영역 자르기 (Subsetting)

### 3.1 왜 영역을 잘라야 하는가?

- **메모리 절약**: 전지구 데이터는 수십 GB, 쿠로시오 영역만 잘라내면 수백 MB
- **계산 속도**: 필요한 영역만 처리하면 분석이 훨씬 빠름
- **연구 집중**: 관심 영역에 집중하여 세부 분석 가능

### 3.2 쿠로시오 박스 정의

연구 로드맵에서 정의한 영역: **115°E-160°E, 20°N-50°N**

```python
# 영역 좌표 정의 (전역 변수로 사용)
LON_MIN, LON_MAX = 115, 160
LAT_MIN, LAT_MAX = 20, 50

def subset_kuroshio(ds, lon_var='lon', lat_var='lat'):
    """
    쿠로시오 확장역으로 데이터 자르기
    
    Parameters:
    -----------
    ds : xarray.Dataset or DataArray
    lon_var, lat_var : str, 좌표 변수 이름
    
    Returns:
    --------
    xarray.Dataset or DataArray
    """
    return ds.sel(
        {lon_var: slice(LON_MIN, LON_MAX),
         lat_var: slice(LAT_MIN, LAT_MAX)}
    )
```

### 3.3 경도 체계 주의사항

**매우 중요**: 데이터마다 경도 체계가 다릅니다!

```python
# 경도 체계 확인
print("경도 범위:", ds.lon.values.min(), "~", ds.lon.values.max())

# Case 1: 0°~360° 체계 (OISST, HYCOM 등 대부분)
# 쿠로시오: 115°E~160°E → 그대로 사용
ds_kuroshio = ds.sel(lon=slice(115, 160), lat=slice(20, 50))

# Case 2: -180°~180° 체계 (일부 데이터)
# 쿠로시오: 115°E~160°E → 그대로 사용 (양수이므로 변환 불필요)
# 만약 대서양 데이터라면 -80°~0° 같은 형태

# 경도 변환 함수 (필요시)
def convert_lon_360_to_180(ds):
    """0-360 체계를 -180~180 체계로 변환"""
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
    return ds.sortby('lon')

def convert_lon_180_to_360(ds):
    """-180~180 체계를 0-360 체계로 변환"""
    ds = ds.assign_coords(lon=(ds.lon % 360))
    return ds.sortby('lon')
```

### 3.4 위도 방향 주의사항

```python
# 위도 순서 확인
print("위도 순서:", ds.lat.values[:5], "...", ds.lat.values[-5:])

# 남→북 순서 (90S to 90N): slice(20, 50) 정상 작동
# 북→남 순서 (90N to 90S): slice(50, 20) 으로 뒤집어야 함

# 자동으로 처리하는 함수
def safe_subset(ds, lon_range, lat_range, lon_var='lon', lat_var='lat'):
    """위도 순서에 관계없이 안전하게 자르기"""
    
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    
    # 위도가 감소하는 순서인지 확인
    if ds[lat_var].values[0] > ds[lat_var].values[-1]:
        lat_slice = slice(lat_max, lat_min)  # 뒤집기
    else:
        lat_slice = slice(lat_min, lat_max)
    
    return ds.sel({lon_var: slice(lon_min, lon_max), lat_var: lat_slice})

# 사용 예시
ds_kuroshio = safe_subset(ds, (115, 160), (20, 50))
```

### 3.5 실습: 영역 자르기 전후 비교

```python
# 원본 데이터 크기
print("원본 크기:")
print(f"  - 위도: {len(ds.lat)} points")
print(f"  - 경도: {len(ds.lon)} points")
print(f"  - 메모리: {ds.sst.nbytes / 1e6:.1f} MB")

# 자른 후 크기
ds_kuroshio = ds.sel(lon=slice(115, 160), lat=slice(20, 50))
print("\n쿠로시오 영역 크기:")
print(f"  - 위도: {len(ds_kuroshio.lat)} points")
print(f"  - 경도: {len(ds_kuroshio.lon)} points")
print(f"  - 메모리: {ds_kuroshio.sst.nbytes / 1e6:.1f} MB")
```

---

## 4. 시간 평균 / 계절 평균

### 4.1 왜 평균을 내는가?

- **기후학적 평균(Climatology)**: 특정 기간의 "평균적인" 상태 파악
- **노이즈 제거**: 일별 변동성을 제거하고 큰 패턴 확인
- **계절 특성**: 계절별 차이 비교 (겨울 vs 여름)

### 4.2 시간 평균 (전체 기간)

```python
# 전체 시간에 대한 평균
sst_mean = ds_kuroshio.sst.mean(dim='time')

# 확인
print("평균 SST 형태:", sst_mean.shape)  # (lat, lon) - time 차원 사라짐
print("평균 SST 범위:", float(sst_mean.min()), "~", float(sst_mean.max()), "°C")
```

### 4.3 월별 평균 (Monthly Mean)

```python
# 월별 평균
sst_monthly = ds_kuroshio.sst.resample(time='1M').mean()

# 또는 groupby 사용
sst_monthly_clim = ds_kuroshio.sst.groupby('time.month').mean(dim='time')
# 결과: 12개월 × lat × lon

print("월별 기후값 형태:", sst_monthly_clim.shape)
```

### 4.4 계절 평균 (Seasonal Mean)

기후학에서 계절 정의:
- **DJF** (겨울): 12월, 1월, 2월
- **MAM** (봄): 3월, 4월, 5월
- **JJA** (여름): 6월, 7월, 8월
- **SON** (가을): 9월, 10월, 11월

```python
# 방법 1: xarray의 season 기능 사용
sst_seasonal = ds_kuroshio.sst.groupby('time.season').mean(dim='time')

print("계절별 평균:")
for season in ['DJF', 'MAM', 'JJA', 'SON']:
    mean_val = float(sst_seasonal.sel(season=season).mean())
    print(f"  {season}: {mean_val:.2f} °C")
```

### 4.5 특정 계절만 추출

```python
# 겨울(DJF)만 추출
def extract_season(ds, season='DJF'):
    """특정 계절의 데이터만 추출"""
    return ds.where(ds['time.season'] == season, drop=True)

sst_winter = extract_season(ds_kuroshio.sst, 'DJF')
sst_summer = extract_season(ds_kuroshio.sst, 'JJA')

print(f"겨울 데이터 개수: {len(sst_winter.time)}")
print(f"여름 데이터 개수: {len(sst_summer.time)}")
```

### 4.6 월별 기후값 (Climatology) 계산

30년 이상의 데이터에서 각 월의 평균을 구하면 "기후값"이 됩니다.

```python
def calculate_monthly_climatology(ds, var_name, start_year=1991, end_year=2020):
    """
    월별 기후값 계산 (WMO 표준: 30년 평균)
    
    Parameters:
    -----------
    ds : xarray.Dataset
    var_name : str, 변수 이름
    start_year, end_year : int, 기후값 기간
    
    Returns:
    --------
    xarray.DataArray with dims (month, lat, lon)
    """
    # 기간 선택
    ds_period = ds.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))
    
    # 월별 평균 (1~12월 각각의 평균)
    climatology = ds_period[var_name].groupby('time.month').mean(dim='time')
    
    return climatology

# 사용 예시
sst_clim = calculate_monthly_climatology(ds_kuroshio, 'sst')
print("기후값 형태:", sst_clim.shape)  # (12, lat, lon)
```

### 4.7 실습: 겨울 vs 여름 SST 비교

```python
# 계절 평균 계산
sst_djf = ds_kuroshio.sst.where(ds_kuroshio['time.season'] == 'DJF', drop=True).mean(dim='time')
sst_jja = ds_kuroshio.sst.where(ds_kuroshio['time.season'] == 'JJA', drop=True).mean(dim='time')

# 차이
sst_diff = sst_jja - sst_djf

print("여름-겨울 SST 차이:")
print(f"  최소: {float(sst_diff.min()):.2f} °C")
print(f"  최대: {float(sst_diff.max()):.2f} °C")
print(f"  평균: {float(sst_diff.mean()):.2f} °C")
```

---

## 5. Anomaly 계산

### 5.1 Anomaly란?

**Anomaly(편차)** = 실제값 - 기후값(Climatology)

- 양의 anomaly: 평년보다 따뜻함/높음
- 음의 anomaly: 평년보다 차가움/낮음

연구에서 anomaly가 중요한 이유:
- **계절 신호 제거**: 여름이 겨울보다 따뜻한 건 당연 → anomaly로 보면 "평년 대비" 변화만 봄
- **기후 변화 탐지**: 평년보다 얼마나 달라졌는지
- **모델 오차 분석**: 모델이 관측보다 얼마나 벗어났는지

### 5.2 월별 Anomaly 계산

```python
def calculate_anomaly(ds, var_name, clim_start=1991, clim_end=2020):
    """
    월별 anomaly 계산
    
    Parameters:
    -----------
    ds : xarray.Dataset
    var_name : str
    clim_start, clim_end : int, 기후값 계산 기간
    
    Returns:
    --------
    xarray.DataArray of anomalies
    """
    # 1. 기후값 계산
    ds_clim_period = ds.sel(time=slice(f'{clim_start}-01-01', f'{clim_end}-12-31'))
    climatology = ds_clim_period[var_name].groupby('time.month').mean(dim='time')
    
    # 2. Anomaly = 원본 - 기후값
    anomaly = ds[var_name].groupby('time.month') - climatology
    
    return anomaly

# 사용
sst_anom = calculate_anomaly(ds_kuroshio, 'sst')
print("Anomaly 범위:", float(sst_anom.min()), "~", float(sst_anom.max()), "°C")
```

### 5.3 Anomaly 시계열 분석

```python
# 영역 평균 anomaly 시계열
sst_anom_ts = sst_anom.mean(dim=['lat', 'lon'])

# 시각화
fig, ax = plt.subplots(figsize=(14, 5))

sst_anom_ts.plot(ax=ax, color='steelblue', linewidth=0.8)
ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax.fill_between(sst_anom_ts.time.values, 0, sst_anom_ts.values, 
                where=sst_anom_ts.values > 0, color='red', alpha=0.3, label='Warm')
ax.fill_between(sst_anom_ts.time.values, 0, sst_anom_ts.values, 
                where=sst_anom_ts.values < 0, color='blue', alpha=0.3, label='Cold')

ax.set_xlabel('Time')
ax.set_ylabel('SST Anomaly (°C)')
ax.set_title('Kuroshio Region SST Anomaly Time Series')
ax.legend()
plt.tight_layout()
plt.savefig('sst_anomaly_timeseries.png', dpi=150)
plt.show()
```

### 5.4 표준화 Anomaly (Standardized Anomaly)

```python
def standardized_anomaly(anomaly):
    """
    표준화 anomaly: (anomaly - mean) / std
    단위가 없어지고, σ(시그마) 단위로 표현됨
    """
    return (anomaly - anomaly.mean()) / anomaly.std()

sst_std_anom = standardized_anomaly(sst_anom)
print("+2σ 이상인 극값 시점 수:", int((sst_std_anom > 2).sum()))
```

---

## 6. Hovmöller Diagram

### 6.1 Hovmöller Diagram이란?

시간(y축) vs 공간(x축, 보통 경도나 위도)에 변수를 색으로 표현한 그림입니다.

**용도**:
- 파동/신호의 전파 방향과 속도 파악
- 시간에 따른 공간 패턴 변화 추적
- 예: SST anomaly가 동쪽으로 전파되는지 확인

### 6.2 경도-시간 Hovmöller (위도 평균)

```python
def hovmoller_lon_time(data, lat_range=None, title='Hovmöller Diagram'):
    """
    경도-시간 Hovmöller diagram
    
    Parameters:
    -----------
    data : xarray.DataArray with dims (time, lat, lon)
    lat_range : tuple, 평균을 낼 위도 범위
    """
    # 위도 평균
    if lat_range:
        data = data.sel(lat=slice(lat_range[0], lat_range[1]))
    
    data_lat_mean = data.mean(dim='lat')
    
    # 그리기
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # pcolormesh로 그리기
    im = data_lat_mean.plot(
        ax=ax,
        x='lon', y='time',
        cmap='RdBu_r',
        center=0,
        cbar_kwargs={'label': f'{data.name} ({data.attrs.get("units", "")})'}
    )
    
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Time')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig, ax

# 사용 예시: 쿠로시오 핵심 위도대(30-40°N) 평균
fig, ax = hovmoller_lon_time(
    sst_anom, 
    lat_range=(30, 40),
    title='SST Anomaly Hovmöller (30-40°N average)'
)
plt.savefig('hovmoller_sst.png', dpi=150)
plt.show()
```

### 6.3 위도-시간 Hovmöller (경도 평균)

```python
def hovmoller_lat_time(data, lon_range=None, title='Hovmöller Diagram'):
    """
    위도-시간 Hovmöller diagram
    """
    if lon_range:
        data = data.sel(lon=slice(lon_range[0], lon_range[1]))
    
    data_lon_mean = data.mean(dim='lon')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    data_lon_mean.plot(
        ax=ax,
        x='lat', y='time',
        cmap='RdBu_r',
        center=0,
        cbar_kwargs={'label': f'{data.name}'}
    )
    
    ax.set_xlabel('Latitude (°N)')
    ax.set_ylabel('Time')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig, ax

# 쿠로시오 본류 경도대(140-150°E) 평균
fig, ax = hovmoller_lat_time(
    sst_anom,
    lon_range=(140, 150),
    title='SST Anomaly Hovmöller (140-150°E average)'
)
plt.savefig('hovmoller_lat_time.png', dpi=150)
plt.show()
```

### 6.4 해석 방법

```
Hovmöller에서 패턴 기울기 해석:

        시간 ↓
        ┌─────────────────┐
      1월│    ╲           │  ← 오른쪽 위에서 왼쪽 아래로 기울어짐
        │     ╲          │    = 서쪽으로 전파 (Westward propagation)
        │      ╲         │
      6월│       ╲        │
        │        ╲       │
     12월│         ╲      │
        └─────────────────┘
           120°E    160°E
           ← 경도 →

- 오른쪽 위 → 왼쪽 아래: 서쪽 전파
- 왼쪽 위 → 오른쪽 아래: 동쪽 전파
- 수직선: 정지 (stationary)
- 기울기가 급할수록 빠른 전파 속도
```

---

## 7. Lat-Lon Map Plotting

### 7.1 Cartopy 기초

Cartopy는 지도 투영과 해안선 등 지리 정보를 처리하는 라이브러리입니다.

```python
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 기본 지도 그리기
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# 지도 요소 추가
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3)

# 영역 설정
ax.set_extent([115, 160, 20, 50], crs=ccrs.PlateCarree())

# 격자선
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
gl.top_labels = False
gl.right_labels = False

plt.show()
```

### 7.2 SST 지도 그리기 (기본)

```python
def plot_sst_map(data, title='SST', cmap='coolwarm', vmin=None, vmax=None):
    """
    SST 분포 지도 그리기
    
    Parameters:
    -----------
    data : xarray.DataArray with dims (lat, lon)
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # 데이터 플롯
    if vmin is None:
        vmin = float(data.min())
    if vmax is None:
        vmax = float(data.max())
    
    im = ax.pcolormesh(
        data.lon, data.lat, data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin, vmax=vmax
    )
    
    # 지도 요소
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=11)
    
    # 컬러바
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', 
                        shrink=0.8, pad=0.02)
    cbar.set_label(f'{data.name} ({data.attrs.get("units", "°C")})')
    
    # 격자선
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, 
                      color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # 영역
    ax.set_extent([115, 160, 20, 50], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax

# 사용 예시
fig, ax = plot_sst_map(sst_mean, title='Mean SST in Kuroshio Region')
plt.savefig('sst_map.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 7.3 Anomaly 지도 (발산 컬러맵)

```python
def plot_anomaly_map(data, title='SST Anomaly', levels=None):
    """
    Anomaly 지도 (0 중심 발산 컬러맵)
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # 대칭 범위 설정
    abs_max = max(abs(float(data.min())), abs(float(data.max())))
    
    # contourf 사용 (더 부드러운 표현)
    if levels is None:
        levels = np.linspace(-abs_max, abs_max, 21)
    
    im = ax.contourf(
        data.lon, data.lat, data,
        transform=ccrs.PlateCarree(),
        levels=levels,
        cmap='RdBu_r',
        extend='both'
    )
    
    # 0 등치선 강조
    ax.contour(
        data.lon, data.lat, data,
        transform=ccrs.PlateCarree(),
        levels=[0],
        colors='black',
        linewidths=1
    )
    
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=11)
    
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', 
                        shrink=0.8, pad=0.02)
    cbar.set_label('Anomaly (°C)')
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, 
                      color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_extent([115, 160, 20, 50], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax

# 특정 시점의 anomaly
sst_anom_snapshot = sst_anom.sel(time='2023-01-15', method='nearest')
fig, ax = plot_anomaly_map(sst_anom_snapshot, title='SST Anomaly (2023-01-15)')
plt.savefig('sst_anomaly_map.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 7.4 SST Gradient (전선 시각화)

쿠로시오 전선(Kuroshio Front)은 SST 경사가 급격한 지역입니다.

```python
def plot_sst_gradient(sst_data, title='SST Gradient'):
    """
    SST gradient 계산 및 시각화
    전선(Front) 위치 파악에 유용
    """
    # gradient 계산 (유한차분)
    # 위도 방향
    dlat = float(sst_data.lat[1] - sst_data.lat[0])  # degree
    dlat_m = dlat * 111000  # meters (1도 ≈ 111km)
    
    # 경도 방향 (위도에 따라 다름)
    dlon = float(sst_data.lon[1] - sst_data.lon[0])
    mean_lat = float(sst_data.lat.mean())
    dlon_m = dlon * 111000 * np.cos(np.radians(mean_lat))
    
    # numpy gradient
    dsst_dy, dsst_dx = np.gradient(sst_data.values, dlat_m, dlon_m)
    
    # magnitude
    gradient_mag = np.sqrt(dsst_dx**2 + dsst_dy**2) * 1000  # per 1000km
    
    # DataArray로 변환
    grad_da = xr.DataArray(
        gradient_mag,
        dims=['lat', 'lon'],
        coords={'lat': sst_data.lat, 'lon': sst_data.lon},
        name='SST_gradient'
    )
    
    # 플롯
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    im = ax.pcolormesh(
        grad_da.lon, grad_da.lat, grad_da,
        transform=ccrs.PlateCarree(),
        cmap='hot_r',
        vmin=0, vmax=5
    )
    
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=11)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('SST Gradient (°C / 1000km)')
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, 
                      color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_extent([115, 160, 20, 50], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax, grad_da

# 겨울철 평균 SST gradient (전선이 강할 때)
fig, ax, grad = plot_sst_gradient(sst_djf, title='Winter SST Gradient (DJF)')
plt.savefig('sst_gradient.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 8. MLD-SST 상관관계 분석

### 8.1 왜 MLD-SST 관계가 중요한가?

**MLD(Mixed Layer Depth, 혼합층 깊이)**는 해양 표층에서 온도/밀도가 균일한 층의 깊이입니다.

연구 로드맵에서 언급: "SST 오차의 70% 이상은 수직 혼합 과정에서 발생"

물리적 관계:
- MLD가 **깊어지면** → 더 많은 물이 섞임 → SST 변화가 완충됨 (변동성 감소)
- MLD가 **얕아지면** → 표층이 고립됨 → SST가 대기 영향에 민감 (변동성 증가)
- **겨울**: 강한 바람 + 냉각 → MLD 깊어짐
- **여름**: 약한 바람 + 가열 → MLD 얕아짐

### 8.2 MLD 데이터 불러오기

```python
# HYCOM에서 MLD 데이터 가져오기
# 또는 ARGO 기반 MLD 기후값 사용

# 예시: HYCOM에서 MLD 추출
ds_hycom = xr.open_dataset('hycom_data.nc')  # 실제 경로로 변경

# MLD 변수명은 데이터마다 다름
# surf_el, mlp_or_kraus, mixed_layer_depth 등 확인 필요
mld = ds_hycom['mixed_layer_depth']  # 예시

# 쿠로시오 영역으로 자르기
mld_kuroshio = mld.sel(lon=slice(115, 160), lat=slice(20, 50))
```

### 8.3 MLD 계절 변동 분석

```python
def analyze_mld_seasonality(mld_data):
    """
    MLD 계절 변동 분석
    """
    # 월별 기후값
    mld_monthly = mld_data.groupby('time.month').mean(dim='time')
    
    # 영역 평균
    mld_monthly_mean = mld_monthly.mean(dim=['lat', 'lon'])
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 월별 MLD 변화
    ax1 = axes[0]
    months = range(1, 13)
    ax1.plot(months, mld_monthly_mean.values, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('MLD (m)')
    ax1.set_title('Monthly Mean MLD in Kuroshio Region')
    ax1.set_xticks(months)
    ax1.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # MLD는 깊이이므로 아래로 증가
    
    # 2. 겨울 vs 여름 MLD 분포
    ax2 = axes[1]
    mld_winter = mld_data.where(mld_data['time.season'] == 'DJF', drop=True).mean(dim='time')
    mld_summer = mld_data.where(mld_data['time.season'] == 'JJA', drop=True).mean(dim='time')
    
    ax2.hist(mld_winter.values.flatten(), bins=30, alpha=0.5, label='Winter (DJF)', density=True)
    ax2.hist(mld_summer.values.flatten(), bins=30, alpha=0.5, label='Summer (JJA)', density=True)
    ax2.set_xlabel('MLD (m)')
    ax2.set_ylabel('Density')
    ax2.set_title('MLD Distribution: Winter vs Summer')
    ax2.legend()
    
    plt.tight_layout()
    return fig

# fig = analyze_mld_seasonality(mld_kuroshio)
# plt.savefig('mld_seasonality.png', dpi=150)
```

### 8.4 MLD-SST 상관관계 계산

```python
def calculate_correlation_map(var1, var2, dim='time'):
    """
    두 변수 간 공간별 상관계수 계산
    
    Parameters:
    -----------
    var1, var2 : xarray.DataArray, 같은 좌표를 가진 데이터
    dim : str, 상관계수를 계산할 차원 (보통 'time')
    
    Returns:
    --------
    xarray.DataArray of correlation coefficients
    """
    # 평균 제거
    var1_anom = var1 - var1.mean(dim=dim)
    var2_anom = var2 - var2.mean(dim=dim)
    
    # 공분산
    covariance = (var1_anom * var2_anom).mean(dim=dim)
    
    # 표준편차
    std1 = var1.std(dim=dim)
    std2 = var2.std(dim=dim)
    
    # 상관계수
    correlation = covariance / (std1 * std2)
    
    return correlation

# MLD와 SST anomaly 상관계수
# 주의: 시간 축이 맞아야 함
# corr_mld_sst = calculate_correlation_map(mld_kuroshio, sst_anom)
```

### 8.5 상관관계 지도 시각화

```python
def plot_correlation_map(corr_data, title='Correlation', significance_mask=None):
    """
    상관계수 지도 시각화
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # 상관계수는 -1 ~ 1
    im = ax.contourf(
        corr_data.lon, corr_data.lat, corr_data,
        transform=ccrs.PlateCarree(),
        levels=np.linspace(-1, 1, 21),
        cmap='RdBu_r',
        extend='neither'
    )
    
    # 유의성 마스크 (해칭)
    if significance_mask is not None:
        ax.contourf(
            corr_data.lon, corr_data.lat, significance_mask,
            transform=ccrs.PlateCarree(),
            levels=[0.5, 1.5],
            hatches=['...'],
            colors='none'
        )
    
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=11)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Correlation Coefficient')
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, 
                      color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_extent([115, 160, 20, 50], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax

# plot_correlation_map(corr_mld_sst, title='Correlation: MLD vs SST Anomaly')
```

### 8.6 시차 상관(Lag Correlation) 분석

MLD 변화가 SST에 영향을 주기까지 시간 지연이 있을 수 있습니다.

```python
def lag_correlation(var1, var2, max_lag=6):
    """
    시차 상관분석
    
    Parameters:
    -----------
    var1, var2 : 1D array (영역 평균 시계열)
    max_lag : int, 최대 시차 (월)
    
    Returns:
    --------
    lags : array of lag values
    correlations : array of correlation at each lag
    """
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = []
    
    for lag in lags:
        if lag < 0:
            corr, _ = stats.pearsonr(var1[:lag], var2[-lag:])
        elif lag > 0:
            corr, _ = stats.pearsonr(var1[lag:], var2[:-lag])
        else:
            corr, _ = stats.pearsonr(var1, var2)
        correlations.append(corr)
    
    return lags, np.array(correlations)

# 영역 평균 시계열
# mld_ts = mld_kuroshio.mean(dim=['lat', 'lon'])
# sst_anom_ts = sst_anom.mean(dim=['lat', 'lon'])

# lags, corr = lag_correlation(mld_ts.values, sst_anom_ts.values, max_lag=6)

# 시각화
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.bar(lags, corr, color='steelblue', edgecolor='black')
# ax.axhline(0, color='black', linewidth=0.5)
# ax.set_xlabel('Lag (months, positive = MLD leads)')
# ax.set_ylabel('Correlation')
# ax.set_title('Lag Correlation: MLD vs SST Anomaly')
# plt.show()
```

---

## 9. 종합 연습 과제

### 과제 1: 쿠로시오 SST 계절 변동성 완전 분석

**목표**: 1년치 OISST 데이터로 쿠로시오 해역의 SST 특성 파악

**수행 단계**:
1. 2023년 OISST 일별 데이터 다운로드 (또는 제공된 샘플 사용)
2. 쿠로시오 영역(115-160°E, 20-50°N) 추출
3. 월별 평균 SST 계산
4. 각 월의 SST 지도 12개 (3×4 subplot)
5. 영역 평균 SST 연간 변화 그래프
6. 결과 해석: 언제 가장 따뜻하고/차가운가? 공간적으로 어디가 가장 변동이 큰가?

### 과제 2: 겨울 vs 여름 SST Gradient 비교

**목표**: 쿠로시오 전선의 계절 변화 분석

**수행 단계**:
1. DJF(겨울)와 JJA(여름) 평균 SST 계산
2. 각 계절의 SST gradient magnitude 계산
3. 2개 패널 지도 (겨울/여름 gradient 비교)
4. 전선 강도(gradient maximum)의 위도 차이 분석
5. 결과 해석: 겨울에 전선이 더 강한 이유는?

### 과제 3: SST Anomaly 전파 분석

**목표**: Hovmöller diagram으로 SST anomaly 전파 특성 파악

**수행 단계**:
1. 5년 이상의 SST 데이터에서 월별 anomaly 계산
2. 30-40°N 위도대 평균 (경도-시간 Hovmöller)
3. 140-150°E 경도대 평균 (위도-시간 Hovmöller)
4. 전파 방향과 대략적인 속도 추정
5. 결과 해석: 동쪽/서쪽 전파가 관측되는가? ENSO 신호가 보이는가?

### 과제 4: MLD-SST 역학 관계 규명

**목표**: 혼합층 깊이와 SST 변동의 관계 정량화

**수행 단계**:
1. HYCOM 또는 ARGO 기반 MLD 데이터 확보
2. MLD 월별 기후값 계산 및 계절 변동 분석
3. SST anomaly와 MLD anomaly의 공간 상관 지도
4. 영역 평균 시계열의 시차 상관 분석
5. 결과 해석: MLD가 SST를 선행(lead)하는가? 상관이 높은 지역은 어디인가?

---

## 부록: 유용한 함수 모음

```python
# 이 파일을 ocean_utils.py로 저장하고 import하여 사용

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 영역 상수
KUROSHIO_BOX = {'lon': (115, 160), 'lat': (20, 50)}

def load_and_subset(filepath, lon_range=None, lat_range=None):
    """데이터 로드 및 영역 추출"""
    ds = xr.open_dataset(filepath)
    if lon_range and lat_range:
        ds = ds.sel(lon=slice(*lon_range), lat=slice(*lat_range))
    return ds

def quick_map(data, title='', cmap='coolwarm', **kwargs):
    """빠른 지도 시각화"""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, **kwargs)
    
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.gridlines(draw_labels=True)
    ax.set_title(title)
    
    return fig, ax

def monthly_climatology(data, dim='time'):
    """월별 기후값 계산"""
    return data.groupby(f'{dim}.month').mean(dim=dim)

def monthly_anomaly(data, dim='time'):
    """월별 anomaly 계산"""
    clim = monthly_climatology(data, dim)
    return data.groupby(f'{dim}.month') - clim

def spatial_correlation(var1, var2, dim='time'):
    """공간별 상관계수 계산"""
    v1_anom = var1 - var1.mean(dim=dim)
    v2_anom = var2 - var2.mean(dim=dim)
    cov = (v1_anom * v2_anom).mean(dim=dim)
    return cov / (var1.std(dim=dim) * var2.std(dim=dim))
```

---

## 다음 단계 예고

이 가이드북의 내용을 완전히 숙달했다면, 다음 단계로:

1. **dask를 이용한 대용량 데이터 처리**: 수십 GB 데이터를 메모리 걱정 없이
2. **MITgcm 출력 파일 다루기**: .data/.meta 바이너리 파일 읽기
3. **PyTorch 기초**: CNN으로 SST 패턴 학습 (이건 이 단계 완료 후!)

---

*작성일: 2026-01-20*
*목적: MITgcm 연구 준비를 위한 해양 데이터 분석 기초*
