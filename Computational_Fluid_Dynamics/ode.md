# ODE 수치해석 정리 (시험 대비)

원본 강의자료를 기준으로 정리했고, **눈에 띄는 수식 오탈자와 개념상 혼동되는 부분은 표준 ODE 수치해석 기준으로 함께 보정**했다.

---

## 0. 먼저: 강의자료 오탈자 / 보정 포인트

> 아래는 PDF 페이지 번호 기준이다.

1. **p.10 Trapezoidal method**
   - 슬라이드 첫 식에 음수 부호가 들어가 있는데, **정확한 식은 플러스**이다.
   - 잘못된 식:
     ```math
     \frac{y(t+h)-y(t)}{h}=-\frac12\left[f(t,y(t))+f(t+h,y(t+h))\right]
     ```
   - 올바른 식:
     ```math
     \frac{y(t+h)-y(t)}{h}=\frac12\left[f(t,y(t))+f(t+h,y(t+h))\right]
     ```

2. **p.13 2nd-order Adams-Bashforth**
   - 마지막 항이 $f(t_{n-1},y_n)$로 되어 있는데, **$f(t_{n-1},y_{n-1})$** 가 맞다.
   - 올바른 식:
     ```math
     y_{n+1}=y_n+\frac{3h}{2}f(t_n,y_n)-\frac{h}{2}f(t_{n-1},y_{n-1})
     ```

3. **p.15 3rd-order Adams-Bashforth**
   - 두 번째 항이 $f(t_{n-1},y_n)$로 되어 있는데, **$f(t_{n-1},y_{n-1})$** 가 맞다.
   - 특성방정식 줄에서도 중간 항에 $\sigma$ 대신 다른 문자가 들어간 듯한데, **일관되게 $\sigma$** 를 써야 한다.
   - 올바른 식:
     ```math
     y_{n+1}=y_n+\frac{23h}{12}f(t_n,y_n)-\frac{16h}{12}f(t_{n-1},y_{n-1})+\frac{5h}{12}f(t_{n-2},y_{n-2})
     ```
   - $y'=-ay$에 대한 특성방정식:
     ```math
     \sigma^3-\left(1-\frac{23}{12}ah\right)\sigma^2-\frac{4}{3}ah\,\sigma+\frac{5}{12}ah=0
     ```

4. **p.16 Improved Euler / RK2 (midpoint form)**
   - 예측값 식의 부호가 잘못되어 있다. **마이너스가 아니라 플러스**이다.
   - 둘째 줄은 $y_n$이 아니라 **$y_{n+1}$** 이어야 한다.
   - 올바른 식:
     ```math
     y_n^\*=y_n+\frac{h}{2}f(t_n,y_n),\qquad
     y_{n+1}=y_n+h\,f\!\left(t_n+\frac{h}{2},\,y_n^\*\right)
     ```

5. **p.17–19 RK2 / Heun**
   - 여러 곳에서 $y_i$가 보이는데, 문맥상 **$y_n$** 이 맞다.
   - Heun 표기에서 $k_1=f(\cdot)$, $k_2=f(\cdot)$로 두면 계수는 **$b_1=c_{11}=1$** 이다.
   - Heun의 올바른 식:
     ```math
     k_1=f(t_n,y_n),\qquad
     k_2=f(t_n+h,\;y_n+h k_1)
     ```
     ```math
     y_{n+1}=y_n+\frac{h}{2}(k_1+k_2)
     ```

6. **p.20 RK4**
   - 첫 줄의 $y_i$는 **$y_n$** 이 맞다.
   - 올바른 식:
     ```math
     y_{n+1}=y_n+\frac{h}{6}(k_1+2k_2+2k_3+k_4)
     ```

7. **p.23 Stiff system**
   - Forward Euler recurrence는
     ```math
     U^{n+1}=(I+hA)U^n
     ```
     이다.
   - 따라서 닫힌형은
     ```math
     U^n=(I+hA)^n U^0
     ```
     이다. 슬라이드의 $U^{n+1}=(I+hA)^nU^0$는 인덱스가 어긋나 있다.

8. **p.12 Leapfrog의 “Accuracy $\sim O(h^0)$”**
   - 이 표기는 **formal order 관점에서는 맞지 않다.**
   - Leapfrog 자체는 **2차 정확도(global $O(h^2)$)** 를 갖는 방법이다.
   - 다만 **$y'=-ay$ 같은 dissipative 문제에서는 parasitic mode 때문에 안정성이 매우 나빠 실전에서는 불안정하게 보일 수 있다.**
   - 즉, **정확도 자체가 0차인 것이 아니라, 안정성이 문제**다.

---

## 1. 강의 전체를 관통하는 큰 그림

연속 문제
```math
y'(t)=f(t,y), \qquad y(t_0)=y_0
```
를 직접 다루기 어렵기 때문에, 시간격자
```math
t_n=t_0+nh,\qquad y_n\approx y(t_n)
```
를 두고 **연속 미분방정식**을 **이산 대수방정식**으로 바꾼다.

강의 초반 슬라이드의 흐름을 한 줄로 요약하면:

- infinite-dimensional $\to$ finite-dimensional
- differential $\to$ algebraic
- continuous $\to$ discrete

즉, **ODE의 해 $y(t)$** 대신 **격자점에서의 근사값 $y_n$** 을 구하는 것이 수치해석의 핵심이다.

---

## 2. Taylor 전개: 유도와 오차분석의 출발점

시험에서 제일 중요한 도구는 Taylor 전개다.

```math
y(t+h)=y(t)+hy'(t)+\frac{h^2}{2}y''(t)+\frac{h^3}{6}y^{(3)}(t)+O(h^4)
```

```math
y(t-h)=y(t)-hy'(t)+\frac{h^2}{2}y''(t)-\frac{h^3}{6}y^{(3)}(t)+O(h^4)
```

따라서 자주 쓰는 차분근사는:

### (1) Forward difference
```math
y'(t)=\frac{y(t+h)-y(t)}{h}+O(h)
```

### (2) Backward difference
```math
y'(t)=\frac{y(t)-y(t-h)}{h}+O(h)
```

### (3) Central difference
```math
y'(t)=\frac{y(t+h)-y(t-h)}{2h}+O(h^2)
```

### (4) 오차를 줄이는 방법
오차가 대략
```math
E(h)\approx C h^p
```
꼴이면 줄이는 방법은 두 가지다.

1. **$h$를 더 작게 잡는다.**
2. **더 높은 차수(order)의 방법을 쓴다.**

예:
- Forward Euler: global error $O(h)$
- Trapezoidal, RK2, AB2: global error $O(h^2)$
- AB3: global error $O(h^3)$
- RK4: global error $O(h^4)$

단, **stiff 문제에서는 단순히 high-order explicit method를 쓴다고 해결되지 않는다.**
그 경우는 **stability restriction** 때문에 implicit method가 더 중요하다.

---

## 3. 오차, consistency, stability, convergence

이 부분은 교수님이 개념적으로 중요하게 보는 포인트다.

## 3.1 Local truncation error (LTE)

한 스텝에서 exact solution을 대입했을 때 생기는 잔차이다.

- 보통 **$p$차 방법**이면
  - local truncation error: $O(h^{p+1})$
  - global error: $O(h^p)$

예:
- Forward Euler: LTE $O(h^2)$, global $O(h)$
- Trapezoidal: LTE $O(h^3)$, global $O(h^2)$

## 3.2 Consistency

쉽게 말하면,
```math
h\to 0 \quad \text{일 때 차분식이 원래 ODE를 제대로 닮아가는가?}
```
를 묻는다.

즉, **local truncation error가 0으로 가면 consistent** 하다.

## 3.3 Stability

작은 오차(초기오차, 반올림오차, 이전 스텝 오차)가 시간이 지나도 폭발하지 않는 성질이다.

안정성은 두 층위가 있다.

### (a) Zero-stability
- 특히 **multistep method** 에서 중요
- $h\to 0$일 때 방법 자체가 오차를 과도하게 증폭시키지 않는가를 보는 개념

### (b) Absolute stability
- 시험에서는 보통 test equation
  ```math
  y'=\lambda y
  ```
  또는 특히
  ```math
  y'=-ay,\qquad a>0
  ```
  에 대하여 본다.
- 한 스텝 증폭인자(amplification factor) $\sigma$가
  ```math
  |\sigma|\le 1
  ```
  이면 안정이라고 본다.

## 3.4 Convergence

```math
h\to 0 \quad \Rightarrow \quad y_n\to y(t_n)
```
이면 convergent 하다.

즉, **격자 간격을 줄이면 수치해가 exact solution로 가는가**가 convergence다.

## 3.5 Stability + Consistency $\Rightarrow$ Convergence

교수님이 말한 요지는 이걸로 정리하면 된다.

- **Lax equivalence theorem**  
  잘 정립된 선형 문제의 유한차분 근사에서  
  **consistency + stability $\Rightarrow$ convergence**

- **ODE의 linear multistep method**에서는 더 정확히
  **Dahlquist equivalence theorem**
  - consistency + zero-stability $\Leftrightarrow$ convergence

시험에서는 보통 다음처럼 기억하면 충분하다.

> **일관성(consistency) + 안정성(stability)이 있으면 수렴성(convergence)이 따라온다.**

단, ODE multistep 문맥에서는 정확히는 **zero-stability**가 핵심이다.

---

## 4. Explicit / Implicit 판정

## 4.1 Explicit method
새로운 값 $y_{n+1}$를 **이미 알고 있는 값들만으로 바로 계산**할 수 있다.

예:
- Forward Euler
- Leapfrog
- AB2, AB3
- RK2, RK4

## 4.2 Implicit method
새로운 값 $y_{n+1}$가 식의 오른쪽에도 들어가 있어서, **매 스텝마다 방정식을 풀어야 한다.**

예:
- Backward Euler
- Trapezoidal

예를 들어 Backward Euler는
```math
y_{n+1}=y_n+h f(t_{n+1},y_{n+1})
```
이므로 $y_{n+1}$가 양변에 있다.

- $f$가 선형이면 직접 풀 수 있다.
- $f$가 비선형이면 Newton method 같은 반복법이 필요하다.

---

## 5. 시험용 표준 test equation: $y'=-ay$

교수님이 특히 강조한 핵심 모델이다.

```math
y'=-ay,\qquad a>0
```

exact solution:
```math
y(t)=y_0 e^{-at}
```

한 스텝 exact amplification factor는
```math
e^{-ah}
```
이다.

이제
```math
\alpha=ah>0
```
라고 두고 각 방법의 수치 증폭인자 $\sigma$를 비교하면,  
**stable / unstable / conditionally stable / unconditionally stable** 를 바로 판정할 수 있다.

---

## 6. 각 방법별 정리

---

## 6.1 Forward Euler (Explicit Euler)

### 유도
Forward difference:
```math
y'(t_n)\approx \frac{y(t_{n+1})-y(t_n)}{h}
```
를 $y'=f(t,y)$에 대입하면
```math
\frac{y_{n+1}-y_n}{h}=f(t_n,y_n)
```
따라서
```math
\boxed{y_{n+1}=y_n+h f(t_n,y_n)}
```

### 성질
- **explicit**
- **self-starting**
- **1차 방법**
- LTE $O(h^2)$, global error $O(h)$

### $y'=-ay$에서의 안정성
```math
y_{n+1}=y_n-hay_n=(1-\alpha)y_n
```
따라서
```math
\boxed{\sigma=1-\alpha}
```

안정조건:
```math
|\sigma|=|1-\alpha|\le 1
```
즉
```math
\boxed{0\le \alpha\le 2}
\qquad\Longleftrightarrow\qquad
\boxed{0\le h\le \frac{2}{a}}
```

### 예시
- $y'=-y$: $h\le 2$
- $y'=-2026y$: $h\le \dfrac{2}{2026}\approx 9.87\times 10^{-4}$

### 해석
Forward Euler는 가장 단순하지만,
- 정확도는 낮고
- stiff 문제에서는 $h$를 엄청 작게 잡아야 한다.

---

## 6.2 Backward Euler

### 유도
Backward difference를 $t_{n+1}$에서 쓰면
```math
y'(t_{n+1})\approx \frac{y(t_{n+1})-y(t_n)}{h}
```
따라서
```math
\frac{y_{n+1}-y_n}{h}=f(t_{n+1},y_{n+1})
```
즉
```math
\boxed{y_{n+1}=y_n+h f(t_{n+1},y_{n+1})}
```

### 성질
- **implicit**
- **self-starting**
- **1차 방법**
- LTE $O(h^2)$, global error $O(h)$

### $y'=-ay$에서의 안정성
```math
y_{n+1}=y_n-ha y_{n+1}
```
```math
(1+\alpha)y_{n+1}=y_n
```
```math
\boxed{\sigma=\frac{1}{1+\alpha}}
```

모든 $\alpha>0$에 대해
```math
0<\frac{1}{1+\alpha}<1
```
이므로
```math
\boxed{\text{unconditionally stable}}
```

### 중요한 해석
- **A-stable**
- 더 나아가 **L-stable**
- stiff 문제에 매우 강하다.

### 주의
unconditionally stable 이라고 해서 **아무리 큰 $h$를 써도 정확한 것은 아니다.**  
안정성과 정확도는 다른 개념이다.

---

## 6.3 Trapezoidal method

### 유도 1: 적분 관점
```math
y_{n+1}=y_n+\int_{t_n}^{t_{n+1}} f(t,y(t))\,dt
```
여기서 적분을 trapezoidal rule로 근사하면
```math
\int_{t_n}^{t_{n+1}} f(t,y(t))\,dt
\approx \frac{h}{2}\left[f(t_n,y_n)+f(t_{n+1},y_{n+1})\right]
```
따라서
```math
\boxed{
y_{n+1}=y_n+\frac{h}{2}\left[f(t_n,y_n)+f(t_{n+1},y_{n+1})\right]
}
```

### 유도 2: Forward Euler + Backward Euler 평균
FE와 BE를 평균낸 형태로 봐도 된다.

### 성질
- **implicit**
- **self-starting**
- **2차 방법**
- LTE $O(h^3)$, global error $O(h^2)$

### $y'=-ay$에서의 안정성
```math
y_{n+1}=y_n-\frac{\alpha}{2}(y_n+y_{n+1})
```
정리하면
```math
\left(1+\frac{\alpha}{2}\right)y_{n+1}
=
\left(1-\frac{\alpha}{2}\right)y_n
```
따라서
```math
\boxed{
\sigma=\frac{1-\alpha/2}{1+\alpha/2}
}
```

모든 $\alpha>0$에 대해 $|\sigma|<1$ 이므로
```math
\boxed{\text{unconditionally stable}}
```

### 추가 개념
- **A-stable**
- 하지만 $\alpha\to\infty$일 때 $\sigma\to -1$ 이므로 **L-stable은 아니다.**
- 매우 stiff한 문제에서는 damping이 충분히 강하지 않을 수 있다.

---

## 6.4 Leapfrog scheme

### 유도
Central difference를 사용하면
```math
y'(t_n)\approx \frac{y(t_{n+1})-y(t_{n-1})}{2h}
```
따라서
```math
\frac{y_{n+1}-y_{n-1}}{2h}=f(t_n,y_n)
```
즉
```math
\boxed{y_{n+1}=y_{n-1}+2h f(t_n,y_n)}
```

### 성질
- **explicit**
- **2-step**
- **self-starting 아님**
- **formal order 2**
- LTE $O(h^3)$, global error $O(h^2)$

### $y'=-ay$에서의 안정성
```math
y_{n+1}=y_{n-1}-2\alpha y_n
```

```math
y_n=C\sigma^n
```
라고 두면
```math
\sigma^{n+1}=\sigma^{n-1}-2\alpha \sigma^n
```
즉
```math
\boxed{\sigma^2+2\alpha \sigma-1=0}
```

근은
```math
\boxed{
\sigma_{\pm}=-\alpha\pm\sqrt{1+\alpha^2}
}
```

여기서
- $\sigma_+\approx 1-\alpha+\alpha^2/2+\cdots$ : physical mode
- $\sigma_-\approx -1-\alpha-\alpha^2/2-\cdots$ : parasitic mode

그런데
```math
|\sigma_-|>1 \qquad (\alpha>0)
```
이므로 $y'=-ay$ 같은 dissipative 문제에서는
```math
\boxed{\text{unstable}}
```
하게 된다.

### 핵심 해석
Leapfrog는 **방법 자체는 2차**이지만,
- 두 개의 root가 생기고
- 그 중 하나가 parasitic mode가 되어
- 오차나 round-off가 섞이면 크게 증폭된다.

그래서 슬라이드에서는 practically unstable하다는 메시지를 주고 싶었던 것으로 보인다.

---

## 6.5 2nd-order Adams-Bashforth (AB2)

### 기본식
```math
\boxed{
y_{n+1}=y_n+\frac{3h}{2}f_n-\frac{h}{2}f_{n-1}
}
\qquad
(f_n:=f(t_n,y_n))
```

즉
```math
\boxed{
y_{n+1}=y_n+\frac{3h}{2}f(t_n,y_n)-\frac{h}{2}f(t_{n-1},y_{n-1})
}
```

### Taylor로 계수 유도
다음 꼴을 가정한다.
```math
y_{n+1}=y_n+h\big(a f_n+b f_{n-1}\big)
```

한편
```math
f_{n-1}=f_n-hf_n'+\frac{h^2}{2}f_n''+O(h^3)
```

따라서
```math
y_{n+1}=y_n+h(a+b)f_n-bh^2 f_n'+\frac{b}{2}h^3 f_n''+\cdots
```

exact expansion은
```math
y_{n+1}=y_n+h f_n+\frac{h^2}{2}f_n'+\frac{h^3}{6}f_n''+\cdots
```

계수 비교:
```math
a+b=1,\qquad -b=\frac12
```
따라서
```math
b=-\frac12,\qquad a=\frac32
```

### 성질
- **explicit**
- **2-step**
- **self-starting 아님**
- **2차 방법**
- LTE $O(h^3)$, global error $O(h^2)$

### $y'=-ay$에서의 안정성
```math
y_{n+1}=y_n-\frac{3}{2}\alpha y_n+\frac12 \alpha y_{n-1}
```
즉
```math
y_{n+1}=\left(1-\frac32\alpha\right)y_n+\frac12 \alpha y_{n-1}
```

```math
y_n=C\sigma^n
```
를 넣으면
```math
\boxed{
\sigma^2-\left(1-\frac32\alpha\right)\sigma-\frac12\alpha=0
}
```

근은
```math
\boxed{
\sigma=\frac12\left[\left(1-\frac32\alpha\right)\pm \sqrt{1-\alpha+\frac94\alpha^2}\right]
}
```

안정구간은
```math
\boxed{0\le \alpha\le 1}
\qquad\Longleftrightarrow\qquad
\boxed{0\le h\le \frac{1}{a}}
```

### 예시
- $y'=-y$: $h\le 1$
- $y'=-2026y$: $h\le \dfrac{1}{2026}\approx 4.94\times 10^{-4}$

### 핵심
Forward Euler보다 더 높은 차수이지만,  
stability restriction은 오히려 더 빡빡할 수 있다.

---

## 6.6 3rd-order Adams-Bashforth (AB3)

### 기본식
```math
\boxed{
y_{n+1}
=
y_n+\frac{23h}{12}f_n-\frac{16h}{12}f_{n-1}+\frac{5h}{12}f_{n-2}
}
```

즉
```math
\boxed{
y_{n+1}
=
y_n+\frac{23h}{12}f(t_n,y_n)
-\frac{16h}{12}f(t_{n-1},y_{n-1})
+\frac{5h}{12}f(t_{n-2},y_{n-2})
}
```

### Taylor로 계수 유도
다음을 가정한다.
```math
y_{n+1}=y_n+h\big(a f_n+b f_{n-1}+c f_{n-2}\big)
```

전개:
```math
f_{n-1}=f_n-hf_n'+\frac{h^2}{2}f_n''-\frac{h^3}{6}f_n'''+\cdots
```

```math
f_{n-2}=f_n-2hf_n'+2h^2 f_n''-\frac{4}{3}h^3 f_n'''+\cdots
```

계수 비교를 하면
```math
a+b+c=1
```
```math
-b-2c=\frac12
```
```math
\frac{b}{2}+2c=\frac13
```

이를 풀면
```math
a=\frac{23}{12},\qquad b=-\frac{16}{12},\qquad c=\frac{5}{12}
```

### 성질
- **explicit**
- **3-step**
- **self-starting 아님**
- **3차 방법**
- LTE $O(h^4)$, global error $O(h^3)$

### $y'=-ay$에서의 안정성
```math
y_{n+1}
=
\left(1-\frac{23}{12}\alpha\right)y_n
+\frac{4}{3}\alpha y_{n-1}
-\frac{5}{12}\alpha y_{n-2}
```

```math
y_n=C\sigma^n
```
를 넣으면
```math
\boxed{
\sigma^3
-
\left(1-\frac{23}{12}\alpha\right)\sigma^2
-
\frac{4}{3}\alpha \sigma
+
\frac{5}{12}\alpha
=0
}
```

안정구간은
```math
\boxed{0\le \alpha\le \frac{6}{11}}
\qquad\Longleftrightarrow\qquad
\boxed{0\le h\le \frac{6}{11a}}
```

### 예시
- $y'=-y$: $h\le 6/11\approx 0.54545$
- $y'=-2026y$: $h\le \dfrac{6}{11\cdot 2026}\approx 2.69\times 10^{-4}$

### 핵심
AB3는 정확도는 더 좋지만, 안정조건은 더 빡세다.  
즉, **“higher-order = 무조건 더 좋다”는 아니다.**

---

## 6.7 RK2: Improved Euler / Midpoint / Heun

강의 PDF에 RK2 family가 같이 나와 있어서 짧게 정리한다.

### (a) Midpoint / Improved Euler
```math
y_n^\*=y_n+\frac{h}{2}f(t_n,y_n)
```
```math
\boxed{
y_{n+1}=y_n+h\,f\!\left(t_n+\frac{h}{2},\,y_n^\*\right)
}
```

### (b) Heun
```math
k_1=f(t_n,y_n),\qquad
k_2=f(t_n+h,\;y_n+h k_1)
```
```math
\boxed{
y_{n+1}=y_n+\frac{h}{2}(k_1+k_2)
}
```

### 일반 RK2 조건
```math
y_{n+1}=y_n+a k_1+b k_2
```
```math
k_1=h f(t_n,y_n),\qquad
k_2=h f(t_n+\alpha h,\;y_n+\beta k_1)
```
2차가 되려면
```math
a+b=1,\qquad \alpha b=\frac12,\qquad \beta b=\frac12
```

### 성질
- **explicit**
- **self-starting**
- **2차 방법**
- $y'=-ay$에서
  ```math
  \sigma=1-\alpha+\frac{\alpha^2}{2}
  ```
  안정조건은
  ```math
  0\le \alpha\le 2
  ```

---

## 6.8 RK4

```math
\boxed{
y_{n+1}=y_n+\frac{h}{6}(k_1+2k_2+2k_3+k_4)
}
```
```math
k_1=f(t_n,y_n)
```
```math
k_2=f\!\left(t_n+\frac{h}{2},\,y_n+\frac{h}{2}k_1\right)
```
```math
k_3=f\!\left(t_n+\frac{h}{2},\,y_n+\frac{h}{2}k_2\right)
```
```math
k_4=f(t_n+h,\;y_n+h k_3)
```

### 성질
- **explicit**
- **self-starting**
- **4차 방법**
- LTE $O(h^5)$, global error $O(h^4)$

### $y'=-ay$에서
```math
\boxed{
\sigma
=
1-\alpha+\frac{\alpha^2}{2}
-\frac{\alpha^3}{6}
+\frac{\alpha^4}{24}
}
```

실수축 음의 방향 stability interval은 대략
```math
\boxed{0\le \alpha \lesssim 2.785}
```

---

## 7. 방법 비교표

| 방법 | explicit / implicit | self-starting | global order | $y'=-ay$에서 안정조건 |
|---|---|---:|---:|---|
| Forward Euler | explicit | O | 1 | $0\le ah\le 2$ |
| Backward Euler | implicit | O | 1 | 모든 $ah>0$ |
| Trapezoidal | implicit | O | 2 | 모든 $ah>0$ |
| Leapfrog | explicit | X | 2 (formal) | $y'=-ay$에는 unstable |
| AB2 | explicit | X | 2 | $0\le ah\le 1$ |
| AB3 | explicit | X | 3 | $0\le ah\le 6/11$ |
| RK2 (Midpoint/Heun) | explicit | O | 2 | $0\le ah\le 2$ |
| RK4 | explicit | O | 4 | $0\le ah\lesssim 2.785$ |

---

## 8. $\sigma$가 여러 개 나올 때 어떻게 해석하나?

교수님이 강조한 “$\sigma$가 두 개 나온다”, “등비수열 꼴로 본다”는 내용은 이 뜻이다.

예를 들어 2-step recurrence
```math
y_{n+1}=A y_n + B y_{n-1}
```
가 있으면
```math
y_n=C\sigma^n
```
를 가정한다.

그러면
```math
\sigma^{n+1}=A\sigma^n+B\sigma^{n-1}
```
즉
```math
\boxed{\sigma^2-A\sigma-B=0}
```
라는 characteristic equation이 나온다.

근이 $\sigma_1,\sigma_2$이면 일반해는
```math
\boxed{
y_n=c_1\sigma_1^n+c_2\sigma_2^n
}
```
가 된다.

### 안정성 판단
- 모든 root에 대해 $|\sigma_i|\le 1$ 이어야 함
- $|\sigma_i|=1$ 인 root는 **simple root** 여야 함

### 왜 simple root가 중요하나?
중근이면
```math
y_n=(c_1+c_2 n)\sigma^n
```
꼴이 되어 $n$ 때문에 커질 수 있다.

### Leapfrog에서의 해석
Leapfrog는 $y'=-ay$에 대해 root가 두 개 나오고,
그중 하나가 parasitic mode가 된다.
그래서 아주 작은 오차도 시간이 지나며 커질 수 있다.

---

## 9. Self-starting이 안 되는 방법: “first step 빌리기”

다음 방법들은 현재값만으로는 다음값을 시작할 수 없다.

- Leapfrog: $y_0, y_1$ 필요
- AB2: $y_0, y_1$ 필요
- AB3: $y_0, y_1, y_2$ 필요

즉, **초기값은 $y_0$ 하나뿐인데 multistep method는 이전 스텝 값이 더 필요**하다.

그래서 시작할 때는 보통
- Forward Euler
- RK2
- RK4

같은 **one-step method**로 $y_1$, $y_2$를 먼저 만들어서 “빌려온다”.

### 주의
주요 방법의 차수를 유지하려면,  
시동값(starting values)도 **너무 낮은 차수로 만들지 않는 것이 좋다.**

예:
- AB2는 RK2로 시작하면 깔끔하다.
- AB3는 RK3 또는 RK4로 $y_1,y_2$를 만드는 것이 안전하다.

---

## 10. Linear multistep 방법의 공통 틀

많은 방법들은 다음 꼴로 쓸 수 있다.
```math
\sum_{j=0}^{k}\alpha_j y_{n+j}
=
h\sum_{j=0}^{k}\beta_j f_{n+j}
```

예:
- FE:
  ```math
  y_{n+1}-y_n=h f_n
  ```
- BE:
  ```math
  y_{n+1}-y_n=h f_{n+1}
  ```
- Trapezoidal:
  ```math
  y_{n+1}-y_n=\frac{h}{2}(f_n+f_{n+1})
  ```
- Leapfrog:
  ```math
  y_{n+1}-y_{n-1}=2h f_n
  ```
- AB2:
  ```math
  y_{n+1}-y_n=h\left(\frac32 f_n-\frac12 f_{n-1}\right)
  ```
- AB3:
  ```math
  y_{n+1}-y_n=h\left(\frac{23}{12}f_n-\frac{16}{12}f_{n-1}+\frac{5}{12}f_{n-2}\right)
  ```

### Consistency 조건 (개념용)
```math
\rho(\xi)=\sum_{j=0}^{k}\alpha_j \xi^j,\qquad
\eta(\xi)=\sum_{j=0}^{k}\beta_j \xi^j
```
라 하면 consistency는
```math
\boxed{\rho(1)=0,\qquad \rho'(1)=\eta(1)}
```
로 판정한다.

### Zero-stability 조건 (개념용)
```math
\rho(\xi)=0
```
의 root $\xi_i$가 모두
```math
|\xi_i|\le 1
```
를 만족하고, unit circle 위의 root는 simple root여야 한다.

---

## 11. Higher-order ODE $\to$ 1차 연립 ODE

슬라이드 후반부의 핵심은 이것이다.

2차 ODE
```math
y''=g(t,y,y')
```
를 바로 풀지 말고, 새로운 변수
```math
u=y,\qquad v=y'
```
를 두면
```math
u'=v,\qquad v'=g(t,u,v)
```
가 되어 **1차 연립 ODE**로 바뀐다.

즉,
```math
\boxed{
\begin{bmatrix}
u\\ v
\end{bmatrix}'=
\begin{bmatrix}
v\\ g(t,u,v)
\end{bmatrix}
}
```

### 슬라이드 예시
```math
y''=-0.3y'-\sin y,\qquad y(0)=\frac{\pi}{2},\quad y'(0)=0
```
에 대해
```math
u=y,\qquad v=y'
```
를 두면
```math
u'=v
```
```math
v'=-0.3v-\sin u
```

Forward Euler를 쓰면
```math
u^{n+1}=u^n+h v^n
```
```math
v^{n+1}=v^n-0.3h v^n-h\sin(u^n)
```

### 선형계의 행렬형
선형 2차 ODE
```math
y''+c y'+k y=s(t)
```
는
```math
\begin{bmatrix}
y\\ v
\end{bmatrix}'
=
\begin{bmatrix}
0 & 1\\
-k & -c
\end{bmatrix}
\begin{bmatrix}
y\\ v
\end{bmatrix}
+
\begin{bmatrix}
0\\ s(t)
\end{bmatrix}
```
처럼 행렬로 쓸 수 있다.

---

## 12. Stiff system

이 부분도 시험 포인트다.

## 12.1 stiff의 핵심 의미
서로 다른 시간스케일이 매우 크게 섞여 있어서,
- 해는 이미 빨리 감쇠하는 성분을 포함하고
- explicit method는 안정성을 위해 $h$를 엄청 작게 잡아야 하는 상황

즉, **해 자체는 별로 안 어려워 보여도 explicit method가 너무 불편해지는 문제**다.

## 12.2 가장 쉬운 예: $y'=-2026y$
exact solution:
```math
y(t)=y_0 e^{-2026 t}
```

Forward Euler 안정조건:
```math
h\le \frac{2}{2026}\approx 9.87\times 10^{-4}
```

즉, 조금만 $h$가 커도 unstable하다.  
이게 stiff 감각의 시작이다.

## 12.3 슬라이드의 행렬 예
```math
u'=-20u-19v,\qquad v'=-19u-20v
```
즉
```math
U' = A U,\qquad
A=
\begin{bmatrix}
-20 & -19\\
-19 & -20
\end{bmatrix}
```

이 행렬의 eigenvalue는
```math
\lambda_1=-39,\qquad \lambda_2=-1
```
이다.

즉, 해는 대략
```math
c_1 e^{-39t} \phi_1 + c_2 e^{-t}\phi_2
```
꼴이다.

- $e^{-39t}$: 매우 빠르게 사라지는 transient
- $e^{-t}$: 상대적으로 천천히 남는 mode

### 왜 explicit Euler가 불편한가?
Forward Euler는
```math
U^{n+1}=(I+hA)U^n
```
이므로 모든 eigenvalue에 대해
```math
|1+h\lambda_i|\le 1
```
가 필요하다.

가장 큰 제약은 $\lambda=-39$에서 나오므로
```math
|1-39h|\le 1
\quad\Longrightarrow\quad
0\le h\le \frac{2}{39}\approx 0.0513
```

해의 느린 스케일은 $O(1)$인데,  
빠른 mode 때문에 explicit Euler는 $h$를 $0.05$ 이하로 잡아야 한다.

이게 stiff problem에서 implicit method를 쓰는 이유다.

## 12.4 stiff에서 어떤 방법이 유리한가?
- Forward Euler: 불리
- AB2, AB3: 더 불리할 수 있음
- Backward Euler: 매우 유리
- Trapezoidal: 안정하지만 매우 stiff할 때 damping이 약할 수 있음

즉, stiff 문제에서는 보통
```math
\boxed{\text{implicit method, 특히 Backward Euler 계열}}
```
이 더 적합하다.

---

## 13. Ill-conditioned matrix

슬라이드 마지막 장의 요점은 이것이다.

**행렬이 ill-conditioned** 하다는 것은,
- 계수나 우변이 아주 조금만 바뀌어도
- 해가 크게 바뀔 수 있다는 뜻이다.

예시:
```math
\begin{bmatrix}
1 & 1\\
1 & 1.001
\end{bmatrix}
\begin{bmatrix}
x\\y
\end{bmatrix}
=
\begin{bmatrix}
2\\2
\end{bmatrix}
```
이면 해는
```math
(x,y)=(2,0)
```

그런데 우변이 아주 조금 바뀌어
```math
\begin{bmatrix}
1 & 1\\
1 & 1.001
\end{bmatrix}
\begin{bmatrix}
x\\y
\end{bmatrix}
=
\begin{bmatrix}
2\\2.001
\end{bmatrix}
```
가 되면 해는
```math
(x,y)=(1,1)
```
이 된다.

즉, 우변 변화는 $0.001$인데 해는 크게 달라진다.

### 왜 이런가?
이 행렬의 고유값은 대략
```math
\lambda_{\min}\approx 4.99875\times 10^{-4},\qquad
\lambda_{\max}\approx 2.0005
```
이어서
```math
\frac{\lambda_{\max}}{\lambda_{\min}}\approx 4002
```
로 매우 크다.

### stiffness와의 차이
- **stiffness**: 시간적 동역학 문제에서 여러 시간스케일 때문에 explicit method가 힘든 것
- **ill-conditioning**: 대수계에서 작은 perturbation이 해를 크게 흔드는 것

둘은 다른 개념이지만, 둘 다 **수치적으로 민감함**을 뜻한다.

---

## 14. 시험장에서 바로 쓰는 판정 프레임

어떤 방법이 나오면 아래 순서로 보면 된다.

### 1단계. 식을 정확히 쓴다
예:
- FE:
  ```math
  y_{n+1}=y_n+h f(t_n,y_n)
  ```
- BE:
  ```math
  y_{n+1}=y_n+h f(t_{n+1},y_{n+1})
  ```
- Trapezoidal:
  ```math
  y_{n+1}=y_n+\frac{h}{2}(f_n+f_{n+1})
  ```

### 2단계. explicit / implicit 판정
- $y_{n+1}$가 오른쪽에 없으면 explicit
- 있으면 implicit

### 3단계. Taylor로 order 판정
- 1차면 global $O(h)$
- 2차면 global $O(h^2)$
- 3차면 global $O(h^3)$
- 4차면 global $O(h^4)$

### 4단계. $y'=-ay$ 대입
- 한 스텝 factor $\sigma$ 구하기
- multistep이면 characteristic equation 구하기

### 5단계. stability 판정
- one-step: $|\sigma|\le 1$
- multistep: 모든 root에 대해 $|\sigma_i|\le 1$, unit-circle root는 simple

### 6단계. self-starting 여부 확인
- FE, BE, Trapezoidal, RK2, RK4: 가능
- Leapfrog, AB2, AB3: 불가능

### 7단계. stiff 문제인지 생각
- $a$가 크면 explicit는 매우 작은 $h$ 필요
- implicit가 유리

---

## 15. 암기용 한 줄 요약

- **Forward Euler**: 가장 단순, explicit, 1차, 조건부 안정
- **Backward Euler**: implicit, 1차, 무조건 안정, stiff에 강함
- **Trapezoidal**: implicit, 2차, 무조건 안정, 정확도 좋음
- **Leapfrog**: explicit, 2차, self-starting 아님, $y'=-ay$에는 불안정
- **AB2**: explicit, 2차, self-starting 아님, $ah\le 1$
- **AB3**: explicit, 3차, self-starting 아님, $ah\le 6/11$
- **RK2**: explicit, 2차, self-starting, $ah\le 2$
- **RK4**: explicit, 4차, self-starting, 안정구간 넓음
- **stability + consistency $\Rightarrow$ convergence**
- **stiff 문제는 implicit를 먼저 떠올려라**
- **higher-order ODE는 1차 연립계로 바꿔라**

---

## 16. 마지막 체크리스트

시험 직전에는 아래 질문에 바로 답할 수 있으면 된다.

1. 각 방법의 식을 **오탈자 없이** 쓸 수 있는가?
2. 각 방법이 **explicit / implicit** 인지 바로 말할 수 있는가?
3. Taylor 전개로 **왜 그 order가 나오는지** 설명할 수 있는가?
4. $y'=-y$, $y'=-2026y$에 대해 **$\sigma$** 또는 characteristic equation을 세울 수 있는가?
5. 그로부터 **stable / unstable / conditionally stable / unconditionally stable** 를 판정할 수 있는가?
6. Leapfrog, AB2, AB3가 **왜 self-starting이 아닌지** 설명할 수 있는가?
7. 2차 ODE를 **1차 연립 ODE**로 바꿀 수 있는가?
8. stiff와 ill-conditioned를 **구별해서 설명**할 수 있는가?

이 8개를 바로 처리할 수 있으면, 이번 범위의 핵심은 거의 다 잡은 것이다.
