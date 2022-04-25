# 평균변화율


```python
from sympy import Symbol, solve

a = 0
b = 2

m = max(a,b)
n = min(a,b)
x = Symbol('X')

fx = 2 *x **2 + 4 * x + 7
fb = fx.subs(x,m)
fa = fx.subs(x,n)

result = (fb - fa) / (m - n)

print(result)
```

    8
    


```python
from sympy import Symbol, solve
# x값이 1에서 k까지 변할때 평균변화율이 15, k값은?
x = Symbol('X')
k = Symbol('K')

fx = x**2
equation = ((k**2 - 1)/(k -1))-15

print(equation)

solve(equation)

```

    -15 + (K**2 - 1)/(K - 1)
    




    [14]




```python
# 평균변화율 코드 구현
import numpy as np

def average(a,b):
    fx = 2 * x ** 2 + 4 * x +7
    fa = fx.subs(x,a)
    fb = fx.subs(x,b)
    
    result = (fa - fb) / (a - b)
    
    return result

print(average(0,2))
```

    8
    


```python
def calc(x, y):
    result1 = f'{x} + {y} = {x+y}'
    result2 = f'{x} - {y} = {x-y}'
    result3 = f'{x} X {y} = {x*y}'
    result4 = f'{x} / {y} = {x/y}'
    return result1, result2, result3, result4

calc(10, 4)
```




    ('10 + 4 = 14', '10 - 4 = 6', '10 X 4 = 40', '10 / 4 = 2.5')




```python
# 두수에 대한 사칙연산 결과값을 출력 

def calc(x, y):
    result1 = f'{x} + {y} = {x+y}'
    result2 = f'{x} - {y} = {x-y}'
    result3 = f'{x} X {y} = {x*y}'
    # 소숫점 2자리까지 처리 
    result4 = f'{x} / {y} = {(x/y):.2f}'
    return result1, result2, result3, result4

calc(10, 4)
```




    ('10 + 4 = 14', '10 - 4 = 6', '10 X 4 = 40', '10 / 4 = 2.50')




```python
from sympy import Symbol, solve, Derivative, diff

x = Symbol('X')
fx = 2 * x **2 + 4 *x +7

fprime = Derivative(fx, x).doit()
n = fprime.subs({x: 3})

print(fx)
print(fprime)

print("fx에서 x = 3에서의 순간변화율은 " , n , "입니다")
```

    2*X**2 + 4*X + 7
    4*X + 4
    fx에서 x = 3에서의 순간변화율은  16 입니다
    


```python
# 미분을 이용할때 보통 diff 함수를 많이 쓴다. Derivative 너무 길어

x = Symbol('X')
fx = 2 * x **2 + 4 *x +7

fprime2 = diff(fx, x)
n = fprime2.subs({x: 3})

print(fx)
print(fprime)

print("fx에서 x = 3에서의 순간변화율은 " , n , "입니다")
```

    2*X**2 + 4*X + 7
    4*X + 4
    fx에서 x = 3에서의 순간변화율은  16 입니다
    

### 퀴즈 

1) 아래 함수식의 도함수를 구하고 x값이 3일때의 미분계수를 구하여라 

$ f(x) = 2x^2 + 4x + 7 $

2) 도함수를 이용하여 미분계수를 구하는 함수를 정의하고 테스트하여라

```
# diff_print(x) 를 호출하면?

함수 ? 의 도함수는 ?  ,  x 값이 ?일때의 미분계수는 ?




```python
# 1)

x = Symbol('X')
fx = 2 * x **2 + 4 *x + 7

fprime2 = diff(fx, x)
n = fprime2.subs({x: 3})

print(fx)
print(fprime)

print("fx에서 x = 3에서의 순간변화율은 " , n , "입니다")
```

    2*X**2 + 4*X + 7
    4*X + 4
    fx에서 x = 3에서의 순간변화율은  16 입니다
    

# 행열
### LATEX로 행렬 표현 하기    

begin 행 \\ 다음 행\\ 다음행 end  


"\begin{vmatrix} 1 & 34 \\ 67 & 23 \end{vmatrix"
-> 이렇게 표현

```
$ \begin{vmatrix} 1 & 34 \\ 67 & 23 \end{vmatrix} $
```

$ \begin{vmatrix} 1 & 34 \\ 67 & 23 \end{vmatrix} $

# Numpy 배열




```python
arr1 = np.array([1,34,55,67,23]) # 1차원
arr2 = np.array([[1,34],[67,23]]) # 2차원
# 2차원 배열은 () 안에 [] 가 있고 그안에 [],[] 있는 형태


print(arr1, type(arr1))
print()
print(arr2,type(arr2))

# 행렬의 차원 확인 
arr1.ndim, arr2.ndim

arr1.shape, arr2.shape
```

    [ 1 34 55 67 23] <class 'numpy.ndarray'>
    
    [[ 1 34]
     [67 23]] <class 'numpy.ndarray'>
    




    ((5,), (2, 2))




```python
# 1차원 -> 2차원
## 넘파이배열.reshape(row, col)

np.array(([10,20,30,40,50,60])).reshape(2,3)

np.array(([10,20,30,40,50,60])).reshape(3,2)
```




    array([[10, 20],
           [30, 40],
           [50, 60]])




```python
# 2차원 -> 1차원 : 배열명.flatten()

arr5 = np.array([[1,4,5,6],[56,23,45,67],[8,4,6,10]])

print(f'array5 = {arr5}, \n차원 = {arr5.ndim}, 구조 = {arr5.shape}')
# 차원은 dim이 아니라 ndim이다.
```

    array5 = [[ 1  4  5  6]
     [56 23 45 67]
     [ 8  4  6 10]], 
    차원 = 2, 구조 = (3, 4)
    

### 퀴즈 )1~12 로 구성된 숫자를 이용하여 아래와 같은 3행 4열의 행렬을 생성하여라  


$ \begin{vmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \end{vmatrix} $





```python
#arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

#print(arr)

arr1 = np.arange(1, 13).reshape(3,4)

# reshape(-1, 5) 는  col의 갯수가 row의 갯수랑 같다.
```

### 퀴즈)  1~50 사이의 범위에서 홀수로 구성된 행열 ? , ? 를 생성하여라 


```python
arr2 = np.arange(1, 51,2).reshape(-1, 5)
arr2 = np.arange(1, 51,2).reshape(5, 5)
arr2 = np.arange(1, 51,2).reshape(5, -1)

arr2
```




    array([[ 1,  3,  5,  7,  9],
           [11, 13, 15, 17, 19],
           [21, 23, 25, 27, 29],
           [31, 33, 35, 37, 39],
           [41, 43, 45, 47, 49]])



## 전치행렬
##### 넘파이배열.T


```python
arr = np.arange(1,7).reshape(3,2)

arr
```




    array([[1, 2],
           [3, 4],
           [5, 6]])




```python
arr.T
```




    array([[1, 3, 5],
           [2, 4, 6]])



## 특수한 벡터  

np.zeros(갯수) => 0으로 구성된 1차원 행렬  
np.zeros([row,col]) => 0으로 구성된 2차원 행렬  
np.ones(갯수) => 1로 구성된 1차원 행렬  
np.ones([row,col]) => 1로 구성된 2차원 행렬 
##### 항등행렬
np.identity(3, dtype = int)
##### 대각선이 1인 행열 
np.identity(n) : n은 1의 갯수


```python
arr2 = np.ones(5)
arr2
```




    array([1., 1., 1., 1., 1.])




```python
arr22 = np.ones([2,3])
arr22
```




    array([[1., 1., 1.]])




```python
np.identity(3, dtype = int)
```




    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]])



## 슬라이싱  

행열이름 [i : j]  
행열이름 [start : end]  

#### 열 단위 추출
행열명[:, 행인덱스]  
arr[:, 2]

#### 특정 열 슬라이싱 
행열명[:, start:end]  
arr[:, 2:4]

#### 특정행 ~ 특정 열 범위 지정 
행열명[start:end, start:end]  
arr[1:3, 2:4]

## 행렬 연산  

np.dot() // 행렬곱



```python
import numpy as np

x = np.array([1, 2])
w = np.array([[1, 3, 5], [2, 4, 6]])

y = np.dot(x, w)
print(y)
```

    [ 5 11 17]
    

# 확률과 통계


```python
import statistics as st
dir(st)

# dir(모듈명이나 모듈의별칭)
# 함수나 속성을 리스트로 반환 
print(dir(st))
```

    ['Counter', 'Decimal', 'Fraction', 'NormalDist', 'StatisticsError', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_coerce', '_convert', '_exact_ratio', '_fail_neg', '_find_lteq', '_find_rteq', '_isfinite', '_normal_dist_inv_cdf', '_ss', '_sum', 'bisect_left', 'bisect_right', 'erf', 'exp', 'fabs', 'fmean', 'fsum', 'geometric_mean', 'groupby', 'harmonic_mean', 'hypot', 'itemgetter', 'log', 'math', 'mean', 'median', 'median_grouped', 'median_high', 'median_low', 'mode', 'multimode', 'numbers', 'pstdev', 'pvariance', 'quantiles', 'random', 'sqrt', 'stdev', 'tau', 'variance']
    


```python
# 표본 생성 
# np.random.randint(start, end, size=갯수)

data = np.random.randint(1, 101, size=100)
data
```




    array([ 18,  81,  76,  94,  10,  38,  43,  49,  86,  90,  61,  61,  82,
            95,  45,  41,  91,  97,  11,  82,  93,  15,  36,  18,  16,  12,
             1,  89,  14,  69,   6,  76,  35,  91,  76,  64, 100,  34,  55,
            35,  64,  64,  83,  90,  83,  31,  63,  59,  40,  55,   6,  79,
             7,  35,  38,  33,  25,  73,  61,  65,  49,  30,  93,  54,  43,
            60,  77,  22,  63,  75,  26,  86,  17,  33,  37,  26,  17,  68,
             7,  10,  29,   7,  22,  13,  53,  18,  70,  11,  53,  56,  53,
            87,  85,   6,  65,  46,  53,  93,  57,  78])




```python
print(f' 평균 = { st.mean(data)}')
print(f' 중앙값 = { st.median(data)}')
print(f' 최빈값 = { st.mode(data)}') 
print(f' 분산 = { st.variance(data)}') 
print(f' 표준편차 = { st.stdev(data)}') 
```

     평균 = 50
     중앙값 = 53.0
     최빈값 = 53
     분산 = 809
     표준편차 = 28.442925306655784
    

# csv 파일 업로드


```python
ls data
```

     C 드라이브의 볼륨: Windows
     볼륨 일련 번호: 8C04-B11C
    
     C:\workspace\step_math\data 디렉터리
    
    2022-04-15  오후 05:42    <DIR>          .
    2022-04-15  오후 05:42    <DIR>          ..
    2022-04-15  오후 05:35               263 data.csv
    2022-04-15  오후 05:35            10,481 sample.xlsx
                   2개 파일              10,744 바이트
                   2개 디렉터리  397,809,737,728 바이트 남음
    


```python
import pandas as pd

# 엑셀 => 데이타프레임

df4 = pd.read_excel('data/sample.xlsx')
df4.shape
```




    (20, 7)




```python
print(dir(pd))
```

    ['BooleanDtype', 'Categorical', 'CategoricalDtype', 'CategoricalIndex', 'DataFrame', 'DateOffset', 'DatetimeIndex', 'DatetimeTZDtype', 'ExcelFile', 'ExcelWriter', 'Flags', 'Float32Dtype', 'Float64Dtype', 'Float64Index', 'Grouper', 'HDFStore', 'Index', 'IndexSlice', 'Int16Dtype', 'Int32Dtype', 'Int64Dtype', 'Int64Index', 'Int8Dtype', 'Interval', 'IntervalDtype', 'IntervalIndex', 'MultiIndex', 'NA', 'NaT', 'NamedAgg', 'Period', 'PeriodDtype', 'PeriodIndex', 'RangeIndex', 'Series', 'SparseDtype', 'StringDtype', 'Timedelta', 'TimedeltaIndex', 'Timestamp', 'UInt16Dtype', 'UInt32Dtype', 'UInt64Dtype', 'UInt64Index', 'UInt8Dtype', '__builtins__', '__cached__', '__doc__', '__docformat__', '__file__', '__getattr__', '__git_version__', '__loader__', '__name__', '__package__', '__path__', '__spec__', '__version__', '_config', '_hashtable', '_is_numpy_dev', '_lib', '_libs', '_np_version_under1p18', '_testing', '_tslib', '_typing', '_version', 'api', 'array', 'arrays', 'bdate_range', 'compat', 'concat', 'core', 'crosstab', 'cut', 'date_range', 'describe_option', 'errors', 'eval', 'factorize', 'get_dummies', 'get_option', 'infer_freq', 'interval_range', 'io', 'isna', 'isnull', 'json_normalize', 'lreshape', 'melt', 'merge', 'merge_asof', 'merge_ordered', 'notna', 'notnull', 'offsets', 'option_context', 'options', 'pandas', 'period_range', 'pivot', 'pivot_table', 'plotting', 'qcut', 'read_clipboard', 'read_csv', 'read_excel', 'read_feather', 'read_fwf', 'read_gbq', 'read_hdf', 'read_html', 'read_json', 'read_orc', 'read_parquet', 'read_pickle', 'read_sas', 'read_spss', 'read_sql', 'read_sql_query', 'read_sql_table', 'read_stata', 'read_table', 'read_xml', 'reset_option', 'set_eng_float_format', 'set_option', 'show_versions', 'test', 'testing', 'timedelta_range', 'to_datetime', 'to_numeric', 'to_pickle', 'to_timedelta', 'tseries', 'unique', 'util', 'value_counts', 'wide_to_long']
    


```python
# csv => 데이타프레임
df = pd.read_csv('data/data.csv')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>name</th>
      <th>kor</th>
      <th>eng</th>
      <th>mat</th>
      <th>bio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>adam</td>
      <td>67</td>
      <td>87</td>
      <td>90</td>
      <td>98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>andrew</td>
      <td>45</td>
      <td>45</td>
      <td>56</td>
      <td>98</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>ben</td>
      <td>95</td>
      <td>59</td>
      <td>96</td>
      <td>88</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>clark</td>
      <td>65</td>
      <td>94</td>
      <td>89</td>
      <td>98</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>dan</td>
      <td>45</td>
      <td>65</td>
      <td>78</td>
      <td>98</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>noel</td>
      <td>78</td>
      <td>76</td>
      <td>98</td>
      <td>89</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>paul</td>
      <td>87</td>
      <td>67</td>
      <td>65</td>
      <td>56</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>walter</td>
      <td>89</td>
      <td>98</td>
      <td>78</td>
      <td>78</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>oscar</td>
      <td>100</td>
      <td>78</td>
      <td>56</td>
      <td>65</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>martin</td>
      <td>99</td>
      <td>89</td>
      <td>87</td>
      <td>87</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2</td>
      <td>hugh</td>
      <td>98</td>
      <td>45</td>
      <td>56</td>
      <td>54</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2</td>
      <td>henry</td>
      <td>65</td>
      <td>89</td>
      <td>87</td>
      <td>78</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>name</th>
      <th>kor</th>
      <th>eng</th>
      <th>mat</th>
      <th>bio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>adam</td>
      <td>67</td>
      <td>87</td>
      <td>90</td>
      <td>98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>andrew</td>
      <td>45</td>
      <td>45</td>
      <td>56</td>
      <td>98</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>ben</td>
      <td>95</td>
      <td>59</td>
      <td>96</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>kor</th>
      <th>eng</th>
      <th>mat</th>
      <th>bio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>12.000000</td>
      <td>12.0000</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.500000</td>
      <td>77.7500</td>
      <td>74.333333</td>
      <td>78.000000</td>
      <td>82.250000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.522233</td>
      <td>20.1184</td>
      <td>18.217541</td>
      <td>15.874508</td>
      <td>16.276726</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>45.0000</td>
      <td>45.000000</td>
      <td>56.000000</td>
      <td>54.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>65.0000</td>
      <td>63.500000</td>
      <td>62.750000</td>
      <td>74.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.500000</td>
      <td>82.5000</td>
      <td>77.000000</td>
      <td>82.500000</td>
      <td>87.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>95.7500</td>
      <td>89.000000</td>
      <td>89.250000</td>
      <td>98.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.000000</td>
      <td>100.0000</td>
      <td>98.000000</td>
      <td>98.000000</td>
      <td>98.000000</td>
    </tr>
  </tbody>
</table>
</div>


