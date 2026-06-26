# Project Notes

## Why I built this

호텔 예약 취소 예측은 데이터 전처리 선택이 모델 성능에 큰 영향을 주는 문제였다. 단순히 모델 하나를 학습하는 것보다, 결측치 처리, 누수 컬럼 제거, 인코딩 방식, 스케일러, 모델 조합을 비교해보는 것이 더 중요하다고 생각했다.

그래서 이 프로젝트는 하나의 최종 모델만 보여주기보다, 여러 전처리/모델 조합을 같은 기준으로 비교하고 왜 그런 전처리 결정을 했는지 남기는 데 초점을 뒀다.

## What was difficult

가장 어려웠던 부분은 “데이터를 얼마나 지울 것인가”였다. 예를 들어 중복 행이 많았지만, 예약 ID가 없어서 정말 중복 입력인지, 같은 조건의 다른 예약인지 알 수 없었다. 무작정 삭제하면 데이터가 크게 줄고 분포가 바뀔 수 있었다.

또 하나 어려웠던 부분은 leakage였다. `reservation_status`, `reservation_status_date`, `assigned_room_type` 같은 컬럼은 예약 취소 여부와 너무 직접적으로 연결되거나, 예약 이후에 알 수 있는 정보일 수 있었다. 이런 컬럼을 넣으면 점수는 올라가지만 실제 예측 문제로는 부정확해진다.

## Issues I ran into

### 1. 중복 행을 삭제할지 고민했다

데이터셋에는 중복처럼 보이는 행이 많았다. 하지만 booking ID가 없어서 실제 중복인지 판단할 수 없었다. 그래서 중복을 삭제하지 않고, README에 그 이유를 명시했다.

### 2. 누수 컬럼을 제거해야 했다

`reservation_status`와 `reservation_status_date`는 취소 결과를 거의 직접 알려준다. `assigned_room_type`도 예약 시점에는 알 수 없거나 사후 운영 결정일 수 있어 누수 가능성이 있었다. 그래서 이 컬럼들을 모델 입력에서 제거했다.

### 3. 전체 조합 실험이 너무 느렸다

OneHotEncoder를 사용하면 feature 수가 크게 늘고, KNN 같은 모델은 느려졌다. 32개 조합을 전부 전체 데이터로 돌리면 시간이 오래 걸렸다. 그래서 stratified 20,000-row sample에서 조합을 비교하고, 선택된 모델은 전체 train set으로 다시 학습해 sanity check를 했다.

### 4. 정확도만 보면 취소 예측을 잘못 해석할 수 있었다

취소/비취소 비율이 완전히 균형적인 문제는 아니어서 accuracy만 보면 놓치는 부분이 있었다. 그래서 balanced accuracy, precision, recall, F1, ROC-AUC를 같이 봤다.

## How I fixed them

- 결측 `children`은 0으로, `country`, `agent`, `company`는 `Unknown`으로 처리했다.
- 총 투숙객 0명, 숙박일 0일, 음수 ADR 같은 비정상 행은 제거했다.
- outcome leakage 가능성이 있는 컬럼은 제거했다.
- `StandardScaler/MinMaxScaler`와 `OneHotEncoder/OrdinalEncoder` 조합을 비교했다.
- Logistic Regression, Decision Tree, KNN의 여러 설정을 같은 cross-validation 기준으로 비교했다.
- sample 기반 선택 후 full train set 재학습으로 결과가 크게 흔들리지 않는지 확인했다.

## What I learned

이 프로젝트를 하면서 모델보다 전처리 결정이 더 중요할 수 있다는 걸 배웠다. 특히 leakage 컬럼을 넣으면 성능이 좋아 보일 수 있지만, 실제 문제 정의에는 맞지 않는다. 점수를 높이는 것보다 “예측 시점에 알 수 있는 정보만 쓰는가”가 더 중요했다.

또 큰 조합 실험을 할 때는 계산 비용을 고려해야 한다는 것도 배웠다. 모든 것을 전체 데이터로 돌리는 것보다, sample로 후보를 좁히고 전체 데이터로 확인하는 방식이 현실적이었다.

## What I would improve next

- 최종 모델의 feature importance나 coefficient를 더 자세히 해석하고 싶다.
- calibration curve를 추가해 예측 확률이 실제 취소율과 맞는지 보고 싶다.
- 웹 데모를 더 작고 재현 가능하게 정리하고 싶다.
- 데이터 split과 preprocessing pipeline을 더 모듈화해서 재실행이 쉽게 만들고 싶다.
