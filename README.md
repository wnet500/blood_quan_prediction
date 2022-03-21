# 적정 수혈량 예측모델 개발
## 1. 초기 설정
### A. 필요 라이브러리 설치
```
pip install -r requirements.py
```

### B. 데이터 정보 업로드
- `data/datasets_info.default.json`에 정보를 채워 넣은 후, `data/datasets_info.json`으로 파일 이름 변경
- 입력할 내용이 없을시 `[]`값으로 두기
- 입력할 정보 종류
  - drop_columns: 모델링에 사용하지 않아 데이텃셋에서 제외할 변수
  - outcome_columns: 모델 학습시 true_label이 될 변수 (아웃컴 변수)
  - preprocessing: 데이터 전처리에 사용할 정보
    - one_hot_columns: one-hot encoding이 필요한 컬럼 (카테고리컬 변수)
    - float_columns: 그 외 숫자 컬럼

### C. 데이터 업로드
- `data`폴더 내에, train sets 파일과 test sets 파일을 각각 업로드
- 업로드한 파일의 이름을 `tests/__init__.py`에 입력    
[예시 - train sets 파일: "train_df.csv", test_sets 파일: "test_df.csv"일 경우]
```
__train_data_file__ = "train_df.csv"
__train_data_file__ = "test_df.csv"
```

## 2. 모델 파라미터 그리드서치 수행
- `tests/test_gridsearch.py`내부에 서치할 파라미터 `grid_params`변경
변경하지 않을 시, 기본으로 설정되어 있는 파라미터 셋으로 서치 진행
- cross validation 코드 수정 (기본: 5-fold cv)
- gpu 사용시, `tree_method=gpu_hist`로 변경
- 그리드 서치 수행
```
pytest -s -v tests/test_gridsearch.py
```
- `output/gridsearch_results` 폴더 내,    
`lr_results.csv`, `rf_results.csv`, `xgb_results.csv` 각 결과 파일 확인

## 3. 최종 모델 학습 및 평가
- 각 모델(알고리즘)의 최적 파라미터와 전체 train sets 데이터를 사용하여 최종모델 학습
- test sets를 사용하여 최종모델 성능 평가
  - 평가 지표: mse, adjusted r2
- gpu 사용시, `tree_method=gpu_hist`로 변경
- 최종 모델 학습 및 평가 수행
```
pytest -s -v tests/test_model_evaluation.py
```
- print되어 화면에 나타나는 정보 확인

## 4. 파이토치 버젼 정보
- torch: `1.10.2+cu113`
- pytorch_lightning: `1.5.10`
