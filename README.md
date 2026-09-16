# ApprenticeProject_2th

Motor Driver `MOVELOG_*.csv` 파일에서 특징량을 추출하고, `I`, `L`, `KI`를 예측하는 제조 AI 머신러닝 예제 프로젝트입니다.

## 목표

- 입력 데이터: `data_train/MOVELOG_*.csv`, `data_validation/MOVELOG_*.csv`, `data_test/MOVELOG_*.csv`
- 입력 변수 X: raw motor log에서 추출한 feature와 SMOVE 조건. 단, 사용자가 입력한 `resistance`는 label shortcut 방지를 위해 학습 입력에서 제외
- 출력 변수 Y: `inertia(I)`, `load(L)`, `ki(KI)`
- 모델: RandomForest 기반 multi-output 회귀/분류 baseline
- 산출물: feature dataset, 성능 평가 report, 학습 모델, PPT 문서

## 실행 순서

```powershell
pip install -r requirements.txt
python src/1_build_dataset.py
python src/1_visualize_dataset.py
python src/2_compare_preprocessing_and_tuning.py
python src/2_ml_train_model.py
python src/2_ml_evaluate_test.py
python src/3_deep_train_model.py
python src/3_deep_evaluate_test.py
python src/4_heuristic_data_analysis.py
python src/5_0_svr_train_validation_split_compare.py
python src/5_1_svr_hyperparameter_only.py
python src/5_2_svr_hyperparameter_scaling.py
python src/5_3_svr_scaling_result_report.py
python src/2_ml_predict_one.py --file data_test/MOVELOG_RUN0001_REP01_T1000_R00_I05_L00_KI03_A03_D03_S0600_PWM1000_CL0_20260913_152838_577.csv
python make_project_ppt.py
```

## 출력 파일

- `outputs/1_dataset_features.csv`: ML 학습용 feature table
- `outputs/1_movelog_raw_summary.csv`: raw 시계열 범위 요약
- `outputs/1_dataset_quality_report.txt`: CSV 품질 점검 report
- `outputs/1_visual_report.html`: 데이터 시각화 report
- `outputs/plots/`: raw waveform, 비교 plot, feature plot 이미지
- `outputs/2_ml_model_i_l_ki.joblib`: RandomForest 모델 bundle
- `outputs/2_ml_evaluation_report.txt`: RandomForest 평가 결과
- `outputs/2_ml_test_evaluation_report.txt`: RandomForest test 전용 최종 평가 결과
- `outputs/2_ml_test_predictions.csv`: RandomForest test 예측 결과
- `outputs/2_ml_feature_importance.csv`: RandomForest feature 중요도
- `outputs/2_preprocessing_tuning_comparison.txt`: scaling/tuning 비교 report
- `outputs/2_preprocessing_tuning_comparison.csv`: scaling/tuning 비교 수치
- `outputs/3_deep_mlp_model.joblib`: MLP 기반 neural network 모델 bundle
- `outputs/3_deep_learning_report.txt`: MLP scaling/hyperparameter/최종 평가 report
- `outputs/3_deep_hyperparameter_results.csv`: MLP 하이퍼파라미터 비교 결과
- `outputs/3_deep_test_evaluation_report.txt`: MLP test 전용 최종 평가 결과
- `outputs/3_deep_test_predictions.csv`: MLP test 예측 결과
- `outputs/4_heuristic_analysis_report.html`: 휴리스틱 feature/label 분석 report
- `outputs/4_heuristic_feature_label_correlation.csv`: feature-label 상관관계 분석 결과
- `outputs/4_heuristic_rule_predictions.csv`: 휴리스틱 rule 기반 예측 결과
- `outputs/5_0_svr_train_validation_split_compare_report.html`: train/validation 분리와 train+validation 병합 학습 전략 비교 report
- `outputs/5_0_svr_train_validation_split_compare_summary.csv`: 두 학습 전략의 test 성능 비교 table
- `outputs/5_1_svr_hyperparameter_only_report.txt`: Data Scaling 없이 SVR 하이퍼파라미터만 비교한 report
- `outputs/5_2_svr_hyperparameter_scaling_report.txt`: SVR 하이퍼파라미터와 Data Scaling 조합 비교 report
- `outputs/5_1_svr_hyperparameter_only_report.html`: Data Scaling 없이 SVR 하이퍼파라미터만 비교한 HTML report
- `outputs/5_2_svr_hyperparameter_scaling_report.html`: SVR 하이퍼파라미터와 Data Scaling 조합 비교 HTML report
- `outputs/5_3_svr_scaling_result_report.html`: Data Scaling 적용 전/후 결과 비교 table/graph report
- `Manufacturing_AI_Motor_ML_Project.pptx`: 과제 발표용 PPT

## 과제 절차 대응

- 데이터셋 획득 및 문제 정의: MOVELOG CSV에서 X/Y 정의
- 데이터 분할 및 비교: `data_train`, `data_validation`, `data_test` 폴더를 명시적으로 분리해서 사용
- 하이퍼파라미터 조정: RandomForest grid search
- 데이터 스케일링 및 이유 설명: tree model은 scaling 불필요, 비교용 StandardScaler pipeline 제공 가능
- 최종 성능 평가: MAE, RMSE, R2, ±1 accuracy로 평가
