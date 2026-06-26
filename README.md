# oht-delay-prediction
Deep learning model to predict real-time traffic congestion and delays for semiconductor AMHS OHT logs.



# 반도체 라인 OHT 정체 및 지연 예측 모델 (딥러닝 기반)

반도체 공장 물류 시스템(AMHS)에서 주행하는 무인 이송 차량(OHT)의 로그 데이터를 활용해서, 라인이 막히거나 지연되는 상황을 미리 예측해 보는 개인 딥러닝 프로젝트입니다.

## 1. 프로젝트 개요
- **문제 상황**: 반도체 생산 라인에서 OHT가 트래픽 때문에 멈추거나 밀리면 전체 공정 효율이 크게 떨어집니다.
- **해결 아이디어**: OHT의 배터리 상태, 라인의 현재 혼잡도, 싣고 있는 웨이퍼 무게 같은 실시간 데이터를 모아서 딥러닝(이진 분류)으로 "정체가 발생할지 안 할지" 선제적으로 예측해 보고자 만들었습니다.

## 2. 데이터 준비(가상 데이터 2,000건)
현실적인 반도체 환경을 녹여내기 위해 파이썬(Pandas/NumPy) 시나리오 기반으로 가상 로그 데이터를 생성하고 전처리했습니다.

- Timestamp: 15초 단위의 실시간 데이터 순서
- OHT_ID: 차량 고유 번호 (OHT_101 ~ OHT_150)
- Line_No: OHT 생산 라인 (1~5번)
- Current_Traffic: 해당 라인의 실시간 혼잡도 (1~10단계)
- Load_Weight: 이송 중인 웨이퍼 박스(Lot) 무게 (5~15kg)
- Battery_Level: 로봇 차량의 현재 배터리 잔량 (15%~100%)

💡 **핵심 포인트 (노이즈 추가)**
처음에는 단순한 조건문 규칙으로만 데이터를 만들었더니 정확도가 100%가 나와서 너무 비현실적이었습니다. 그래서 현실 공장처럼 예측 불가능한 예외 상황을 만들기 위해, 이항 분포(np.random.binomial) 모델을 써서 혼잡도와 배터리에 비례하는 '확률형 노이즈'를 데이터셋에 추가 반영했습니다.

## 3. 기술 스택 및 모델 구조
- **사용 언어**: Python 3.12 (Jupyter Notebook)
- **사용한 라이브러리**: TensorFlow, Scikit-learn, Pandas, NumPy, Matplotlib
- **딥러닝 모델 설계**:
  - 데이터 안정성을 위해 'StandardScaler'로 피처 정규화 선행
  - TensorFlow Sequential 모델 활용
  - **은닉층 1**: Dense Layer (노드 16개, ReLU)
  - **은닉층 2**: Dense Layer (노드 8개, ReLU)
  - **출력층**: Dense Layer (노드 1개, Sigmoid로 확률값 도출)
  - **설정**: Adam 옵티마이저 / Binary Crossentropy 손실 함수

## 4. 분석 결과 및 배운 점
- 노이즈가 섞인 까다로운 조건에서도 20 Epoch 동안 돌리면서 Loss가 안정적으로 떨어지는 것을 확인했습니다.
- **최종 검증 정확도(Accuracy)**: 70.50%
- 단순한 데이터로 학습시키는 것보다, 현실적인 노이즈를 섞고 전처리를 해주는 과정이 딥러닝 모델 성능에 얼마나 큰 영향을 미치는지 직접 경험해 볼 수 있었습니다.

## 5. 실행 및 확인 방법
- 저장소 내의 'oht_delay_prediction.ipynb' 파일을 클릭하시면 데이터 생성 과정부터 최종 학습 곡선(Loss/Accuracy 그래프)까지 전체 실행 결과를 바로 웹에서 확인하실 수 있습니다.

