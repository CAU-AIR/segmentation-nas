# segmentation-nas
Neural Architecture Search for Segmentation
<img src="https://img.shields.io/badge/AutoML-0B2C4A?style=flat&logo=AutoML&logoColor=white"/>

<img src="https://img.shields.io/badge/python-3776AB?style=flat&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=Docker&logoColor=white"/>


## 연구 개요
자동차 제조 과정에서 차량의 골격을 만드는 단계에서는 접착제를 이용한 부착이 필수적이다. 하지만, 기계를 사용하여 도포하더라도 항상 일정한 양을 도포하는 것은 매우 어렵다. 이로 인해 접착제가 너무 적게 도포되거나, 너무 많이 도포되는 문제가 발생한다. 접착제가 적게 도포되는 경우 차체의 안정성이 저하될 수 있으며, 반대로 너무 많이 도포되는 경우 제조 비용이 증가하고, 실러가 지저분하게 튀어나와 추가적인 공정이 필요하다.

기존에는 룰 베이스 모델을 활용하여 실러의 도포량을 측정하였으나, 이 방법은 낮은 성능과 환경 변화에 따른 인식 부족으로 인해 개선이 필요하다. 이를 해결하기 위해 딥러닝 기반 모델 중 하나인 신경망 구조 검색(Neural Architecture Search, NAS)을 도입하여 성능을 개선하고자 한다. 

본 연구의 주요 목표는 다음과 같다.
- 기존 방법에서 잘 인식되지 않는 핫스탬핑 영역에서의 정확도 향상
- 실러의 폭을 측정하여 실러의 양 측정(결여, 과잉 여부 판별)
- 학습된 차종뿐만 아니라, 새로운 차종에 대해서도 일관된 성능 보장

이 연구를 통해 자동차 제조 과정에서 접착제 도포 문제를 해결함으로써, 생산 효율성을 높이고 제조 비용을 절감하는데 기여하고자 한다.

![전체개요](https://github.com/CAU-AIR/segmentation-nas/assets/97182733/09603fe5-c7bf-46cc-bae0-e28a95d3383f)

## 다양한 차종에 강인한 단일 구조용 접착제 검사 시스템 구축
기존 룰 베이스 연구의 경우, 차종이 달라지면 접착제 인식률이 낮아지는 문제가 있었다. 이를 해결하기 위해 다양한 차종을 모델에 학습시켜, 차종이 변하여 주변 환경이 변하더라도 높은 성능을 유지할 수 있는 검사 시스템을 구축하는 것을 목표로 한다.

- 차종이 바뀌더라도 높은 성능 유지
- 새로운 차종이 추가되더라도, 접착제 인식률에 대한 성능 유지

![단일구조용접착제검사](https://github.com/CAU-AIR/segmentation-nas/assets/97182733/2574ef87-c450-4f98-abbf-9098ea82be10)

## 제조 환경 변화(핫스탬핑 데이터)에 강인한 딥러닝 모델 개발
제조 환경에 따라 배경 영역이 어두워 접착제를 구분하기 어려운 부분을 핫스탬핑 영역이라 하며, 기존 룰 베이스 연구의 경우 해당 영역에서의 인식률이 매우 낮아 접착제의 양을 확인할 수 없다는 문제가 있었다. 이를 해결하기 위해 핫스탬핑 영역에서도 강인한 모델을 개발하는 것을 목표로 한다.


![핫스탬핑데이터](https://github.com/CAU-AIR/segmentation-nas/assets/97182733/4a2b789e-c739-43b5-9123-7d26438f6660)

## End-to-End 파이프라인 구축
접착제 영역을 추론하는 단계부터, 접착제의 양을 예측하는 과정까지 전체 프로세스를 쉽게 이용할 수 있도록 end-to-end 파이프라인을 구축하고자 한다.
본 서비스는 윈도우10을 기반으로 제공하며, 도커를 기반으로 구축되었다.

### 데이터 준비

### 

![e2e](https://github.com/CAU-AIR/segmentation-nas/assets/97182733/ac9dba64-54c4-4882-8cfb-0bc7bf21066a)