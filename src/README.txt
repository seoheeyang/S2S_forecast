서울대학교_손석우 교수님 연구실

__MAML코드_________________________________________

학습 전 준비: 
input_summer_dt_19502004_nmax.py : 본 학습 입력장 만드는 파일입니다.
target_summer_dt_19502004_nmax2.py : 본 학습 타겟 만드는 파일입니다.
ini_summer_target_dt_19502004_nmax2.py : 사전학습 타겟 만드는 파일입니다. 사전학습은 연차보고회 자료에 적힌 듯이, 여름철 한국 기온과 연관있는 기후 인자들을 대상으로 (타겟으로) 사전훈련을 하며 가중치를 만듭니다. 
ini_input_for_dt_nmax.py: 사전학습 입력장 만드는 파일입니다. 
이 때 사전학습 입력장으로는 예측 전 월의 1~14일 중, 중복 허용하는 랜덤 14개일을 뽑아 평균했습니다. 

MAML
본학습
nmax2 파일 기준으로 학습합니다. Exloss 적용된 내용입니다. (연차보고회 자료 참고)
MAML.py: 모델 코드
setup_MAML_new.py: 모델 코드
run_MAML_YSH.sh 실행 시, setup_MAML_new.py->MAML.py실행. 

사전학습
run_ini_MAML_YSH.sh실행하여 ini_setup_MAML.py 코드 실행

앙상블 개수에 따라 만드시면 됩니다. 

__ K-TempCast__________________________________________

이 때 사전학습은 오토인코더 사용하며 내용은 첨부드린 논문 참고해주시면 됩니다.

input_diff.py: 입력장 코드
target_summer_dt_19502004_std.py: 타겟 자료 생성 코드
ysh_lab_augu_dt_19502004_std_2nd_19502025.nc
ysh_lab_july_dt_19502004_std_2nd_19502025.nc
ysh_lab_june_dt_19502004_std_2nd_19502025.nc
ysh_z500_diff_tzuv_inp_july_dt_19502004_std_2nd.nc
ysh_z500_diff_tzuv_inp_june_dt_19502004_std_2nd.nc
ysh_z500_diff_tzuv_inp_may_dt_19502004_std_2nd.nc
5, 6, 7월 입력장, 6, 7, 8월 카겟 코드
step1, step2파일: 오토인코더 학습을 위한 데이터 생성
ktemp_train_autoencoder_new_z500.py 사전학습 실행코드->core폴더: 사전학습 (오토인코더) 모델 코드

run_ktemp.sh실행 시, ktemp.py실행->Model/KTEMPCAST.py실행 (KTEMPCAST.py는 Model폴더 내로 넣어주셔야합니다.)
utils폴더: 모델 돌 때 필요한 폴더.
src/폴더 내에서 drawgl.py 그리면 era5 & glosea6와 비교 시계열 그래프 생성 (na, ea케이스로 바꿔서 비교 가능)
src/폴더 내에서 trnmax_new.py 그리면 era5& glosea6와 삼분위 비교 그래프 및 F1-Score 생성 (na ,ea케이스도 한번에)
