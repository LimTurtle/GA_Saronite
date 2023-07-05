# Source Code

## 1. 데이터
- playlist의 'id', 'songs', 'like_cnt' 추출
- 'songs'를 기준으로 split
- 'songs'를 'song_id'로 개명

- song meta의 'id', 'album_id', 'song_gn_dtl_gnr_basket' 추출
- 'id'를 'song_id'로, 'song_gn_dtl_gnr_basket'을 'genre'로 개명

### 선택사항
- 세부장르를 대장르로 변경
    - 세부 장르(현재 'genre') 데이터의 대장르 정보를 추출
        대장르의 번호는 앞 두개의 숫자이다. (GN<U>16</U>01)
    - 개수가 가장 많은 대장르를 선택, 동일한 경우 대장르의 숫자가 작은 것을 선택

- playlist와 song meta의 데이터를 병합
- target은 playlist의 'id' 학습 데이터는 그외의 것으로 선택

## 2. 모델
- 학습
    - target과 학습데이터를 knn 모델에 학습
        n_neighbors 옵션은 1 (이유 : knn의 점수가 높게 측정되었다)
    
- 예측
    - test.json 파일의 playlist들 각각에 대해 1의 과정 수행 및 knn 모델 예측
        이유 : playlist 각각에 대해 음악 추천을 해야한다
    - knn 모델이 예측한 id 각각의 개수 확인
    - 가장 많은 id(playlist)에서 test playlist에 없는 노래 추출 (최대 30개)
