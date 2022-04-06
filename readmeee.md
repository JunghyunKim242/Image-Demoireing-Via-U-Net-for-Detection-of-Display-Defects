
# demoire network manual
- 실행은 python main.py 로 해주시면 됩니다.
- train 과 test의 차이는 main.py 안에서 설정하시면 됩니다. 
- CUDA version 은 11.2 version 을 사용했습니다. 

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch

```
$ conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

- 다른 모듈 download
```
$ pip install -r requiremets.txt
```

- dataset의 경로설정의 main.py 안에서 traindata_path 와 testdata_path를지정 해주시면 됩니다. 
- 설치와 환경설정이 완료 되셨다면, 실행하시면 됩니다.
```
$ python main.py 
```