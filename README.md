##### 학습환경설정

- anaconda 가상환경 설정

```
conda create -n cvclass python=3.7
conda activate cvclass
conda install -c conda-forge opencv -y
pip install opencv-contrib-python
conda install ipykernel -y
```

- vs code settings

가상환경에서 파이썬을 실행시켜야하기에 git bash에서 다음과 같이 ~/.bashrc 파일 수정

```
## . /c/ProgramData/Anaconda3/etc/profile.d/conda.sh
conda activate
alias activate="conda activate"
``` 