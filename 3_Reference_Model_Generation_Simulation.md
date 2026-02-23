앞에서 살펴본 Reference Model을 대상으로 Python을 통해 test_data.txt를 생성하고, 해당 데이터를 ModelSim 환경에서 시뮬레이션하여 동작 검증 및 정확도 검증을 수행하였습니다.

## test_data 출력 

Directory 내 Python 폴더로 이동하면 여러 Python 스크립트가 존재합니다. 

전체 파일을 모두 분석하기에는 범위가 넓어, 본 절에서는 test data 생성과 직접적으로 관련된 파일만 확인하였습니다.

​

genTestData.py는 test data를 생성하는 스크립트이며, 해당 파일을 열어보면 가장 먼저 출력 경로를 설정하는 부분이 확인됩니다.

    outputPath = "../CNN-MNIST-Arty-Z7/CNN-MNIST-Arty-Z7.sim/sim_1/behav/xsim/"
    headerFilePath = "../CNN-MNIST-Arty-Z7/CNN-MNIST-Arty-Z7.sim/sim_1/behav/xsim/"
  ​
위 경로는 생성된 test_data.txt 및 관련 헤더 파일이 저장될 ModelSim(XSIM) 시뮬레이션 디렉터리를 지정합니다.

실제 ModelSim이 실행되는 디렉터리로 출력 경로를 직접 지정하였습니다.

      outputPath = "C:/NPU_study/ref/"
      headerFilePath = "C:/NPU_study/ref/"
  
이를 통해 genTestData.py 실행 시 생성되는 모든 test data 파일이 ModelSim 시뮬레이션 디렉터리 내에 직접 생성되며, RTL testbench에서 별도의 경로 수정 없이 해당 파일을 바로 참조할 수 있습니다.

### genTestData.py 실행 

경로를 절대 경로로 수정한 이후에는 Python 스크립트의 위치에서 직접 실행하면 됩니다.

1. Command Prompt 실행

    Windows CMD 또는 PowerShell을 실행합니다.

2. Python 스크립트가 있는 디렉터리로 이동

          cd C:\NPU_study\CNN-Handwritten-Digit-MNIST-main\Network\Vivado\Python
          (※ genTestData.py가 위치한 정확한 경로로 이동)
  
3. 스크립트 실행

          python genTestData.py
  
※ numpy Error 발생 시

    만약 해당 오류가 발생한다면, 이는 Python 실행 환경에 NumPy 라이브러리가 설치되어 있지 않을 경우에 발생합니다.

    genTestData.py는 Reference Model 기반의 연산 및 test data 생성을 위해 NumPy를 사용하므로, 사전에 해당 라이브러리가 설치되어 있어야 합니다.

      Traceback (most recent call last):
        File "C:\NPU_study\CNN-Handwritten-Digit-MNIST-main\Network\Vivado\Python\genTestData.py", line 11, in <module>
          import numpy as np
      ModuleNotFoundError: No module named 'numpy'
  
### 해결 방법

Command Prompt 또는 PowerShell을 실행한 뒤, 아래 명령어를 통해 NumPy를 설치합니다.

      python -m pip install numpy
  
이후 3. 스크립트 실행을 재실행 합니다.

4. 실행 결과 확인

    정상적으로 실행되면 다음 파일들이 생성됩니다. 정상적으로 실행되면 지정한 출력 디렉터리에 test data 파일이 생성됩니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_26.png" width="100"/>

<div align="left">


​    다만 스크립트를 기본 설정 그대로 실행할 경우, MNIST test set 전체(총 10,000개)를 기준으로 파일이 생성되므로 test_data_XXXX.txt가 10,000개 출력됩니다. 

​    시뮬레이션 시간을 단축하거나 특정 개수만 검증하고자 하는 경우, genAllTestData() 함수의 반복 범위를 제한하여 원하는 개수만 생성하도록 수정할 수 있습니다.

    ​예를 들어 100개만 생성하고자 할 경우, 아래와 같이 for 문을 수정합니다.

        # for i in range(len(test_inputs)):
        for i in range(100):
        
    위와 같이 for i in range(100)으로 변경하면, test_data_0000.txt부터 test_data_0099.txt까지 총 100개의 test data 파일만 생성되며, 필요한 검증 범위에 맞춰 원하는 개수로 조정할 수 있습니다.

### ModelSim Simulation

1. ModelSim 설치

ModelSIm 설치방법은 아래 link를 참조하시면 됩니다. 

https://blog.naver.com/mini9136/224140269011

2. ModelSim 실행

ModelSim을 실행하면 다음과 같은 초기 화면이 표시됩니다. 좌측에는 Project 및 Library 구조가 나타나며, 우측에는 명령어 입력을 위한 Transcript 창이 기본으로 표시됩니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_27.png" width="400"/>

<div align="left">

3. Create Project 

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_28.png" width="400"/>

<div align="left">

상단 메뉴에서 File → New → Project를 선택하면 Project 생성 창이 표시됩니다. 이때 Project Name은 testbench 이름과 동일하게 top_sim으로 설정하는 것이 관리 측면에서 용이합니다. 이후 Project Location에서 ModelSim 실행 디렉터리를 지정한 뒤 OK를 선택하면 Project가 생성됩니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_29.png" width="400"/>

<div align="left">

4. Add File

Create Project가 완료되면 Add file to Project 창이 자동으로 표시됩니다. 이때, 기존에 다운로드한 Reference Model의 RTL 코드가 위치한 다음 경로를 선택합니다.

해당 디렉터리를 선택한 후, directory 내의 모든 RTL 파일을 추가(Open) 하여 Project에 포함시킵니다.

5. Compile

RTL 파일을 Project에 추가하면 다음과 같이 화면이 전환되며, 추가된 RTL 파일 목록이 표시됩니다.

이후 상단 메뉴에서 Compile → Compile All을 선택하여 전체 RTL 파일에 대해 컴파일을 진행합니다.

※ File을 찾지 못해 발생하는 Error

컴파일을 진행했을 때 no such file 오류가 발생하는 경우가 있습니다.

이는 top_sim.v에서 참조하는 include.v 파일을 ModelSim이 찾지 못해 발생하는 문제입니다.

이 경우, top_sim.v 내 include.v를 포함하는 부분을 다음과 같이 수정합니다.

`include "include.v"
또한, 

작업 Directory(ModelSim 실행 디렉터리) 내에 tb 및 rtl 디렉터리의 모든 파일을 함께 위치시키면 해당 오류를 해결할 수 있습니다.


​

6. Simulation

컴파일이 문제없이 완료되면 Simulation을 진행합니다.

​

top_sim.v를 선택한 후 Simulate를 클릭하면 Start Simulation 창이 표시됩니다.

해당 창에서 work → top_sim을 선택하면 Simulation 화면이 실행됩니다.

​


Waveform을 확인하기 위해 View → Wave를 선택하여 Wave 창을 엽니다.

이후 확인하고자 하는 신호를 Wave 창으로 추가한 뒤, Simulate → Run을 실행하면 해당 신호의 Waveform을 확인할 수 있습니다.

​


​

7. 최종 Simulation 결과

 Simulation을 완료한 후 하단 Console 창을 확인하면, 

실제 input data에 대한 연산 결과와 Expected 결과를 비교하는 과정을 확인할 수 있습니다.

​

두 결과가 일치하는 경우 이를 정상으로 판단하고, 전체 결과를 기반으로 Accuracy를 산출합니다.

​

본 Project에서는 전체 100개 sample 중 99개가 일치하여, 99%의 정확도를 보임을 확인하였습니다.


etc. 추가적으로 확인해 볼 사항

1. test_data가 없는 경우에도, Warning이 발생하지 않는 경우

​


top_sim.v의 testbench는 MaxTestSamples 값을 기준으로 for-loop 형태로 동작하며,

각 iteration마다 test data 파일명을 생성하여 처리를 시도합니다.

​

이 과정에서 파일 존재 여부($fopen 성공 여부)를 별도로 확인하지 않기 때문에,

test data 파일이 존재하지 않더라도 Simulation은 중단되지 않습니다.

​

그 결과 ModelSim 상에서는 별도의 Warning 없이 Simulation이 진행될 수 있습니다.

​

2. Simulation 시, 발생하는 Warning


Simulation 중 출력되는 Warning은 주로 모듈 인스턴스화 과정에서

port width와 연결된 signal width가 서로 일치하지 않을 때 발생합니다.

​

ModelSim은 이러한 경우 자동으로 bit-width를 조정하여 Simulation을 진행합니다.

​

본 Project에서는 해당 Warning이 최종 결과 및 Accuracy에는 영향을 주지 않았습니다.
