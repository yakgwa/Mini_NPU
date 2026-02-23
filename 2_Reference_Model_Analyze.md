## Reference Model 소개

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_21.jpg" width="400"/>

MNIST 손글씨 Dataset​

<div align="left">MNIST Data Set을 입력으로 받아 숫자(0~9)를 분류하는 Neural Network 모델입니다.
    
- GitHub (Reference Code)

    https://github.com/yeshvanth-m/CNN-Handwritten-Digit-MNIST

- 관련 영상

    https://youtu.be/NJgFl8gsZzM?si=ek-daA4LbSv3Ywsa
    
    https://youtu.be/rw_JITpbh3k?si=10GmqDOyCYjxIFBF

※ 시작하기 전에, Testbench 확인의 중요성

프로젝트를 분석하기 위해서는 먼저 전체 구조를 파악해야 합니다. 이를 위해 가장 효과적인 방법 중 하나는 Testbench를 우선적으로 확인하는 것입니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_22.png" width="400"/>

​<div align="left">Testbench는 RTL 코드를 Simulation 하기 위해 작성되는 Test File 입니다. Testbench는 Stimulus를 통해 DUT(Design Under Test)에 입력을 인가하고, DUT의 출력은 Output Checker에서 검증하는 구조로 구성됩니다.

​이 과정에서 설계자가 정의한 기능은 Time Sequence에 따라 전개됩니다. 따라서 Testbench의 Stimulus 흐름을 기준으로 RTL signal path를 따라 분석하는 것이 효과적입니다.

## Reference Model Testbench (top_sim.v) 분석

Reference Model의 Testbench인 top_sim.v를 분석해보겠습니다.

​### 1. Interface 구조

TB를 확인하면, DUT는 두 가지 인터페이스를 기준으로 설계되어 있습니다.

- AXI-Lite → Weight, Bias, Control register, Result register 접근
- AXI-Stream → Input feature (MNIST 784 pixel) 전달
  
역할을 구분하면 다음과 같습니다.

- AXI-Lite   : Parameter / Control path
- AXI-Stream : Data path (Inference input)
- AXI-Lite는 Weight/Bias와 같은 파라미터 로딩, 제어 레지스터 설정, 결과 레지스터 read를 담당하는 low-bandwidth control interface입니다.

반면 AXI-Stream은 784개의 MNIST pixel 값을 순차적으로 전달하는 streaming data interface입니다. 대량의 feature data를 한 방향으로 연속 전송하는 구조에 적합합니다.

### 2. Stimulus 관점에서의 TB 구성

현재 Testbench에서는 AXI-Stream protocol을 검증하는 형태로 stimulus를 구성하지 않습니다.

외부 AXI master가 Stream을 구동하는 구조가 아니라, test_data_xxxx.txt 파일을 $readmemb로 읽어 내부 신호(in, in_valid)에 직접 주입하는 방식으로 구성되어 있습니다.

​따라서 본 분석에서는 AXI-Stream protocol 자체보다는, "파일 기반 입력 데이터가 DUT로 어떻게 주입되고, 그에 따라 inference 결과가 어떻게 검증되는지" 에 초점을 맞추어 Stimulus 흐름을 정리하겠습니다.

### 3. 전체 동작 흐름 개요

top_sim.v는 다음과 같은 순서로 동작합니다.

    Reset
    → (선택) Weight / Bias preload
    → test_data 파일 로드
    → 784 pixel Stream 전송
    → intr 대기
    → Result register read
    → 정답 비교 및 Accuracy 누적
    → 반복
    이제 각 단계를 자세히 살펴보겠습니다.
    
### 4. Reset Sequence

DUT의 reset 포트는 다음과 같이 연결되어 있습니다.

    .s_axi_aresetn(reset)
    
aresetn이므로 active-low reset 구조입니다.

Testbench의 reset 동작은 다음과 같습니다.

    1. reset = 0 → reset asserted
    
    2. 100ns 대기
    
    3. reset = 1 → reset deassert
    
    4. writeAxi(28, 0) → soft reset clear

​    즉, Hardware reset → reset release → 내부 control register 초기화

이 과정을 통해 DUT는 inference를 수행할 준비 상태로 진입합니다.

### 5. Weight / Bias 설정 단계 (Preload)

pretrained가 정의되지 않은 경우에만 수행됩니다. 이 단계는 inference 이전에 parameter를 AXI-Lite로 로딩하는 과정입니다.

### 5.1 Weight 설정

각 layer와 neuron에 대해:

    1. writeAxi(12, k) → Layer 선택
    
    2. writeAxi(16, j) → Neuron 선택
    
    3. Weight 파일(fim.<neuron>_<layer>_w)을 $readmemb로 로드
    
    4. Weight를 순차적으로 writeAxi(0, data)로 write
    
    즉, DUT 내부 weight memory를 AXI-Lite를 통해 초기화합니다.

### 5.2 Bias 설정

각 neuron에 대해:

    1. Bias 파일(fim.<neuron>_<layer>_b) 로드
    
    2. writeAxi(4, data)로 bias write

    이 과정까지 완료되면 DUT는 fully parameterized된 상태가 됩니다.

### 6. Test Data 기반 Stimulus 생성

이제 실제 inference 검증이 시작됩니다.

​각 iteration마다 다음 파일이 생성됩니다.

test_data_0000.txt
test_data_0001.txt
...
파일 구성은 다음과 같습니다.

index 0 ~ 783  : 784 pixel 값
index 784      : 정답 label
​
### 7. Input Feature 전송 (AXI-Stream 경로)

sendData() task 내부 동작을 시간 순서로 정리하면 다음과 같습니다.

### 7.1 파일 로드

$readmemb(fileName, in_mem);
in_mem[0:784]에 pixel 및 label 저장

### 7.2 Pixel Stream 전송

for (t = 0; t < 784; t++)
    in       <= in_mem[t];
    in_valid <= 1;
    
→ 총 784 cycle 동안 연속 전송

→ in_valid는 전송 구간 동안 계속 1

→ axis_in_data_ready는 TB에서 사용하지 않음

즉, DUT가 매 cycle 입력을 수용한다고 가정한 구조입니다.

### 7.3 Label 저장

루프 종료 후:
in_valid <= 0;
expected = in_mem[784];
마지막 index(784)는 DUT에 전송되지 않으며, 검증을 위한 expected label로만 사용됩니다.

### 8. Inference 완료 대기

입력 전송이 완료되면 Testbench는 다음 신호를 기다립니다.

@(posedge intr);

→ intr는 DUT의 inference 완료 신호입니다. 따라서 동작 흐름은 다음과 같습니다.

Pixel Stream 입력
→ 내부 MAC 연산
→ Layer 연산 완료
→ intr assert
Testbench는 interrupt 기반으로 결과 준비 완료를 판단합니다.

### 9. Result Read 및 정확도 계산

intr가 발생하면 다음을 수행합니다.

readAxi(8);

→  AXI-Lite를 통해 Result register read
→ 결과는 axiRdData에 저장

이후,

if (axiRdData == expected)
    right++;
    
→ 정답 여부를 비교하여 accuracy를 누적합니다.

### 10. 반복 수행

위 과정은 다음 조건만큼 반복됩니다.

MaxTestSamples = 100
→ 즉, 100개의 test sample에 대해:

파일 로드
→ Pixel 전송
→ intr 대기
→ 결과 read
→ 정답 비교
를 반복하며 최종 Accuracy를 출력합니다.

### 11. 정리

top_sim.v의 Testbench는 단순히 파일을 읽어 데이터를 넣는 구조가 아니라,

- AXI-Lite 기반 Parameter preload

- AXI-Stream 기반 Feature injection

- Interrupt 기반 completion synchronization

- AXI-Lite 기반 result readback

- Label 비교를 통한 Accuracy 계산

이라는 완전한 inference verification flow를 구성하고 있습니다.

##Reference Model Top (zynet.v) 분석

다음으로 Reference Model의 Top module인 zynet.v를 분석해보겠습니다.​

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_23.png" width="400"/>

H/W Architecture

​<div align="left">

+ 공통 설정

    + include.v를 include하여 parameter(dataWidth, layer neuron 수 등)를 사용

    - axis_in_data_ready는 항상 1로 고정되어, 입력 stream을 항상 수신 가능한 상태(Valid)로 유지.

    - intr는 out_valid에 연결되어, 모든 연산이 끝난 시점에서의 결과를 외부에 알림.

​

+ 입력 데이터 수신

    + Testbench는 axis_in_data와 axis_in_data_valid를 통해 입력 Pixel 데이터를 전달.

    - axis_in_data_valid가 assert된 구간이 Layer 1 연산의 입력 window로 동작.

​

+ Layer 1 연산 

    + Layer_1은 입력 stream(axis_in_data, axis_in_data_valid)을 받아 Fully-Connection 연산을 수행
    
    - 연산결과는 vector 형태(x1_out)로 생성 → o1_valid가 output vector의 유효 시점을 알림.

​

+ Controller (Layer 출력 직렬화: vector → scalar)

    + 각 Layer 출력은 vector 형태이므로, 다음 Layer에 입력으로 전달하기 위해 직렬화(serialize).
    
    - 이를 위해 각 Layer 뒤에는 IDLE/SEND 2-state FSM이 배치.
    
        + IDLE
    
            - oX_valid[0]가 assert되면, Layer 출력 vector(xX_out)을 holdData_X에 저장
        
            - 상태를 SEND로 전환
    
        - SEND

            - holdData_X의 LSB(dataWidth)를 out_data_X로 출력
            
            - holdData_X를 dataWidth만큼 shift
            
            - data_out_valid_X를 assert하여 다음 Layer 입력 valid로 사용
            
            - 지정된 neuron 수만큼 전송 후 IDLE 복귀
            
            - 즉, 각 Layer의 vector 출력은 out_data_X / data_out_valid_X 형태의 scalar stream으로 변환 후, 다음 Layer로 전달.

​

+ Layer 2 연산

    + Layer_2는 Layer 1 serialize된 출력(out_data_1, data_out_valid_1)을 입력으로 받아 연산.
    
    - 연산결과는 vector(x2_out)로 출력되며, o2_valid로 유효 시점을 표시.
    
    - 이후 동일한 FSM 구조를 통해 vector → scalar로 변환되어 Layer 3로 전달.

​

+ Layer 3 연산 (최종 FC Layer)

    + Layer_3는 Layer 2 serialize된 출력(out_data_2, data_out_valid_2)을 입력으로 받아 연산.
    
    - 결과는 최종 score vector(x3_out)로 출력되며, o3_valid가 유효 시점을 나타냄.

​

+ 최종 결과 선택: maxFinder

    + maxFinder는 최종 layer 출력 vector(x3_out)를 입력으로 사용.
    
    - vector 내 값 중 최댓값을 갖는 index를 선택하여 분류 결과(out) 생성.
    
    - out_valid는 최종 분류 결과의 유효 시점을 나타내며, 동시에 intr 신호로 외부에 전달.

+ 한 줄 요약

    + zyNet은 입력 stream을 받아 FC Layer 연산을 통해 vector를 생성하고, FSM 기반 Serialization을 거쳐 다음 Layer로 전달하는 구조.
    
    - 최종 Layer의 score vector에 대해 maxFinder가 class index를 선택하여 결과를 출력.
    
    - 상위에서는 include.v를 include하여 Neuron Network 전반에서 공통 parameter를 사용.​

## Reference Model Sub (neuron.v) 분석

Top module의 각 Layer는 다수의 neuron instance로 구성됩니다.

neuron.v는 입력 stream을 받아 dot-product(MAC) + bias + activation을 수행하는 최소 연산 Unit에 해당합니다.

이제 Reference Model의 Sub module인 neuron.v를 분석해보겠습니다.​

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_24.png" width="400"/>

H/W Architecture

<div align="left">

+ 공통 설정
    
    + include.v include로 공통 parameter 사용.
    
    - neuron 인스턴스는 layerNo, neuronNo, numWeight 등으로 구분됨.

+ 입력 데이터 수신

    + 입력 데이터: myinput
    
    - 입력 유효 구간: myinputValid
    
    - myinputValid는 단순 valid가 아니라, weight read enable + 누적(accumulate) 조건으로 사용됨.

+ Weight / Bias 로딩 방식

    + Weight 로딩

        + runtime 로딩 인터페이스 존재: weightValid, weightValue, config_layer_num, config_neuron_num

        - 해당 neuron 선택 조건: (config_layer_num==layerNo) & (config_neuron_num==neuronNo)
        
        - 조건 만족 시, Weight_Memory에 write 수행(wen, w_addr, w_in).

    + Bias 로딩

        + pretrained 정의 시: biasFile에서 $readmemb를 통해 preload.

        - pretrained 미정의 시: biasValid와 config match 조건을 만족할 경우 runtime 로딩.

​

Weight_Memory (stream 대응)

ren은 myinputValid에 직접 연결되어, 입력이 valid인 cycle에서만 Weight Memory read 수행.

read address r_addr는 myinputValid가 유지되는 동안 매 cycle마다 1씩 증가.

결과적으로 입력 index와 weight index가 cycle 단위로 1:1 대응되도록 정렬됨.

​

MAC 연산 및 Accumulation (valid 기반)

입력은 myinputd로 1-cycle 지연되어 multiply 연산에 사용됨.

multiply 연산: mul <= myinputd * w_out

accumulate enable 조건: mux_valid = mult_valid

mux_valid가 assert된 cycle에서만 sum <= sum + mul 수행.

누적 과정에서 overflow 발생 시 saturation 처리 포함.

​

Input 종료 검출 및 BiasAdd (단 1회)

input 종료 조건은 length 신호가 아니라, valid falling edge 기반으로 판단됨.

muxValid_f <= !mux_valid & muxValid_d

종료 이벤트와 모든 weight 처리 완료 조건: (r_addr == numWeight) & muxValid_f

위 조건에서 sum <= sum + bias 수행(1회).

해당 시점에 sigValid 생성 후 outvalid로 전달.

​

Activation 선택 (compile-time)

actType에 따라 activation 블록 결정(generate).

sigmoid 설정 시 Sig_ROM 사용.

그 외의 경우 ReLU 사용.

activation 결과는 최종 출력 out으로 연결됨.

​

한 줄 요약

neuron은 입력 stream을 받아 weight memory 기반 dot-product 연산을 수행하고,

valid 기반 누적(MAC)과 bias add를 거쳐 activation을 적용한 단일 출력 값을 생성하는 최소 연산 unit.

상위 Layer에서는 이러한 neuron 출력들을 모아 vector 형태로 구성함.

Summary

본 글에서 손글씨 인식 Reference Model의 구조와 동작 흐름을 Testbench, Top module, Sub module(neuron) 순서로 정리했습니다.

​

Reference Model은 입력 stream 수신, FC 연산, vector 생성 및 serialization을 거쳐 

다음 layer로 전달되는 명확한 파이프라인 구조를 가집니다.

​

다만 FC 기반 구조는 연산이 시간 축으로만 수행되는 1-D 방식이기 때문에 병렬 처리가 어렵고, 

입력 차원이 커질수록 성능과 PPA 측면에서 비효율이 커집니다.

​

이후 내용에서는 이러한 한계를 개선하기 위해 Weight-Stationary 기반 Systolic Array 구조로 전환하고, 

이에 따른 dataflow와 H/W Architecture를 살펴보도록 하겠습니다.

FC 기반 Reference Model의 Systolic Array 적용

다음으로 Reference Model의 기존 1-D Fully Connected(FC) 구조를2-D 구조인 Systolic Array 기반 구현으로 대체해 보겠습니다.

Systolic Array의 개념과 연산 방식은 아래 링크와 뒤에 이어질 내용을 참고해 주세요.

​

https://blog.naver.com/mini9136/224141619115


 
[Mini-NPU] 개념 정리 (Neural Network, Systolic Array)
Preliminarie 본격적인 Reference Model 분석에 앞서, 이후 내용을 이해하는 데 필요한 기본 개념들을 ...

blog.naver.com

왜 FC 대신 Systolic Array인가

기존 Reference Model의 Fully Connected(FC) 구조에서 각 neuron은 하나의 입력과 하나의 weight를 받아 곱셈을 수행하고,

이를 입력 개수만큼 순차적으로 누적(accumulate)하는 방식으로 동작합니다.

​

이 구조는

연산 흐름이 직관적이고

RTL 구현 및 디버깅이 비교적 단순하다는 장점을 가집니다.

​

그러나 연산이 1-Dimension으로 순차 수행되기 때문에, 연산량이 증가할수록 성능이 입력 길이에 직접적으로 종속되며,

PPA(Power, Performance, Area) 관점에서는 효율적인 구조라고 보기 어렵습니다.

​

특히,

동일한 weight가 여러 cycle 동안 반복 사용되고

MAC 연산이 시간 축으로만 확장되는 구조는 하드웨어 병렬성을 충분히 활용하지 못하는 한계를 가집니다.

​

FC 구조와 Systolic Array 구조 비교

이러한 특성 차이는 FC 구조와 Systolic Array 구조를 비교하면 보다 명확해집니다.

구분

Fully Connected

Systolic Array

연산 차원

1-D (시간 축)

2-D (공간 + 시간)

MAC 수행 방식

순차 수행

공간적 병렬 수행

Weight 사용

반복 재사용 (시간적)

PE 내부 고정(preload)

병렬성 활용

제한적

높음

성능 확장성

입력 길이에 종속

Array 크기에 따라 확장

PPA 관점

비효율적

상대적으로 효율적

​

Systolic Array 도입 배경

이러한 한계를 개선하기 위해, Reference Model의 FC 연산을 Systolic Array 구조로 대체하고자 합니다.

​

Systolic Array는 다수의 Processing Element(PE)를 2-Dimensional array 형태로 배치하고,

데이터가 시간에 따라 규칙적으로 이동하면서 연산이 수행되는 구조입니다.

​

이 구조에서는

다수의 MAC 연산이 공간적으로 병렬 수행되고

데이터 이동과 연산이 구조적으로 결합되며

그 결과 성능 및 에너지 효율 측면에서 이점을 확보할 수 있습니다.

​

즉, 기존 FC 구조가 시간 축(1D)으로 연산을 확장했다면,

Systolic Array는 공간 + 시간(2D)으로 연산을 전개하는 구조라고 볼 수 있습니다.

​

FC 연산을 Systolic Array에 적용하기 위한 고려사항

다만 Fully Connected 연산은 1-D dot-product 특성을 가지므로, Systolic Array에 직접 매핑할 수는 없습니다.

​

따라서,

연산을 tile 단위로 분해하고

입력과 weight를 어떤 방식으로 배치할지에 대한

Dataflow 정의가 필요합니다.

​

본 구현에서는 Weight-Stationary(WS) Dataflow를 사용합니다.

Weight를 tile 단위로 분할하여 Systolic Array 내부 PE에 preload하고, 입력 데이터만을 시간에 따라 전달하는 방식을 선택하였습니다.

​


4 x 4 Systolic Array (Weight Stationoary)를 적용한 예상 Block-Diagram

이와 같은 구조를 통해 기존 FC 기반 Reference Model 대비 연산 병렬성을 효과적으로 확장할 수 있으며, 

입력 차원 증가에 따른 성능 저하를 완화할 수 있습니다.

​

이후에는 

Systolic Array가 적용된 Reference Model의 전체 Dataflow와 각 Block의 역할을 중심으로 설계를 구체화해보도록 하겠습니다.
