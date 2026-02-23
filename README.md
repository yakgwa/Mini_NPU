<div align="center">

<!-- logo -->
<img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/images_1.jpeg" width="400"/>

### Mini NPU 🖍️

[<img src="https://img.shields.io/badge/-readme.md-important?style=flat&logo=google-chrome&logoColor=white" />]() [<img src="https://img.shields.io/badge/release-v1.0.0-ㅎㄱㄷ두?style=flat&logo=google-chrome&logoColor=white" />]() 
<br/> [<img src="https://img.shields.io/badge/프로젝트 기간-2026.02.23~2026.03.03-fab2ac?style=flat&logo=&logoColor=white" />]()

</div> 

## 📝 소개
Mini NPU Architeture & Design & Verfication

다음과 같은 내용을 작성할 수 있습니다.
- Google TPU
- 프로젝트 화면 구성
- 사용한 기술 스택
- 기술적 이슈와 해결 과정
- 프로젝트 팀원

필요한 기술 스택에 대한 logo는 [Mini NPU 폴더](/rtl)에서 다운로드 받을 수 있습니다.

<br />

## Prologue - GPU 시대에 등장한 TPU
최근 몇 년간 AI 시장에서는 NVIDIA의 GPU가 여전히 강세를 보이고 있습니다.

GPU는 딥러닝(Deep-Learning) 계산에 잘 맞도록 계속 발전해 왔고, CUDA라는 개발 환경도 매우 잘 갖춰져 있습니다.

이 덕분에 논문을 쓰는 연구자든, 실제 서비스를 만드는 개발자든 딥러닝은 GPU로 처리하는 것이 자연스러운 선택이 되었습니다.​ 그런데 이런 GPU 중심의 흐름 속에서, 한 가지 흥미로운 사례가 등장합니다. 

바로  Google의 TPU(Tensor Processing Unit)입니다.

https://www.opinionnews.co.kr/news/articleView.html?idxno=128677&utm_source=chatgpt.com

TPU는 2016년 처음 공개된 이후, 구글 내부의 대규모 딥러닝 추론(inference) 작업을 처리하기 위해 별도로 설계된 ASIC(주문형 반도체)입니다.

GPU와 비교했을 때 TPU는 inference workload에 최적화된 연산 구조와 메모리 구성, 연산과 메모리 간 데이터 이동을 줄여 높은 throughput을 달성, 높은 energy efficiency를 목표로 설계되었습니다.​

이번 글에서는 TPU 논문을 중심으로, 해당 architecture가 어떤 설계 의도를 가지고 있으며, H/W 구조가 어떻게 구성되어 있는지를 살펴보겠습니다.

## Abstract 

많은 컴퓨터 구조 설계자들은 이제 성능과 에너지 효율을 동시에 크게 개선하기 위해서는 범용 프로세서가 아닌 도메인 특화 하드웨어, 즉 이미 학습된 AI 모델을 기반으로 실제 예측을 수행하는 추론 단계에 최적화된 특수 목적 하드웨어가 필요하다고 보고 있습니다.

​2015년부터 Google 데이터센터에 실제로 배치되어 사용 중인 TPU(Tensor Processing Unit)가 그 중 하나입니다. 

​TPU의 핵심은 65,536개의 8-bit MAC으로 구성된 Matrix Multiply Unit입니다. 이를 통해 최대 92 TOPS(Tera Operations Per Second)의 연산 성능을 제공합니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_2.jpg" width="400"/>

<div align="center">Multiply와 Adder, Accumulate로 이루어진 MAC 구조 

​

<div align="left">TPU는 deterministic execution model을 사용하며, 99th-percentile response time(p99 latency) 요구사항에 더 잘 맞는 구조입니다.
→ 이는 추론 workload가 반복적이고 입력 크기와 연산 패턴이 고정되어 실행과 결과가 예측 가능하기 때문이며, 실제 서비스 환경에서 정해진 시간 내에 전체 요청의 99%가 응답되는 요구사항을 안정적으로 만족할 수 있습니다.

이러한 설계 선택 덕분에, TPU는 많은 MAC과 큰 on-chip memory를 포함하고 있음에도 불구하고 상대적으로 작고 저전력인 칩으로 구현될 수 있었습니다.

논문에서는 TPU를 동일한 데이터센터 환경에서 사용된 Intel Haswell CPU와 NVIDIA K80 GPU와 비교합니다.

​평가에 사용된 workload는 TensorFlow로 작성된 실제 production inference application이며, MLP, CNN, LSTM 모델을 포함해 데이터센터 추론 워크로드의 약 95%를 대표합니다.
→ 실제 데이터센터에서 돌아가는 대표 추론 모델을 사용해, 실제 서비스 환경을 거의 그대로 반영했음을 나타냅니다.

​그 결과, 일부 application에서의 낮은 자원 활용률에도 불구하고 TPU는 평균적으로 CPU나 GPU 대비 15~30배 빠른 성능을 보였고, 전력 효율(TOPS/Watt) 역시 30~80배 더 높게 나타났습니다.​

## 1. Introduction — Neural Network와  inference환경

대규모 데이터와 이를 처리할 수 있는 컴퓨팅 인프라는 머신러닝, 특히 Deep Neural Network(DNN)의 발전을 가능하게 했습니다.

​신경망(Neural Network)은 입력의 weighted sum에 non-linear activation function을 적용하는 인공 뉴런(artificial neuron)을 기본 단위로 하며, 

이러한 neuron들이 layer로 연결되어 한 layer의 output이 다음 layer의 input으로 전달되는 구조를 가집니다.

DNN에서 “deep”이라는 개념은 다수의 layer가 중첩된 구조에서 비롯되며, cloud 환경의 대규모 data set과 GPU의 높은 computing power는더 많은 layer와 더 큰 model을 학습할 수 있는 기반을 제공했습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_3.jpg" width="400"/>

<div align="center">Artificial Neuron

<div align="center">입력(Input)에 가중치(Weight)를 곱해 합산한 뒤, 그 결과에 activation function을 적용하는 구조

​

<div align="left">이제 다시 신경망의 동작 과정으로 돌아가 보면, 신경망에는 크게 두 가지 단계가 존재합니다.

Training: 모델의 구조는 고정한 채, 정답이 있는 데이터를 반복적으로 입력하며 예측 결과가 맞아지도록 weight 값을 학습하는 단계

Inference: Training을 통해 이미 학습된 weight를 그대로 사용하여, 새로운 입력이 주어졌을 때 정답을 분류하거나 결과를 예측하는 단계.

​Training 단계에서는 weight 학습 과정의 수치적 안정성을 위해 높은 정밀도의 floating-point 연산이 요구되며, 현재 대부분의 training은 16-bit floating-point 연산을 기반으로 수행됩니다.

​반면, 논문에서 다루는 데이터센터 inference workload의 경우 이미 학습된 weight를 사용하므로, quantization을 통해 8-bit 정수 연산으로 변환하더라도 대부분의 경우 충분한 정확도를 유지할 수 있습니다.

​이러한 선택은  Power, Performance, Area(PPA) 측면에서 유리한 설계 trade-off로 설명됩니다.

​### 1.1 데이터센터에서 사용되는 Neural Network 유형

현재 데이터센터 추론 환경에서 널리 사용되는 신경망은 크게 세 가지 유형으로 나눌 수 있습니다.

- MLP (Multi-Layer Perceptron: 각 layer가 이전 layer의 모든 출력과 fully connected로 연결되는 구조
- CNN (Convolutional Neural Network): 공간적으로 인접한 입력에 대해 동일한 weight를 반복 적용하는 구조
- RNN / LSTM (Recurrent Neural Network): 이전에 처리한 정보를 기억하면서, 시간 순서대로 데이터를 하나씩 처리하는 구조

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_4.png" width="400"/>

​

<div align="left">

### 1.2 추론 workload의 현실적인 제약

논문에서 사용하는 여섯 개의 benchmark application은 앞서 정리한 세 가지 신경망 유형(MLP, CNN, LSTM)을 대표하며, Google 데이터센터 추론(inference) 워크로드의 약 95%를 차지합니다.

이러한 inference application들은 대부분 검색, 번역, 추천과 같은 user-facing service의 일부로 동작하며,

그 결과 추론 환경에서는 처리량(throughput)보다 응답 시간(response time)이 훨씬 더 중요한 성능 지표로 작용합니다.

또한 각 모델은 수백만에서 수천만 개에 이르는 weight를 포함하고 있어,

이 weight에 접근하는 데 소요되는 시간과 에너지 비용이 전체 성능을 제한하는 주요 요인이 될 수 있습니다.

이는 performance와 power 측면에서 모두 불리하게 작용합니다. Batch 단위로 동일한 weight를 재사용하면 이러한 접근 비용을 amortize하여 효율을 높일 수 있지만,

실제 서비스 환경에서는 엄격한 latency 제한으로 인해 batch size를 크게 증가시키는 것이 항상 가능한 선택은 아닙니다.

이러한 배경에서 본 논문은, 데이터센터 추론 환경의 특성에 보다 적합한 하드웨어 구조로서 Tensor Processing Unit(TPU) 아키텍처를 제시합니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_5.png" width="400"/>

Neural Network 유형별 비교:

Layer 수와 Weight 수를 중심으로 구조를 중점적으로 Check.

<div align="left">

## 2. TPU Origin, Architecture, Implementation, and Software

### 2.1 TPU의 등장 배경

Google은 2006년부터 데이터센터에 GPU/FPGA/ASIC 도입을 검토했지만, 당시에는 특수 하드웨어가 필요한 workload가 제한적이라 대규모 데이터센터의 미사용 연산 자원으로 사실상 ‘공짜’에 가깝게 처리할 수 있다고 판단했습니다. 하지만 2013년, 음성 검색을 하루 3분만 사용해도 DNN 기반 음성 인식이 데이터센터 연산 수요를 2배로 늘릴 수 있다는 전망이 나오면서, CPU 기반 인프라로는 비용 부담이 급격히 커질 것이 명확해졌습니다.

이에 Google은 추론(inference)용 custom ASIC을 빠르게 개발하는 최우선 프로젝트를 시작했고, 학습(training)은 상용 GPU를 구매해 대응했습니다. 

​목표는 GPU 대비 10배의 cost-performance 개선이었으며, 그 결과 TPU는 설계–검증–제작–데이터센터 배치까지 약 15개월 만에 완료되었습니다.

​### 2.2 시스템 구성과 설계 방향

TPU는 deployment 지연 가능성 최소화를 위해, CPU와 tightly integrated된 구조가 아닌 PCIe I/O bus 상의 coprocessor (특정 연산 수행 보조 Processor)로 설계되었으며, GPU와 마찬가지로 기존 서버 아키텍처를 변경하지 않고 데이터센터에 바로 장착할 수 있는 형태를 채택했습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_6.png" width="400"/>

​

<div align="left">

또한 하드웨어 설계와 디버깅을 단순화하기 위해, TPU가 자체적으로 instruction을 fetch하는 방식이 아니라 host CPU가 실행할 TPU instruction을 전달하는 구조를 사용합니다.

이러한 설계로 인해 TPU는 GPU보다는 전통적인 FPU(floating-point unit) coprocessor에 더 가까운 성격을 가집니다.

→ GPU는 내부에서 instruction을 fetch–decode–execute하며 실행 흐름을 스스로 관리하는 programmable processor인 반면, TPU는 실행 제어를 host CPU에 맡기고, 전달받은 명령에 따라 연산만 수행하는 구조를 채택했습니다.

이로 인해 TPU는 복잡한 fetch·decode·control logic을 제거할 수 있었으며, 하드웨어 구조와 디버깅 복잡도를 크게 감소시켰습니다.

설계 목표는 host CPU와의 interaction을 최소화하면서 전체 inference 모델을 TPU 내부에서 모두 실행하는 것이었으며,

추후 등장할 NN workload까지 고려한 유연성을 확보하는 것이었습니다.

### 2.3 TPU의 전체 구조 개요

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_7.png" width="400"/>

TPU Block Diagram

​<div align="left">TPU instruction은 PCIe Gen3 x16 bus를 통해 host에서 instruction buffer로 전달되며, 내부 block들은 일반적으로 256-byte-wide data path로 연결되어 있습니다.
→ TPU를 제어하는 host(일반적으로 CPU)가 수행할 연산과 데이터 이동에 대한 지시사항(Instruction)을 생성해 TPU로 전달합니다.

이때 host와 TPU는 표준 interface인 PCIe Gen3 x16을 통해 연결되며, 전달된 instruction은 TPU 내부의 instruction buffer (그림의 Instr Block)에 저장된 뒤 순차적으로 해석·실행됩니다.
→ 대규모 행렬 곱 연산 시 연속적인 데이터 공급과 peak utilization(연산 유닛이 매 cycle마다 쉬지 않고 동작)을 유지하기 위해, 내부 block 간을 매우 넓은 data path로 연결하는 구조를 채택했습니다. TPU 구조의 중심에는 Matrix Multiply Unit이 위치합니다.

이 unit은 256×256 MAC array 로 구성되어 있으며, signed 또는 unsigned 8-bit integer multiply & add 연산을 수행합니다.

​연산 결과인 16-bit product는 matrix unit 아래에 위치한 4 MiB 크기의 32-bit Accumulator에 누적됩니다.

이 accumulator 메모리는 4096개의 256-element accumulator를 저장할 수 있도록 구성되어 있습니다.

※ 4096은 roofline model에서 도출된 knee point(약 1350 operations/byte)를 기준으로 최소 요구치를 2048로 올림한 뒤, 컴파일러가 peak performance 상태에서도 double buffering을 활용할 수 있도록 이를 다시 두 배로 확장해 결정되었습니다.
 →8-bit activation과 8-bit weight의 곱셈 결과는 최대 16-bit로 생성되며,

행렬 곱 과정에서 생성되는 partial sum은 여러 차례 누적되며, 이러한 최종 누적 결과는 32-bit Accumulator에 저장됩니다.
→ Matrix Multiply Unit은 한 cycle에 256개의 출력 값을 한 묶음(256-element row)으로 생성해 Accumulator로 전달하며,

Accumulator 메모리는 이러한 256-element row를 여러 개 (4096개) 동시에 누적·보관할 수 있도록 구성되어 있습니다.

### 2.4 Matrix Multiply Unit의 동작 특성

Matrix Multiply Unit은 clock cycle당 하나의 256-element partial sum을 생성합니다. 연산에 사용되는 bit width에 따라 처리 속도는 다음과 같이 달라집니다.

- 8-bit weight + 8-bit activation → full speed
- 8-bit / 16-bit 혼용 → half speed
- 16-bit / 16-bit → quarter speed
→ bit width가 증가할수록 하나의 MAC 연산이 더 많은 하드웨어 자원을 점유하게 되며, 그 결과 동일한 하드웨어에서 처리 가능한 연산 throughput이 감소합니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_8.png" width="400"/>
 
Matrix Unit (Systolic array) 예시

<div align="left">Matrix Unit은 cycle당 256개의 value를 read/write할 수 있으며, matrix multiplication과 convolution을 모두 수행할 수 있습니다.

이 unit은 64 KiB 크기의 weight tile과, tile 전환 시의 지연을 숨기기 위한 double buffering용 tile을 함께 보유합니다.

이는 새로운 tile을 로드하는 데 필요한 256 cycle의 latency를 연산과 겹쳐 숨기기 위한 설계입니다.

→ tile이란 Matrix Unit이 한 번에 처리하도록 chip 내에 고정해 두는 weight의 작업 단위를 의미합니다. 전체 weight를 모두 chip 내부에 저장하기에는 용량 제약이 있으므로, TPU는 계산에 필요한 weight만을 tile 단위로 분할해 순차적으로 로드합니다.

이때 double buffering을 적용해, 하나의 tile로 연산을 수행하는 동안 다음 tile을 미리 로드함으로써 tile 교체 과정에서 발생하는 256 cycle의 지연을 연산과 겹쳐 숨길 수 있도록 설계되었습니다.

​TPU는 대부분의 값이 실제 연산에 사용되는 dense 행렬(대부분의 원소가 0이 아닌 행렬)을 대상으로 설계되었으며, chip을 빠르게 개발하고 성능을 안정적으로 확보하기 위해 sparse 연산 지원은 의도적으로 제외되었습니다.

​Weight는 Chip 내부 Weight FIFO를 통해 공급되며, 이 FIFO는 Chip 외부에 위치한 8 GiB DRAM 기반 Weight Memory에서 데이터를 읽어옵니다.
→ off-chip DRAM(그림 내 DDR3 DRAM Block) 접근 시 발생하는 수백~수천 cycle의 지연을 피하기 위해, weight를 미리 FIFO에 적재함으로써 Matrix Unit에 연속적인 데이터 공급이 가능하도록 설계되었습니다.

​Inference 단계에서 weight는 read-only이며, 이 용량은 여러 model을 동시에 활성화하기에 충분합니다.
→ training 단계에서 이미 결정된 weight를 사용하므로 inference 과정에서는 값이 변경되지 않습니다. →따라서 하드웨어 설계에서는 weight memory를 read-only로 가정하고 최적화할 수 있습니다.​

### 2.5 Unified Buffer와 데이터 이동

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_9.png" width="400"/>
 
TPU Floorplan

<div align="left">Matrix Unit의 입력과 중간 결과는 24 MiB 크기의 chip 내부 Unified Buffer에 저장되며, 이 buffer는 Matrix Unit의 입력으로 직접 사용될 수 있습니다.
→ Unified Buffer(UB)는 chip 내부에 위치한 대용량 software-managed scratchpad memory로, Matrix 연산에 사용되는 activation과 partial sum을 저장합니다. UB에 저장된 데이터는 DRAM을 경유하지 않고 Matrix Unit으로 직접 공급되도록 설계되었습니다.

​CPU host memory와 Unified Buffer 사이의 데이터 이동은 programmable DMA controller가 담당합니다.
→ DMA controller는 Matrix Unit의 연산과 독립적으로 동작하며, 연산이 진행되는 동안 다음 layer의 입력이나 다음 tile을 사전에(prefetch)하여 Unified Buffer로 전송할 수 있습니다. 이를 통해 데이터 이동 latency를 연산과 겹쳐 숨길 수 있고, Matrix Unit은 가능한 한 연산에만 집중하도록 유지됩니다.

Die floorplan을 보면, Unified Buffer는 die 면적의 약 1/3, Matrix Multiply Unit은 약 1/4를 차지하여, 전체 datapath가 die 면적의 약 2/3를 구성합니다.

반면, control logic은 전체의 **약 2%**에 불과합니다.

→ 이는 TPU가 die 면적의 대부분을 datapath와 데이터 저장·이동을 위한 구조에 할당하고, control logic은 최소화함으로써, 연산 유닛의 utilization을 높이고 데이터 재사용과 처리량(throughput)을 극대화하는 방향으로 설계되었음을 보여줍니다.

### 2.6 Instruction과 실행 모델

TPU instruction은 비교적 느린 PCIe bus를 효율적으로 사용하기 위해 repeat field를 포함한 CISC 형태를 따르며, 

이로 인해 평균 CPI(Cycle Per Instrucion)는 약 10~20 cycle 수준으로 나타납니다.

→ 짧은 instruction을 자주 보내는 RISC 방식보다는 하나의 instruction이 많은 연산을 반복 수행 가능한 구조를 사용합니다.

TPU의 instruction 수는 최소화되어 있으며, 전체 동작은 다음 다섯 가지 핵심 instruction을 중심으로 구성됩니다.

- Read_Host_Memory:  

→ CPU host memory(DRAM)에 저장된 입력 (activation) 를 Chip 내부 Unified Buffer(UB)로 전송합니다.

- Read_Weights: 

→ Weight Memory로부터 weight를 미리 읽어, Matrix Unit 입력으로 사용되는 Weight FIFO에 저장합니다.

- Matrix_Multiply / Convolve: 

→ Unified Buffer의 activation을 Matrix Unit에 공급하여, 연산 결과를 Accumulator에 누적하는 

matrix multiplication 또는 convolution을 수행합니다.

- Activate (ReLU, Sigmoid 등)

→ Accumulator 연산 결과에 nonlinear function을 적용한 뒤 UB로 저장하며, 

필요 시 추가 hardware를 통해 pooling operation을 수행합니다.

- Write_Host_Memory

→ Unified Buffer에 저장된 결과 데이터를 CPU host memory(DRAM)로 기록합니다.

TPU microarchitecture의 기본 철학은 Matrix Multiply Unit이 stall 없이 최대한 바쁘게 유지되도록 하는 것입니다.

이를 위해 TPU는 CISC instruction에 대해 4-stage pipeline을 사용하며, 서로 다른 instruction의 실행을 overlap시켜 latency를 숨기는 구조를 채택합니다.

이때 Read_Weights instruction은 decoupled-access/execute 방식을 따르며, weight fetch가 완료되기 전에 instruction 자체는 먼저 완료될 수 있습니다.

다만, input activation 또는 weight 데이터가 준비되지 않은 경우 Matrix Unit은 stall하게 됩니다.
→ Read_Weights instruction은 weight를 가져오라는 요청만 먼저 발행하고, 

실제 weight 데이터의 도착 여부와는 독립적으로 instruction 완료가 가능합니다. 

그러나 Matrix Unit이 실제 연산을 수행하기 위해서는 input activation과 weight가 모두 준비되어 있어야 하므로, 필요한 데이터가 준비되지 않은 경우 연산 단계에서는 stall이 발생합니다.

​또한 네트워크 layer 간에는 데이터 의존성이 존재하므로, 이전 layer의 activation이 Unified Buffer에 모두 준비될 때까지 다음 MatrixMultiply instruction은 명시적인 synchronization을 기다리며 RAW stall이 발생할 수 있습니다.
→ 이는 단순한 memory access 지연이 아니라, layer 간 Read-After-Write(RAW) 데이터 의존성에 의해 발생하는 stall입니다. 

즉, instruction은 issue 및 완료될 수 있으나, 이전 layer의 결과 activation이 Unified Buffer에 완전히 기록되기 전까지는 다음 MatrixMultiply 연산이 시작될 수 없습니다.

​이로 인해 TPU의 pipeline overlap은 전통적인 RISC pipeline처럼 고정된 1-cycle 단위가 아니라, instruction에 따라 수백~수천 cycle 동안 하나의 stage를 점유하는 형태로 동작합니다.

### 2.7 Systolic Execution

대용량 SRAM 접근은 산술 연산에 비해 훨씬 높은 전력 소모를 유발합니다. 이에 따라 TPU는 Unified Buffer 접근을 최소화하고 데이터 재사용을 극대화하기 위한 수단으로 systolic execution을 채택했습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_10.png" width="400"/>
 
Google TPU의 systolic execution

<div align="left">이 그림을 이해하기 위해서는, 먼저 systolic array 구조에 대해 이해할 필요가 있습니다.

Systolic execution은 다수의 Processing Element(PE)를 배열 형태로 연결하고, activation과 weight를 PE 간에 전달하면서 중간 결과를 외부 메모리에 다시 저장하지 않고 Matrix Unit 내부에서 연속적으로 multiply-accumulate 연산을 수행하는 방식입니다.

각 PE는 activation과 weight를 입력으로 받아 MAC 연산을 수행하고, partial sum을 누적하는 동시에 activation 또는 결과를 인접한 PE로 전달합니다.

이러한 구조를 통해 연산에 필요한 데이터는 PE 간 이동을 통해 재사용되며, Unified Buffer에 대한 반복적인 read/write가 크게 줄어듭니다.

​

연산 방법은 다음과 같습니다.

아래 그림들은 추후 실제 Mini-TPU 설계에서 사용할 2×2 systolic array 구조를 기반으로,systolic array에서 matrix multiplication이 시간에 따라 어떻게 진행되는지를 설명하기 위한 것입니다.

​

이때 행렬 A와 행렬 B가 주어지며,

그 곱 결과인 행렬 C = A × B를 계산하는 과정을 예로 설명합니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_11.png" width="400"/>
 
2×2  행렬 곱 연산 예시

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_12.png" width="400"/>

2×2 systolic array

<div align="left">​

​


Previous imageNext image
Time step 1

첫 번째 PE로 입력된 a11과 b11이 PE 내부에서 곱셈 연산을 수행하며,

그 결과인 partial sum이 PE 내부 accumulator register에 저장됩니다.

(그림에는 accumulator가 명시적으로 표시되지 않았으나, 연산 결과는 PE 내부에 유지됩니다.)

​

Time step 2

이전 단계에서 사용된 activation과 weight는 각각 인접한 PE로 전달되어 새로운 곱셈 연산을 시작합니다.

동시에 첫 번째 PE에는 새로운 입력 a12와 b21이 도착하여,

기존 accumulator 값에 추가로 누적됩니다.

​

Time step 3 ~ 4

이후 단계에서도 동일한 방식으로 activation과 weight가 systolic array 내부를 따라 이동하며,

각 PE에서는 multiply-accumulate 연산이 연속적으로 수행됩니다.

​

이 과정을 통해 최종적으로 각 PE의 accumulator에는

행렬 곱 결과에 해당하는 값이 저장되며, 전체 행렬 C가 완성됩니다.

​

※ PE 내부 저장 vs 외부 Accumulator에 대한 설계 고민

위의 2×2 systolic array 예시는 partial sum을 PE 내부에 저장하는 구조를 기준으로 설명되었습니다.

​

하지만 Google TPU와 같이 PE 외부에 별도의 accumulator block을 두는 구조를 적용하려면,

partial sum을 어디에 저장하고 어떻게 전달할지에 대한 구조적·데이터 경로 설계가 필요합니다.

​

이는 단순히 PE 구조의 차이에서 비롯되는 문제가 아니라,어떤 데이터를 PE 내부에 유지하느냐에 따라 실행 방식이 달라지는 문제입니다.

​

앞서 살펴본 2×2 systolic array 예시는

PE 간 데이터가 이동하며 multiply-accumulate 연산이 수행되는 기본 동작을 보여주기 위한 예시입니다.

​

이때 중요한 점은, 

이러한 동작이 PE 구조 자체의 차이에서 비롯되는 것이 아니라, 어떤 데이터를 PE 내부에 유지하느냐에 따라 달라진다는 점입니다.

​

동일한 PE 배열을 사용하더라도, partial sum, weight, activation 중 어떤 데이터를 PE에 고정(stationary) 해 두느냐에 따라

서로 다른 실행 방식이 정의될 수 있으며, 이를 dataflow라고 부릅니다.

​

대표적으로 Output-Stationary, Weight-Stationary, Input-Stationary 방식이 존재합니다. 


systolic execution dataflow type

Output-Stationary PE (OS)

partial sum이 PE에 고정되어있으며,

activation과 weight가 PE로 입력되어 이동합니다.

​

Weight-Stationary PE (WS)

weight가 PE에 고정되어있으며,

activation이 입력으로 들어와 이동을 합니다.

주로 TPU에서 사용됩니다.

​

Input-Stationary PE (IS)

activation이 PE에 고정되어있으며,

weight가 입력으로 들어와 이동을 합니다.

​

TPU는 이 중에서 Weight-Stationary(WS) 성향의 구조를 채택합니다.

이는 inference 단계에서 weight가 read-only이며 재사용률이 높다는 특성을 활용하기 위함입니다.

→ weight를 PE에 고정하고 activation만 이동시킬 경우, Unified Buffer 및 off-chip memory 접근을 효과적으로 줄일 수 있습니다.

​

이러한 dataflow 선택은 단일 PE의 동작 방식에만 영향을 주는 것이 아니라,

연산이 tile 단위로 반복되고 array 전체로 확장될 때의 실행 패턴에도 직접적인 영향을 미칩니다.

​

2×2 array에서 관찰한 데이터 이동과 누적 방식은,

더 큰 systolic array로 확장되면 diagonal wavefront 형태로 array 전체를 따라 전파됩니다. ** Section 2.4 그림 참조

이 과정에서 각 PE는 서로 다른 시점의 연산 단계에 위치하게 되며,연산의 시작과 종료 시점이 PE마다 자연스럽게 어긋나게 됩니다.


Google TPU의 systolic execution

이 과정에서 각 PE는 서로 다른 시점의 연산 단계에 위치하게 되며, 연산의 시작과 종료 시점이 PE마다 자연스럽게 어긋나게 됩니다. 

그 결과 별도의 복잡한 제어 없이도 control과 data가 자연스럽게 pipeline되며, 

pipeline이 충분히 채워진 이후에는 매 cycle마다 다수의 PE가 동시에 활성화됩니다.

​

논문에서 언급되는 *“256개의 input이 동시에 read되고, 256개의 accumulator가 즉시 update되는 것처럼 동작한다”*라는 표현은, 

실제로 모든 입력이 한 번에 메모리에서 읽히거나 모든 accumulator가 동일한 시점에 갱신된다는 의미는 아닙니다. 

​

이는 대규모 systolic array에서 여러 연산이 시간 축에서 overlap되어 실행되며

pipeline이 충분히 채워진 상태에서 Matrix Multiply Unit이 높은 utilization으로 지속적으로 동작하고 있음을 표현한 것입니다.

​

즉, wavefront 기반 실행과 충분히 채워진 pipeline에 의해 

Matrix Multiply Unit은 매 cycle마다 유효한 연산을 수행하며, 가능한 한 idle 없이 연속적으로 동작하도록 설계되었습니다. 

이는 Matrix Multiply Unit을 최대한 쉬지 않고 동작시키려는 TPU의 설계 목표가 실제 실행 단계에서 구현된 모습이라 볼 수 있습니다

​

한편, 이러한 systolic execution은 software 입장에서는 연산 결과의 정확성만 보장되면 되며, 

내부 아키텍처 구현 방식과는 직접적인 관련이 없습니다.

​

Software는 내부적으로 systolic array가 어떻게 구성되어 있는지, 또 어떤 dataflow가 사용되는지를 인지할 필요 없이

연산 결과의 정확성만을 보장하면 됩니다.

​

반면, 성능(performance) 관점에서는 systolic execution의 특성이 매우 중요합니다.

​

Matrix Unit의 실행 시간, 데이터 준비 시점, tile 전환 타이밍에 따라 Matrix Multiply Unit의 utilization이 결정되며, 

이는 TPU 전체 성능에 직접적인 영향을 미칩니다. 

​

즉, systolic execution은 software에 의해 직접 제어되거나 노출되지는 않지만, 

데이터 이동과 연산의 overlap 정도(latency hiding)에 따라 실제 성능을 좌우하는 핵심 요인으로 작용합니다.

​

2.8 Software Stack

TPU는 기존 CPU/GPU 환경에서 사용하던 소프트웨어를 큰 수정 없이 그대로 사용할 수 있도록 설계되었습니다.

​

Application은 TensorFlow로 작성되며, 

동일한 API를 통해 GPU 또는 TPU용으로 컴파일될 수 있어 기존 모델을 TPU로 쉽게 옮길 수 있습니다.

​

TPU 소프트웨어는 크게 Kernel Driver와 User Space Driver로 나뉩니다.

Kernel Driver는 메모리 관리와 interrupt 처리 등 필수적인 기능만 담당하는 얇은 계층으로, 안정성을 우선합니다.

​

반면 User Space Driver는 TPU 실행을 실제로 준비하고 제어하는 계층으로,

연산 설정, 데이터 형태 변환, 그리고 고수준 API를 TPU가 이해할 수 있는 명령어로 변환·컴파일하는 역할을 수행합니다.

​

이 때문에 User Space Driver는 단순한 드라이버라기보다는 TPU 전용 컴파일러와 실행 환경에 가깝습니다.

​

TPU 프로그램은 처음 실행될 때 모델 컴파일과 weight 로딩이 한 번 수행되며,

이 결과는 캐시되어 이후 동일한 모델을 실행할 때는 별도의 준비 과정 없이 곧바로 최대 성능으로 실행됩니다.

​

또한 TPU는 입력부터 출력까지의 전체 연산을 가능한 한 chip 내부에서 끝내도록 설계되어,

외부와의 데이터 전송(PCIe I/O)을 최소화하고 실제 연산에 사용하는 시간 비율을 최대화합니다. 

​

실제 연산은 주로 layer 단위로 수행되며, instruction 및 연산의 overlapped execution을 통해

Matrix Multiply Unit이 non-critical-path 작업으로 인해 idle 상태에 빠지는 것을 최대한 방지합니다.

→ 이로 인해 TPU는 개발자에게는 GPU와 유사한 가속기로 보이지만, 

실제로는 OS가 직접 세밀하게 제어하는 범용 프로세서가 아닌, 

실행 전 연산 순서와 데이터 흐름이 대부분 결정되는 연산 중심의 coprocessor로 동작합니다.

​

Software는 correctness만을 보장하면 되며, 실제 성능은 컴파일 단계에서 결정된 실행 스케줄과 데이터 이동에 의해 좌우됩니다.

+ 추가 (26.01.06.)

스터디를 진행하며 생긴 추가 질문들은 아래에 정리하였습니다.

​

질문사항

1. HW설계 관점에서 floating point vs fixed point 차이는?

→ 

Floating-point arithmetic는 넓은 dynamic range와 높은 numerical stability를 제공하지만, 

hardware implementation cost가 큽니다. 

​

Exponent alignment, mantissa normalization, rounding과 같은 추가적인 logic이 필요하므로, 

area, power, 그리고 critical path latency가 증가합니다. 

이러한 특성으로 인해 floating-point 연산은 주로 training phase에서 사용됩니다.

​

반면 fixed-point (integer) arithmetic는 표현 가능한 dynamic range는 제한적이지만, 

multiplier와 adder 중심의 단순한 datapath로 구현할 수 있어 Power, Performance, Area(PPA) 측면에서 매우 효율적입니다. 

​

이미 학습된 weight를 사용하는 inference phase에서는 quantization을 적용하더라도 충분한 accuracy를 유지하는 경우가 많아, fixed-point 연산이 널리 사용됩니다.

​

2. Double Buffering 이란?

→ 

Double Buffering은 computation과 data transfer를 overlap하여 memory access latency를 숨기기 위한 기법입니다. 

​

하나의 buffer에서 computation이 수행되는 동안, 

다른 buffer에서는 subsequent tile, activation, 또는 weight data를 미리 load(prefetch)합니다. 

​

이를 통해 computation이 data fetch 완료를 기다리며 stall되는 상황을 방지할 수 있습니다.

​

TPU에서는 weight tile buffer, Unified Buffer(UB), accumulator memory 등 multiple memory hierarchy levels에서 double buffering이 적용되어, 

Matrix Multiply Unit(MMU)이 data starvation으로 인해 idle 상태에 빠지는 것을 최소화하도록 설계되었습니다. 

​

이러한 구조는 computation throughput을 안정적으로 유지하고, overall utilization을 극대화하는 데 기여합니다.

​

관련자료: https://blog.naver.com/bbineekim/222873251460

​

3. Systolic Array의 장점은? Systolic Array를 사용하지 않는다면 어떤 식으로 구현되나?

→ 

Systolic Array의 가장 큰 장점은 

data movement를 구조적으로 최소화하면서 compute unit utilization을 지속적으로 유지할 수 있다는 점입니다. 

​

Activation과 weight는 Processing Element(PE) 간에 local forwarding 방식으로 전달되며 반복적으로 data reuse가 이루어지고, partial sum은 array 내부에서 in-place accumulation됩니다. 

​

이로 인해 연산 과정에서 DRAM이나 상위 memory hierarchy에 대한 접근이 크게 감소하며, 

compute unit은 data starvation 없이 continuous execution이 가능합니다. 

​

이러한 특성은 matrix multiplication과 같이 MAC operation이 대량으로 반복되는 workload에서 높은 energy efficiency와 안정적인 throughput을 제공하는 데 특히 효과적입니다.

​

반면 systolic array를 사용하지 않는 경우에는, 

centralized register file이나 shared memory에서 데이터를 읽어 각 compute unit으로 분배하는 구조, 혹은 GPU와 같이 

scheduler-driven execution model을 사용하는 방식으로 구현됩니다.

​

이러한 구조는 execution flexibility는 높지만, 연산마다 데이터가 memory hierarchy를 반복적으로 이동해야 하며, 

이를 관리하기 위한 dynamic scheduling과 control logic overhead가 필수적으로 수반됩니다. 

​

그 결과 data movement overhead와 control complexity가 증가하고, 

동일한 성능을 달성하기 위해 더 많은 power and area cost가 요구되는 경우가 많습니다.

​

4. MLP에서 Hidden Layer와 weight 갯수는 어떻게 정하는가?

→ 댓글 참고.

​

5. AI동작은 Training, Inference 로 구분되는데, 둘다 HW로 만드는건지?

→ 댓글 참고.

이번 글에서는 Google의 TPU 논문

In-Datacenter Performance Analysis of a Tensor Processing Unit

을 바탕으로 분석을 진행했습니다.

​

CPU·GPU·TPU의 실제 시뮬레이션 결과에 대한 내용은 본문에서 다루지 않았으며,

해당 내용은 논문의 Section 3 이후에서 확인할 수 있습니다.

​

자세한 분석은 기회가 된다면 추후 별도의 글로 다룰 예정입니다.

​

CPU·GPU·TPU 시뮬레이션 비교 분석: (추후 작성 예정)​

추가로, PPT 자료가 필요하신 경우 아래 링크를 참고하시면 됩니다.

https://courses.grainger.illinois.edu/cs533/sp2025/notes/tpu_arch.pdf

다음 글에서는 아래 두 가지 방향 중 하나로 내용을 이어갈 예정입니다.

Gemmini 논문을 분석하여 systolic array 구조를 보다 자세히 살펴보는 방향

해당 논문을 참고해 실제 PE Block을 구현해보는 방향

### 화면 구성
|화면 명|
|:---:|
|<img src="https://user-images.githubusercontent.com/80824750/208456048-acbf44a8-cd71-4132-b35a-500047adbe1c.gif" width="450"/>|
|화면에 대한 설명을 입력합니다.|


|화면 명|
|:---:|
|<img src="https://user-images.githubusercontent.com/80824750/208456234-fb5fe434-aa65-4d7a-b955-89098d5bbe0b.gif" width="450"/>|
|화면에 대한 설명을 입력합니다.|

<br />

## ⚙ 기술 스택
> skills 폴더에 있는 아이콘을 이용할 수 있습니다.
### Front-end
<div>
<img src="https://github.com/yewon-Noh/readme-template/blob/main/skills/JavaScript.png?raw=true" width="80">
<img src="https://github.com/yewon-Noh/readme-template/blob/main/skills/React.png?raw=true" width="80">
<img src="https://github.com/yewon-Noh/readme-template/blob/main/skills/JWT.png?raw=true" width="80">
</div>

### Infra
<div>
<img src="https://github.com/yewon-Noh/readme-template/blob/main/skills/AWSEC2.png?raw=true" width="80">
</div>

### Tools
<div>
<img src="https://github.com/yewon-Noh/readme-template/blob/main/skills/Github.png?raw=true" width="80">
<img src="https://github.com/yewon-Noh/readme-template/blob/main/skills/Notion.png?raw=true" width="80">
</div>

<br />

## 🤔 기술적 이슈와 해결 과정
- CORS 이슈
    - [Axios message: 'Network Error'(CORS 오류)](https://leeseong010.tistory.com/117)
- api 호출 시 중복되는 헤더 작업 간소화하기
    - [axios interceptor 적용하기](https://leeseong010.tistory.com/133)
- axios 요청하기
    - [axios delete 요청 시 body에 data 넣는 방법](https://leeseong010.tistory.com/111)

<br />

## 💁‍♂️ 프로젝트 팀원
|Backend|Frontend|
|:---:|:---:|
| ![](https://github.com/yewon-Noh.png?size=120) | ![](https://github.com/SeongHo-C.png?size=120) |
|[노예원](https://github.com/yewon-Noh)|[이성호](https://github.com/SeongHo-C)|
