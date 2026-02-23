## Systolic Array - Dataflow 개념 정리 & Array 확장

### 기존 Systolic Array의 이론적 연산

다음은 Output Stationary 방식을 기준으로 직접 연산 과정을 정리한 그림입니다.

- 2 x 2 Systolic Array

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_46.jpg" width="400"/>

<div align="left">

- 4 x 4 Systolic Array

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_47.jpg" width="400"/>

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_58.jpg" width="400"/>

<div align="left">

### Output Stationary (OS) 구조의 핵심

이 구조의 가장 큰 특징은 이름 그대로 "최종 결과값(Output)이 생성되는 위치가 고정된다"라는 점입니다.

​- 고정된 데이터 (Stationary):
 - Partial Sum (C): PE 내부 레지스터에 고정되어, 연산이 완료될 때까지 계속 누적됩니다.

- 이동하는 데이터 (Moving):
 - Input (A): 좌측 → 우측으로 이동
 - Weight (B): 상단 → 하단으로 이동

- 동기화 (Synchronization) 
    정확한 연산 타이밍을 맞추기 위해 입력 데이터($A, B$)는 대각선 형태의 Wavefront을 그리며 진입합니다. 이를 위해 각 행/열마다 Shift Delay를 주어 데이터가 순차적으로 PE에 도달하도록 제어합니다.

### Weight Stationary (WS) 구조의 핵심
반면, Google TPU 등이 채택한 WS 방식은 "가중치(Weight)를 PE 안에 고정한다"라는 점에서 OS 방식과 정반대의 흐름을 가집니다.

​- 고정된 데이터 (Stationary):
 - Weight (B): 연산 시작 전, 각 PE 레지스터에 미리 로드(Pre-load) 되어 고정되고, 연산이 끝날 때까지 밖으로 이동하지 않습니다.

- 이동하는 데이터 (Moving):
 - Input (A): 좌측 → 우측으로 이동하며 각 PE의 Weight와 연산됩니다.
 - Partial Sum (C): PE에 머무르지 않고 상단 → 하단으로 흐르며 값이 누적됩니다. (OS와의 가장 큰 차이점)

이를 시각적으로 나타내면 다음과 같습니다. C = A × W 연산 시, Input과 Weight가 만나 생성되는 결과 행렬 C의 구조입니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_48.png" width="400"/>

<div align="left">

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_49.png" width="400"/>

<div align="left">

### OS

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_50.png" width="400"/>

<div align="left">

### WS

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_51.png" width="400"/>

<div align="left">

###실제 연산 값 적용 및 확장_(1): 1×4 Weight-Stationary Systolic Array

Reference Model의 입력은 784개의 픽셀 데이터(숫자 이미지 1장)로 이루어진 벡터입니다.

​이 데이터를 처리하는 기존 Reference Model의 FC Layer 구조를 살펴보면, Layer 1은 30개, Layer 2는 20개의 출력 뉴런을 가집니다. 

이 중 Layer 1의 연산을 행렬식으로 표현하면 아래와 같습니다. 이제 실제 Systolic Array 구조로 구현해 보겠습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_52.png" width="400"/>

<div align="left">​

1 × 4 또는 4 × 4 Systolic Array를 그대로 적용할 경우, Array 크기 제약으로 인해 모든 데이터를 한 번에 처리하기는 어렵습니다. 따라서 입력 데이터와 Weight를 tile 단위로 분해​하여, 이를 순차적으로 계산하는 방식으로 접근합니다.

​먼저, 1 × 4 Systolic Array 구조를 기준으로 tile 1이 적용된 경우를 살펴보겠습니다. 해당 구조에서의 데이터 배치와 연산 흐름은 아래 그림과 같습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_53.png" width="400"/>

1 x 4 Systolic Array 예상 구현

<div align="left">​

위 그림에서 각 PE에는 tile 1에 해당하는 weight의 일부 구간만이 고정되어 저장됩니다. 구체적으로, Tile 1에서는 각 PE가 서로 다른 weight index w_{*,0}, w_{*,1}, w_{*,2}, w_{*,3} 중 하나의 column을 담당하며,

해당 column에 속한 784개의 weight 값이 시간축에 따라 순차적으로 적용됩니다. 즉, PE 내부에서는 tile 단위로 선택된 weight column이 고정되고 cycle마다 입력 a_i가 전달됨에 따라 동일한 PE에서 w_{i, j} × a_i 연산이 반복 수행됩니다.

​이때 “weight가 고정된다”는 것은 tile 1이 처리되는 동안 PE가 담당하는 weight column이 변하지 않는다는 의미이며, 각 cycle마다 서로 다른 weight 값이 사용되더라도, 그 값들은 모두 동일한 weight column에 속한 값들입니다.

<mark>** 1×4 Array에서 확인해야 할 사항</mark>

Q1. Input과 Weight의 구성

    Batch Size를 1로 가정하면, 입력 데이터는 1 × 784 크기의 입력 벡터 1개로 구성되며, 입력 요소 a_0, a_1, ~ a_783은 매 cycle마다 serial 하게 입력됩니다. Weight Matrix은 784 × 30이며 (Output Feature = 30, Input Dimension = 784), 1 × 4 Array 구조에서는 한 번에 4개의 Output Feature만 처리할 수 있습니다.

​    따라서 전체 30개의 Output Feature를 계산하기 위해 Weight Matrix를 Column 단위로 분할하여 총 8회의 Tile 연산을 순차적으로 수행합니다.

​Q2. 1 × 4 Array 구조는 WS 인가, OS 인가?

    Input과 Weight가 매 cycle마다 PE에 공급되는 형태만 보면 Output Stationary(OS) 구조로 오해하기 쉽습니다. 그러나 아키텍처 관점에서는 Weight Stationary(WS)로 분류하는 것이 타당합니다.

    그 이유는 다음과 같습니다. 각 PE는 Weight Matrix의 특정 column을 전담하며 해당 Weight는 PE 간에 전달되거나 이동하지 않습니다. Input 데이터만이 Systolic 하게 PE를 따라 흐르며 연산을 수행합니다.
​
    Batch Size = 1 환경에서는 cycle마다 다른 weight 값이 사용되기 때문에 Weight가 이동하는 것처럼 보일 수 있습니다. 그러나 이는 Weight가 흐르는 것이 아니라, 동일한 PE 위치에서 연산 순서에 따라 값이 갱신되는 것입니다. 즉, Weight의 공간적 위치가 고정되어 있다는 점에서 본 구조의 본질은 WS에 해당합니다.

​Q3. Tile 교체 시점과 Pipeline을 활용한 효율 극대화
    마지막 입력 a_783이 주입된 이후에도, Systolic Array의 전파 특성으로 인해 PE 위치에 따라 연산 종료 시점에는 차이가 발생합니다.

​

모든 PE의 연산이 끝난 뒤 다음 Tile을 로드할 경우, 일부 PE는 동작하지 않고 대기하게 되어 파이프라인 효율이 저하됩니다.

​

이를 방지하기 위해, 각 PE가 자신의 연산을 완료하는 즉시 다음 Tile에 대한 Weight를 로드하고 연산을 시작하는

Wavefront 기반 Tile 교체 방식을 적용합니다.

​

하드웨어적으로는 Double Buffering(또는 Shadow Register) 구조를 사용하여 

현재 Tile 연산과 다음 Tile의 Weight 로딩을 중첩 수행함으로써, Cycle 손실 없는 연속 처리가 가능합니다.

실제 연산 값 적용 및 확장_(2): 4×4 Weight-Stationary Systolic Array

앞서 1 × 4 Systolic Array에서의 연산 흐름을 살펴보았으니, 이제 이를 4 × 4 Systolic Array로 확장해 보겠습니다.

4 × 4 구조로 확장하면서는 batch 크기를 10으로 늘려, 여러 입력 샘플을 동시에 처리하는 형태로 구성합니다.


​

4 × 4 Systolic Array에서의 mapping은 다음과 같습니다.


Column(열) 방향으로는 출력 뉴런 4개를 병렬로 처리하며, 하나의 tile은 출력 4개 column에 해당합니다.

Row(행) 방향으로는 batch 샘플을 최대 4개까지 동시에 처리합니다.

​

따라서 4 × 4 Array는 한 번에 4개 샘플 × 4개 출력에 대한 연산 결과를 생성합니다.

​

Batch가 10인 경우, 한 번에 처리할 수 있는 row 수를 초과하므로 입력 샘플을 3개의 group으로 나누어 순차적으로 처리합니다.

또한 출력 뉴런은 총 30개이기 때문에, column 4개씩을 묶어 총 8개의 tile로 분할합니다.

​

이와 같이 4 × 4 Systolic Array에서는 row 방향으로 batch를, column 방향으로 출력 뉴런을 병렬화하고,

tile과 group 단위로 연산을 반복 수행하게 됩니다.

​

다만 WS 방식에서는 row 방향의 의미가 앞서 언급한 batch 병렬성과는 다릅니다.​

​

4개의 row는 서로 다른 샘플을 처리하는 것이 아니라, 입력 벡터 내에서 서로 다른 feature(row index)를 담당합니다.

즉, 4 × 4 Systolic Array는 한 번에 입력 채널 4개 (k ∼ k+3)와 출력 채널 4개 (j ∼ j+3)의 교차 영역에 대한 연산을 수행합니다.

​

Input Mapping: Row = Input Feature

Row마다 서로 다른 Feature를 할당합니다. 

샘플 10개는 한꺼번에 들어가지 않고, 마치 컨베이어 벨트처럼 시간 순서대로 하나씩 연속해서(Streaming) Array로 주입됩니다.

Row 0: Sample_{0 ~ 9}의 Input[k] Stream

Row 1: Sample_{0 ~ 9}의 Input[k + 1] Stream

Row 2: Sample_{0 ~ 9}의 Input[k + 2] Stream

Row 3: Sample_{0 ~ 9}의 Input[k + 3] Stream

​

Diagonal Data Setup (Systolic Schedule)

올바른 누적 연산을 위해 입력 데이터는 대각선 형태(Diagonal)로 시간차를 두고 주입됩니다. (Skewing)

Cycle t: Row 0에 Sample_0 진입

Cycle t+1: Row 1에 Sample_0 진입 (Row 0은 Sample_1)

Cycle t+2: Row 2에 Sample_0 진입 ...

이렇게 하면 Sample_0에 대한 Wavefront이 대각선으로 Array를 훑고 지나가게 됩니다.

​

입력 전파 (Horizontal Shift)

각 PE는 a_reg를 가지며, 입력 값은 매 cycle마다 우측으로 전달됩니다.

a_reg <= left_neighbor_a_reg
이 흐름을 통해 하나의 입력 데이터가 동일 Row에 있는 4개의 Output Column PE에 차례대로 재사용됩니다.

​

Weight는 어떻게 Pre-load 되는가

WS에서 “Weight가 고정된다"라는 의미는 "Batch(Sample)이 모두 지나갈 때까지 레지스터 값이 변하지 않음"을 뜻합니다.

​

4 × 4 Array의 각 PE는 Weight Matrix의 4 × 4 Array Sub-Block을 담당합니다.

현재 처리 중인 입력 인덱스가 k ~ k+3이고, 출력 인덱스가 j ~ j+3이라면

PE(r, c)는 W[k+r][j+c] 값을 내부 레지스터에 저장(Latch)합니다.

이 값은 Batch 10개가 모두 처리되는 동안(최소 10 Cycle + Latency) 절대 변하지 않고 고정됩니다.

입력이 파이프라인을 타고 계속 흐르는 동안, PE는 고정된 가중치와 곱셈을 수행합니다.

​

Weight는 Tile 교체 시점에만 Weight Buffer에서 PE로 로드됩니다.

​

psum 경로: Vertical Accumulation (수직 누적)

WS 구조에서 psum은 위에서 아래로 흐릅니다.

각 PE는 자신의 곱셈 결과를 위에서 내려온 부분합에 더해서 아래로 전달합니다.

​

각 PE는 매 사이클 다음 연산을 수행합니다.


psum_{in}: 위쪽 PE (row_{r-1})에서 내려온 값

psum_{out}: 아래쪽 PE (row_{r+1})로 전달할 값 (최하단 Row는 Array 외부로 출력)

​

이 구조에서는 서로 다른 Feature(k ~ k+3)들의 곱이 수직으로 합쳐집니다.

​

Sample_0가 최상단 Row 0을 지나 Row 3을 빠져나올 때,


위와 같이 4개 채널에 대한 부분합(Partial Sum)이 완성되어 나옵니다.

​

Tile 교체: Double Buffering 및 Psum 관리

WS 방식에서는 K(Input Feature 784) 차원을 따라 Loop를 돌아야 하므로, psum의 관리가 중요합니다.

​

Weight Double Buffer

Batch 10개를 처리하는 동안 Weight는 고정되지만, 

다음 4개 채널(k+4 ~ k+7)을 처리하려면 새로운 Weight가 필요합니다.

Active Bank: 현재 연산 중인 W_{tile} 공급

Shadow Bank: 연산 도중 다음 W_{next_tile}을 미리 로드 (Pre-fetch)

Batch 10개가 Array를 모두 통과하면 Bank Swap을 수행하여 즉시 다음 연산을 시작합니다.

​

Psum External Accumulation (외부 누적)

Array 내부에서는 4개 채널에 대한 합만 계산되므로, 

전체 784개 채널에 대한 합은 Array 외부의 Accumulator가 담당합니다.

Array 하단 출력: 4개 채널의 부분합

외부 동작: Global_Psum_Buffer[sample_id] += Array_Output

이렇게 K=0 ~ 783 loop가 모두 끝나면 최종 결과가 완성됩니다.

 

Batch=10 전체 처리 Schedule

전체 연산은 다음 3중 loop로 정리됩니다. (loop 순서가 데이터 재사용을 결정합니다)

​

1. Outer Loop (Output Tile): 출력 Feature 30개를 4개씩 분할 (총 8회)

Target Output Column j ~ j+3 설정

​

2. Middle Loop (Input Feature Tile): 입력 Feature 784개를 4개씩 분할 (총 196회)

해당하는 4 × 4 Weight Tile 로드 (Pre-load)

​

3. Inner Loop (Batch Stream): Sample 0 ~ 9 (총 10회)

10개의 Sample을 연속으로 주입 (Streaming)

PE는 고정된 Weight로 10번의 연산 수행

결과값(psum)은 하단으로 배출되어 외부 Buffer에 누적

​

실행 흐름 예시:

Tile Setup: W[0..3][0..3] 로드

Streaming: Sample 0~9가 차례대로 Row 0~3을 대각선으로 통과

Accumulation: 하단으로 나온 결과(Sample 0~9의 부분합)를 메모리에 저장

Swap: Weight W[4..7][0..3] 으로 교체 (Double Buffer)

Repeat: Sample 0~9 다시 주입하여 이전 결과에 누적 ...

마무리

이번 글에서는 Systolic Array를 적용하는 과정에서

Output-Stationary(OS) 와 Weight-Stationary(WS) 를 헷갈리기 쉬운 부분을 중심으로 개념을 한 번 정리해 보았습니다.

​

이를 바탕으로 1-D 1×4 Weight-Stationary Systolic Array의 연산 흐름을 먼저 살펴보고,

이 구조를 4×4 Systolic Array로 확장했을 때 dataflow가 어떻게 변하는지도 함께 정리했습니다.

​

다음 글에서는 Verilog(또는 SystemVerilog)를 사용해 이제 실제로 RTL 설계 단계로 넘어가 보려 합니다.

​

구체적으로는 다음과 같은 내용을 순서대로 다룰 예정입니다.

PE(Processing Element) 구조 설계

1-D PE Chain (1×4 Array)

2-D Systolic Array (4×4 Array)

Controller 구조 (Systolic Data Skewing 제어)

Activation Function

​

이 과정을 통해, 이번 글에서 정리한 dataflow와 연산 모델이 RTL 수준에서 어떤 형태로 구현되는지 차근차근 이어가 보겠습니다.
