## Preliminarie

본격적인 Reference Model 분석에 앞서, 이후 내용을 이해하는 데 필요한 기본 개념들을 간단히 정리합니다.

​Neural Network를 공부하다 보면 DNN, CNN, RNN, MLP와 같은 model architecture 수준의 개념부터, MatMul, GEMM, Systolic Array와 같은 computation 및 hardware 관점의 개념까지, 서로 다른 성격과 level of abstraction을 가진 용어들이 동시에 등장합니다.

​이들 모두가 Neural Network와 관련된 개념이다 보니, 어떤 용어가 computation을 의미하는지, 어떤 용어가 model architecture를 의미하는지, 어떤 용어가 hardware implementation과 직접적으로 연결되는지 처음 접하는 입장에서는 이를 구분하기가 쉽지 않습니다.

이에 따라 본격적인 분석에 앞서, 관련 개념들을 role과 level of abstraction 관점에서 정리했습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_17.png" width="400"/>

​

<div align="left">
- Model / Algorithm Level (Neural Network architecture) : 
  가장 상위 레벨에서는 Neural Network가 어떤 구조로 구성되어 있는지를 정의합니다. 이 레벨의 핵심은 layer의 종류, connection 방식, 그리고 data flow​이며, 구체적인 computation 구현 방식이나 hardware 구현 세부 사항은 다루지 않습니다.

​- DNN (Deep Neural Network) :
  여러 개의 hidden layer를 가지는 Neural Network를 포괄적으로 지칭하는 개념입니다. 특정한 구조를 의미하기보다는, 네트워크의 depth(depth-wise complexity)를 강조하는 상위 분류(umbrella term)에 가깝습니다.

​- MLP (Multi-Layer Perceptron) : 
  multiple fully connected layer로 구성된 가장 기본적인 DNN architecture입니다. 내부 연산은 MatMul + Bias Add + Activation 형태로 직접 표현되며, Neural Network 연산이 MatMul 기반으로 구현됨을 가장 직관적으로 보여주는 예시입니다.

​- CNN (Convolutional Neural Network) : 
  convolution operation을 중심으로 구성된 Neural Network architecture입니다. 개념적으로는 convolution 연산을 사용하지만, 실제 실행 단계에서는 image-to-column 등의 변환을 통해 MatMul 형태로 변환되어 계산됩니다.

- RNN (Recurrent Neural Network) : 
  sequence data 처리를 위해 recurrent connection을 가지는 Neural Network architecture입니다. 시간 축 방향으로 state가 반복적으로 전달되며, 내부 계산은 time-step 단위의 반복적인 MatMul과 accumulation으로 구성됩니다. 이 레벨에서는 사용되는 computation의 종류보다는, Neural Network의 구조와 구성 방식에 초점을 둡니다. 

- Mathematical / Operation Level (Mathematical computation) :
  다음 레벨에서는 Neural Network 내부에서 수행되는 계산을 수학적 연산 관점에서 바라봅니다.

​- MatMul (Matrix Multiplication) : 
  MatMul은 행렬 곱 연산 그 자체를 의미하는 가장 근본적인 computation unit입니다. Fully Connected, Convolution, Attention과 같은 연산들은 모두 MatMul 형태로 환원될 수 있습니다. 이 레벨에서는 hardware 구조나 execution 방식보다는, “무엇을 계산하는가”에 대한 수학적 정의가 핵심입니다.

- Algorithm / Interface Level (Execution-oriented representation) :
  수학적 정의만으로는 컴퓨터에서 효율적인 실행이 어렵기 때문에, 이를 실행 관점에서 정형화한 표현이 필요합니다.

- GEMM (General Matrix Multiplication) : 
  GEMM은 MatMul과 accumulation을 포함하는 표준 연산 형태로, 일반적으로 C = α A × B + β C 형식으로 정의됩니다. GEMM은 CPU의 BLAS library, GPU kernel, TPU/NPU accelerator 등에서 공통적으로 사용되는 standard computation interface입니다.

​“Neural Network 연산을 GEMM으로 변환한다” 는 표현은 MatMul을 실제 hardware에서 효율적으로 실행 가능한 형태로 재구성한다는 의미로 이해하면 됩니다.

 - Hardware Architecture Level (Hardware implementation) : 
  가장 하위 레벨에서는 GEMM과 같은 연산을 hardware에서 어떻게 빠르고 효율적으로 수행할 것인가를 다룹니다. 이 레벨의 핵심은 연산의 수학적 정의가 아니라, data movement, parallelism, utilization입니다.

​- Systolic Array : 
Systolic Array는 다수의 Processing Element (PE)를 2D array 형태로 배치한 hardware 구조입니다.Input data와 weight가 인접한 PE로 cycle 단위로 전달되며, 각 PE는 multiply–accumulate (MAC) operation을 매 cycle 수행합니다.

​이 구조의 목적은 data와 weight를 PE 내부 및 인접 PE 간에 전달함으로써 memory access를 최소화하고, 동일한 data를 여러 PE에서 반복 활용하여 data reuse를 극대화하는 데 있습니다. 

​또한 pipeline 기반으로 연산을 수행함으로써 매 cycle마다 연산 결과를 지속적으로 출력할 수 있으며, 이를 통해 전체 throughput을 향상시킵니다.

​결과적으로 Systolic Array는 GEMM을 고효율로 실행하기 위한 대표적인 hardware architecture입니다.
