## Archteture Review

Neural Network의 연산 흐름을 이해하기 위해, DNN accelerator 연구용 오픈소스 프로젝트인 Gemmini를 살펴봅니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_39.png" width="400"/>

Gemmini​

<div align="left">

### Gemmini

Gemmini는 DNN accelerator를 독립적인 연산 블록으로 보지 않고, system과 software stack의 일부로 다룹니다. 

이를 통해 accelerator와 system, software 간의 상호작용이 DNN 성능에 미치는 영향을 분석할 수 있도록 설계된 플랫폼입니다

​Gemmini는 Chisel HDL을 사용해 개발되었습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_40.png" width="100"/>

<div align="left">

### Architecture

Gemmini는 RoCC (Rocket Custom Coprocessor) accelerator로 구현되며,

CPU가 실행하는 RISC-V custom instruction을 통해 직접 제어됩니다.

​

해당 instruction은 표준 RISC-V ISA에 포함되지 않은 사용자 정의 명령으로, 

CPU 프로그램 흐름 내에서 Gemmini의 동작을 트리거하는 역할을 합니다.

​

Gemmini는 Rocket Chip 또는 BOOM tile의 RoCC port에 연결되며, 이를 통해 CPU와 가속기 간 명령 및 상태 전달이 이루어집니다.

또한 별도의 전용 메모리를 사용하지 않고, System Bus를 통해 CPU의 memory system에 접근하여 L2 cache와 연동됩니다.

​

즉, Gemmini는 독립적인 연산 IP가 아니라, CPU가 명령어로 제어하며 동일한 메모리 시스템을 공유하는 가속기로 설계되었습니다.

Accelerator의 핵심에는 matrix multiplication을 수행하는 systolic array가 위치합니다. 

​

Gemmini는 기본적으로 output-stationary (OS)와 ​weight-stationary (WS) dataflow를 모두 지원하며,

programmer는 이를 runtime에 선택하거나 elaboration time에 고정(hardening)하여 사용할 수 있습니다.

​

Systolic array의 입력과 출력은 banked SRAM으로 구성된 scratchpad에 저장되며,

host CPU가 접근하는 main memory와 scratchpad 간의 데이터 이동은 DMA engine이 담당합니다.

** Banked SRAM은 하나의 큰 메모리를 사용하는 대신, SRAM을 여러 개의 독립적인 bank로 나누어 구성한 메모리 구조입니다.

** DMA (Direct Memory Access) engine은 CPU 개입 없이 메모리 간 데이터 이동을 전담하는 하드웨어 블록입니다.

​

Weight-stationary dataflow에서는 systolic array 외부에 accumulator가 필요하므로, 

adder units가 포함된 additional SRAM bank가 추가됩니다. 

​

systolic array는 accumulator의 임의 주소로 결과를 저장하거나 새로운 input을 읽을 수 있습니다. 

​

또한 해당 accumulator SRAM은 scratchpad와 유사하게 사용되며, 

bias 로딩 등을 위해 main memory와 DMA를 통한 직접 데이터 전송을 지원합니다.

​

마지막으로 Gemmini는 다음을 포함합니다.

activation function(ReLU, ReLU6) 적용

power-of-2 scaling을 통한 quantized workload 지원

→ 

정수 기반 연산에서 스케일링을 곱셈이 아닌 bit shift 연산으로 처리할 수 있도록 지원하여, 

quantized DNN 연산을 효율적으로 수행합니다.

output-stationary dataflow를 위한 matrix transpose 기능을 수행하는 peripheral circuitry

→ 

output-stationary dataflow에서 요구되는 데이터 배치를 맞추기 위해, 

행렬을 전치(transpose)하는 전용 주변 회로를 포함하여 systolic array 입력 형태를 하드웨어에서 직접 변환합니다.

​

​

Generator Parameters

Gemmini에서 중요하게 고려되는 주요 parameters는 다음과 같습니다.

​

Systolic array dimensions (tileRows, tileColumns, meshRows, meshColumns)

Systolic array는 2-level hierarchy로 구성됩니다. 

각 tile은 fully combinational 구조이며, tile들로 구성된 mesh 사이에는 pipeline registers가 삽입됩니다. ​

​

이러한 구조를 통해 tile 내부 연산과 tile 간 데이터 전달이 분리됩니다.


Dataflow parameters (dataflow)

Gemmini의 systolic array가 output-stationary(OS), weight-stationary(WS) 중 어떤 dataflow를 사용할지 결정합니다. 

또한 두 dataflow를 모두 지원하도록 설정할 경우, programmer가 runtime에 dataflow를 선택할 수 있습니다.

​

Scratchpad and accumulator memory parameters (sp_banks, sp_capacity, acc_capacity)​

Gemmini scratchpad memory와 accumulator memory의 특성을 결정합니다. 

전체 memory capacity(KiB 단위)와 scratchpad가 몇 개의 bank로 분할되는지가 이 parameters로 정의됩니다.

​

Type parameters (inputType, outputType, accType)

Gemmini 내부를 흐르는 data-type을 결정합니다. 

​

예를 들어 inputType은 8-bit fixed-point일 수 있으며, 

matrix multiplication 중 partial accumulation을 담당하는 accType은 32-bit integer로 설정될 수 있습니다.

​

outputType은 processing element(PE) 간에 전달되는 데이터 타입을 정의하며, 

예를 들어 8-bit multiplication 결과가 16-bit로 확장되어 PE 간에 전달될 수 있습니다.

​

사용 가능한 datatype 예시는 다음과 같습니다.

SInt(8.W) : signed 8-bit integer

UInt(32.W) : unsigned 32-bit integer

Float(8, 24) : single-precision IEEE floating-point

​

Floating-point datatype을 사용하는 경우, pe_latency parameter​를 함께 조정해야 할 수 있습니다. 

이는 PE 내부에 삽입되는 shift register의 개수를 지정하며, 

multiply-accumulate 연산이 single cycle 내에 완료되지 않는 경우 필요합니다.

​

Access-execute queue parameters 

(ld_queue_length, st_queue_length, ex_queue_length, rob_entries)

Gemmini는 access-execute decoupling을 위해 load, store, execute instruction queue를 각각 분리하여 구현합니다. 

​

각 queue의 크기는 access와 execute 간 decoupling 수준을 결정합니다.

또한 reorder buffer(ROB) 를 포함하며, ROB entry 수는 dependency management에 영향을 미칩니다.

​

DMA parameters (dma_maxbytes, dma_buswidth, mem_pipeline)

Gemmini는 DMA engine을 통해 main memory와 scratchpad, accumulator 간의 data transfer를 수행합니다. 

​

DMA transaction 크기는 해당 parameters로 결정됩니다.

이 parameters는 Rocket Chip SoC system parameters와 밀접하게 연관되며, 

예를 들어 dma_buswidth는 SystemBusKey beatBytes, dma_maxbytes는 cacheblockbytes parameter와 연결됩니다.

​

Gemmini는 elaboration-time에 선택적으로 포함할 수 있는 optional features도 제공합니다.

​

Scaling during “move-in” operations (mvin_scale_args, mvin_scale_acc_args)​

DRAM 또는 main memory에서 scratchpad로 데이터를 이동시키는 mvin 과정에서, 데이터에 scaling factor를 적용할 수 있습니다. 

​

해당 parameters는 scaling factor의 datatype과 scaling 방식를 정의합니다.

이 parameters가 None으로 설정되면, 해당 기능은 elaboration-time에 비활성화됩니다.

​

Scratchpad input과 accumulator input에 동일한 scaling을 적용하는 경우, 

mvin_scale_shared를 true로 설정하여 multiplier와 functional unit을 공유할 수 있습니다.

Major Components​

Decoupled Access / Execute Architecture

Gemmini는 decoupled access/execute architecture를 채택하고 있으며, 

이는 memory-access와 execute instruction이 hardware 내 서로 다른 영역에서 concurrently 수행됨을 의미합니다.

​

이를 위해 Gemmini는 hardware를 크게 세 개의 controller로 분리합니다.

ExecuteController

Matrix multiplication과 같은 execute-type ISA command를 수행합니다. 

내부에는 systolic array와 transposer가 포함됩니다.

​

LoadController

Main memory에서 Gemmini의 private scratchpad 또는 accumulator로 데이터를 이동하는 모든 load instruction을 담당합니다.

​

StoreController

Gemmini의 private SRAM에서 main memory로 데이터를 이동하는 store instruction을 담당합니다. 

또한 Gemmini는 pooling을 memory write 과정에서 수행하므로, max-pooling instruction 역시 이 controller에서 처리됩니다.

​

세 controller는 programmer로부터 직접 전달된 ISA command를 decode하고 실행하며, scratchpad 및 accumulator SRAM을 공유합니다.

​

Scratchpad and Accumulator

Gemmini는 systolic array의 input과 output을 private SRAMs에 저장하며, 이를 각각 scratchpad와 accumulator라고 부릅니다.

​

일반적으로 input은 scratchpad에 저장되고, partial sum 및 final result는 accumulator에 저장됩니다.

​

Scratchpad와 accumulator는 모두 Scratchpad.scala에서 instantiate되며,

scratchpad bank는 ScratchpadBank

accumulator bank는 AccumulatorMem

모듈로 구현됩니다.

​

각 scratchpad/accumulator SRAM의 row는 DIM elements로 구성되며, DIM은 systolic array 가로 방향의 PE 개수를 의미합니다. 

​

각 element는 Gemmini가 처리하는 하나의 scalar value입니다.

scratchpad element type: inputType (default: 8-bit integer)

accumulator element type: accType (default: 32-bit integer)

​

예를 들어 default configuration(16×16 systolic array)에서는

scratchpad row width = 16 × bits(inputType) = 128 bits

accumulator row width = 16 × bits(accType) = 512 bits

입니다.

​

Accumulator는 scratchpad보다 복잡한 구조를 가지며,

in-place accumulation을 위한 adder

scaler

activation function unit

을 포함합니다.

​

Scaling 및 activation은 accType 값을 inputType으로 변환하여 읽어낼 때 적용되며, 

이는 한 layer의 partial sum을 다음 layer의 low-bitwidth input으로 변환하기 위해 사용됩니다.

​

Systolic Array and Transposer

ExecuteController 내부의 MeshWithDelays 모듈은

systolic array (Mesh)

transposer

input alignment를 위한 delay registers 를 포함합니다.

​

MeshWithDelays는 매 cycle마다 A, B, D matrix의 row를 입력받아,

C = A × B + D 를 row 단위로 출력합니다.

​

Weight-stationary mode:

B는 preload되고, A와 D가 streaming됩니다.

Output-stationary mode:

D가 preload되고, A와 B가 streaming됩니다.

​

A, B, D는 모두 inputType이며, C는 outputType입니다.

C를 scratchpad에 저장하면 inputType으로 cast되고, accumulator에 저장하면 accType으로 cast됩니다.

​

Weight-stationary mode에서는 inputType D가 partial sum을 표현하기에 bitwidth가 부족하므로, 

일반적으로 D는 zero-matrix를 사용하고 partial sum은 accumulator SRAM에서 누적됩니다.

​

Input(A, B, D)은 정확한 cycle에 정확한 PE에 도달하도록 shift-register delay를 거쳐 전달됩니다.


Systolic array 자체는 Mesh.scala에 구현되어 있으며,

Tile + PE로 구성된 two-tier hierarchy 구조입니다.

Tile 간에는 pipeline register가 존재하고,

각 Tile 내부는 combinational PEs로 구성됩니다.


Transposer는 간단한 systolic 구조로 구현되며, 

output-stationary mode에서는 programmer가 transpose를 명시하지 않더라도 항상 사용됩니다. 

​

이는 scratchpad row layout과 systolic array input 요구사항 간의 mismatch를 해결하기 위함입니다.


​

DMA

Gemmini는 두 개의 DMA를 포함합니다.

main memory → private SRAM

private SRAM → main memory

​

DMA는 virtual address를 사용하며, 

TLB를 통해 physical address로 변환됩니다. TLB miss 시에는 host CPU와 공유하는 PTW를 사용합니다.

​

DMA는 large memory request를 여러 개의 TileLink transaction으로 분할하며, 

request size는 power-of-2 alignment를 만족해야 합니다.

​

성능 관점에서 Gemmini는 요청 개수를 최소화하는 방향으로 DMA를 설계합니다.

​

DMAWriter는 memory write 과정에서 max-pooling을 수행하기 위한 comparator를 포함합니다.

​

ROB (Reorder Buffer)

Decoupled architecture로 인해 Load/Store/ExecuteController는 서로 out-of-order로 동작할 수 있습니다.

​

Gemmini는 ROB를 통해 controller 간 instruction hazard를 감지합니다.

Instruction은 dependency가 해소된 이후에만 각 controller로 issue됩니다.

​

단, 같은 controller 내부의 instruction은 program-order로 issue되며, 해당 controller가 내부 hazard를 책임집니다.

​

Matmul / Conv Loop Unrollers

Gemmini의 systolic array는 최대 DIM × DIM 크기의 matmul만 직접 처리할 수 있습니다.

​

더 큰 matmul 또는 convolution은 tiling이 필요합니다.

이를 programmer가 직접 처리하는 부담을 줄이기 위해, Gemmini는 CISC-style high-level ISA instruction을 제공하며, 

이를 통해 matmul/conv를 자동으로 tile 및 unroll합니다.

​

이 기능은 LoopMatmul, LoopConv 모듈로 구현됩니다.

해당 모듈은 FSM 기반으로 동작하며,

double-buffering

ROB 상태 모니터링

을 통해 memory access와 compute 간 overlap을 극대화합니다.

Memory Addressing Scheme

Gemmini의 private memory는 row-addressed 방식으로 구성됩니다.

​

각 row는 DIM elements로 이루어지며, DIM은 systolic array 가로 방향의 PE 개수를 의미합니다(기본 설정에서는 16).

Scratchpad의 element type은 inputType입니다.

Accumulator의 element type은 accType입니다.

​

모든 Gemmini private memory address는 32-bit이며, 상위 bit에는 다음과 같은 의미가 부여됩니다.

Bit 31 (MSB)

0: scratchpad addressing

1: accumulator addressing

​

Bit 30: accumulator write 시에만 의미를 가집니다.

0: overwrite

1: accumulate (기존 값에 누적)

​

Bit 29: accumulator read 시에만 의미를 가집니다.

0: inputType으로 scaled-down 된 값 read

1: accType 값 그대로 read

​

Bit 29가 1인 경우에는 activation function 및 scaling이 적용되지 않습니다.


Gemmini는 main memory 접근 시 software-visible virtual address를 사용하며, 

physical address translation은 Gemmini 내부에서 TLB를 통해 programmer에게 투명하게 처리됩니다.

Core Matmul Sequences

Gemmini에서 하나의 matrix multiplication은 두 단계로 수행됩니다.

matmul.preload

matmul.compute

​

이는 systolic array에 유지되어야 하는 데이터(B 또는 D)를 명확히 구분하기 위한 구조입니다.

Output-stationary(OS)

D matrix를 preload

partial sum은 systolic array 내부에 유지

​

Weight-stationary(WS)

B matrix를 preload

partial sum은 accumulator SRAM에 저장

​

이 preload/compute 분리는 ISA 길이 제한 때문이기도 하지만, 

본질적으로는 systolic array state 유지를 위한 hardware 모델을 반영한 것입니다.

​

또한, compute 단계에서는

previously preloaded data를 재사용하거나

이전 결과 위에 누적(accumulated compute)

하는 방식이 지원됩니다.
