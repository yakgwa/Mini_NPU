앞서 Reference와 Proposed 구조에 대해 10,000개의 sample을 대상으로 inference를 수행하고 결과를 비교했습니다.

이번 글에는 PPA (Power, Performance, Area) 관점에서 두 구조를 비교하기 위해 Synthesis와 Implementation을 수행하고, 그 결과를 기반으로 구조적 차이와 개선 여부를 분석했습니다.

## Reference Design: Synthesis / Implementation Results

### .xdc 생성

Implementation에 필요한 Design Constraint를 정의하기 위해 XDC 파일을 작성합니다.

이 과정에서는 FPGA의 physical pin과 top-level port 간의 연결 관계를 설정할 뿐 아니라, I/O standard, clock, timing constraint 등 구현에 필요한 제약 조건도 함께 정의합니다.

​현재 보유하고 있는 FPGA인 Zybo Z7-20의 Spec에 맞춰 .xdc (Xilinx Design Constraints) 파일을 작성했습니다.

    ## =========================================================
    ## ZYBO Z7-20 constraints for design_1_wrapper external ports
    ## =========================================================
    
    ## 1) I/O standard
    set_property IOSTANDARD LVCMOS33 [get_ports {axis_in_data_0[*]}]
    set_property IOSTANDARD LVCMOS33 [get_ports axis_in_data_valid_0]
    set_property IOSTANDARD LVCMOS33 [get_ports axis_in_data_ready_0]
    set_property IOSTANDARD LVCMOS33 [get_ports intr_0]
    
    ## 2) axis_in_data_0[7:0] -> Pmod JA[7:0]
    ## Pmod JA pins (Digilent Zybo Z7-20 Master XDC)
    set_property PACKAGE_PIN N15 [get_ports {axis_in_data_0[0]}]
    set_property PACKAGE_PIN L14 [get_ports {axis_in_data_0[1]}]
    set_property PACKAGE_PIN K16 [get_ports {axis_in_data_0[2]}]
    set_property PACKAGE_PIN K14 [get_ports {axis_in_data_0[3]}]
    set_property PACKAGE_PIN N16 [get_ports {axis_in_data_0[4]}]
    set_property PACKAGE_PIN L15 [get_ports {axis_in_data_0[5]}]
    set_property PACKAGE_PIN J16 [get_ports {axis_in_data_0[6]}]
    set_property PACKAGE_PIN J14 [get_ports {axis_in_data_0[7]}]
    
    ## 3) valid -> SW0, ready -> LED1, intr -> LED0
    ## SW0=G15, LED0=M14, LED1=M15
    set_property PACKAGE_PIN G15 [get_ports axis_in_data_valid_0]   ;# SW0
    set_property PACKAGE_PIN M15 [get_ports axis_in_data_ready_0]   ;# LED1
    set_property PACKAGE_PIN M14 [get_ports intr_0]                 ;# LED0
    
Zybo Z7-20 보드의 Pmod JA 및 on-board I/O(SW, LED)를 기준으로 AXI-Stream 입력 신호와 동작 상태 확인용 신호를 FPGA pin에 할당했습니다. 

​이와 함께, 외부 인터페이스에 사용되는 모든 포트에 LVCMOS33 I/O standard를 적용하여 보드 환경에 맞는 electrical constraint를 설정했습니다.

### Block Design HDL Wrapper 생성

Create Block Design에서 Zynq Processing System IP를 추가한 뒤, 동작 주파수를 100 MHz로 설정합니다.

이후 Top Module인 zyNet을 Add Module로 추가하고, Run Connection Automation을 수행한 뒤 필요한 pin을 수동으로 연결하면 다음과 같은 Block Design이 구성됩니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_119.png" width="400"/>

Reference Block Design 

<div align="left">

다음으로 IP 간 인터페이스 연결 및 설정에 문제가 없는지 확인하기 위해 Validate Design을 수행합니다.

​검증이 정상적으로 완료되면 Create HDL Wrapper를 통해 Top-level HDL Wrapper를 생성합니다.

### Synthesis / Implementation

앞서 생성한 HDL Wrapper 기반 Block Design에 대해 Synthesis와 Implementation을 진행했습니다.

Synthesis는 정상적으로 완료되었지만, 다음과 같은 Critical Warning이 발생했습니다.

    [PSU-1]  Parameter : PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_0 has negative value -0.050 . PS DDR interfaces might fail when entering negative DQS skew values. 
    
해당 경고는 Zynq PS 설정 과정에서 DDR Controller의 DQS-to-CLK skew 값이 음수로 계산되었음을 의미합니다.

​Board Preset 기반 설정에서는 DDR timing 계산 과정에서 이 값이 소량의 음수로 계산되는 경우가 있으며, Zynq PS의 DDR Controller는 초기화 단계에서 timing calibration(read/write leveling) 을 수행하므로 

이러한 skew 값은 대부분 runtime에서 자동으로 보정됩니다. 

​따라서 본 warning은 DDR timing 계산 과정에서 발생하는 안내 수준의 메시지로, 일반적인 보드 preset 환경에서는 실제 동작에 큰 영향을 주지 않는 경우가 많습니다.

​Synthesis 이후 Implementation을 수행하면 설계된 로직이 FPGA 디바이스 내부 자원에 실제로 배치됩니다.

​아래 그림은 Implementation 완료 후 FPGA 디바이스 상의 placement 결과를 나타낸 것이며, RTL 로직이 PL 영역의 CLB(Configurable Logic Block) 에 배치된 것을 확인할 수 있습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_120.png" width="400"/>

<div align="left">


## Proposed Design: Synthesis / Implementation Results (임시 조치)

### .xdc 생성

앞서 Reference에서 XDC를 작성한 것과 동일하게, 이번에는 NPU_Top의 port 구성에 맞춰 Zynq Z7-20의 spec을 반영한 .xdc (Xilinx Design Constraints) 파일을 작성했습니다.

    ## =========================================================
    ## Zybo Z7-20 XDC for PPA-oriented check
    ## Target ports:
    ##   i_start_inference_0
    ##   i_input_valid_0
    ##   i_input_pixels_0[31:0]
    ##   o_done_interrupt_0
    ## =========================================================
    
    ## =========================================================
    ## 1) I/O Standard
    ## =========================================================
    set_property IOSTANDARD LVCMOS33 [get_ports i_start_inference_0]
    set_property IOSTANDARD LVCMOS33 [get_ports i_input_valid_0]
    set_property IOSTANDARD LVCMOS33 [get_ports {i_input_pixels_0[*]}]
    set_property IOSTANDARD LVCMOS33 [get_ports o_done_interrupt_0]
    
    ## =========================================================
    ## 2) 32-bit input mapping
    ## JA + JB + JC + JD = 32 GPIO
    ## Based on Digilent Zybo Z7 Master XDC
    ## =========================================================
    
    ## JA[7:0] -> i_input_pixels_0[7:0]
    set_property PACKAGE_PIN N15 [get_ports {i_input_pixels_0[0]}]
    set_property PACKAGE_PIN L14 [get_ports {i_input_pixels_0[1]}]
    set_property PACKAGE_PIN K16 [get_ports {i_input_pixels_0[2]}]
    set_property PACKAGE_PIN K14 [get_ports {i_input_pixels_0[3]}]
    set_property PACKAGE_PIN N16 [get_ports {i_input_pixels_0[4]}]
    set_property PACKAGE_PIN L15 [get_ports {i_input_pixels_0[5]}]
    set_property PACKAGE_PIN J16 [get_ports {i_input_pixels_0[6]}]
    set_property PACKAGE_PIN J14 [get_ports {i_input_pixels_0[7]}]
    
    ## JB[7:0] -> i_input_pixels_0[15:8]
    set_property PACKAGE_PIN V8  [get_ports {i_input_pixels_0[8]}]
    set_property PACKAGE_PIN W8  [get_ports {i_input_pixels_0[9]}]
    set_property PACKAGE_PIN U7  [get_ports {i_input_pixels_0[10]}]
    set_property PACKAGE_PIN V7  [get_ports {i_input_pixels_0[11]}]
    set_property PACKAGE_PIN Y7  [get_ports {i_input_pixels_0[12]}]
    set_property PACKAGE_PIN Y6  [get_ports {i_input_pixels_0[13]}]
    set_property PACKAGE_PIN V6  [get_ports {i_input_pixels_0[14]}]
    set_property PACKAGE_PIN W6  [get_ports {i_input_pixels_0[15]}]
    
    ## JC[7:0] -> i_input_pixels_0[23:16]
    set_property PACKAGE_PIN V15 [get_ports {i_input_pixels_0[16]}]
    set_property PACKAGE_PIN W15 [get_ports {i_input_pixels_0[17]}]
    set_property PACKAGE_PIN T11 [get_ports {i_input_pixels_0[18]}]
    set_property PACKAGE_PIN T10 [get_ports {i_input_pixels_0[19]}]
    set_property PACKAGE_PIN W14 [get_ports {i_input_pixels_0[20]}]
    set_property PACKAGE_PIN Y14 [get_ports {i_input_pixels_0[21]}]
    set_property PACKAGE_PIN T12 [get_ports {i_input_pixels_0[22]}]
    set_property PACKAGE_PIN U12 [get_ports {i_input_pixels_0[23]}]
    
    ## JD[7:0] -> i_input_pixels_0[31:24]
    set_property PACKAGE_PIN T14 [get_ports {i_input_pixels_0[24]}]
    set_property PACKAGE_PIN T15 [get_ports {i_input_pixels_0[25]}]
    set_property PACKAGE_PIN P14 [get_ports {i_input_pixels_0[26]}]
    set_property PACKAGE_PIN R14 [get_ports {i_input_pixels_0[27]}]
    set_property PACKAGE_PIN U14 [get_ports {i_input_pixels_0[28]}]
    set_property PACKAGE_PIN U15 [get_ports {i_input_pixels_0[29]}]
    set_property PACKAGE_PIN V17 [get_ports {i_input_pixels_0[30]}]
    set_property PACKAGE_PIN V18 [get_ports {i_input_pixels_0[31]}]
    
    ## =========================================================
    ## 3) Control / status
    ## =========================================================
    
    ## Switches
    set_property PACKAGE_PIN G15 [get_ports i_start_inference_0] ;# SW0
    set_property PACKAGE_PIN P15 [get_ports i_input_valid_0]     ;# SW1
    
    ## LED
    set_property PACKAGE_PIN M14 [get_ports o_done_interrupt_0]  ;# LED0

### Block Design HDL Wrapper 생성

Reference와 동일하게 Zynq Processing System IP를 추가한 뒤, NPU_Top을 Add Module로 포함하고 Run Connection Automation을 수행했습니다. 

​이후 필요한 pin은 수동으로 연결하여 Block Design을 구성했습니다. 다만 NPU_Top 모듈에는 AXI 관련 인터페이스가 정의되어 있지 않기 때문에, AXI 관련 IP는 제외하고 다음과 같이 Block Design을 구성했습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_121.png" width="400"/>

<div align="left">

NPU_Top이 .sv 형식(SystemVerilog)이었기 때문에 Add Module 단계에서 hide 처리되어 Block Design에서 인식되지 않는 문제가 있었습니다.

​이를 해결하기 위해 파일을 .v 형식으로 변환하고, SystemVerilog 문법인 '0을 [dataWidth-1:0] 형태로 수정했습니다.

​이후 Processor System Reset IP를 추가했으며, 입력 및 출력 신호는 External Port로 선언하여 Block Design에서 사용할 수 있도록 구성했습니다.

### Synthesis / Implementation

    초기 Synthesis / Implementation 결과, Implementation 과정에서 Error 및 Critical Warning이 발생했습니다.
    
    [Place 30-415] IO placement is infeasible. The design requires more IO pins than are available on the device.
    
    [DRC UCIO-1] Unconstrained Logical Port
    Port 'o_result_class_0_0[31:0]' has no user assigned pin location.
    
    [DRC NSTD-1] Unspecified I/O Standard
    
원인을 확인한 결과, Top-level I/O 개수가 Zybo Z7-20 보드의 최대 I/O 수(125개)를 초과한 것이었습니다.

​특히 result_class_* 출력 신호가 32-bit × 4개(총 128-bit)로 선언되어 있어, 전체 I/O 수가 163개까지 증가하면서 Implementation 단계에서 I/O 자원 부족으로 인한 오류가 발생했습니다.

​최종적으로는 AXI 인터페이스를 추가하여 PS와 PL 간 데이터 전달을 처리하는 구조로 확장할 예정이지만, 현재 설계에서는 AXI 인터페이스가 포함되어 있지 않아 결과 데이터를 외부 I/O로 직접 출력하도록 구성되어 있었습니다.

​

​

(기재 예정)
