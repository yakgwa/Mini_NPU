앞서 Reference와 Proposed 구조에 대해 10,000개의 sample을 대상으로 inference를 수행하고 결과를 비교했습니다.

이번 글에는 PPA (Power, Performance, Area) 관점에서 두 구조를 비교하기 위해 Synthesis와 Implementation을 수행하고, 그 결과를 기반으로 구조적 차이와 개선 여부를 분석했습니다.

## Reference Design: Synthesis / Implementation Results

### Block Design 구성

Vivado에서 Zynq Processing System IP를 추가한 뒤 동작 주파수를 100MHz로 설정합니다. 이후 Top Module인 zyNet을 Add Module로 추가하고, AXI Interconnect 및 Processor System Reset IP를 연결하여 Block Design을 구성하였습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_124.png" width="400"/>

Reference Block Design

<div align="left">

IP 간 연결 검증을 위해 Validate Design을 수행하고, 완료 후 Create HDL Wrapper를 통해 Top-level HDL Wrapper를 생성합니다.

### .xdc 생성

Implementation에 필요한 Design Constraint를 정의하기 위해 XDC 파일을 작성합니다.

이 과정에서는 FPGA의 physical pin과 top-level port 간의 연결 관계를 설정할 뿐 아니라, I/O standard, clock, timing constraint 등 구현에 필요한 제약 조건도 함께 정의합니다.

​현재 보유하고 있는 FPGA인 Zybo Z7-20의 Spec에 맞춰 .xdc (Xilinx Design Constraints) 파일을 작성했습니다.(Zybo Z7-20 Pmod JA 핀 기준으로 XDC를 작성)

    ## IOSTANDARD
    set_property IOSTANDARD LVCMOS33 [get_ports {axis_in_data_0[*]}]
    set_property IOSTANDARD LVCMOS33 [get_ports axis_in_data_valid_0]
    set_property IOSTANDARD LVCMOS33 [get_ports axis_in_data_ready_0]
    set_property IOSTANDARD LVCMOS33 [get_ports intr_0]
    
    ## Pin 할당 — axis_in_data_0[7:0] → Pmod JA
    set_property PACKAGE_PIN N15 [get_ports {axis_in_data_0[0]}]
    set_property PACKAGE_PIN L14 [get_ports {axis_in_data_0[1]}]
    set_property PACKAGE_PIN K16 [get_ports {axis_in_data_0[2]}]
    set_property PACKAGE_PIN K14 [get_ports {axis_in_data_0[3]}]
    set_property PACKAGE_PIN N16 [get_ports {axis_in_data_0[4]}]
    set_property PACKAGE_PIN L15 [get_ports {axis_in_data_0[5]}]
    set_property PACKAGE_PIN J16 [get_ports {axis_in_data_0[6]}]
    set_property PACKAGE_PIN J14 [get_ports {axis_in_data_0[7]}]
    
    ## axis_in_data_valid_0 → SW0 / axis_in_data_ready_0 → LED1 / intr_0 → LED0
    set_property PACKAGE_PIN G15 [get_ports axis_in_data_valid_0]
    set_property PACKAGE_PIN M15 [get_ports axis_in_data_ready_0]
    set_property PACKAGE_PIN M14 [get_ports intr_0]
    
    ## Input/Output delay
    set_input_delay  -clock [get_clocks *] -max 2.0 [get_ports {axis_in_data_0[*]}]
    set_input_delay  -clock [get_clocks *] -max 2.0 [get_ports axis_in_data_valid_0]
    set_output_delay -clock [get_clocks *] -max 2.0 [get_ports axis_in_data_ready_0]
    set_output_delay -clock [get_clocks *] -max 2.0 [get_ports intr_0]
    
### Synthesis 결과

※ Synthesis 과정에서 다음 Critical Warning이 발생하였습니다.

    [PSU-1] Parameter : PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_0 has negative value -0.050


Zynq PS Board Preset 기반 설정에서 DDR DQS-to-CLK skew 값이 음수로 계산될 때 발생하는 경고입니다. 
PS DDR Controller가 초기화 단계에서 timing calibration을 수행하므로 실제 동작에 영향을 주지 않습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_125.png" width="400"/>

REF Post-Synthesis Utilization Graph

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_126.png" width="400"/>

REF Post-Synthesis Utilization Table

<div align="left">


※ REF Post-Synthesis에서 LUT/FF/BRAM이 표시되지 않고 IO만 나타나는 현상이 있었습니다. 이는 Block Design 기반 구조에서 zyNet 내부가 Wrapper 계층에 가려져 Synthesis 단계 utilization에 반영되지 않는 경우로, Implementation 결과에서 정확한 수치를 확인할 수 있습니다.

### Implementation 결과

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_127.png" width="400"/>

REF Post-Implementation Utilization Graph

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_128.png" width="400"/>

REF Post-Implementation Utilization Table

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_129.png" width="400"/>

REF Implementation Device View

<div align="left">





