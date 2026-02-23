## Proposed Model

이전에는 Reference Model의 debug 과정에서 단일 input sample(test_data_0000, class 7)에 대한 output을 도출하였으며, 이를 Golden value로 정의하였습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_83.png" width="400"/>

<div align="left">

이번에는 Proposed Model의 initial 코드를 simulation한 뒤 발생한 error를 수정하고, 동일 input에 대해 Golden value와 동일한 output이 나오도록 구성합니다.

### Revision History

    Proposed Model 분석 과정 (Update Log)
    test_data_0000 기준으로 Reference Model에서 결과값 추출 → Golden value 정의 (Done)
    
    Proposed Model 1차 Simulation (Done)
    Layer 1 결과 불일치, Layer 2/3 output = X(Unknown)
    조치: PE 내 unsigned → signed 치환 후 2차 진행
    
    Proposed Model 2차 Simulation (Done)
    Layer 1 불일치 지속, Layer 2/3 X(Unknown) 유지
    조치: waveform만으로 한계 → TB debug/monitor 추가 후 3차 진행
    
    Proposed Model 3차 Simulation (Done)
    TB debug로 MAC/acc 동작 확인(PE(0,0) 기준 MAC 동작 자체는 정상)
    k_cnt 기준 monitor에서 출력 일부 누락 이슈 확인(추가 flush 필요)
    조치: Bias 처리 이슈 확인 후 4차 진행
    
    Proposed Model 4차 Simulation (Done)
    조치: Bias scaling을 Golden과 동일하게 수정(<<< dataWidth)
    결과: Layer 1 PASS, Layer 2/3 X(Unknown) 유지
    
    Proposed Model 5차 Simulation (Done)
    조치: TB debug를 Layer 2/3까지 확장
    결과: Layer 2/3에서 weight 첫 값이 X로 출력되는 현상 확인
    조치: Global_Buffer 경로/저장 방식 점검
    
    Proposed Model 6차 Simulation (Done)
    조치: Layer 2 결과를 Global_Buffer 상위 영역(+128)에 저장하도록 수정 → layer 간 overwrite 회피
    결과: Layer 3 output 생성은 확인되나, L2/L3 X(Unknown) 잔존
    추가 조치: address assign이 특정 state에서만 갱신되던 부분 수정
    
    Proposed Model 7차 Simulation (Done)
    조치: Weight_Bank(1-cycle latency)와 input path timing mismatch 해결
    Global_Buffer read data를 buf_r_data_d1로 1-cycle delay하여 정렬
    TB debug monitor도 1-cycle latency 반영
    결과: Layer 2 Golden match, Layer 3 output 생성 확인(최종 결과는 불일치)
    
    Proposed Model 8차 Simulation (Done / PASS)
    조치: overflow 처리 수정
    Activation_Unit: truncation → saturation 적용
    PE accumulator: 동일하게 saturation 적용
    
    결과: 최종 PASS

## Proposed Model Dataflow & Block Diagram

해당 Model은 다음과 같은 module들로 구성되어 있습니다.

    Simulation Environment
    └── TB_NPU_Top.sv
        │   * 역할: NPU_Top을 DUT로 instantiate하고,
        │           sample input을 적용하여 end-to-end simulation을 수행합니다.
        │
        ├── include.v
        │       * 역할: network 및 layer 관련 parameter를 정의합니다.
        │               (dimension, neuron count, bit-width 등)
        │
        ├── DUT : NPU_Top.sv
        │       * 역할: 전체 NPU의 top-level module로, datapath 및 control을 통합합니다.
        │               
        │
        │   ├── Systolic_Array.sv
        │   │       * 역할: MAC 연산을 수행하는 systolic array datapath입니다.
        │   │    │
        │   │    └──── PE.sv
        │   │       * 역할: systolic array를 구성하는 processing element(MAC unit)입니다.
        │   │
        │   ├── Global_Buffer.sv
        │   │       * 역할: input 및 Layer 1 / 2에서의 output을 저장합니다.
        │   │
        │   ├── Weight_Memory.sv
        │   │       * 역할: weight storage 및 read interface를 제공합니다.
        │   │
        │   ├── Weight_Bank.sv
        │   │       * 역할: layer별 weight bank를 구성하고 관리합니다.
        │   │
        │   ├── Bias_Bank.sv
        │   │       * 역할: neuron별 bias 값을 저장하고 제공합니다.
        │   │
        │   ├── Activation_Unit.sv
        │   │       * 역할: MAC 결과에 대한 activation 처리를 수행합니다.
        │   │
        │   │    ├── ReLU.sv
        │   │    │       * 역할: ReLU activation function을 구현합니다.
        │   │    │
        │   │    └── Sig_ROM.sv
        │   │            * 역할: sigmoid activation을 위한 ROM 기반 lookup table입니다.
        │   │    
        │   └── maxFinder.sv
        │           * 역할: 최종 output vector 중 maximum value의 index(0~9)를 추출하여
        │                   classification 결과를 생성합니다.
        │
        └── Sample_Data
                * 역할: simulation에 사용되는 sample data를 포함합니다.
                        bias, weight, sigmoid lookup value 등이 저장됩니다.

Dataflow 관점에서 Block Diagram을 구성하면, 전체 구조를 그림과 같이 표현할 수 있습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_85.png" width="400"/>

<div align="left">
                       
1) Input stream

test_data_0000.txt는 img_mem_0[0:784]에 load되며, index 0~783은 pixel data, index 784는 label(expected value)로 사용됩니다.

​CALC_L1 상태에서는  DUT 내부 k_cnt를 address로 사용하여, 매 cycle마다 input pixel 4개를 packing하여 compute block에 입력합니다.

​이를 통해 여러 input pixel을 동시에 처리할 수 있는 병렬 연산이 가능하도록 구성되어 있습니다.(단, 해당 구조는 systolic array가 아니라 broadcast-based spatial parallel MAC array 구조입니다.)

​2) Weight fetch: Weight_Memory → Weight_Bank (1-cycle latency)

Weight_Bank는 layer별로 다수의 Weight_Memory instance를 생성하며, w_addr(= k_cnt)에 해당하는 weight를 읽어옵니다.

​Weight_Memory의 read 동작은 always @(posedge clk) 블록에서 수행되며, wout <= mem[radd] 형태로 출력이 register에 저장되므로, weight read는 1-cycle latency를 가집니다.

​Weight_Bank는 row 당 PE 개수 기준(본 Model에서는 4×4 구조)에 맞추어, 4개의 neuron에 대응하는 weight를 선택하고 이를 w_out_packed 형태로 묶어 출력합니다.

​이를 통해 연산 block으로 공급되는 weight가 병렬로 입력되도록 정렬합니다.

→ 1번과 2번에서 input과 weight를 packed 형태로 묶는 이유는, 한 cycle에 동시에 처리되는 병렬 연산의 단위를 RTL interface 수준에서 정의하기 위함입니다.

​이를 통해 각 lane이 동일한 Timing으로 동작하도록 보장할 수 있으며, 설계 확장 시 병렬도 변경이 용이하고, Synthesis 및 Place & Route 단계에서도 안정적인 품질을 확보할 수 있습니다.

3) Compute

Systolic_Array는 4×4 PE를 병렬로 instantiate한 구조이며, 각 PE는 입력 a와 b의 곱(mul)을 누산(psum)하는 MAC 연산을 수행합니다.

​각 PE는 내부 accumulator sum을 가지며, en이 1일 때마다 sum <= sum + mul 연산을 수행하도록 설계되어 있습니다.

​기존 구현에서는 overflow 발생 시 상위 비트가 잘리면서 wrap-around가 발생하는 구조였지만,  해당 부분을 수정하여 현재는 saturation 로직을 적용하였습니다. 

​이에 따라 overflow가 발생하면 표현 가능한 signed 범위의 최대값 또는 최소값으로 clamp되도록 처리됩니다.

​4) Bias + Activation: Bias_Bank + Activation_Unit

Bias_Bank는 bias 값을 .mif 파일에서 읽어오며, 선택된 4개의 bias를 packing하여 출력합니다. 이때 bias는 psum의 Q-format에 맞추기 위해  left shift를 적용하여 fractional bit를 정렬합니다. 
(논리적으로 {bias, 8'b0}와 동일)

    <<< dataWidth
    
Activation_Unit에서는 다음 순서로 연산이 수행됩니다.
- psum_in + bias_in 연산 수행
- overflow 방지를 위한 saturating add 적용
- act_sel에 따라 Sig_ROM 또는 ReLU activation 선택

​ReLU는 posedge clk 기반의 register output을 가지며, Sig_ROM 역시 address가 posedge clk에 갱신됩니다. 따라서 activation 경로는 전체적으로 1-cycle latency를 갖는 구조입니다.

​5) Global_Buffer 저장 및 Layer 간 Buffering

각 layer의 연산은 계산 단계(CALC_Lx)와 저장 단계(BUFFER_WR_Lx)로 구분됩니다.

​CALC_Lx (Computation Phase)
- input과 weight를 이용해 MAC 연산을 수행합니다.
- 누산(accumulation)을 통해 각 neuron의 output을 생성합니다.
- Activation function이 적용되며, 해당 block의 pipeline delay를 포함한 activation latency가 존재합니다.

​BUFFER_WR_Lx (Write Phase)
- Activation 결과를 Global_Buffer에 순차적으로 write합니다.
- Activation latency를 고려하여 write addressoutput data가 정확히 time-aligned 되도록 제어합니다.
- Data valid timing과 address increment timing이 일치하도록 설계되어 있습니다.

Global_Buffer 동작 특성
- 각 layer의 intermediate result를 저장하는 buffer입니다.
- Write는 sequential 방식으로 수행됩니다.
- Read는 combinational path로 구성되어 address 입력에 대해 즉시 data가 출력됩니다.
  
요약하면, 연산 단계(CALC_Lx)와 저장 단계(BUFFER_WR_Lx)를 분리하고, activation latency를 address 제어 로직에서 보정하여 write alignment를 보장하는 구조입니다.

​각 layer의 output은 Global_Buffer에 저장된 후 다음 layer의 input으로 재사용되며, layer 간 데이터 경계는 memory를 기준으로 분리됩니다. 

​이에 따라 multi-layer 연산은 layer 단위로 순차 수행됩니다.

​7) Output scan 및 classification

모든 layer의 연산이 완료되면 최종 10개의 output 값이 maxFinder로 전달됩니다. maxFinder는 각 output 값을 비교하여 maximum value의 index를 추출하고, 이를 최종 classification 결과로 결정합니다.

도출된 index는 입력과 함께 제공된 expected value(label)와 비교되며, 이를 통해 전체 연산 결과의 정합 여부를 검증합니다.

### Proposed Model - 1st Simulation

initial code에 대해 Simulation을 수행하면 연산 결과는 그림과 같이 출력됩니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_86.png" width="400"/>

<div align="left">

현재 Layer 2와 Layer 3의 output이 X(Unknown) 상태로 나타나며, Layer 1의 연산 결과 또한 Golden value와 일치하지 않음을 확인할 수 있습니다.

이에 따라 Systolic_Array 구현 사례와 동일하게, unsigned로 선언된 부분을 signed로 수정한 뒤 simulation을 재진행하였습니다.

### Proposed Model - 2nd Simulation

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_87.png" width="400"/>

<div align="left">

1차 시뮬레이션 이후, unsigned로 선언되어 있던 신호를 signed로 수정하여 재시뮬레이션을 수행하였으나, Layer 1의 출력값은 여전히 Golden value와 불일치하였고, Layer 2 및 Layer 3의 출력은 X(Unknown) 상태로 관측되었습니다.

​초기에는 waveform을 통해 실제 연산이 정상적으로 수행되는지를 확인하고자 하였으나, Pipe-line 및 다수의 내부 신호로 인해 waveform 기반 분석에는 한계가 있다고 판단하였습니다. 

​이에 따라 Testbench에 debug 및 monitor logic을 추가하여 각 단계의 연산 결과를 출력·검증하는 방식으로 분석을 진행하였습니다. (아래에 제시된 코드는 최종 Debug Monitor Logic입니다.)

### TB_NPU_Top.sv

            ///////////////////////////////////////////////////////////////////
            // --- Debug Monitor: L1/L2/L3 k-by-k + latency snapshots
            //
            // DUT 모델:
            //  - L1: raw k=1..cur_input_len (TB는 원상복귀되어 k_cnt 그대로 투입)
            //  - L2/L3: buf_r_data_d1(1-cycle) 때문에 유효 MAC은 raw k=2..cur_input_len+1
            //
            // 출력:
            //  - L2/L3 MAC 로그는 k를 사람이 보기 쉽게 1..cur_input_len로 표시(k_eff=raw-1)
            //  - raw k도 같이 표시
            //  - flush 스냅샷은 1회만 출력
            ///////////////////////////////////////////////////////////////////
        
            localparam int FLUSH_CYC = 2;  // LAT1=1, LAT2=2 (TB 관측용)
        
            // DUT state encoding
            localparam int CALC_L1 = 1;
            localparam int CALC_L2 = 3;
            localparam int CALC_L3 = 5;
        
            // ---- internal regs (L1) ----
            reg     l1_debug_active;
            integer l1_debug_cnt;
            reg     l1_wait_final;
            integer l1_flush_cnt;
        
            // ---- internal regs (L2) ----
            reg     l2_debug_active;
            integer l2_debug_cnt;
            reg     l2_wait_final;
            integer l2_flush_cnt;
        
            // ---- internal regs (L3) ----
            reg     l3_debug_active;
            integer l3_debug_cnt;
            reg     l3_wait_final;
            integer l3_flush_cnt;
        
            always @(posedge clk) begin
                if (!rst_n) begin
                    l1_debug_active <= 0; l1_debug_cnt <= 0; l1_wait_final <= 0; l1_flush_cnt <= 0;
                    l2_debug_active <= 0; l2_debug_cnt <= 0; l2_wait_final <= 0; l2_flush_cnt <= 0;
                    l3_debug_active <= 0; l3_debug_cnt <= 0; l3_wait_final <= 0; l3_flush_cnt <= 0;
                end else begin
        
                    // ============================================================
                    // L1 (CALC_L1 = 1)
                    // ============================================================
                    if (prev_state != CALC_L1 && dut.state == CALC_L1 && dut.group_cnt == 0) begin
                        l1_debug_active <= 1'b1;
                        l1_debug_cnt    <= 0;
                        l1_wait_final   <= 1'b0;
                        l1_flush_cnt    <= 0;
        
                        $display("\n============================================================");
                        $display("[TB_DEBUG] Start L1 stream debug (grp0). Printing k=1..%0d", dut.cur_input_len);
                        $display("============================================================");
                    end
        
                    if (prev_state == CALC_L1 && dut.state != CALC_L1) begin
                        if (l1_debug_active) begin
                            $display("============================================================");
                            $display("[TB_DEBUG] End L1 MAC debug. Printed %0d lines", l1_debug_cnt);
                            $display("============================================================\n");
                        end
                        l1_debug_active <= 1'b0;
                    end
        
                    if (l1_debug_active && dut.state == CALC_L1 && dut.group_cnt == 0 &&
                        dut.k_cnt >= 1 && dut.k_cnt <= dut.cur_input_len) begin
        
                        l1_debug_cnt <= l1_debug_cnt + 1;
        
                        $display("[L1_MAC] t=%0t k=%0d/%0d | in0=%0d w0=%0d | mul00=%0d acc00=%0d",
                            $time, dut.k_cnt, dut.cur_input_len,
                            $signed(dut.sys_row_in[`dataWidth-1:0]),
                            $signed(dut.sys_col_in[`dataWidth-1:0]),
                            $signed(dut.debug_mul_00),
                            $signed(dut.pe_res_unpacked[0][0])
                        );
        
                        if (dut.k_cnt == dut.cur_input_len && !l1_wait_final) begin
                            l1_wait_final <= 1'b1;
                            l1_flush_cnt  <= 0;
                        end
                    end
        
                    if (l1_wait_final) begin
                        l1_flush_cnt <= l1_flush_cnt + 1;
        
                        if (l1_flush_cnt == (FLUSH_CYC-1)) begin
                            $display("[L1_ACC_AFTER_LAT1] t=%0t | k_end=%0d +1 | acc00=%0d",
                                $time, dut.cur_input_len,
                                $signed(dut.pe_res_unpacked[0][0])
                            );
                        end
        
                        if (l1_flush_cnt == FLUSH_CYC) begin
                            $display("[L1_AU_AFTER_LAT2 ] t=%0t | k_end=%0d +2 | psum_sel=%0d bias_sel=%0d | sum=%0d sat=%0d | act=%0d | acc00=%0d | sel(img=%0d neu=%0d)",
                                $time, dut.cur_input_len,
                                $signed(dut.au_in_psum),
                                $signed(dut.au_in_bias),
                                $signed(dut.au.bias_add_res),
                                $signed(dut.au.sum_saturated),
                                $signed(dut.act_out_val),
                                $signed(dut.pe_res_unpacked[0][0]),
                                dut.in_img_idx, dut.in_neu_idx
                            );
                            l1_wait_final <= 1'b0;
                        end
                    end
                    // ============================================================
                    // L2 (CALC_L2 = 3) : 유효 MAC raw k=2..cur_input_len+1, 표시 k_eff=raw-1
                    // ============================================================
                    if (prev_state != CALC_L2 && dut.state == CALC_L2 && dut.group_cnt == 0) begin
                        l2_debug_active <= 1'b1;
                        l2_debug_cnt    <= 0;
                        l2_wait_final   <= 1'b0;
                        l2_flush_cnt    <= 0;
        
                        $display("\n============================================================");
                        $display("[TB_DEBUG] Start L2 stream debug (grp0). Printing k=1..%0d", dut.cur_input_len);
                        $display("============================================================");
                    end
        
                    if (prev_state == CALC_L2 && dut.state != CALC_L2) begin
                        if (l2_debug_active) begin
                            $display("============================================================");
                            $display("[TB_DEBUG] End L2 MAC debug. Printed %0d lines", l2_debug_cnt);
                            $display("============================================================\n");
                        end
                        l2_debug_active <= 1'b0;
                    end
        
                    if (l2_debug_active && dut.state == CALC_L2 && dut.group_cnt == 0 &&
                        dut.k_cnt >= 2 && dut.k_cnt <= (dut.cur_input_len + 1)) begin
        
                        l2_debug_cnt <= l2_debug_cnt + 1;
        
                        $display("[L2_MAC] t=%0t k=%0d/%0d (raw=%0d) | in0=%0d w0=%0d | mul00=%0d acc00=%0d",
                            $time,
                            (dut.k_cnt-1), dut.cur_input_len, dut.k_cnt,
                            $signed(dut.sys_row_in[`dataWidth-1:0]),
                            $signed(dut.sys_col_in[`dataWidth-1:0]),
                            $signed(dut.debug_mul_00),
                            $signed(dut.pe_res_unpacked[0][0])
                        );
        
                        if (dut.k_cnt == (dut.cur_input_len + 1) && !l2_wait_final) begin
                            l2_wait_final <= 1'b1;
                            l2_flush_cnt  <= 0;
                        end
                    end
        
                    if (l2_wait_final) begin
                        l2_flush_cnt <= l2_flush_cnt + 1;
        
                        if (l2_flush_cnt == (FLUSH_CYC-1)) begin
                            $display("[L2_ACC_AFTER_LAT1] t=%0t | k_end=%0d (raw_end=%0d) +1 | acc00=%0d",
                                $time,
                                dut.cur_input_len,
                                (dut.cur_input_len + 1),
                                $signed(dut.pe_res_unpacked[0][0])
                            );
                        end
        
                        if (l2_flush_cnt == FLUSH_CYC) begin
                            $display("[L2_AU_AFTER_LAT2 ] t=%0t | k_end=%0d (raw_end=%0d) +2 | psum_sel=%0d bias_sel=%0d | sum=%0d sat=%0d | act=%0d | acc00=%0d | sel(img=%0d neu=%0d)",
                                $time,
                                dut.cur_input_len,
                                (dut.cur_input_len + 1),
                                $signed(dut.au_in_psum),
                                $signed(dut.au_in_bias),
                                $signed(dut.au.bias_add_res),
                                $signed(dut.au.sum_saturated),
                                $signed(dut.act_out_val),
                                $signed(dut.pe_res_unpacked[0][0]),
                                dut.in_img_idx, dut.in_neu_idx
                            );
                            l2_wait_final <= 1'b0;
                        end
                    end
        
        
                    // ============================================================
                    // L3 (CALC_L3 = 5) : 유효 MAC raw k=2..cur_input_len+1, 표시 k_eff=raw-1
                    // ============================================================
                    if (prev_state != CALC_L3 && dut.state == CALC_L3 && dut.group_cnt == 0) begin
                        l3_debug_active <= 1'b1;
                        l3_debug_cnt    <= 0;
                        l3_wait_final   <= 1'b0;
                        l3_flush_cnt    <= 0;
        
                        $display("\n============================================================");
                        $display("[TB_DEBUG] Start L3 stream debug (grp0). Printing k=1..%0d", dut.cur_input_len);
                        $display("============================================================");
                    end
        
                    if (prev_state == CALC_L3 && dut.state != CALC_L3) begin
                        if (l3_debug_active) begin
                            $display("============================================================");
                            $display("[TB_DEBUG] End L3 MAC debug. Printed %0d lines", l3_debug_cnt);
                            $display("============================================================\n");
                        end
                        l3_debug_active <= 1'b0;
                    end
        
                    if (l3_debug_active && dut.state == CALC_L3 && dut.group_cnt == 0 &&
                        dut.k_cnt >= 2 && dut.k_cnt <= (dut.cur_input_len + 1)) begin
        
                        l3_debug_cnt <= l3_debug_cnt + 1;
        
                        $display("[L3_MAC] t=%0t k=%0d/%0d (raw=%0d) | in0=%0d w0=%0d | mul00=%0d acc00=%0d",
                
해당 Logic에서는 Layer 1 / 2 / 3에서 MAC 연산 결과, accumulator 누적값, bias add 및 saturation을 거친 activation 출력까지의 각 단계 결과를 순차적으로 모니터링합니다.

​Top-level FSM의 state 값을 기준으로 연산 구간을 Layer 단위로 분리하여 관찰하며, CALC_L1(=1), CALC_L2(=3), CALC_L3(=5) state를 각각 Layer 1, Layer 2, Layer 3의 연산 구간으로 mapping 합니다. 또한 CALC_Lx 상태 진입 시 group_cnt == 0 조건에서만 debug stream을 활성화하여, Layer 단위로 연산 결과를 분리하였습니다.

각 Layer의 연산 구간에서는 k_cnt를 기준으로, input과 weight에 대한 MAC 연산이 k index에 따라 반복 수행되는 과정을 순차적으로 출력합니다.

​Layer 1에서는 k = 1 .. cur_input_len 범위에서, input과 weight를 기준으로 PE(0,0)에서 관측되는 mul00 및 acc00를 출력하여 k index에 따른 MAC 연산 동작을 확인합니다.

​반면 Layer 2와 Layer 3는 내부 buffer(buf_r_data_d1)로 인해 1-cycle latency가 존재하므로, 유효 MAC 구간을 raw k 기준 k = 2 .. cur_input_len + 1로 정의하고, 편의를 위해 k_eff = raw k - 1를 함께 표시하여 출력합니다.

​MAC log에서는 mul00(multiplier output)와 acc00(accumulator output)을 확인하여 multiplication result가 올바른지와 accumulation 과정에서 overflow, incorrect signed/unsigned interpretation, 또는 X propagation이 발생하는지를 k index 단위로 확인합니다.

​또한 MAC 종료 이후에도 내부 pipeline 처리로 인해 최종 output이 즉시 확정되지 않는 점을 고려하여, FLUSH_CYC = 2로 flush observation 구간을 설정하고 추가 snapshot을 출력합니다.

​MAC 종료 후 1 cycle 시점에는 ACC_AFTER_LAT1 snapshot을 통해 accumulator에 최종 반영된 값을 확인하며, 2 cycle 시점에는 AU_AFTER_LAT2 snapshot을 통해 psum_sel, bias_sel, bias add result(sum), saturation result(sat), activation output(act), 그리고 acc00를 함께 출력합니다. 

이때 output target 식별을 위해 img_idx와 neu_idx도 함께 기록합니다. 이를 통해 해당 logic은 MAC result, accumulator accumulation, bias add, saturation, activation output에 이르는 각 processing stage의 결과를 cycle 단위로 확인할 수 있도록 구성되었으며, waveform만으로는 확인하기 어려운 computation mismatch나 X propagation 발생 지점을 보다 세부적으로 분석할 수 있습니다.

### Proposed Model - 3rd Simulation

Debug Monitor logic을 추가한 이후, 검증한 연산 결과는 다음과 같습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_88.png" width="400"/>

<div align="left">

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_89.png" width="400"/>

MAC 연산의 마지막 output 값이 누락되어 있으나, 당시 input과 weight가 모두 0이므로, 전체 결과에는 이상 없음.

<div align="left">

input과 weight가 입력된 이후 multiplication이 수행되고, 1-cycle 뒤에 accumulation이 정상적으로 이루어지는 것을 확인하여, MAC 연산 과정 자체에는 문제가 없음을 확인하였습니다. 

​이후 문제 발생 지점을 분석하는 과정에서, 다른 스터디원의 분석 내용을 참고하여 Bias_Bank 부분에 문제가 있음을 확인하였습니다.

​

### Bias_Bank.sv

                case(layer_idx)
                    1: begin
                        for(k=0; k<4; k=k+1) begin
                            if ((neuron_group*4 + k) < `numNeuronLayer1) sel_b[k] = w_b_l1[neuron_group*4 + k];
                        end
                    end
                    2: begin
                        for(k=0; k<4; k=k+1) begin
                            if ((neuron_group*4 + k) < `numNeuronLayer2) sel_b[k] = w_b_l2[neuron_group*4 + k];
                        end
                    end
                    3: begin
                        for(k=0; k<4; k=k+1) begin
                            if ((neuron_group*4 + k) < `numNeuronLayer3) sel_b[k] = w_b_l3[neuron_group*4 + k];
                        end
                    end
                endcase
            end    
        
            always @(posedge clk) begin
                b_out_packed <= { 
                    {dataWidth{sel_b[3][dataWidth-1]}}, sel_b[3],       // 최상위 bit (MSB)를 dataWidth 만큼 복제하고 뒤에는 기존 sel_b 값 붙이기
                    {dataWidth{sel_b[2][dataWidth-1]}}, sel_b[2],       // 예시로 dataWidth=8인 경우, sel_b[3]이 8'b1111_1010 (-6)라면,
                                                                        // {dataWidth{sel_b[3][dataWidth-1]}}는 8'b1111_1111가 되고, 최종적으로 16'b1111_1111_1111_1010이 됨
                    {dataWidth{sel_b[1][dataWidth-1]}}, sel_b[1],
                    {dataWidth{sel_b[0][dataWidth-1]}}, sel_b[0]
                };
            end

### 개선포인트

1. 기존 neuron_group * 4 연산을 neuron_group << 2로 변경하여 resource 사용을 줄이고, logic depth 감소를 통해 critical path를 완화하며 가독성을 개선하였습니다.

​2. Bias 연산은 multiplication 결과가 누적된 partial sum의 최종 값에 bias를 더하는 방식으로 수행되며, 이 과정에서 psum과의 datawidth를 맞추기 위해 bit extension이 필요합니다. 

​기존 코드에서는 bias 값에 대해 sign extension을 적용하여 bit width를 확장하였으나, 이 방식은 bias의 Q-format을 유지하지 못한다는 문제가 있습니다.

    ex) {dataWidth{sel_b[3][dataWidth-1]}} = {8{1'b1}} = 8'b1111_1111인 경우, sign extension을 적용하면 최종적으로 16'b1111_1111_1111_1010으로 확장됩니다. 

이는 sign bit를 단순 복제하여 정수값 −6을 16-bit signed 정수로 표현한 결과로, bias가 가지는 fractional bit 정보를 전혀 반영하지 않습니다. (fraction bit란 2진수 소수점 이하 자릿수를 의미합니다.) 반면, MAC의 psum은 이미 Q-format(예: Q(16,8))으로 누적된 값이므로, 두 값을 직접 더할 경우 binary point 위치가 서로 달라져 실제 의도한 bias magnitude와 다른 값이 더해지게 됩니다.
이러한 문제를 해결하기 위해 sign extension 대신 left shift를 적용하여 bias에 fractional bit를 추가하도록 수정하였으며, 이를 통해 bias를 MAC의 psum Q-format(Q(16,8))과 정렬시켜 올바른 bias add가 수행되도록 하였습니다.

    ex) $signed(sel_b[3]) <<< 8을 적용할 경우, sel_b[3] = 8'b1111_1010이 signed 값 −6으로 해석된 뒤 8-bit left shift가 적용되어, 결과적으로 −6 << 8 = −1536, 즉 16'b1111_1010_0000_0000으로 변환됩니다.

            always @(*) begin
                // default (latch 방지)
                sel_b[0] = '0;
                sel_b[1] = '0;
                sel_b[2] = '0;
                sel_b[3] = '0;
        
                case (layer_idx)
                    1: begin
                        for (k = 0; k < 4; k = k + 1) begin
                            if (((neuron_group << 2) + k) < `numNeuronLayer1)           // 4의 배수개 + index k가 layer neuron 개수를 넘지 않을 경우,        
                                sel_b[k] = w_b_l1[(neuron_group << 2) + k];             // sell_b에 해당 bias 할당
                        end                                                             // 넘어갈 경우, sel_b는 이미 '0'으로 초기화 되었으므로 0 유지
                    end
                    2: begin
                        for (k = 0; k < 4; k = k + 1) begin
                            if (((neuron_group << 2) + k) < `numNeuronLayer2)
                                sel_b[k] = w_b_l2[(neuron_group << 2) + k];
                        end
                    end
                    3: begin
                        for (k = 0; k < 4; k = k + 1) begin
                            if (((neuron_group << 2) + k) < `numNeuronLayer3)
                                sel_b[k] = w_b_l3[(neuron_group << 2) + k];
                        end
                    end
                    default: begin
                        // keep sel_b = 0
                    end
                endcase
            end
        
            // bias signing extension 및 packing
            // weight와 다르게 bias에서는 dataWidth 확장이 필요하므로, sign extension을 적용하여 packing
            // Original Code : Q(8,0) -> Q(16,0)  (sign-extension only)
            // * Q(total ,fractional bit)로 표기함.
        
            // Modified Code : Q(8,0) -> Q(16,8)
            //  - bias를 dataWidth만큼 left shift하여 fractional bit를 추가, fractional bit란 2진 소수점 이하 자릿수를 의미.
            //  - MAC psum(Q(16,8))과 Q-format을 맞추기 위함
            //  - Golden 모델의 bias 처리({bias, 8'b0})와 bit-level로 동일
        
            reg signed [2*dataWidth-1:0] bias_scaled [0:3];
        
            always_comb begin
            for (int i=0; i<4; i++) begin
                bias_scaled[i] = $signed(sel_b[i]) <<< dataWidth;  // (= {sel_b[i], {dataWidth{1'b0}}})
            end
            end
        
            always_ff @(posedge clk) begin
            b_out_packed <= { bias_scaled[3], bias_scaled[2], bias_scaled[1], bias_scaled[0] };
            end
​
### Proposed Model - 4th Simulation

3차 시뮬레이션 이후, Bias_Bank를 수정한 결과는 다음과 같습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_90.png" width="400"/>

<div align="left">

Layer 1의 연산 결과는 Golden value와 동일함을 확인했으나. 여전히 Layer 2와 Layer 3에서는 X가 출력됨을 확인하였습니다.

==========================================================================================

번외로, 앞서 TB에서 설명한 것과 같이 k_cnt 기준으로 debug monitor를 구성하다 보니, 최종 output 일부가 누락되는 문제가 발생하였습니다.

​아래 waveform을 보면 state가 CALC에서 다음 state로 전이되는 시점에 debug_active가 함께 deassert되며, pipeline latency에 의해 아직 출력되지 않은 값(약 2-cycle 분)이 capture되지 못하고 누락됨을 확인할 수 있습니다.

​이를 해결하기 위해 debug point의 기준을 단순히 k_cnt 구간에 고정하지 않고, latency를 고려한 “유효 연산 구간”과 “flush 구간”을 함께 포함하도록 수정하였습니다.

​구체적으로 state 전이 직후에도 pipeline에 남아 있는 결과를 확인할 수 있도록, 마지막 MAC 이후 FLUSH_CYC만큼 추가로 snapshot을 출력하여 acc(LAT1) 및 Activation_Unit output(LAT2)까지 확인 가능하도록 보강하였습니다.(최종 TB 코드 참고)

​(해당 TB 설계 시에는 simulation 결과를 기반으로 2-cycle flush를 임의로 적용하였으며, 추후 이론적인 latency 분석을 통해 이에 대한 검증이 필요합니다.)

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_91.png" width="400"/>

<div align="left">

### Proposed Model - 5th Simulation

앞서 설계한 TB를 Layer 2와 Layer 3에 대한 debug monitor까지 확장하였습니다. 그 결과, L2와 L3 모두에서 weight의 첫 번째 값이 X로 출력되며, 이로 인해 전체 output에 영향을 미치고 있음을 확인하였습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_92.png" width="400"/>

Layer 2

<div align="left">

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_93.png" width="400"/>

Layer 3

<div align="left">

연산 흐름을 분석한 결과, Layer 2와 Layer 3의 input은 모두 Layer 1의 연산 output임을 확인하였으며, 해당 문제는 각 layer의 연산 결과를 저장하는 Global_Buffer와 관련된 부분에서 발생한 것으로 판단했습니다.

​다시 DUT 내부 FSM을 살펴보면, BUFFER_WR_L1(=2), BUFFER_WR_L2(=4), BUFFER_WR_L3(=6) state는 각 layer의 연산 결과를 Global_Buffer에 write하는 단계입니다.

​

### NPU_Top.sv

                        BUFFER_WR_L1: begin
                            // Layer1 결과를 Global_Buffer에 저장하는 단계
                            // - write_seq_cnt = 1..16 동안, (img 0..3) x (neuron_in_group 0..3) = 16개 결과를 순차 저장
                            // - wr_img_idx / wr_neu_idx는 write_seq_cnt-1 기준으로 계산되어 write 주소를 결정
                            // - wr_global_neuron_idx = group_cnt*4 + wr_neu_idx (현재 neuron group의 전역 neuron index)
                            // - cur_neuron_total을 초과하는 neuron index는 저장 스킵 (마지막 group 처리)
                            // - write_seq_cnt == 16이면:
                            //   · 마지막 group이면 다음 레이어(CALC_L2)로 전이 및 파라미터 갱신
                            //   · 아니면 group_cnt++ 후 다시 CALC_L1로 돌아가 다음 neuron group 계산
                            if (write_seq_cnt > 0 && write_seq_cnt <= 16) begin
                                if (wr_global_neuron_idx < cur_neuron_total) begin 
                                    // WRITE TO BUFFER 1
                                    buf_wen <= 1;
                                    buf_w_addr <= (wr_img_idx * 32) + wr_global_neuron_idx;
                                    buf_w_data <= act_out_val; 
                                end
                            end
        
                            if (write_seq_cnt == 16) begin
                                if ((group_cnt + 1) * 4 >= cur_neuron_total) begin
                                    state <= CALC_L2;
                                    cur_layer_num <= 2;
                                    cur_input_len <= `numNeuronLayer1; 
                                    cur_neuron_total <= `numNeuronLayer2; 
                                    cur_act_sel <= 0; 
                                    group_cnt <= 0;
                                    k_cnt <= 0;
                                    pe_rst <= 1;
                                end else begin
                                    state <= CALC_L1;
                                    group_cnt <= group_cnt + 1;
                                    k_cnt <= 0;
                                    pe_rst <= 1;
                                end
                                write_seq_cnt <= 0;
                            end else begin
                                write_seq_cnt <= write_seq_cnt + 1;
                            end
                        end
                
해당 state에서는 write_seq_cnt를 1..16까지 증가시키며, (img 0..3) × (neuron_in_group 0..3)에 해당하는 총 16개의 activation output (이전 state의 output)을 순차적으로 저장합니다.

​이때 write address는 write_seq_cnt-1를 기준으로 계산된 wr_img_idx와 wr_neu_idx를 사용하여 결정됩니다. wr_neu_idx는 현재 neuron group 내에서의 local index를 의미하며, 이를 wr_global_neuron_idx = group_cnt*4 + wr_neu_idx 형태로 변환함으로써, 전체 layer 기준에서의 global neuron index를 생성합니다.

​즉, group 단위로 나누어 처리된 neuron 결과를 Global_Buffer 상의 연속된 neuron address 공간에 올바르게 mapping하기 위한 계산입니다.

​또한, 마지막 group에서는 

        wr_global_neuron_idx < cur_neuron_total
        
조건을 통해 neuron 개수를 초과하는 항목에 대해서는 write를 수행하지 않도록 처리합니다.​

유효한 neuron index에 대해서만 buf_wen을 assert하며, buf_w_addr 주소에 data를 저장하게 됩니다.

        buf_w_addr = (wr_img_idx * 32) + wr_global_neuron_idx
        buf_w_data = act_out_val
​
즉, 각 sample(img)별로 neuron output이 Global_Buffer 상에서 연속된 address 공간에 저장되도록 구성되어 있습니다.

write_seq_cnt == 16이 되면, 현재 group이 마지막 group인지 여부를 

        (group_cnt + 1) * 4 >= cur_neuron_total

조건으로 판단합니다. 

마지막 group인 경우에는 다음 layer의 계산 state(CALC_L2, CALC_L3 등)로 전이하면서, cur_layer_num, cur_input_len, cur_neuron_total, cur_act_sel을 다음 layer에 맞게 갱신하고, group_cnt와 k_cnt를 초기화한 뒤 pe_rst를 assert하여 다음 연산을 준비합니다.

​반대로 마지막 group이 아닌 경우에는 group_cnt를 증가시킨 후 동일 layer의 계산 state로 복귀하여, 다음 neuron group에 대한 연산을 계속 수행합니다.

각 layer의 연산 결과는 BUFFER_WR_Lx state에서 Global_Buffer의 memory address 공간에 저장됩니다. 초기 구현에서는 Layer 1과 Layer 2가 동일한 address 영역을 사용하도록 되어 있어, layer 간 결과가 overwrite되거나 이후 연산에서 잘못된 값을 read하는 문제가 발생하였습니다. 

​이를 방지하기 위해 BUFFER_WR_L2에서는 Layer 2의 저장 address에 +128 offset을 적용하여, Layer 2 결과를 Global_Buffer의 상위 영역(128~)에 저장하도록 수정하였습니다.

### NPU_Top.sv

                        BUFFER_WR_L2: begin
                            if (write_seq_cnt > 0 && write_seq_cnt <= 16) begin
                                if (wr_global_neuron_idx < cur_neuron_total) begin
                                    // WRITE TO BUFFER 2
                                    buf_wen <= 1;
                                    // Original:
                                    // buf_w_addr <= (wr_img_idx * 32) + wr_global_neuron_idx;
                                    buf_w_addr <= (wr_img_idx * 32) + wr_global_neuron_idx + 128; 
                                    //
                                    // [Debug / Temporary Fix]
                                    // - L2 결과가 기존에 addr=0부터 저장되면서,
                                    //   이후 L3 연산 시 L1/L3 결과 영역과 충돌하여 X가 발생함
                                    // - 이를 분리하기 위해 L2 결과를 Global_Buffer의 상위 영역(128~)에 저장
                                    // - offset(+128)을 적용하여 L2 결과가 L3에서 사용되는 base=0 영역과 겹치지 않도록 함
                                    //
                                    // NOTE:
                                    // - 현재는 디버깅을 위해 offset을 하드코딩한 상태
                                    // - 이후 ping-pong buffering 구조(buf_rd_base / buf_wr_base)로 정리 필요
        
                                    buf_w_data <= act_out_val;
                                end
                            end
        
                            if (write_seq_cnt == 16) begin
                                if ((group_cnt + 1) * 4 >= cur_neuron_total) begin
                                    state <= CALC_L3;
                                    cur_layer_num <= 3;
                                    cur_input_len <= `numNeuronLayer2; 
                                    cur_neuron_total <= `numNeuronLayer3; 
                                    cur_act_sel <= 0; 
                                    group_cnt <= 0;
                                    k_cnt <= 0;
                                    pe_rst <= 1;
                                end else begin
                                    state <= CALC_L2;
                                    group_cnt <= group_cnt + 1;
                                    k_cnt <= 0;
                                    pe_rst <= 1;
                                end
                                write_seq_cnt <= 0;
                            end else begin
                                write_seq_cnt <= write_seq_cnt + 1;
                            end
                        end

또한 Layer 3 연산 시에는 Layer 2 결과가 저장된 영역을 read해야 하므로, CALC_L3 state에서만 read address에 동일한 +128 offset을 적용하도록 분기 처리하였습니다.(현재 offset 적용은 layer 간 결과 충돌을 회피하기 위한 임시 조치이며, 향후에는 Global_Buffer를 read 영역과 write 영역으로 분리한 ping-pong buffering 구조를 도입하여, layer 전이 시 buffer 역할을 교대함으로써 address 충돌을 구조적으로 방지할 계획입니다.)

        assign buf_r_addr[0] = (state == CALC_L3) ? (0 * 32) + k_cnt + 128 : (0 * 32) + k_cnt;
        assign buf_r_addr[1] = (state == CALC_L3) ? (1 * 32) + k_cnt + 128 : (1 * 32) + k_cnt;
        assign buf_r_addr[2] = (state == CALC_L3) ? (2 * 32) + k_cnt + 128 : (2 * 32) + k_cnt;
        assign buf_r_addr[3] = (state == CALC_L3) ? (3 * 32) + k_cnt + 128 : (3 * 32) + k_cnt;
​
### Proposed Model - 6th Simulation

5차 simulation 이후, 예상과 같이 Layer 2의 output이 정상적으로 Layer 3의 input으로 전달되는 것을 확인하였습니다.

​

다만 4차 simulation에서 확한 input과 weight 간 mismatch의 영향으로, 

Layer 2와 Layer 3에서는 여전히 X(Unknown)이 발생하고 있음을 확인할 수 있습니다.


MAC 연산의 첫 cycle을 기준으로 분석한 결과, 원인은 weight path와 input path 간의 latency 정렬에 있음을 확인하였습니다.

​

Weight는 Weight_Memory에서 posedge clk 기준으로 read되므로 Weight_Bank 출력(w_bank_out)에 1-cycle latency가 

존재하는 반면, input은 k_cnt == 0 시점부터 즉시 Systolic_Array로 공급되어 두 경로 간 timing mismatch가 발생하였습니다. 

​

이로 인해 초기 MAC 연산에서 input–weight 불일치가 발생하고, 

해당 mismatch가 accumulator 및 이후 pipeline stage로 전파되면서 X(Unknown)가 생성되었습니다.

​

이를 해결하기 위해 Global_Buffer에서 read된 input data에 대해 1-cycle pipeline register(buf_r_data_d1)를 추가하여, 

input path에도 weight path와 동일한 latency를 부여하였습니다. 

​

수정된 코드에서는 Layer 2 및 Layer 3 연산 구간에서 direct buffer output(buf_r_data) 대신 지연된 버전(buf_r_data_d1)을 

row input으로 사용합니다.

// Global_Buffer read-data 1-cycle latency (align with Weight_Bank latency)
reg [dataWidth-1:0] buf_r_data_d1 [3:0];

always @(posedge clk) begin
    for (rr = 0; rr < 4; rr = rr + 1) begin
        buf_r_data_d1[rr] <= buf_r_data[rr];
    end
end
이와 같이 weight path(Weight_Memory → Weight_Bank)와 input path(Global_Buffer → buf_r_data_d1)에 

동일한 pipeline depth를 적용함으로써, 

Systolic_Array에서 input과 weight가 동일한 k_cnt 기준으로 time-aligned되어 MAC 연산이 수행되도록 수정하였습니다.

​

추가적으로 TB Debug Monitor에서도 1-cycle latency를 고려하여 monitoring 되도록 코드를 수정하였습니다.

(** 최종 TB 코드 참고)

​

Proposed Model - 7th Simulation

7회차 simulation에서 Layer 2의 연산 결과가 Golden value와 일치함을 확인하였습니다.

또한 Layer 3에서도 output이 정상적으로 생성되었으나, 최종 결과와는 아직 일치하지 않음을 확인하였습니다.



Layer 2 결과


Layer 3

기대값이 7임에도 불구하고, 최종 결과가 6에 대해 127로 출력되는 현상을 확인하였습니다.

​

이를 해결하기 위해 연산 경로를 상세히 분석한 결과,

weight.mif에 정의된 값과 실제 MAC 연산 과정에서 사용되는 weight 값이 일치하지 않는 구간이 존재함을 확인하였습니다.

​

이로 인해 MAC 연산 초기에 예상과 다른 곱셈 결과가 발생하였고, 해당 오차가 누적되면서 최종 결과에 영향을 미치고 있었습니다.

​

문제 분석 과정에서 특히 overflow 처리 방식에 주목하였으며,

기존 구현에서는 MAC 누산 과정 및 bias add 이후 단계에서 overflow 발생 시 상위 bit를 단순히 truncation하는 방식으로 

처리하고 있음을 확인하였습니다.

​

<Activation_Unit.sv>

    always @(*) begin
        // eliminate MSB(overflow bit) by truncation
        // -> overflow 시 sign bit가 잘려 wrap-around 발생
        sum_saturated = bias_add_res[2*dataWidth-1:0];
    end
이 경우 overflow가 발생하면 MSB가 잘리면서 wrap-around가 발생하고, 

결과적으로 부호가 뒤집히거나 값이 크게 왜곡되는 문제가 발생할 수 있습니다.

​

초기에는 Activation 단계에서 발생하는 overflow가 원인일 가능성을 고려하여,

bias add 이후의 결과에 대해 saturation 처리를 적용하였습니다.

    // ================================================================
    // [MODIFIED CODE] (saturating add to prevent wrap-around)
    // ----------------------------------------------------------------
    // Saturating add for psum + bias
    //
    // psum_in과 bias_in은 W-bit signed 값이며,
    // 더한 결과는 최악의 경우 W+1 bit가 필요함.
    //
    // 기존 truncate 방식은 overflow 시 MSB가 잘려
    // wrap-around(부호 뒤집힘)가 발생할 수 있음.
    //
    // 이를 방지하기 위해 결과를 W-bit signed 범위로
    // saturation 처리함.
    //
    // bias_add_res : (W+1)-bit full-precision sum
    // sum_saturated: W-bit saturated result
    // ----------------------------------------------------------------
    localparam int W = 2*dataWidth;                         // psum/bias 입력 폭

    localparam signed [W-1:0] SAT_MAX = (1 <<< (W-1)) - 1;  // 최대값 (W-bit signed)
    localparam signed [W-1:0] SAT_MIN = - (1 <<< (W-1));    // 최소값 (W-bit signed)

    // saturation logic (wrap-around 방지)
    always @(*) begin
        if (bias_add_res > SAT_MAX)
            sum_saturated = SAT_MAX;        // positive overflow
        else if (bias_add_res < SAT_MIN)
            sum_saturated = SAT_MIN;        // negative overflow
        else
            sum_saturated = bias_add_res[W-1:0]; // 정상 범위
    end
    // ----------------------------------------------------------------
그러나 Activation_Unit에서의 overflow 처리만으로는 문제가 완전히 해결되지 않았으며,

동일한 wrap-around 문제가 PE 내부의 accumulator 경로에서도 발생할 수 있음을 확인하였습니다.

​

<PE.sv>

    assign full_sum  = {sum[W-1], sum} + {mul[W-1], mul};

    always @(posedge clk) begin
        if (rst) begin
            sum <= '0;
        end else if (en) begin
            if (full_sum > SAT_MAX)
                sum <= SAT_MAX;      // positive overflow clamp
            else if (full_sum < SAT_MIN)
                sum <= SAT_MIN;      // negative overflow clamp
            else
                sum <= full_sum[W-1:0];
        end
    end
이에 따라 Activation_Unit뿐만 아니라 PE 내부의 accumulator 경로에도 동일하게 saturation 기반 overflow 처리를 적용하였으며,

이를 통해 MAC 누산 과정 전반에서 wrap-around로 인한 값 왜곡을 방지하도록 수정하였습니다.

​

Proposed Model - 8th Simulation


============================================================
[TB] Layer 1 Calculation Complete!
[ 
  72   17   95   30    0   37  116   67  127   17 
     0    0    0  115    0  127    0   55    0   36 
     0   18   17   35   84    1   73    3   12   95  
]
============================================================


============================================================
[TB] Layer 2 Calculation Complete!
[ 
 118  122  111    5    7  125    0   64  113   86 
    91  125   73    1   78   40  118  105    0  125  
]
============================================================


============================================================
[TB] Layer 3 Calculation Complete!
[ 
   0    0    0    1    0    0    0  126    0    0  
]
============================================================

============================================================
[TB] Inference DONE at time 68105000
------------------------------------------------------------
[TB] [Image 0] Result: 7 | Expected: 7
[TB] >>> PASS <<<
============================================================

7차 Simulation에서 코드 수정을 반영한 결과, 최종적으로 PASS함을 확인하였습니다.

향후 Ver.1 → Ver.2 개선점

1. 다중 Sample 입력 처리 검증

Ver.1에서는 단일 sample뿐만 아니라 최대 4개의 sample을 병렬로 처리할 수 있도록 TB 구조를 재설계하였습니다.

첨부파일TB_Full_NPU_Top.sv파일 다운로드
실제로 Sample 4개에 대해서도 최종 classification 결과는 모두 PASS함을 확인하였습니다.


다만 Layer별 연산 과정을 debug하는 과정에서, 일부 intermediate value가 X(Unknown)으로 관측되는 문제가 확인되었습니다.


============================================================
[TB] Layer 1 Calculation Complete! (sample0 only)
[ 
  72   17    x    x    0   37    x    x  127   17 
     x    x    0  115    x    x    0   55    x    x 
     0   18    x    x   84    1    x    x   12   95  
]
============================================================


============================================================
[TB] Layer 2 Calculation Complete! (sample0 only)
[ 
 118  122    x    x    7  125    x    x  113   86 
     x    x   73    1    x    x  118  105    x    x  
]
============================================================


============================================================
[TB_DEBUG] Start L3 stream debug (sample0). Printing k=1..20
============================================================
[L3_MAC] t=66845000 k=1/20 (raw=2) | in0=118 w0=35 | mul00=4130 acc00=0
[L3_MAC] t=66855000 k=2/20 (raw=3) | in0=122 w0=-6 | mul00=-732 acc00=4130
[L3_MAC] t=66865000 k=3/20 (raw=4) | in0=111 w0=13 | mul00=1443 acc00=3398
[L3_MAC] t=66875000 k=4/20 (raw=5) | in0=5 w0=17 | mul00=85 acc00=4841
[L3_MAC] t=66885000 k=5/20 (raw=6) | in0=7 w0=41 | mul00=287 acc00=4926
[L3_MAC] t=66895000 k=6/20 (raw=7) | in0=125 w0=-43 | mul00=-5375 acc00=5213
[L3_MAC] t=66905000 k=7/20 (raw=8) | in0=0 w0=-36 | mul00=0 acc00=-162
[L3_MAC] t=66915000 k=8/20 (raw=9) | in0=64 w0=21 | mul00=1344 acc00=-162
[L3_MAC] t=66925000 k=9/20 (raw=10) | in0=113 w0=-43 | mul00=-4859 acc00=1182
[L3_MAC] t=66935000 k=10/20 (raw=11) | in0=86 w0=-39 | mul00=-3354 acc00=-3677
[L3_MAC] t=66945000 k=11/20 (raw=12) | in0=91 w0=-26 | mul00=-2366 acc00=-7031
[L3_MAC] t=66955000 k=12/20 (raw=13) | in0=125 w0=-48 | mul00=-6000 acc00=-9397
[L3_MAC] t=66965000 k=13/20 (raw=14) | in0=73 w0=4 | mul00=292 acc00=-15397
[L3_MAC] t=66975000 k=14/20 (raw=15) | in0=1 w0=-52 | mul00=-52 acc00=-15105
[L3_MAC] t=66985000 k=15/20 (raw=16) | in0=78 w0=-17 | mul00=-1326 acc00=-15157
[L3_MAC] t=66995000 k=16/20 (raw=17) | in0=40 w0=15 | mul00=600 acc00=-16483
[L3_MAC] t=67005000 k=17/20 (raw=18) | in0=118 w0=-19 | mul00=-2242 acc00=-15883
[L3_MAC] t=67015000 k=18/20 (raw=19) | in0=105 w0=56 | mul00=5880 acc00=-18125
[L3_MAC] t=67025000 k=19/20 (raw=20) | in0=0 w0=-16 | mul00=0 acc00=-12245
[L3_MAC] t=67035000 k=20/20 (raw=21) | in0=125 w0=0 | mul00=0 acc00=-12245
============================================================
[TB_DEBUG] End L3 MAC debug. Printed 20 lines
============================================================

[L3_ACC_AFTER_LAT1] t=67055000 | k_end=20 (raw_end=21) +1 | acc00=-12245
[L3_AU_AFTER_LAT2 ] t=67065000 | k_end=20 (raw_end=21) +2 | psum_sel=-8217 bias_sel=-5632 | sum=-13849 sat=-13849 | act=0 | acc00=-12245

============================================================
[TB] Layer 3 Calculation Complete! (sample0 only)
[ 
   0    0    x    x    0    0    x    x    0    0  
]
============================================================

============================================================
[TB] Inference DONE at time 68105000
------------------------------------------------------------
[TB] [Image 0] Result: 7 | Expected: 7
[TB] >>> PASS <<<
------------------------------------------------------------
[TB] [Image 1] Result: 2 | Expected: 2
[TB] >>> PASS <<<
------------------------------------------------------------
[TB] [Image 2] Result: 1 | Expected: 1
[TB] >>> PASS <<<
------------------------------------------------------------
[TB] [Image 3] Result: 0 | Expected: 0
[TB] >>> PASS <<<
------------------------------------------------------------
============================================================
$finish called at time : 68305 ns : File "C:/Users/mini9/Desktop/NPU/initial_v1/TB_Full_NPU_Top.sv" Line 202
해당 현상은 최종 classification 결과에는 영향을 주지 않았으나,

다중 sample 입력 시 Layer별 MAC 결과, accumulator 값, activation 출력 등

중간 연산 값 일부가 X(Unknown)으로 관측되는 문제가 확인되었습니다.

​

따라서 Ver.2에서는 다중 sample 입력 조건에서도 Layer별 MAC 결과와 intermediate value가 X 없이 일관되게 관측되도록,

DUT 및 Testbench 구성을 재정비할 필요가 있습니다.

​

2. Ver.1 Compute 구조의 한계

현재 Ver.1의 compute 구조는 전통적인 의미의 Systolic Array 구조라고 보기는 어렵습니다.

​

Wavefront skew가 반영되지 않아, input과 weight가 동일 cycle에 모든 PE로 동시에 공급되며

연산이 병렬적으로 수행되는 SIMD 형태에 가까운 구조로 동작하기 때문입니다.

​

즉, row/column 방향으로 data가 cycle마다 이동하는 systolic dataflow 구조가 아니라,

동일 cycle에 분해된 input과 weight가 각 PE에 직접 전달되는 방식입니다.

​

Ver.2에서는 input과 weight 경로에 cycle 단위 data skew(wavefront propagation)를 도입하여,

각 PE가 시간적으로 offset된 data를 처리하도록 구성함으로써 보다 전형적인 Systolic Array 구조로 개선할 계획입니다.

