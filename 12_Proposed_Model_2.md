## Proposed Model: Ver.2

이전 글에서는 initial 코드를 기반으로, 실제 inference가 수행되는 mini-NPU를 구현하였습니다.

다만 해당 모델은 초기 설계 목표였던 Systolic Array 기반 OS(Output Stationary) dataflow가 아니라, 입력 데이터를 동일 cycle에 모든 PE로 동시에 전달하는 Broadcast 방식으로 동작합니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_98.png" width="400"/>

Ver.1 – Global Broadcast Dataflow

<div align="left">

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_99.png" width="400"/>

Ver.2 – Systolic Array (Output Stationary) Dataflow

<div align="left">

따라서 이번 글에서는 Ver.1 아키텍처를 기반으로, 실제 SA의 OS dataflow에 맞는 연산 구조로 재구성하고 구현 과정을 정리했습니다.

Proposed Model v.2는 3차 Simulation까지 진행한 이후 Roll-back 한 뒤, 재설계를 진행하였습니다. 최종 모델의 Debug 과정을 확인하려면 “Proposed Model v.2 – 4th Simulation”부터 참고하시면 됩니다.(Project Ver.1 File Download)

### Revision History

    Proposed Model v2 – Update Log
    1차 Simulation
    → w_pipe 도입으로 weight 1-cycle latency가 추가되며 input–weight timing mismatch 및 X 발생.
    
    2차 Simulation
    → weight latency 증가에 대한 enable 보정 미흡으로 MAC에 X 누적 지속.
    
    3차 Simulation
    → lane별 delay 조정 시도에도 X 문제 해결되지 않아 구조 복잡도 증가로 Roll-back 결정.
    
    4차 Simulation (After Roll-back)
    → w_pipe를 NPU_Top으로 이동하고 hierarchy를 단순화하여 X 제거, 대신 1-cycle latency 발생.
    
    5차 Simulation
    → input alignment 보정 과정에서 Global_Buffer invalid read로 X 유입 확인 및 gating 적용.
    
    6차 Simulation
    → flush cycle 도입으로 state 조기 전환 문제 해결했으나 첫 값 drop 및 tail 누락 발생.
    
    7차 Simulation
    → data_valid와 shift_en 분리 및 drain 확장으로 첫 값 drop과 tail 누락 해결.
    
    8차 Simulation (Final)
    → 4-lane 및 100-sample 검증 결과 98% accuracy로 정상 동작 확인.
    
### Proposed Model v.2 - 1st Simulation

1. PE 내부에 data_pipe 추가
    각 PE가 MAC 연산을 수행하는 동시에 입력 데이터를 인접 PE로 전달할 수 있도록, PE 내부에 data_pipe 구조를 추가하였습니다.

        // data_pipe register
        logic signed [dataWidth-1:0] a_reg, b_reg;
        
        // forwarding to adjacent PE
        assign a_out = a_reg;
        assign b_out = b_reg;
        
        // input latch (1-cycle pipeline)
        always_ff @(posedge clk) begin
            if (rst) begin
                a_reg <= '0;
                b_reg <= '0;
            end 
            else if (en) begin
                a_reg <= a_in;
                b_reg <= b_in;
            end
        end

    이후 weight 경로에 data skew를 적용하기 위해, Weight_Bank.sv에 다음과 같은 pipeline 구조를 추가하였습니다.(당시에는 sample 1개에 대한 추론만 수행하였으므로 input data skew는 차순위 검토 대상으로 두었습니다.)

        // -------------------------------------------------
        // Weight Data Skew Pipeline
        // -------------------------------------------------
        
        reg  signed [dataWidth-1:0] w_pipe [0:3][0:3];
        wire signed [4*dataWidth-1:0] w_wavefront_packed;
        
        // diagonal tap (wavefront alignment)
        assign w_wavefront_packed = {
            w_pipe[3][3],
            w_pipe[2][2],
            w_pipe[1][1],
            w_pipe[0][0]
        };
        
        integer lane, st;
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                for (lane = 0; lane < 4; lane = lane + 1)
                    for (st = 0; st < 4; st = st + 1)
                        w_pipe[lane][st] <= '0;
            end 
            else if (pe_rst) begin
                for (lane = 0; lane < 4; lane = lane + 1)
                    for (st = 0; st < 4; st = st + 1)
                        w_pipe[lane][st] <= '0;
            end 
            else begin
                for (lane = 0; lane < 4; lane = lane + 1) begin
                    for (st = 3; st > 0; st = st - 1)
                        w_pipe[lane][st] <= w_pipe[lane][st-1];
                    w_pipe[lane][0] <= w_lane[lane];
                end
            end
        end

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_100.png" width="400"/>

Weight Skew Pipeline (w_pipe) 구조와 Diagonal Tap (w_wavefront_packed) 동작

<div align="left">

해당 pipeline은 4×4 shift register 구조로 구성하였으며, 각 lane의 weight가 매 cycle마다 한 단계씩 이동하도록 설계하였습니다.

​lane index에 따라 서로 다른 pipeline stage를 tap하여, lane[0]~[3]에 대해 0 ~ 3-cycle skew가 형성되도록 구성하였습니다.

​최종적으로 w_pipe[0][0], w_pipe[1][1], w_pipe[2][2], w_pipe[3][3]를 묶어 출력함으로써, lane별 0~3 cycle의 time skew가 의도적으로 반영된 weight wavefront를 구성하고, PE array 입력 타이밍이 순차적으로 정렬되도록 맞추었습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_101.png" width="400"/>

<div align="left">

그러나 Golden과 불일치하였으며 tile 0 이후 출력이 X로 발생하였고, L2 및 L3 구간에서도 input은 정상적으로 관측되었으나 weight 신호는 X로 출력되는 현상이 확인되었습니다.

​이에 따라, PE(0,0)에서 수행되는 MAC 연산 로그를 확인하였습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_102.png" width="400"/>

<div align="left">

L1 연산 로그 분석 결과, 빨간색 영역 이후로 input과 weight의 cycle 정렬이 이루어지지 않아 Mismatch가 발생하고 있음을 확인하였습니다.

​이는 Weight_Bank에 전체 lane에 대한 w_pipe를 추가하면서, weight가 pipeline register를 한 단계 더 거치도록 구조가 변경되어 의도치 않은 1-cycle latency가 삽입되었기 때문입니다.

weight의 1-cycle 지연을 보정하기 위해 v1과 동일하게 input에 1-cycle delay를 추가한 후 Simulation을 재수행하였습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_103.png" width="400"/>

<div align="left">

또한 L2, L3의 경우 global buffer를 통해 결과가 다음 layer의 input으로 전달되므로, 

        Global buffer(1-cycle latency) + weight_pipe(1-cycle latency) = Total(2-cycle latency)

에 의해 총 2-cycle latency가 발생함을 고려하여 input 경로에 2-stage register를 추가하여 보정하였습니다. 

        for (rr = 0; rr < 4; rr = rr + 1)
            buf_r_data_d1[rr] <= buf_r_data[rr];
            buf_r_data_d2[rr] <= buf_r_data_d1[rr];

보정 이후 L2 및 L3 구간에서도 input과 weight가 동일 cycle에 정렬됨을 확인하였습니다.

### Proposed Model v.2 - 2nd Simulation

1st Simulation에서 4개 연산 결과 이후 X가 출력되는 문제가 발생하였으며, 초기에는 연산 수행 state_1 (CALC_L1) → 결과 저장 state_2 (BUFFER_WR_L1) → tile 변경과 함께 state_1으로 복귀하는 전이 구간에서 X 값이 유입된 것으로 원인을 추정하였습니다.

​이를 확인하기 위해 Ver.1과 Ver.2의 state 전이 구간을 비교 분석하였습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_104.png" width="400"/>

Ver.1 State Transition 구간

<div align="left">

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_105.png" width="400"/>

Ver.2 State Transition 구간

<div align="left">

Ver.1과 Ver.2의 state 전이 구간을 각각 분석한 결과, state 전이 자체에는 문제가 없음을 확인하였습니다.

그러나 b_in이 X인 상태에서 en이 활성화되어 MAC 연산이 수행되었고, 그 결과 X가 누적되는 현상이 발생하고 있음을 확인하였습니다.

​weight 경로에 1-cycle latency가 추가되었으나, 해당 지연에 맞춰 enable 타이밍이 조정되지 않아 valid하지 않은 weight가 MAC에 사용되었기 때문으로 추정하였습니다.

​이에 따라 tile 1~8 구간에서 발생한 X(Unknown) 문제를 해결하기 위해, 기존 Systolic_Array 설계를 참고하여 enable 제어 및 data 정렬 방식을 재구성하였습니다.

​이에 따라 기존의 

        PE → Systolic_Array → NPU_Top

구조를 

        PE → PE_Systolic_cell → Systolic_Array → NPU_Top
        
형태로 변경하고, data_pipe를 별도 모듈로 분리하는 구조 수정도 시도하였습니다. 

그러나 이러한 아키텍처 변경 이후에도 출력 값에는 유의미한 개선이 확인되지 않았습니다.

### Proposed Model v.2 - 3rd Simulation

2nd Simulation 이후, w_pipe로 인해 전파되는 X를 제거하기 위해 기존 4×4 w_pipe 구조를 수정하였습니다.

        // -------------------------------------------------
        // Weight Data Skew Pipeline (Modified)
        // Spec:
        //  lane0 : sel_w bypass (0-cycle)
        //  lane1 : w_pipe[1][0] -> 1-cycle latency
        //  lane2 : w_pipe[2][1] -> 1-cycle latency + 1-cycle delay
        //  lane3 : w_pipe[3][2] -> 1-cycle latency + 2-cycle delay
        // -------------------------------------------------
        
        reg  signed [dataWidth-1:0] w_pipe [0:3][0:3];
        wire signed [4*dataWidth-1:0] w_wavefront_packed;
        
        // tap selection based on spec
        assign w_wavefront_packed = {
            w_pipe[3][2],   // lane3 : 1 + 2
            w_pipe[2][1],   // lane2 : 1 + 1
            w_pipe[1][0],   // lane1 : 1 + 0
            sel_w[0]        // lane0 : bypass (0)
        };
        
        integer lane, st;
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                for (lane = 0; lane < 4; lane = lane + 1)
                    for (st = 0; st < 4; st = st + 1)
                        w_pipe[lane][st] <= '0;
            end
            else if (pe_rst) begin
                for (lane = 0; lane < 4; lane = lane + 1)
                    for (st = 0; st < 4; st = st + 1)
                        w_pipe[lane][st] <= '0;
            end
            else begin
                // lane0 : not used (optional clear for waveform readability)
                for (st = 0; st < 4; st = st + 1)
                    w_pipe[0][st] <= '0;
        
                // lane1~3 : shift only within stages [0:2]
                // (stage[3] unused/cleared)
                for (lane = 1; lane < 4; lane = lane + 1) begin
                    w_pipe[lane][2] <= w_pipe[lane][1];
                    w_pipe[lane][1] <= w_pipe[lane][0];
                    w_pipe[lane][0] <= sel_w[lane];
        
                    w_pipe[lane][3] <= '0;
                end
            end
        end

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_106.png" width="400"/>

<div align="left">

기존 구조에서는 모든 lane에 동일한 depth의 pipeline을 적용하였으나, 수정안에서는 lane0을 w_pipe를 거치지 않는 bypass 경로로 변경하고, lane1~3에 대해서는 pipeline stage를 순차적으로 감소시켜 delay를 축소하는 방식으로 재구성하였습니다.

        lane0: sel_w bypass (지연 없음)
        lane1: w_pipe(0) → 1-cycle latency
        lane2: w_pipe(1) → 1-cycle latency + 1-cycle delay
        lane3: w_pipe(2) → 1-cycle latency + 2-cycle delay

해당 구조 적용 시 X 전파 문제가 해소될 것으로 예상하였으나, 실제 결과에서는 lane0을 제외한 lane1~3 구간에서 여전히 X가 출력되는 현상이 확인되었습니다..

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_107.png" width="400"/>

<div align="left">

lane0에 해당하는 PE는 정상적으로 동작하였으나, input이 1-cycle 지연되어 도달하는 현상이 관측되었습니다. 

​반면 lane1~3에 속한 PE는 2nd Simulation과 동일하게 en 활성 시점에 operand(b_in)가 X 상태였고, 그 결과 MAC 연산 결과가 X로 전파되는 현상이 지속되었습니다.

해당 지점에서 구조 변경이 누적되며 설계 complexity가 증가하였고, 부분적인 수정으로는 문제 해결이 어렵다고 판단하였습니다.

이에 따라 initial Ver.1로 roll-back한 후, PE에 data_pipe가 추가된 시점부터 다시 분석을 시작하였습니다.

### Proposed Model v.2 - 4th Simulation

Roll-back 이전에 발생한 문제의 원인이 w_pipe로 확인됨에 따라, 해당 로직을 Weight_Bank에서 제거하고 NPU_Top으로 이동하여 재구성하였습니다. 

​이와 함께 분리되어 있던 PE와 PE_Systolic_Cell을 다시 단일 PE로 통합하고, PE 내부에 data_pipe를 재적용하여 module hierarchy를 단순화하였습니다.(PE_Systolic_Cell을 제거하고 PE로 통합함으로써 MAC 연산과 data_pipe 경로를 하나의 모듈에서 처리하도록 정리하였고, 이에 따라 enable·reset·latency를 일관되게 관리할 수 있게 되었습니다. 또한 data_pipe의 1-cycle latency가 PE 내부에 고정되면서 상위 모듈에서 별도의 delay 보정 로직을 둘 필요가 줄어들었고, 전체 module hierarchy도 단순해졌습니다.)

이후 Simulation을 재수행한 결과, L1·L2·L3 전 구간에서 X 없이 모든 신호가 정상적으로 동작함을 확인하였습니다. 

다만 w_pipe 적용으로 인해 모든 layer에서 weight 경로에 1-cycle latency가 추가로 발생함을 확인하였습니다.

### Proposed Model v.2 - 5th Simulation

4th Simulation에서 발생한 1-cycle latency를 보정하기 위해, 

L1 ~ L3 input 경로에 1-cycle delay를 추가하여 input과 weight를 재정렬하였습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_108.png" width="400"/>

valid data(input: 72) 도달 전 구간에서 X가 유입되는 현상. (Input skew 적용 이전) 

<div align="left">

하지만 input 경로에 1-cycle delay를 추가하는 과정에서, 초기 구간에서 X(Unknown)이 발생하는 문제가 확인되었습니다.

​원인 분석 결과, write state에서 Global_Buffer read address가 유효 범위를 벗어나는 구간이 존재하였고, 이때 발생한 X가 r_data 경로를 통해 input pipeline으로 전파되는 것을 확인하였습니다. (Roll-back 이전 X도 동일 원인으로 추후 확인.)

​이에 따라 Global_Buffer read 경로에서의 X 전파를 차단하기 위해, read address가 CALC_L2, CALC_L3, OUTPUT_SCAN 상태에서만 유효하도록 gating을 적용하였습니다.

        assign buf_r_addr[0] = (calc_l2)  ? (0*32 + k_cnt) :
                               (calc_l3)  ? (0*32 + k_cnt + 128) :
                               (out_scan) ? (0*32 + k_cnt) :
                                            32'd0;
        
        assign buf_r_addr[1] = (calc_l2)  ? (1*32 + k_cnt) :
                               (calc_l3)  ? (1*32 + k_cnt + 128) :
                               (out_scan) ? (1*32 + k_cnt) :
                                            32'd0;
        
        assign buf_r_addr[2] = (calc_l2)  ? (2*32 + k_cnt) :
                               (calc_l3)  ? (2*32 + k_cnt + 128) :
                               (out_scan) ? (2*32 + k_cnt) :
                                            32'd0;
        
        assign buf_r_addr[3] = (calc_l2)  ? (3*32 + k_cnt) :
                               (calc_l3)  ? (3*32 + k_cnt + 128) :
                               (out_scan) ? (3*32 + k_cnt) :
                                            32'd0;
                                            
또한 buf_r_data_d1(필요 시 buf_r_data_d2 포함)을 단일 always 블록에서만 갱신하도록 구조를 정리하였습니다. 

​이를 통해 write state 동안 발생할 수 있는 invalid read 값(0/X)이 내부 경로로 전파되는 것을 차단하였습니다.

        // 기존 inject 조건 (문제 원인)
        wire inject_en_l23 = calc_l23 &&
                             (k_cnt >= 1) &&
                             (k_cnt <= cur_input_len);
        
        // read-data staging도 inject_en에 의해 간접적으로 gating
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                for (rr = 0; rr < 4; rr = rr + 1)
                    buf_r_data_d1[rr] <= '0;
            end else begin
                if (inject_en_l23) begin
                    for (rr = 0; rr < 4; rr = rr + 1)
                        buf_r_data_d1[rr] <= buf_r_data[rr];
                end
            end
        end

해당 수정 반영 이후 Simulation 결과, Global_Buffer로부터 전파되던 X는 제거되었으며 L2·L3 구간의 출력도 정상 값으로 안정화되었습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_109.png" width="400"/>

<div align="left">

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_110.png" width="400"/>

<div align="left">

다만 마지막 MAC 연산 결과가 완전히 누적되지 않는 현상이 확인되었으며, 해당 영향으로 인해 L2 및 L3의 최종 결과에서도 ACCU(accumulate) 및 AU(activate) 값에 오차가 발생하였습니다.

### Proposed Model v.2 - 6th Simulation

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_111.png" width="400"/>

30번째 값 미반영. (Input skew 적용 이전)

<div align="left">

Waveform 분석 결과, 30번째 값이 accumulator에 반영되어야 하는 시점에 FSM이 다음 state로 전환되고 있음을 확인하였습니다.

즉, 마지막 MAC 결과가 fully accumulate 되기 전에 state가 변경되면서 최종 누적 결과 일부가 반영되지 않는 문제가 발생하였습니다.

​이에 따라 연산 종료 이후에도 pipeline 내부 데이터가 완전히 전파(drain)되도록, 추가적인 pipeline flush cycle을 도입하였습니다.

        localparam int FLUSH_CYCLES = 6;
        wire [31:0] calc_end_k = cur_input_len + 32'd1 + FLUSH_CYCLES;
        
FLUSH_CYCLES를 6으로 설정한 이유는 SA 내부에 row skew(최대 3-cycle)와 weight skew(최대 3-cycle)가 동시에 존재하기 때문입니다.

worst-case 기준으로 두 skew가 중첩될 경우 최대 6-cycle의 전파 시간이 필요하므로, 마지막 MAC 결과가 fully 반영되도록 6-cycle의 flush 구간을 추가하였습니다.

​이를 통해 input 주입이 종료된 이후에도 일정 cycle 동안 state를 유지함으로써, wavefront가 완전히 전파되도록 보장하였습니다.

​OS(Output Stationary) dataflow 특성에 맞도록 기존 roadcast 기반 input 구조에 skew를 적용하여, 각 lane에 데이터가 동시에 인가되지 않도록 하고 시간차를 두고 순차적으로 주입되도록 구조를 수정하였습니다.

​또한 pipeline 구조 적용에 따라 기존 L2/L3에서 사용하던 buf_r_data_d2는 제거하고 buf_r_data_d1만 사용하도록 변경하여, 불필요한 1-cycle delay를 제거하였습니다.

        // wavefront: row pipe
        else if (shift_en) begin
            for (rlane = 0; rlane < 4; rlane = rlane + 1) begin
                for (rstg = 3; rstg > 0; rstg = rstg - 1)
                    row_pipe[rlane][rstg] <= row_pipe[rlane][rstg-1];
                row_pipe[rlane][0] <= push_en ? in_lane[rlane] : '0;
            end
        end
​
### Proposed Model v.2 - 7th Simulation

앞서 state 전환 시 충분한 flush cycle을 부여하고 input skew를 적용하였으나, 추가적인 문제가 발생하였습니다.

​첫째, input skew는 정상적으로 적용되었음에도 불구하고 첫 번째 출력 값이 0으로 나타나는 현상이 확인되었습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_112.png" width="400"/>

<div align="left">

Waveform 확인 결과, k_cnt==0 구간에서 입력될 첫 번째 input 값(72)이 buf_r_data_d1에 저장되지 못하고 그대로 소실되는 현상이 확인되었습니다.

        // 기존 inject 조건 (문제 원인)
        wire inject_en_l23 = calc_l23 && (k_cnt >= 1) && (k_cnt <= cur_input_len);

기존 구조에서는 L2/L3의 read 데이터가 inject_en이 asserted될 때만 저장되도록 되어 있었으며, inject_en_l23 조건에 k_cnt >= 1이 포함되어 있었습니다. 

​이로 인해 k_cnt==0인 첫 cycle에서는 inject_en_l23가 0으로 유지되었고, 해당 시점의 read 데이터(addr0, 값 72)가 Global Buffer에서 buf_r_data_d1으로 저장되지 못한 채 그대로 누락되었습니다.

​이를 해결하기 위해 read 데이터 저장 조건을 k_cnt >= 1 기반 enable 구조에서 k_cnt <= cur_input_len 조건으로 변경하였습니다.

        // 새 capture 조건 (k_cnt==0 포함)
        wire data_valid_l23 = calc_l23 &&
                              (k_cnt <= cur_input_len);
        
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                for (rr = 0; rr < 4; rr = rr + 1)
                    buf_r_data_d1[rr] <= '0;
            end else begin
                if (calc_l23) begin
                    if (data_valid_l23) begin
                        for (rr = 0; rr < 4; rr = rr + 1)
                            buf_r_data_d1[rr] <= buf_r_data[rr];
                    end
                end else begin
                    for (rr = 0; rr < 4; rr = rr + 1)
                        buf_r_data_d1[rr] <= '0;
                end
            end
        end
        
그 결과 k_cnt==0인 첫 cycle의 데이터가 정상적으로 buf_r_data_d1에 capture되었으며, 첫 번째 MAC 연산 결과 또한 정상적으로 확인되었습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_113.png" width="400"/>

<div align="left">

두 번째로, 기존 구조에서는 inject_en이 k_cnt <= cur_input_len 구간까지만 유지되어 shift 동작이 조기에 종료되는 문제가 있었습니다.

​그 결과 마지막 wavefront가 Systolic Array 내부를 끝까지 전파하지 못하고 중간에서 차단되어, 최종 누적 결과 일부가 반영되지 않는 현상이 발생하였습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_114.png" width="400"/>

<div align="left">

Waveform 분석 결과, inject_en이 deassert되는 시점에 row/weight pipe의 shift 동작도 함께 중단되는 것을 확인할 수 습니다.

그 결과 마지막 유효 데이터가 PE array의 끝까지 전파되기 전에 pipeline이 정지하였고, 해당 구간의 MAC 연산이 수행되지 못한 채 0이 출력되는 문제가 발생하였습니다.

​lane별로 0/1/2/3-cycle의 input skew가 존재하므로, 가장 늦게 도착하는 lane3의 마지막 데이터는

        k_cnt = cur_input_len + MAX_LANE_SKEW

시점에야 pipe에 진입하게 됩니다.

​그러나 기존 구조에서는 inject_en_l23가 다음과 같이 정의되어 있었습니다.

        // 기존 shift/inject window (문제 구조)
        wire inject_en_l23 = calc_l23 &&
                             (k_cnt >= 1) &&
                             (k_cnt <= cur_input_len);

이로 인해 k_cnt <= cur_input_len 구간까지만 shift가 유지되었고, tail 구간에서 도달해야 할 마지막 wavefront가 SA 내부를 끝까지 전파하기 전에 shift가 중단되는 문제가 발생하였습니다.

이를 해결하기 위해 shift/inject window를 cur_input_len + MAX_LANE_SKEW까지 확장하여, tail 데이터가 완전히 전파되도록 수정하였습니다.

        localparam int MAX_LANE_SKEW = 4;
        
        // shift window 확장 (tail drain 보장)
        wire shift_en_l23 = calc_l23 &&
                            (k_cnt >= 1) &&
                            (k_cnt <= (cur_input_len + MAX_LANE_SKEW));

다만 shift window를 단순히 확장하면 k_cnt > cur_input_len 구간에서도 Global Buffer read가 지속되어, invalid/미초기화 데이터(0/X)가 pipe로 유입될 수 있습니다. 

​이 값이 그대로 전파되면 accumulator까지 전달되어 X contamination이 발생합니다.

​이를 방지하기 위해 enable을 propagation(shift) 와 stage0 injection(push) 로 분리하였습니다. 

shift는 tail drain을 위해 유지하되, 유효 데이터 주입은 k_cnt <= cur_input_len 구간에서만 허용하도록 수정하였습니다.

        // 유효 데이터 capture window
        wire data_valid_l23 = calc_l23 &&
                              (k_cnt <= cur_input_len);
        
        // propagation window (drain 포함)
        wire shift_en_l23 = calc_l23 &&
                            (k_cnt >= 1) &&
                            (k_cnt <= (cur_input_len + MAX_LANE_SKEW));
        
        wire shift_en = inject_en_l1 | shift_en_l23;
        wire push_en  = inject_en_l1 | data_valid_l23;

이때 drain 구간(k_cnt > cur_input_len)에서는 shift_en은 유지되지만 push_en은 deassert되므로, stage0에는 0(bubble)을 주입하여 invalid 값이 pipe로 전파되지 않도록 차단하였습니다.

​해당 수정 이후 마지막 MAC 결과가 정상적으로 누적되는 것을 확인하였으며, sample_0000에 대해 PASS 결과를 확인하였습니다.

        ============================================================
        [TB] Inference DONE at time 69065000
        ------------------------------------------------------------
        [TB] [Image 0] Result: 7 | Expected: 7
        [TB] >>> PASS <<<
        ============================================================
​
### Proposed Model v.2 - 8th Simulation

7th Simulation에서 sample_0000에 대한 결과를 확인한 이후 sample_0000 ~ sample_0003을 동시에 입력하여 Systolic Array의 OS dataflow가 병렬 환경에서도 정상적으로 연산을 수행하는지 검증하였습니다.

                // Load test data into image memories
                $readmemb("test_data_0000.txt", img_mem_0);
                $readmemb("test_data_0001.txt", img_mem_1);
                $readmemb("test_data_0002.txt", img_mem_2);
                $readmemb("test_data_0003.txt", img_mem_3);

또한, 각 sample에 대한 연산이 완료(done) 되었을 때 해당 결과를 확인할 수 있도록 TB에 monitor 로직을 추가하였으며, 

4개 lane의 최종 inference 결과를 동시에 검증하였습니다.

                if (done) begin
                    $display("============================================================");
                    $display("[TB] Inference DONE at time %0t", $time);
        
                    repeat(20) @(posedge clk);
        
                    $display("------------------------------------------------------------");
                     // ---- lane0: 기존 그대로 ----
                    $display("[TB] [Image 0] Result: %0d | Expected: %0d", results[0], img_mem_0[784]);
                    if (results[0] == img_mem_0[784]) $display("[TB] >>> PASS <<<"); else $display("[TB] >>> FAIL <<<");
        
                    // ---- lane1~3: DONE 결과만 추가 출력 ----
                    $display("------------------------------------------------------------");
                    $display("[TB] [Image 1] Result: %0d | Expected: %0d", results[1], img_mem_1[784]);
                    if (results[1] == img_mem_1[784]) $display("[TB] >>> PASS <<<"); else $display("[TB] >>> FAIL <<<");
                    $display("[TB] [Image 2] Result: %0d | Expected: %0d", results[2], img_mem_2[784]);
                    if (results[2] == img_mem_2[784]) $display("[TB] >>> PASS <<<"); else $display("[TB] >>> FAIL <<<");
                    $display("[TB] [Image 3] Result: %0d | Expected: %0d", results[3], img_mem_3[784]);
                    if (results[3] == img_mem_3[784]) $display("[TB] >>> PASS <<<"); else $display("[TB] >>> FAIL <<<");
        
        
                    $display("============================================================");
                    $finish;
                end

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_115.png" width="400"/>

<div align="left">

최종적으로 sample 4개에 대해서도 PASS 결과를 확인하였습니다.

### 100-Sample Inference Accuracy

이제 기존 Reference Model과 정확도 비교를 위해, Sample 100개에 대해 inference를 수행하였습니다. (TB_NPU_Top.sv파일 다운로드)

test_data_0000.txt부터 test_data_0099.txt까지 총 100개 입력을 처리하도록 TB를 수정하였습니다. 우선 전체 테스트 개수를 100으로 정의하고, Batch index와 정확도 집계를 위한 변수를 추가하였습니다.

        localparam int NUM_SAMPLES = 100;
        integer sample_idx;
        integer pass_cnt;
        integer total_cnt;

NUM_SAMPLES는 전체 입력 개수를 의미합니다. sample_idx는 현재 처리 중인 배치의 시작 번호이며, pass_cnt와 total_cnt는 100개 결과에 대한 정확도 계산을 위한 누적 변수입니다.

​본 설계는 4-lane 병렬 구조이므로, 한 번의 inference에서 4개의 입력 파일을 동시에 읽도록 load_batch_4() task를 추가하였습니다.

        task automatic load_batch_4(input int base_idx);
            string f0, f1, f2, f3;
        begin
            f0 = $sformatf("test_data_%04d.txt", base_idx + 0);
            f1 = $sformatf("test_data_%04d.txt", base_idx + 1);
            f2 = $sformatf("test_data_%04d.txt", base_idx + 2);
            f3 = $sformatf("test_data_%04d.txt", base_idx + 3);
        
            for (i=0; i<=784; i=i+1) begin
                img_mem_0[i] = '0;
                img_mem_1[i] = '0;
                img_mem_2[i] = '0;
                img_mem_3[i] = '0;
            end

base_idx를 기준으로 파일명을 자동 생성하여 4개를 동시에 로드하도록 구성하였습니다. 또한 파일을 읽기 전에 메모리를 초기화하여 이전 Batch의 데이터가 남지 않도록 하였습니다.

​여러 배치를 연속 실행할 경우 DUT 내부 state나 done 신호가 남아 있을 수 있으므로, 배치마다 reset을 수행하도록 하였습니다.

        task automatic batch_reset();
        begin
            rst_n <= 1'b0;
            repeat(5) @(posedge clk);
            rst_n <= 1'b1;
            repeat(5) @(posedge clk);
        end
        endtask

이를 통해 각 배치가 항상 동일한 초기 조건에서 시작되도록 하였습니다.

start_inference는 DUT가 level 기반으로 신호를 인식하는 경우를 고려하여 2 cycle 동안 유지하도록 구성하였습니다.

        task automatic kick_inference(input int hold_cycles);
        begin
            start_inference <= 1'b1;
            repeat(hold_cycles) @(posedge clk);
            start_inference <= 1'b0;
        end
        endtask
        
실제 실행 시에는 다음과 같이 2 cycle을 유지하도록 하였습니다.

        kick_inference(2);
        
메인 루프에서는 sample_idx를 4씩 증가시키며 총 25회 배치를 수행하도록 구성하였습니다.

        for (sample_idx = 0; sample_idx < NUM_SAMPLES; sample_idx += 4) begin
            load_batch_4(sample_idx);
        
            batch_reset();
            kick_inference(2);
        
            wait(done === 1'b1);
            repeat(20) @(posedge clk);
        
            $display("[TB] Batch base=%0d DONE t=%0t", sample_idx, $time);
            ...
        end
        
각 배치에서 4개 입력을 로드한 뒤, reset → start → done 대기 → 결과 안정화 순으로 동작하도록 하였습니다.

각 lane의 inference 결과를 해당 입력의 label과 비교하여 PASS/FAIL을 집계하였습니다.

        total_cnt++;
        if (results[0] == img_mem_0[784]) begin
            pass_cnt++;
            $display("[TB] [Image %0d] PASS | Result=%0d Expected=%0d",
                     sample_idx+0, results[0], img_mem_0[784]);
        end else begin
            $display("[TB] [Image %0d] FAIL | Result=%0d Expected=%0d",
                     sample_idx+0, results[0], img_mem_0[784]);
        end

마지막으로 전체 100개에 대한 PASS/TOTAL을 출력하여 Reference Model과의 정확도를 확인하였습니다.

        $display("[TB] ALL DONE. PASS=%0d / TOTAL=%0d (FAIL=%0d)",
                 pass_cnt, total_cnt, (total_cnt-pass_cnt));
                 
이후 해당 TB를 적용한 100개 샘플에 대한 inference 결과를 정리하였습니다.

        ------------------------------------------------------------
        [TB] Batch base=0 DONE t=69385000
        [TB] [Image 0] PASS | Result=7 Expected=7
        [TB] [Image 1] PASS | Result=2 Expected=2
        [TB] [Image 2] PASS | Result=1 Expected=1
        [TB] [Image 3] PASS | Result=0 Expected=0
        [TB] Progress: pass=4 / total=4
        ------------------------------------------------------------
        [TB] Loading batch: 4..7
        ------------------------------------------------------------
        [TB] Batch base=4 DONE t=138725000
        [TB] [Image 4] PASS | Result=4 Expected=4
        [TB] [Image 5] PASS | Result=1 Expected=1
        [TB] [Image 6] PASS | Result=4 Expected=4
        [TB] [Image 7] PASS | Result=9 Expected=9
        [TB] Progress: pass=8 / total=8
        ------------------------------------------------------------
        [TB] Loading batch: 8..11
        ------------------------------------------------------------
        [TB] Batch base=8 DONE t=208065000
        [TB] [Image 8] PASS | Result=5 Expected=5
        [TB] [Image 9] PASS | Result=9 Expected=9
        [TB] [Image 10] PASS | Result=0 Expected=0
        [TB] [Image 11] PASS | Result=6 Expected=6
        [TB] Progress: pass=12 / total=12
        ------------------------------------------------------------
        [TB] Loading batch: 12..15
        ------------------------------------------------------------
        [TB] Batch base=12 DONE t=277405000
        [TB] [Image 12] PASS | Result=9 Expected=9
        [TB] [Image 13] PASS | Result=0 Expected=0
        [TB] [Image 14] PASS | Result=1 Expected=1
        [TB] [Image 15] PASS | Result=5 Expected=5
        [TB] Progress: pass=16 / total=16
        ------------------------------------------------------------
        [TB] Loading batch: 16..19
        ------------------------------------------------------------
        [TB] Batch base=16 DONE t=346745000
        [TB] [Image 16] PASS | Result=9 Expected=9
        [TB] [Image 17] PASS | Result=7 Expected=7
        [TB] [Image 18] FAIL | Result=8 Expected=3
        [TB] [Image 19] PASS | Result=4 Expected=4
        [TB] Progress: pass=19 / total=20
        ------------------------------------------------------------
        [TB] Loading batch: 20..23
        ------------------------------------------------------------
        [TB] Batch base=20 DONE t=416085000
        [TB] [Image 20] PASS | Result=9 Expected=9
        [TB] [Image 21] PASS | Result=6 Expected=6
        [TB] [Image 22] PASS | Result=6 Expected=6
        [TB] [Image 23] PASS | Result=5 Expected=5
        [TB] Progress: pass=23 / total=24
        ------------------------------------------------------------
        [TB] Loading batch: 24..27
        ------------------------------------------------------------
        [TB] Batch base=24 DONE t=485425000
        [TB] [Image 24] PASS | Result=4 Expected=4
        [TB] [Image 25] PASS | Result=0 Expected=0
        [TB] [Image 26] PASS | Result=7 Expected=7
        [TB] [Image 27] PASS | Result=4 Expected=4
        [TB] Progress: pass=27 / total=28
        ------------------------------------------------------------
        [TB] Loading batch: 28..31
        ------------------------------------------------------------
        [TB] Batch base=28 DONE t=554765000
        [TB] [Image 28] PASS | Result=0 Expected=0
        [TB] [Image 29] PASS | Result=1 Expected=1
        [TB] [Image 30] PASS | Result=3 Expected=3
        [TB] [Image 31] PASS | Result=1 Expected=1
        [TB] Progress: pass=31 / total=32
        ------------------------------------------------------------
        [TB] Loading batch: 32..35
        ------------------------------------------------------------
        [TB] Batch base=32 DONE t=624105000
        [TB] [Image 32] PASS | Result=3 Expected=3
        [TB] [Image 33] PASS | Result=4 Expected=4
        [TB] [Image 34] PASS | Result=7 Expected=7
        [TB] [Image 35] PASS | Result=2 Expected=2
        [TB] Progress: pass=35 / total=36
        ------------------------------------------------------------
        [TB] Loading batch: 36..39
        ------------------------------------------------------------
        [TB] Batch base=36 DONE t=693445000
        [TB] [Image 36] PASS | Result=7 Expected=7
        [TB] [Image 37] PASS | Result=1 Expected=1
        [TB] [Image 38] PASS | Result=2 Expected=2
        [TB] [Image 39] PASS | Result=1 Expected=1
        [TB] Progress: pass=39 / total=40
        ------------------------------------------------------------
        [TB] Loading batch: 40..43
        ------------------------------------------------------------
        [TB] Batch base=40 DONE t=762785000
        [TB] [Image 40] PASS | Result=1 Expected=1
        [TB] [Image 41] PASS | Result=7 Expected=7
        [TB] [Image 42] PASS | Result=4 Expected=4
        [TB] [Image 43] PASS | Result=2 Expected=2
        [TB] Progress: pass=43 / total=44
        ------------------------------------------------------------
        [TB] Loading batch: 44..47
        ------------------------------------------------------------
        [TB] Batch base=44 DONE t=832125000
        [TB] [Image 44] PASS | Result=3 Expected=3
        [TB] [Image 45] PASS | Result=5 Expected=5
        [TB] [Image 46] PASS | Result=1 Expected=1
        [TB] [Image 47] PASS | Result=2 Expected=2
        [TB] Progress: pass=47 / total=48
        ------------------------------------------------------------
        [TB] Loading batch: 48..51
        ------------------------------------------------------------
        [TB] Batch base=48 DONE t=901465000
        [TB] [Image 48] PASS | Result=4 Expected=4
        [TB] [Image 49] PASS | Result=4 Expected=4
        [TB] [Image 50] PASS | Result=6 Expected=6
        [TB] [Image 51] PASS | Result=3 Expected=3
        [TB] Progress: pass=51 / total=52
        --------------------------
        ------------------------------------------------------------
        [TB] Loading batch: 52..55
        ------------------------------------------------------------
        [TB] Batch base=52 DONE t=970805000
        [TB] [Image 52] PASS | Result=5 Expected=5
        [TB] [Image 53] PASS | Result=5 Expected=5
        [TB] [Image 54] PASS | Result=6 Expected=6
        [TB] [Image 55] PASS | Result=0 Expected=0
        [TB] Progress: pass=55 / total=56
        ------------------------------------------------------------
        [TB] Loading batch: 56..59
        ------------------------------------------------------------
        [TB] Batch base=56 DONE t=1040145000
        [TB] [Image 56] PASS | Result=4 Expected=4
        [TB] [Image 57] PASS | Result=1 Expected=1
        [TB] [Image 58] PASS | Result=9 Expected=9
        [TB] [Image 59] PASS | Result=5 Expected=5
        [TB] Progress: pass=59 / total=60
        ------------------------------------------------------------
        [TB] Loading batch: 60..63
        ------------------------------------------------------------
        [TB] Batch base=60 DONE t=1109485000
        [TB] [Image 60] PASS | Result=7 Expected=7
        [TB] [Image 61] PASS | Result=8 Expected=8
        [TB] [Image 62] PASS | Result=9 Expected=9
        [TB] [Image 63] PASS | Result=3 Expected=3
        [TB] Progress: pass=63 / total=64
        ------------------------------------------------------------
        [TB] Loading batch: 64..67
        ------------------------------------------------------------
        [TB] Batch base=64 DONE t=1178825000
        [TB] [Image 64] PASS | Result=7 Expected=7
        [TB] [Image 65] PASS | Result=4 Expected=4
        [TB] [Image 66] PASS | Result=6 Expected=6
        [TB] [Image 67] PASS | Result=4 Expected=4
        [TB] Progress: pass=67 / total=68
        ------------------------------------------------------------
        [TB] Loading batch: 68..71
        ------------------------------------------------------------
        [TB] Batch base=68 DONE t=1248165000
        [TB] [Image 68] PASS | Result=3 Expected=3
        [TB] [Image 69] PASS | Result=0 Expected=0
        [TB] [Image 70] PASS | Result=7 Expected=7
        [TB] [Image 71] PASS | Result=0 Expected=0
        [TB] Progress: pass=71 / total=72
        ------------------------------------------------------------
        [TB] Loading batch: 72..75
        ------------------------------------------------------------
        [TB] Batch base=72 DONE t=1317505000
        [TB] [Image 72] PASS | Result=2 Expected=2
        [TB] [Image 73] PASS | Result=9 Expected=9
        [TB] [Image 74] PASS | Result=1 Expected=1
        [TB] [Image 75] PASS | Result=7 Expected=7
        [TB] Progress: pass=75 / total=76
        ------------------------------------------------------------
        [TB] Loading batch: 76..79
        ------------------------------------------------------------
        [TB] Batch base=76 DONE t=1386845000
        [TB] [Image 76] PASS | Result=3 Expected=3
        [TB] [Image 77] PASS | Result=2 Expected=2
        [TB] [Image 78] PASS | Result=9 Expected=9
        [TB] [Image 79] FAIL | Result=3 Expected=7
        [TB] Progress: pass=78 / total=80
        ------------------------------------------------------------
        [TB] Loading batch: 80..83
        ------------------------------------------------------------
        [TB] Batch base=80 DONE t=1456185000
        [TB] [Image 80] PASS | Result=7 Expected=7
        [TB] [Image 81] PASS | Result=6 Expected=6
        [TB] [Image 82] PASS | Result=2 Expected=2
        [TB] [Image 83] PASS | Result=7 Expected=7
        [TB] Progress: pass=82 / total=84
        ------------------------------------------------------------
        [TB] Loading batch: 84..87
        ------------------------------------------------------------
        [TB] Batch base=84 DONE t=1525525000
        [TB] [Image 84] PASS | Result=8 Expected=8
        [TB] [Image 85] PASS | Result=4 Expected=4
        [TB] [Image 86] PASS | Result=7 Expected=7
        [TB] [Image 87] PASS | Result=3 Expected=3
        [TB] Progress: pass=86 / total=88
        ------------------------------------------------------------
        [TB] Loading batch: 88..91
        ------------------------------------------------------------
        [TB] Batch base=88 DONE t=1594865000
        [TB] [Image 88] PASS | Result=6 Expected=6
        [TB] [Image 89] PASS | Result=1 Expected=1
        [TB] [Image 90] PASS | Result=3 Expected=3
        [TB] [Image 91] PASS | Result=6 Expected=6
        [TB] Progress: pass=90 / total=92
        ------------------------------------------------------------
        [TB] Loading batch: 92..95
        ------------------------------------------------------------
        [TB] Batch base=92 DONE t=1664205000
        [TB] [Image 92] PASS | Result=9 Expected=9
        [TB] [Image 93] PASS | Result=3 Expected=3
        [TB] [Image 94] PASS | Result=1 Expected=1
        [TB] [Image 95] PASS | Result=4 Expected=4
        [TB] Progress: pass=94 / total=96
        ------------------------------------------------------------
        [TB] Loading batch: 96..99
        ------------------------------------------------------------
        [TB] Batch base=96 DONE t=1733545000
        [TB] [Image 96] PASS | Result=1 Expected=1
        [TB] [Image 97] PASS | Result=7 Expected=7
        [TB] [Image 98] PASS | Result=6 Expected=6
        [TB] [Image 99] PASS | Result=9 Expected=9
        [TB] Progress: pass=98 / total=100
        ------------------------------------------------------------
        
        ============================================================
        [TB] ALL DONE. PASS=98 / TOTAL=100 (FAIL=2)
        ============================================================
        
검증 결과, 100개 sample 기준 inference accuracy는 98%로 측정되었습니다. Reference Model의 accuracy 99% 대비 1% 낮지만, 전체적으로 정상적으로 inference 동작이 수행됨을 확인하였습니다.

### Ver.2 설계 한계 및 개선 방향

1. 기능 확장 과정에서의 구조 복잡도 증가

초기 v1 구조를 기반으로 기능을 확장하는 과정에서 weight pipe, row pipe, enable control, latency compensation logic 등이 단계적으로 추가되었고, 그 결과 설계 구조가 점진적으로 복잡해졌습니다.

​이로 인해 data path와 control path가 여러 module에 분산되고 signal flow tracing이 어려워졌으며 debug 및 change impact 분석 복잡도가 증가하는 문제가 발생하였습니다.

​구조 복잡도가 일정 수준을 넘어서면서 설계 안정성이 저하된다고 판단하였고, 한 차례 Roll-back을 수행하여 Ver.1으로 복원한 뒤 구조를 재정비한 후 다시 확장을 진행하였습니다.

​이와 같은 상황은 실무에서도 빈번하게 발생합니다. 대부분의 개발은 clean-slate 설계가 아니라 legacy RTL 기반의 확장 형태로 이루어지며, 기존 구조의 안정성을 유지하기 위해 wrapper 또는 추가 logic 형태로 기능을 덧붙이는 방식이 일반적입니다.

이는  단기 안정성 측면에서는 유리하지만, 기능이 누적될수록 patch 형태의 코드가 증가하여 구조적 복잡도 상승 문제가 발생합니다.

​따라서 실무에서는 일정 수준 이상의 기능이 축적되면 부분 refactoring을 통해 data path와 control path를 재정렬하고, latency를 명시적으로 정의하여 구조를 재정비한다고 합니다. 

​즉, 기능 추가 → 안정화 → 구조 정리(refactoring) 의 반복 사이클로 관리합니다.

​이번 프로젝트에서도 Ver.2 단계에서 구조 복잡도를 인지하였고, signal 흐름을 단순화하고 latency 정렬을 명시적으로 재구성하는 방향으로 일부 구조를 재정의하였습니다.

​다만 module 간 의존성이 완전히 분리되지는 않았으며, 향후에는 control centralized 구조로 재정렬하여 구조 단순화를 추가로 진행할 예정입니다.

### What’s Next

이번에는 mini-NPU를 구현하면서 겪었던 debug 과정과 구조 안정화 작업을 중심으로 정리해보았습니다. 다음에는 조금 더 확장된 관점에서 설계를 살펴보려고 합니다.

1. 먼저 Ver.2 RTL을 동작 흐름과 제어 신호 기준으로 다시 정리해보겠습니다. state 전이와 data propagation이 실제로 어떻게 연결되는지, 구조 관점에서 차근히 풀어볼 예정입니다.

2. 이후에는 Sample 10,000개 기반 정량 분석을 통해 Reference Model 대비 accuracy 차이를 비교하고, 오차가 발생하는 지점을 분석하여 개선 방향을 정리해보겠습니다.

3. 또한 실제 Synthesis / Implementation 결과를 기반으로 PPA(Area / Timing / Power)를 확인하고, 구조 변경이 물리적 특성에 어떤 영향을 주었는지도 함께 살펴볼 예정입니다.

4. 마지막으로 통신 Protocol을 추가하여 외부 입력 데이터를 수신하고, 이를 기반으로 inference를 수행하는 구조까지 확장해볼 계획입니다.

이 과정을 통해 이번 설계가 단순히 “동작하는 RTL” 수준을 넘어 정량적으로 어느 정도 개선되었는지를 직접 검증해보겠습니다. (mini-NPU 설계가 완료!!)
