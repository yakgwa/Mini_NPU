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

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_122.png" width="400"/>

<div align="left">

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

### Proposed Model v.2 - 4.1th w_pipe — NPU_Top으로 이동

w_pipe 로직을 Weight_Bank에서 제거하고 NPU_Top 내부에서 직접 관리하도록 재구성하였습니다. 이를 통해 Weight_Bank는 weight 값 출력만 담당하고, skew 로직은 상위 모듈에서 통합 제어하는 구조로 정리하였습니다.

    // (10) wavefront: weight pipe — NPU_Top 내부
    wire signed [dataWidth-1:0] w_lane [3:0];
    
    assign w_lane[0] = w_bank_out[1*dataWidth-1 : 0*dataWidth];
    assign w_lane[1] = w_bank_out[2*dataWidth-1 : 1*dataWidth];
    assign w_lane[2] = w_bank_out[3*dataWidth-1 : 2*dataWidth];
    assign w_lane[3] = w_bank_out[4*dataWidth-1 : 3*dataWidth];
    
    reg  signed [dataWidth-1:0] w_pipe [0:3][0:3];
    wire signed [4*dataWidth-1:0] w_wavefront_packed;
    
    assign w_wavefront_packed = {
        w_pipe[3][3],
        w_pipe[2][2],
        w_pipe[1][1],
        w_pipe[0][0]
    };
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (lane = 0; lane < 4; lane = lane + 1)
                for (st = 0; st < 4; st = st + 1)
                    w_pipe[lane][st] <= '0;
        end else begin
            if (pe_rst) begin
                for (lane = 0; lane < 4; lane = lane + 1)
                    for (st = 0; st < 4; st = st + 1)
                        w_pipe[lane][st] <= '0;
            end else if (shift_en) begin
                for (lane = 0; lane < 4; lane = lane + 1) begin
                    for (st = 3; st > 0; st = st - 1)
                        w_pipe[lane][st] <= w_pipe[lane][st-1];
                    w_pipe[lane][0] <= push_en ? w_lane[lane] : '0;
                end
            end
        end
    end

### Proposed Model v.2 - 4.2th PE 단일화 — data_pipe 통합

분리되어 있던 PE와 PE_Systolic_Cell을 다시 단일 PE로 통합하고, MAC 연산과 data_pipe(a_reg, b_reg) 경로를 하나의 모듈에서 처리하도록 정리하였습니다.

    // PE.sv — data_pipe + MAC 통합 구조
    logic signed [dataWidth-1:0] a_reg, b_reg;
    
    // forwarding to adjacent PE
    assign a_out = a_reg;
    assign b_out = b_reg;
    assign mul   = a_reg * b_reg;
    
    always_ff @(posedge clk) begin
        if (rst) begin
            a_reg <= '0;
            b_reg <= '0;
            sum   <= '0;
        end
        else begin
            if (en) begin
                a_reg <= a_in;
                b_reg <= b_in;
            end
            if (clr)      sum <= '0;
            else if (en)  sum <= sat_add_w(sum, mul);
        end
    end

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

worst-case 기준으로 두 skew가 중첩될 경우 최대 6-cycle의 전파 시간이 필요하므로, 마지막 MAC 결과가 fully 반영되도록 6-cycle의 flush 구간을 추가하였습니다.(해당 flush cycle 설정은 이후 10,000개 sample 검증 과정에서 pipeline drain이 불충분한 것으로 확인되었으며, accuracy mismatch의 원인이 되었습니다)

​이를 통해 input 주입이 종료된 이후에도 일정 cycle 동안 state를 유지함으로써, wavefront가 완전히 전파되도록 보장하였습니다.

​또한 OS(Output Stationary) dataflow 특성에 맞도록 기존 roadcast 기반 input 구조에 skew를 적용하여, 각 lane에 데이터가 동시에 인가되지 않도록 하고 시간차를 두고 순차적으로 주입되도록 구조를 수정하였습니다.

​아울러 pipeline 구조 적용에 따라 기존 L2/L3에서 사용하던 buf_r_data_d2는 제거하고 buf_r_data_d1만 사용하도록 변경하여, 불필요한 1-cycle delay를 제거하였습니다.

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

        // 기존 구조 (k_cnt >= 1 조건)
        k_cnt:           0    1    2    3   ...  30
        inject_en:       0    1    1    1   ...   1
        buf_r_data:     [72] [x1] [x2] [x3] ... [x30]  ← GB에서 읽힘
        buf_r_data_d1:   X  [72] [x1] [x2] ... [x29]  ← k_cnt==0 때 저장 안 됨 (누락)
    
        // 수정 후 (k_cnt <= cur_input_len 조건)
        k_cnt:           0    1    2    3   ...  30
        data_valid:      1    1    1    1   ...   1
        buf_r_data:     [72] [x1] [x2] [x3] ... [x30]
        buf_r_data_d1:  [72] [x1] [x2] [x3] ... [x29]  ← k_cnt==0부터 정상 저장

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

6차의 flush cycle은 마지막 데이터가 pipe에 진입한 이후 SA 내부를 전파하는 시간을 확보하는 것입니다.

반면 여기서 다루는 tail 누락은 그보다 앞 단계, 즉 마지막 데이터 자체가 shift_en 조기 종료로 인해 pipe에 진입조차 하지 못하는 문제입니다.

        [ 6차 flush cycle 문제 ]
        k_cnt:     ...  29   30   31   32   33   34      ← calc_end_k = 30+1+6 = 37
        shift_en:  ...   1    0    0    0    0    0      ← k_cnt>30에서 꺼짐
        pipe:      ... [x29][---][---][---][---][---]   ← 마지막 데이터 pipe 진입 후
        SA 전파:   ...                 [x29 전파 중]    ← flush 없으면 state 전환으로 차단

        [ 7.2 shift window 조기 종료 문제 ]
        lane:       0    1    2    3
        skew:      +0   +1   +2   +3  cycle

        k_cnt=30 (cur_input_len):
          lane0 마지막 데이터 → k_cnt=30에 pipe 진입 ✓
          lane1 마지막 데이터 → k_cnt=31에 pipe 진입 필요
          lane2 마지막 데이터 → k_cnt=32에 pipe 진입 필요
          lane3 마지막 데이터 → k_cnt=33에 pipe 진입 필요

        shift_en (기존): k_cnt <= cur_input_len(30) 까지만 유지
          → lane1~3 마지막 데이터: pipe 진입 전에 shift 중단 ✗ (tail 누락)

        shift_en (수정): k_cnt <= cur_input_len + MAX_LANE_SKEW(4) 까지 유지
          → lane0~3 마지막 데이터: 모두 정상 pipe 진입 ✓

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

        k_cnt:      29   30   31   32   33
        shift_en:    1    1    1    1    1   ← tail 구간도 유지
        push_en:     1    1    0    0    0   ← cur_input_len 이후 차단
        stage0:    [x29][x30][ 0 ][ 0 ][ 0 ] ← bubble 주입
        pipe 전파:       [x30이 SA 끝까지 이동]

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
        
검증 결과, 100개 sample 기준 inference accuracy는 98%로 측정되었습니다. Reference Model의 accuracy 99% 대비 1% 낮지만, 전체적으로 정상적으로 inference 동작이 수행됨을 확인하였습니다.(초기 100개 sample 검증에서는 98%로 Reference(99%) 대비 1% 차이에 그쳐 문제를 인지하지 못하였으나, 이후 10,000개 sample로 검증 범위를 확장하는 과정에서 accuracy mismatch의 원인을 확인하였습니다)

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

### 적용 사항

|모듈|변경 내용|이유|
|------|---|---|
|maxFinder.sv|unsigned → signed 비교 / reset 추가|unsigned 비교 시 음수 값 오작동 가능, reset 누락|
|Sig_ROM.sv|address 변환: 덧셈/뺄셈 → MSB XOR|32-bit 정수 산술로 인한 address 오버플로우 가능성|
|Global_Buffer.sv|async read → sync read / depth 파라미터화|async read는 BRAM 추론 불가(distributed RAM으로 합성됨)|
|PE.sv|제한적|높음|
|Activation_Unit.sv|동일하게 패키지 import|상동|
|sat_arith_pkg.sv|신규 추가 (공용 패키지)|중복 제거 및 유지보수 일원화|
|Systolic_Array.sv|clr 포트 추가/.clr(rst) → .clr(pe_clr) 분리|기존 .clr(rst) 연결로 rst/clr 독립 제어 불가|
|NPU_Top.sv|done_interrupt level → pulse / buf_r_data_d1 제거/pe_clr 독립 구동/OUTPUT_SCAN k_cnt 오프셋 수정|중복 트리거 방지/sync GB 전환 후 이중 지연 제거/clr 분리 반영/sync GB 1-cycle latency 반영|
|NPU_Wrapper.sv|신규 추가|합성 시 IO pin 축소(result 32bit → 4bit 외부 노출, 내부 32bit 유지)|

- maxFinder — signed 비교 수정
  
기존 maxFinder는 output class index 비교 시 unsigned 방식으로 처리하고 있었습니다. 

이 경우 signed 값으로 표현된 activation 출력에서 음수 값이 양수보다 크게 인식되어 오작동할 수 있습니다. 

    // 기존 (unsigned 비교)
    if (inDataBuffer[counter*inputWidth+:inputWidth] > maxValue)
    
    // 수정 (signed 비교)
    if ($signed(inDataBuffer[counter*inputWidth +: inputWidth]) >
        $signed(maxValue))

또한 reset 시 내부 상태가 초기화되지 않는 문제도 함께 수정하였습니다.

    // 수정 — reset 추가
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            o_data_valid <= 1'b0;
            o_data       <= 32'd0;
            maxValue     <= {inputWidth{1'b0}};
            counter      <= 0;
        end
        ...
    end

- Sig_ROM — MSB XOR 방식으로 address 변환

기존 Sig_ROM은 signed input x를 unsigned address로 변환할 때 덧셈/뺄셈 방식을 사용하였습니다. 

이 경우 32-bit 정수 산술 과정에서 inWidth 범위를 초과하는 address가 생성될 수 있었습니다.

    // 기존 (오버플로우 가능)
    if ($signed(x) >= 0)
        y <= x + (2**(inWidth-1));
    else
        y <= x - (2**(inWidth-1));
    
    // 수정 (MSB XOR — overflow 없음)
    y <= x ^ (1 << (inWidth-1));

※ MSB XOR 방식은 부호 비트만 반전하므로 가장 음수(-512) → 0, 0 → 512, 가장 양수(511) → 1023으로 깔끔하게 매핑됩니다.  

- Global_Buffer — sync read 전환 및 depth 파라미터화

기존 Global_Buffer는 async read 방식으로 구현되어 있었습니다. 

Vivado는 async read 메모리를 BRAM이 아닌 distributed RAM으로 추론하여, 불필요하게 많은 LUT를 소모하는 문제가 있었습니다.

    // 기존 (async read — distributed RAM 추론)
    assign r_data_0 = mem[r_addr_0];
    
    // 수정 (sync read — BRAM 추론)
    always @(posedge clk) begin
        r_data_0 <= mem[r_addr_0];
        r_data_1 <= mem[r_addr_1];
        r_data_2 <= mem[r_addr_2];
        r_data_3 <= mem[r_addr_3];
    end

또한 메모리 depth를 파라미터로 분리하여 재사용성을 향상시켰습니다.

    module Global_Buffer #(
        parameter dataWidth = 8,
        parameter DEPTH     = 256   // 파라미터화
    )(
        ...
        input [$clog2(DEPTH)-1:0]  w_addr,
        input [$clog2(DEPTH)-1:0]  r_addr_0,
        ...

※ sync read로 전환 시 1-cycle read latency가 추가됩니다. 

※ 이에 따라 NPU_Top에서 기존에 사용하던 buf_r_data_d1 staging register를 제거하였습니다. 

​※ sync GB가 이미 동등한 1-cycle latency를 제공하므로 이중 지연이 발생하지 않습니다.

- sat_arith_pkg — 공용 패키지 분리

기존 PE.sv와 Activation_Unit.sv에는 동일한 sat_add_w 함수가 각각 별도로 정의되어 있었습니다. 

두 모듈 중 하나를 수정할 경우 다른 모듈에도 동일한 수정이 필요한 유지보수 위험이 존재하였습니다.

    // sat_arith_pkg.sv — 공용 패키지
    package sat_arith_pkg;
        function automatic logic signed [15:0] sat_add_16(
            input logic signed [15:0] x,
            input logic signed [15:0] y
        );
            ...
        endfunction
    endpackage
    
    // PE.sv / Activation_Unit.sv — 패키지 import
    module PE
        import sat_arith_pkg::*;
    ...

※ 현재 sat_add_16은 dataWidth=8 기준 W=16으로 고정되어 있습니다.

※ dataWidth 파라미터를 변경할 경우 패키지 함수도 함께 수정이 필요하며, 이 부분은 향후 개선 사항으로 남겨두었습니다.    

- PE / Systolic_Array — clr/rst 분리

기존 Systolic_Array에서는 PE의 clr 포트에 rst 신호를 직접 연결하고 있었습니다. 

이 구조에서는 reset과 accumulator clear가 항상 동시에 동작하여, 독립적인 clr 제어가 불가능합니다.

    // 기존 (clr/rst 동일 신호)
    PE pe_inst (
        .rst (rst),
        .clr (rst),   // rst와 동일하게 연결
        ...
    );
    
    // 수정 (clr/rst 분리)
    PE pe_inst (
        .rst (rst),
        .clr (clr),   // 별도 신호로 분리
        ...
    );

Systolic_Array에 clr 포트를 별도로 추가하고, NPU_Top에서 pe_clr 신호를 독립적으로 구동하도록 수정하였습니다.

- NPU_Top — done_interrupt pulse화

기존 done_interrupt는 DONE state에 진입한 이후 계속 1을 유지하는 level 신호였습니다. 

이 경우 후단 로직에서 done을 엣지로 인식하지 않으면 중복 트리거가 발생할 수 있습니다.

    // 기존 (level 신호 — DONE state 내내 유지)
    DONE: begin
        done_interrupt <= 1;
    end
    
    // 수정 (1-cycle pulse)
    // FSM 루프 상단에서 매 cycle 리셋
    done_interrupt <= 1'b0;
    ...
    // DONE 진입 시 1-cycle만 assert
    if (k_cnt == 11) begin
        mf_valid_pulse <= 1'b1;
        state          <= DONE;
        done_interrupt <= 1'b1;   // 다음 cycle에 자동으로 0으로 복귀
    end

- NPU_Wrapper — IO pin 축소

합성 시 NPU_Top의 result 포트(32bit × 4 = 128핀)가 FPGA 구현 시 pin 제약 문제를 야기할 수 있었습니다. 

NPU_Wrapper를 추가하여 외부 노출은 4bit로 축소하되, 내부 32bit는 유지하여 향후 AXI 확장에 대비하였습니다.

    // NPU_Wrapper.sv
    module NPU_Wrapper (
        ...
        output logic [3:0]  result_0,   // 외부: 4bit
        output logic [3:0]  result_1,
        output logic [3:0]  result_2,
        output logic [3:0]  result_3
    );
        wire [31:0] full_result_0, full_result_1,
                    full_result_2, full_result_3;
    
        NPU_Top npu ( ... );   // 내부: 32bit 유지
    
        assign result_0 = full_result_0[3:0];
        assign result_1 = full_result_1[3:0];
        assign result_2 = full_result_2[3:0];
        assign result_3 = full_result_3[3:0];
    endmodule

- FLUSH_CYCLES 이론적 근거

최종 코드에서 FLUSH_CYCLES = 8로 설정되어 있습니다. 이 값의 이론적 근거는 다음과 같습니다.   

|구성 요소|최대 지연|설명|
|------|---|---|
|Row (input) skew|3 cycle|lane0~3 중 lane3이 가장 늦게 SA에 도달 (3-cycle skew)|
|Col (weight) skew|3 cycle|w_pipe diagonal tap — lane3이 3-cycle 지연|
|SA 내부 전파|1 cycle|	PE 내부 data_pipe 1-cycle latency|
|여유분|1 cycle|경계 조건 마진

### What’s Next

이번에는 mini-NPU를 구현하면서 겪었던 debug 과정과 구조 안정화 작업을 중심으로 정리해보았습니다. 다음에는 조금 더 확장된 관점에서 설계를 살펴보려고 합니다.

1. 먼저 Ver.2 RTL을 동작 흐름과 제어 신호 기준으로 다시 정리해보겠습니다. state 전이와 data propagation이 실제로 어떻게 연결되는지, 구조 관점에서 차근히 풀어볼 예정입니다.

2. 이후에는 Sample 10,000개 기반 정량 분석을 통해 Reference Model 대비 accuracy 차이를 비교하고, 오차가 발생하는 지점을 분석하여 개선 방향을 정리해보겠습니다.

3. 또한 실제 Synthesis / Implementation 결과를 기반으로 PPA(Area / Timing / Power)를 확인하고, 구조 변경이 물리적 특성에 어떤 영향을 주었는지도 함께 살펴볼 예정입니다.

4. 마지막으로 통신 Protocol을 추가하여 외부 입력 데이터를 수신하고, 이를 기반으로 inference를 수행하는 구조까지 확장해볼 계획입니다.

이 과정을 통해 이번 설계가 단순히 “동작하는 RTL” 수준을 넘어 정량적으로 어느 정도 개선되었는지를 직접 검증해보겠습니다. (mini-NPU 설계가 완료!!)
