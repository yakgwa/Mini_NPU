## Systolic Array 설계_(OS) PE, 1-D Array

이전 글에서는 WS(Weight-Stationary) dataflow 기반으로 설계하였으나, 본 글에서는 OS(Output-Stationary) dataflow를 적용하여 설계를 진행하였습니다.

## 1. PE

### DUT
    module mac_pe #(
      parameter int DATA_W = 8,
      parameter int ACC_W  = 2*DATA_W  
    )(
      input  logic                  clk,
      input  logic                  rst_n,
    
      input  logic                  clr,      // 누산 초기화
      input  logic                  en,       // 연산 enable
    
      input  logic signed [DATA_W-1:0]     a,        // operand 1
      input  logic signed [DATA_W-1:0]     b,        // operand 2
    
      output logic signed [ACC_W-1:0]      mul,      // a*b 결과
      output logic signed [ACC_W-1:0]      acc_sum   // 누산 결과
    );
    
      // 곱셈기: Combi
      always_comb begin                               // always_comb (= assign mul = a * b;)
        mul = a * b;
      end
      
    
      // 누산기: acc_sum <= acc_sum + mul
      
      always_ff @(posedge clk or negedge rst_n) begin // always_ff (= always)
        if (!rst_n) begin
          acc_sum <= '0;
        end
        else begin
          if (clr) begin
            acc_sum <= '0;
          end
          else if (en) begin                         // en=0이면 acc_sum 유지
            acc_sum <= acc_sum + mul;
          end
        end
      end
    
    endmodule

### TB
        `timescale 1ns/1ps
        
        module tb_mac_pe;
        
          localparam int  DATA_W   = 8;
          localparam int  ACC_W    = 2*DATA_W;
          localparam time SIM_TIME = 2000ns;  // 해당 시간동안 run
        
          logic                  clk;
          logic                  rst_n;
          logic                  clr;
          logic                  en;
          logic signed [DATA_W-1:0]     a;
          logic signed [DATA_W-1:0]     b;
          logic signed [ACC_W-1:0]      mul;
          logic signed [ACC_W-1:0]      acc_sum;
        
          mac_pe #(
            .DATA_W (DATA_W),
            .ACC_W  (ACC_W)
          ) dut (
            .clk     (clk),
            .rst_n   (rst_n),
            .clr     (clr),
            .en      (en),
            .a       (a),
            .b       (b),
            .mul     (mul),
            .acc_sum (acc_sum)
          );
        
          //==========================================================
          // Clock / Reset
          //==========================================================
          initial begin
            clk = 1'b0;
            forever #5 clk = ~clk;  // 100MHz
          end
        
          initial begin
          // reset assert 한 뒤, 30 있다가 reset deassert
          // 모든 always_ff logic이 의도한 초기값으로 정렬됨.
            rst_n = 1'b0;
            clr   = 1'b0;
            en    = 1'b0;
            a     = '0;
            b     = '0;
        
            #30;
            rst_n = 1'b1;
          end
        
          //==========================================================
          // Reference Model & Checker (Watchpoint)
          //==========================================================
          logic signed [ACC_W-1:0] ref_mul;
          logic signed [ACC_W-1:0] ref_sum;
        
          int cycles_checked;
          int err_mul_cnt;
          int err_acc_cnt;
        
          // ref model: DUT와 동일한 동작을 testbench 안에서 구현
          assign ref_mul = a * b; //always -> assign (data-mismatch debugging point)
        
          always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
              ref_sum <= '0;
            end else begin
              if (clr) begin
                ref_sum <= '0;
              end else if (en) begin
                ref_sum <= ref_sum + ref_mul;
              end
              // en=0이면 유지
            end
          end
        
          // Checker: DUT vs ref 비교
          always_ff @(posedge clk) begin
            if (!rst_n) begin
              cycles_checked <= 0;
              err_mul_cnt    <= 0;
              err_acc_cnt    <= 0;
            end else begin
              cycles_checked++;
        
              // WP1: mul 비교
              if (mul !== ref_mul) begin
                err_mul_cnt++;
                $display("ERROR! WP1 MUL mismatch: dut=%0d, ref=%0d, time=%0d ns",
                         mul, ref_mul, $time);
              end else begin
                $display("PASS!  WP1 MUL match   : dut=%0d, ref=%0d, time=%0d ns",
                         mul, ref_mul, $time);
              end
        
              // WP2: acc_sum 비교
              if (acc_sum !== ref_sum) begin
                err_acc_cnt++;
                $display("ERROR! WP2 ACC_SUM mismatch: dut=%0d, ref=%0d, time=%0d ns",
                         acc_sum, ref_sum, $time);
              end else begin
                $display("PASS!  WP2 ACC_SUM match   : dut=%0d, ref=%0d, time=%0d ns",
                         acc_sum, ref_sum, $time);
              end
            end
          end
        
          //==========================================================
          // Time-based Random Stimulus
          //==========================================================
          initial begin
            // reset 해제 직후의 불안정한 구간 (일부 F/F가 reset branch 구간) or test stimulus가 안정적으로 시작되도록.
            @(posedge rst_n);
            @(posedge clk);
        
            $display("[TB] Time-based Random Stress START, duration = %0d ns", SIM_TIME);
        
            fork
              // 랜덤 입력 생성 쓰레드
              begin
                forever begin
                  // operand 랜덤
                  // a = $urandom_range(0, (1<<DATA_W)-1); //(1<<8)-1 = 255 (code reuse)
                  // b = $urandom_range(0, (1<<DATA_W)-1);
                  // 기존 unsigned 대신 signed random 값 출력.
                     a = $signed($urandom_range(-(1<<(DATA_W-1)), (1<<(DATA_W-1))-1));
                     b = $signed($urandom_range(-(1<<(DATA_W-1)), (1<<(DATA_W-1))-1));
        
        
                  // en 랜덤
                  en = $urandom_range(0, 1);
        
                  // clr는 낮은 확률로만 발생 (예: 5%)
                  if ($urandom_range(0,99) < 5) begin
                    clr = 1'b1;
                  end else begin
                    clr = 1'b0;
                  end
                  @(posedge clk);
                end
              end
        
              // 시간 제한 쓰레드
              begin
                #SIM_TIME;
                disable fork;  // 위 랜덤 쓰레드 종료
              end
            join_any
        
            // idle 몇 사이클
            en  = 1'b0;
            clr = 1'b0;
            a   = '0;
            b   = '0;
            repeat (5) @(posedge clk);
        
            //========================================================
            // Summary
            //========================================================
            $display("==================================================");
            $display("[TB] Simulation Summary");
            $display("  Simulation time    : %0d ns", SIM_TIME);
            $display("  Cycles checked     : %0d",    cycles_checked);
            $display("  MUL mismatches     : %0d",    err_mul_cnt);
            $display("  ACC_SUM mismatches : %0d",    err_acc_cnt);
            if (err_mul_cnt == 0 && err_acc_cnt == 0) begin
              $display("  RESULT             : PASS");
            end else begin
              $display("  RESULT             : FAIL");
            end
            $display("==================================================");
        
            $finish;
          end
        
        endmodule

### Simulation Result

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_64.png" width="400"/>

<div align="left">

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_65.png" width="200"/>

<div align="left">

<mark>Check Point !!/<mark>
- data width : ACC_W는 기본적으로 DATA_W * 2로 정의합니다. 이는 입력 a와 b의 곱셈 결과 bit width가 입력 bit width의 두 배가 되기 때문입니다. 그러나 곱셈 결과를 여러 번 누산(accumulation)하는 구조에서는 추가적인 bit width 고려가 필요합니다.

    누산 횟수가 증가할수록 중간 합의 최대값이 커지며, 누산 과정에서 overflow가 발생할 수 있습니다. 따라서 실제 연산기 수준의 설계에서는 단순히 ACC_W = DATA_W * 2로는 충분하지 않습니다. 누산 동작을 안정적으로 지원하기 위해 accumulator bit width를 더 크게 설정합니다.

    ​예를 들어 DATA_W = 8인 경우, ACC_W = 16 대신 2의 거듭제곱 단위인 ACC_W = 32로 설정하여 누산 과정에서의 overflow 여유를 확보합니다.

- signed / unsigned : 연산을 수행하는 과정에서 stimulus는 음수 값을 가질 수 있습니다. 이때 signal이 unsigned로 선언되어 있을 경우, 동일한 bit pattern이라도 값이 양수로 해석되어 연산 결과가 의도와 다르게 계산될 수 있습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_66.png" width="400"/>

unsigned 연산 결과

<div align="left">

특히 곱셈 및 누산 연산에서는 signed 해석 여부가 결과에 직접적인 영향을 미치며, unsigned 설정 시 sign bit가 magnitude의 일부로 취급되어 잘못된 연산 결과를 유발할 수 있습니다.

​따라서 입력 operand(a, b)와 곱셈 결과, accumulator는 모두 signed로 선언하여 음수 값에 대한 연산이 올바르게 수행되도록 합니다. 이를 통해 2’s complement 표현을 기반으로 한 곱셈 및 누산 동작을 정확하게 모델링할 수 있습니다.

## 2. 1-D Array

### DUT
        module pe_chain_1d #(
            parameter int DATA_W = 8,
            parameter int ACC_W = 2*DATA_W,
            parameter int NUM_PE = 4
        )(
            input logic                     clk,
            input logic                     rst_n,
        
            input logic                     clr,    
            input logic                     en,
        
            input logic signed [DATA_W-1:0]     in_data,
            output logic signed [DATA_W-1:0]    out_data,
        
            input logic  [DATA_W-1:0]           weight [0:NUM_PE-1],    // [ACC_W-1:0] → 각 원소의 비트 폭 (packed array, 비트 벡터)
                                                                        // [0:NUM_PE-1] → 배열 인덱스 범위 (unpacked array)
            // for monitoring
            output logic signed [ACC_W-1:0]     pe_mul   [0:NUM_PE-1],
            output logic signed [ACC_W-1:0]     pe_acc_sum [0:NUM_PE-1]
        );
            // =============================================================
            // 데이터 파이프라인
            // =============================================================
        
            logic [DATA_W-1:0] data_pipe [0:NUM_PE-1]; // 데이터 파이프라인 레지스터 (SV에서는 logic 사용이 권장됨)
        
            assign out_data = data_pipe[NUM_PE-1]; // 마지막 PE의 출력 데이터를 모니터링용으로 연결
        
            // ** 데이터 파이프라인 레지스터
                // Verilog에서는 i++ 대신 i = i + 1 사용
                // genvar는 generate 블록 내에서만 사용되는 변수
                // for문은 NUM_PE 개수만큼 하드웨어를 복제하여 생성
            generate
                for (genvar i = 0; i < NUM_PE; i++) begin   
                    if (i==0) begin                         // 첫 stage는 이전 stage가 없으므로 (i-1이 불가능) 별도 처리합니다.
                        always_ff @(posedge clk or negedge rst_n) begin
                            if (!rst_n) begin
                                data_pipe[0] <= '0;         // rst_n==0이면 data_pipe[0]를 0으로 초기화
                            end else if (en) begin
                                data_pipe[0] <= in_data;    // rst_n==1이고 en==1이면 data_pipe[0]에 in_data를 샘플링, en==0이면 이전 값을 유지                                    
                            end
                        end
                    end else begin                          // 두 번째 stage부터는 이전 stage의 값을 받아옴
                        always_ff @(posedge clk or negedge rst_n) begin
                            if (!rst_n) begin
                                data_pipe[i] <= '0;
                            end else if (en) begin
                                data_pipe[i] <= data_pipe[i-1];     // rst_n==1이고 en==1이면 data_pipe[i]가 data_pipe[i-1]로 업데이트, en==0이면 hold
                            end
                        end
                    end
                end
            endgenerate
        
            // 각 stage에 mac_pe 인스턴스
            generate
            for (genvar i = 0; i < NUM_PE; i++) begin
                mac_pe #(
                .DATA_W (DATA_W),
                .ACC_W  (ACC_W)
                ) u_mac_pe (
                .clk     (clk),
                .rst_n   (rst_n),
                .clr     (clr),
                .en      (en),
                .a       (data_pipe[i]),
                .b       (weight[i]),
                .mul     (pe_mul[i]),
                .acc_sum (pe_acc_sum[i])
                );
            end
            endgenerate
        endmodule
        
        // 본 블록은 단일 입력 스트림(in_data)을 NUM_PE 단계로 구성된 파이프라인을 통해
        // 클록 단위로 순차 전달하는 역할을 한다. en이 활성화된 경우에만 데이터가
        // 한 stage씩 전진하며, 각 stage의 출력은 대응되는 mac_pe의 입력으로 연결되어
        // 동일한 입력 데이터가 시간차를 두고 각 PE에서 연산되도록 한다.

### TB
        `timescale 1ns/1ps  
        
        module tb_pe_chain_1d;
            localparam int  DATA_W   = 8;       // parameter와 localparam의 차이: parameter는 모듈 인스턴스화 시 변경 가능, localparam은 변경 불가
                                                // DUT에서는 외부에서 설정받기 위해 parameter를 쓰고, TB에서는 그 설정을 고정된 기준으로 사용하기 위해 localparam을 쓴다.
            localparam int  ACC_W    = 2*DATA_W;
            localparam int  NUM_PE   = 4;
            localparam time SIM_TIME = 3000ns;  // 이 시간 동안 랜덤 스트레스
        
            // =========================================================
            // DUT와 연결될 신호 선언
            // =========================================================    
        
            logic                        clk;
            logic                        rst_n;        
            logic                        clr;
            logic                        en;
            logic signed [DATA_W-1:0]    in_data;
            logic signed [DATA_W-1:0]    out_data;
            logic signed [DATA_W-1:0]    weight [0:NUM_PE-1];
        
            logic signed [ACC_W-1:0]     pe_mul   [0:NUM_PE-1];
            logic signed [ACC_W-1:0]     pe_acc_sum [0:NUM_PE-1];
        
            pe_chain_1d #(
                .DATA_W (DATA_W),
                .ACC_W  (ACC_W),
                .NUM_PE (NUM_PE)
            ) dut (
                .clk        (clk),
                .rst_n      (rst_n),
                .clr        (clr),
                .en         (en),
                .in_data    (in_data),
                .out_data   (out_data),
                .weight     (weight),
                .pe_mul     (pe_mul),
                .pe_acc_sum (pe_acc_sum)
            );
        
            // =========================================================
            // Clock / Reset Generation
            // =========================================================
            initial begin
                clk = 0;
                forever #5 clk = ~clk; // 10ns 주기 클럭 생성
            end
        
            initial begin
                rst_n = 0;
                clr   = 0;
                en    = 0;     
                in_data = '0;
        
                // weight 초기화
                // 예: weight[0]=1, weight[1]=2, weight[2]=3, weight[3]=4
                // initial의 0 초기화는 시뮬레이션 안정화용이고, 
                // weight를 non-zero로 설정하는 것은 연산 경로와 PE 동작을 검증하기 위함이다.
                for (int wi = 0; wi < NUM_PE; wi++) begin
                    weight[wi] = wi + 1;    
        
                end 
        
                #30;            // 시뮬레이션 시작 후 30ns 동안 rst_n=0 유지
                rst_n = 1;      // 그 후 rst_n=1로 해제
            end
        
            //==========================================================
            // Reference Model (DUT와 1:1 정합)
            // - ref_data_pipe: DUT data_pipe와 동일하게 en==1일 때만 전진
            // - ref_sum: reset/clr/en 우선순위 동일
            //==========================================================
            logic signed [DATA_W-1:0] ref_data_pipe [0:NUM_PE-1];
            logic signed [ACC_W-1:0]  ref_sum       [0:NUM_PE-1];
        
            // Data pipe (DUT와 동일하게 구현)
            generate
                for (genvar gi = 0; gi < NUM_PE; gi++) begin : gen_ref_pipe
                    if (gi == 0) begin
                        always_ff @(posedge clk or negedge rst_n) begin
                        if (!rst_n) begin
                            ref_data_pipe[0] <= '0;
                        end else if (en) begin
                            ref_data_pipe[0] <= in_data;
                        end
                    end
                    end else begin
                        always_ff @(posedge clk or negedge rst_n) begin
                            if (!rst_n) begin
                                ref_data_pipe[gi] <= '0;
                            end else if (en) begin
                                ref_data_pipe[gi] <= ref_data_pipe[gi-1];
                            end
                        end
                    end
                end
            endgenerate
        
            // Accumulator reference (reset > clr > en > hold)
            // DUT의 PE는 내부 accumulator에 값을 저장·누산하지만, 이는 검증 대상이므로 TB에서 직접 참조하지 않는다.
            // Ref model은 DUT 내부 상태에 의존하지 않고, 관측 가능한 입력 및 제어 신호를 기반으로
            // 누산 동작을 독립적으로 재현하여 기대값(expected result)을 생성한다.
            // 생성된 기대값은 DUT 출력과의 비교 검증에 사용된다.
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int i = 0; i < NUM_PE; i++) begin
                        ref_sum[i] <= '0;
                    end
                end else if (clr) begin
                    for (int i = 0; i < NUM_PE; i++) begin
                        ref_sum[i] <= '0;
                    end
                end else if (en) begin
                    for (int i = 0; i < NUM_PE; i++) begin
                        ref_sum[i] <= ref_sum[i] + (ref_data_pipe[i] * weight[i]);
                    end
                end
            end
        
            //==========================================================
            // Expected MUL (WP1)
            // - mul이 조합이면 ref_mul_comb
            // - mul이 레지스터면 ref_mul_q (en==1에서만 업데이트/hold)
            //==========================================================
            logic signed [ACC_W-1:0] ref_mul_comb [0:NUM_PE-1];
            logic signed [ACC_W-1:0] ref_mul_q    [0:NUM_PE-1];
        
            // combi logical mul
            generate
                for (genvar gi = 0; gi < NUM_PE; gi++) begin : gen_ref_mul
                always_comb begin
                    ref_mul_comb[gi] = ref_data_pipe[gi] * weight[gi];
                end
                end
            endgenerate

         // registered mul, 1 cycle delay
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int i = 0; i < NUM_PE; i++) begin
                        ref_mul_q[i] <= '0;
                    end
                end else if (en) begin
                    for (int i = 0; i < NUM_PE; i++) begin
                        ref_mul_q[i] <= ref_data_pipe[i] * weight[i];
                    end
                end
            end
        
            //==========================================================
            // Checker / Scoreboard
            //==========================================================
            int cycles_checked;
            int err_mul_cnt;
            int err_acc_cnt;
        
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                cycles_checked <= 0;
                err_mul_cnt    <= 0;
                err_acc_cnt    <= 0;
                end else begin
                cycles_checked <= cycles_checked + 1;
        
                for (int i = 0; i < NUM_PE; i++) begin
                    // WP1: mul 비교
                    if (pe_mul[i] !== exp_mul(i)) begin
                    err_mul_cnt <= err_mul_cnt + 1;
                    $display("ERROR! [PE%0d] MUL mismatch: dut=%0d (0x%0h), ref=%0d (0x%0h), data=%0d, w=%0d, en=%0b, clr=%0b, t=%0t",
                            i, pe_mul[i], pe_mul[i], exp_mul(i), exp_mul(i), ref_data_pipe[i], weight[i], en, clr, $time);
                    end
        
                    // WP2: acc_sum 비교
                    if (pe_acc_sum[i] !== ref_sum[i]) begin
                    err_acc_cnt <= err_acc_cnt + 1;
                    $display("ERROR! [PE%0d] ACC mismatch: dut=%0d (0x%0h), ref=%0d (0x%0h), en=%0b, clr=%0b, t=%0t",
                            i, pe_acc_sum[i], pe_acc_sum[i], ref_sum[i], ref_sum[i], en, clr, $time);
                    end
                end
                end
            end
        
            //==========================================================
            // Random Stimulus
            //==========================================================
            initial begin
                @(posedge rst_n);
                @(posedge clk);
        
                $display("[TB] 1D PE Chain Random Stress START, duration=%0d ns, MUL_IS_REGISTERED=%0b", SIM_TIME, MUL_IS_REGISTERED);
        
                fork
                // 랜덤 입력 생성
                begin
                    forever begin
                    // Signed random input over DATA_W-bit 2's complement range
                    // [-(1<<(DATA_W-1)) , (1<<(DATA_W-1))-1] == [-2^(DATA_W-1) , 2^(DATA_W-1)-1]
                    in_data = signed'($urandom_range(-(1<<(DATA_W-1)), (1<<(DATA_W-1))-1));
        
                    // en 랜덤
                    en = $urandom_range(0, 1);
        
                    // clr: 5% 확률
                    clr = ($urandom_range(0,99) < 5);
        
                    @(posedge clk);
                    end
                end
        
                // 시간 제한
                begin
                    #SIM_TIME;
                    disable fork;
                end
                join_any
        
                // Summary
                $display("==================================================");
                $display("[TB] 1D PE Chain Simulation Summary");
                $display("  Simulation time    : %0d ns", SIM_TIME);
                $display("  Cycles checked     : %0d",    cycles_checked);
                $display("  MUL mismatches     : %0d",    err_mul_cnt);
                $display("  ACC mismatches     : %0d",    err_acc_cnt);
                $display("  RESULT             : %s", (err_mul_cnt==0 && err_acc_cnt==0) ? "PASS" : "FAIL");
                $display("==================================================");
        
                $finish;
            end
        
        endmodule

### Simulation Result

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_67.png" width="400"/>

<div align="left">

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_68.png" width="400"/>

<div align="left">

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_69.png" width="200"/>

<div align="left">

<mark>Check Point !!/<mark>
- NUM_PE : 1-D PE chain 구성부터는 NUM_PE 파라미터가 추가됩니다. NUM_PE는 Chain에 포함되는 PE의 개수를 의미합니다. 이후 generate 문 내부의 for 문을 통해 NUM_PE 값만큼 PE instance를 복제하여 1-D 구조를 구성합니다.
- data_pipe : data_pipe[0:NUM_PE-1]는 입력 in_data를 clock 단위로 한 stage씩 delay시키기 위한 pipeline register 배열입니다. en=1일 때만 pipeline이 동작하며, 매 clock마다 data_pipe[0]에는 새로운 입력이 저장되고, data_pipe[i]에는 이전 stage의 값(data_pipe[i-1])이 전달됩니다. en=0이면 모든 stage는 현재 값을 유지하여 데이터 이동이 정지됩니다. 이 구조로 인해 동일한 input stream이 시간차를 두고 각 PE에 도달하며, data_pipe[i]가 mac_pe의 입력 a에 연결되어 각 PE는 서로 다른 시점의 입력 데이터를 처리하게 됩니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_70.png" width="200"/>

<div align="left">







