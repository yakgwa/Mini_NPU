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
- data width: ACC_W는 기본적으로 DATA_W * 2로 정의합니다. 이는 입력 a와 b의 곱셈 결과 bit width가 입력 bit width의 두 배가 되기 때문입니다. 그러나 곱셈 결과를 여러 번 누산(accumulation)하는 구조에서는 추가적인 bit width 고려가 필요합니다.

    누산 횟수가 증가할수록 중간 합의 최대값이 커지며, 누산 과정에서 overflow가 발생할 수 있습니다. 따라서 실제 연산기 수준의 설계에서는 단순히 ACC_W = DATA_W * 2로는 충분하지 않습니다. 누산 동작을 안정적으로 지원하기 위해 accumulator bit width를 더 크게 설정합니다.

    ​예를 들어 DATA_W = 8인 경우, ACC_W = 16 대신 2의 거듭제곱 단위인 ACC_W = 32로 설정하여 누산 과정에서의 overflow 여유를 확보합니다.

- signed / unsigned:

    연산을 수행하는 과정에서 stimulus는 음수 값을 가질 수 있습니다. 이때 signal이 unsigned로 선언되어 있을 경우, 동일한 bit pattern이라도 값이 양수로 해석되어 연산 결과가 의도와 다르게 계산될 수 있습니다.












