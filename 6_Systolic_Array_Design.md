## Systolic Array 설계_(WS) PE, 1-D Array

앞에서 Dataflow에 따른 연산 개념에 대한 정리를 진행하였고, 이를 바탕으로 이제부터 Verilog HDL을 사용하여 각 구성 요소를 단계적으로 설계해보겠습니다.

## 1. PE

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_59.png" width="400"/>

PE Weight Stationary

<div align="left">

### DUT
    module ws_pe_cell #(
      parameter int DATA_W = 8,
      parameter int ACC_W  = 2*DATA_W
    )(
      input  logic                 clk,
      input  logic                 rst_n,
    
      input  logic                 en,       // run enable (stall)
      input  logic                 w_load,   // weight update
      input  logic [DATA_W-1:0]    w_in,
    
      input  logic [DATA_W-1:0]    a_in,
      output logic [DATA_W-1:0]    a_out,
    
      input  logic [ACC_W-1:0]     psum_in,  // unused (1D)
      output logic [ACC_W-1:0]     psum_out, // product
    
      output logic [DATA_W-1:0]    w_reg_mon,
      output logic [ACC_W-1:0]     mul_mon
    );
    
      logic [DATA_W-1:0] w_reg;
      logic [DATA_W-1:0] w_eff;
      logic [ACC_W-1:0]  mul_next;
    
      assign w_reg_mon = w_reg;
    
      // weight bypass (alignment 핵심)
      assign w_eff    = (w_load) ? w_in : w_reg;
      assign mul_next = ACC_W'(a_in) * ACC_W'(w_eff);
      assign mul_mon  = mul_next;
    
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          w_reg    <= '0;
          a_out    <= '0;
          psum_out <= '0;
        end else if (en) begin
          if (w_load) w_reg <= w_in;
          a_out    <= a_in;
          psum_out <= mul_next; // product only
        end
      end
    
    endmodule

### Testbench
        `timescale 1ns/1ps
        module tb_ws_pe_cell;
        
          localparam int DATA_W = 8;
          localparam int ACC_W  = 32;
        
          logic clk, rst_n;
          logic en;
          logic w_load;
          logic [DATA_W-1:0] a_in, w_in;
          logic [DATA_W-1:0] a_out;
          logic [ACC_W-1:0]  psum_out;
          logic [DATA_W-1:0] w_reg_mon;
          logic [ACC_W-1:0]  mul_mon;
        
          logic [ACC_W-1:0] psum_in;
          assign psum_in = '0;
        
          ws_pe_cell #(
            .DATA_W(DATA_W),
            .ACC_W (ACC_W)
          ) dut (
            .clk      (clk),
            .rst_n    (rst_n),
            .en       (en),
            .w_load   (w_load),
            .w_in     (w_in),
            .a_in     (a_in),
            .a_out    (a_out),
            .psum_in  (psum_in),
            .psum_out (psum_out),
            .w_reg_mon(w_reg_mon),
            .mul_mon  (mul_mon)
          );
        
          initial clk = 0;
          always #5 clk = ~clk;
        
          function automatic int calc_mul(input int a, input int w);
            return a * w;
          endfunction
        
          int exp_w_reg;
          int exp_a_out;
        
          // *** 1-cycle delayed expected product ***
          int exp_pprod_pipe;   // what should appear on DUT outputs THIS cycle
          int exp_pprod_next;   // computed from inputs of THIS cycle, will appear NEXT cycle
        
          task automatic step_and_check_1cyc_delay(
            input int t,
            input bit i_en,
            input bit i_w_load,
            input int i_a,
            input int i_w
          );
            int w_eff;
        
            // drive inputs (sampled at next posedge)
            en     = i_en;
            w_load = i_w_load;
            a_in   = DATA_W'(i_a);
            w_in   = DATA_W'(i_w);
        
            // compute "next" expected product from current inputs
            w_eff = (i_w_load) ? i_w : exp_w_reg;
        
            if (i_en) begin
              exp_a_out     = i_a;
              exp_pprod_next= calc_mul(i_a, w_eff);
              if (i_w_load) exp_w_reg = i_w;
            end else begin
              // stall이면 DUT hold => 다음에도 이전 값 유지
              exp_a_out      = exp_a_out;
              exp_pprod_next = exp_pprod_pipe; // 다음에도 같은 값이 유지된다고 모델링
              exp_w_reg      = exp_w_reg;
            end
        
            @(posedge clk);
            #1;
        
            // a_out / w_reg는 DUT가 en일 때 갱신되는 레지스터이므로 "현재 사이클 관측값"과 비교
            if (a_out !== DATA_W'(exp_a_out)) begin
              $display("[FAIL][t=%0d] a_out got=%0d exp=%0d", t, a_out, exp_a_out);
              $fatal;
            end
        
            if (w_reg_mon !== DATA_W'(exp_w_reg)) begin
              $display("[FAIL][t=%0d] w_reg got=%0d exp=%0d", t, w_reg_mon, exp_w_reg);
              $fatal;
            end
        
            // *** product outputs are checked against 1-cycle delayed expectation ***
            if (psum_out !== ACC_W'(exp_pprod_pipe)) begin
              $display("[FAIL][t=%0d] psum_out got=%0d exp(delayed)=%0d", t, psum_out, exp_pprod_pipe);
              $fatal;
            end
        
            if (mul_mon !== ACC_W'(exp_pprod_pipe)) begin
              $display("[FAIL][t=%0d] mul_mon got=%0d exp(delayed)=%0d", t, mul_mon, exp_pprod_pipe);
              $fatal;
            end
        
            // shift pipeline for next cycle check
            exp_pprod_pipe = exp_pprod_next;
        
          endtask
        
          initial begin
            rst_n   = 0;
            en      = 0;
            w_load  = 0;
            a_in    = '0;
            w_in    = '0;
        
            exp_w_reg      = 0;
            exp_a_out      = 0;
            exp_pprod_pipe = 0;
            exp_pprod_next = 0;
        
            repeat (3) @(posedge clk);
            rst_n = 1;
        
            // 초기 한 사이클은 delayed check 때문에 "0이 나오는 것"이 정상
            step_and_check_1cyc_delay(0, 1, 1, 3, 7);   // next=21, but now expects 0
            step_and_check_1cyc_delay(1, 1, 0, 4, 99);  // now expects 21
            step_and_check_1cyc_delay(2, 1, 1, 5, 2);   // now expects 28
            step_and_check_1cyc_delay(3, 1, 0, 6, 123); // now expects 10
            step_and_check_1cyc_delay(4, 0, 1, 9, 8);   // stall: expects hold(12)
            step_and_check_1cyc_delay(5, 1, 0, 7, 0);   // expects hold(12) then next=14
        
            $display("[PASS] ws_pe_cell 1-cycle-delayed product check passed.");
            $finish;
          end
        
        endmodule

### Result

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_60.png" width="400"/>

<div align="left">

Simulation 결과 및 waveform 분석을 통해, 입력된 activation과 preload된 weight가 정상적으로 곱셈 연산을 수행하며, 해당 결과가 상단에서 전달된 partial sum(임의의 초기값)에 누적되어 기대한 출력이 생성됨을 확인하였습니다.

## 2.1-Dimension 1×4 Systolic Array

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_61.png" width="400"/>

<div align="left">

앞서 분석한 1-D 1×4 Weight-Stationary(WS) Systolic Array를 구현하였습니다. 본 구조는 앞서 설계한 4개의 Processing Element(PE)로 구성되며, 각 연산 단계에서 생성되는 partial sum을 저장하기 위한 accumulator를 포함합니다.

### DUT
- ws_pe_cell.sv
- systolic_array_1x4_ws.sw
        module systolic_1x4_chain #(
          parameter int DATA_W  = 8,
          parameter int ACC_W   = 32,
          parameter int NUM_COL = 4
        )(
          input  logic                 clk,
          input  logic                 rst_n,
          input  logic                 en,
        
          input  logic [DATA_W-1:0]    a_in,
        
          input  logic                 w_load [0:NUM_COL-1],
          input  logic [DATA_W-1:0]    w_in   [0:NUM_COL-1],
        
          output logic [DATA_W-1:0]    a_pipe    [0:NUM_COL-1],
          output logic [DATA_W-1:0]    w_reg_mon [0:NUM_COL-1],
          output logic [ACC_W-1:0]     pprod     [0:NUM_COL-1]
        );
        
          genvar j;
          generate
            for (j = 0; j < NUM_COL; j++) begin : GEN_COL
              logic [DATA_W-1:0] a_in_j;
        
              if (j == 0) assign a_in_j = a_in;
              else        assign a_in_j = a_pipe[j-1];
        
              ws_pe_cell #(
                .DATA_W (DATA_W),
                .ACC_W  (ACC_W)
              ) u_pe (
                .clk       (clk),
                .rst_n     (rst_n),
                .en        (en),
                .w_load    (w_load[j]),
                .w_in      (w_in[j]),
                .a_in      (a_in_j),
                .a_out     (a_pipe[j]),
                .psum_in   ('0),
                .psum_out  (pprod[j]),
                .w_reg_mon (w_reg_mon[j]),
                .mul_mon   ()
              );
            end
          endgenerate
        
        endmodule

- col_accumulators.sv
        module col_accumulators #(
          parameter int ACC_W   = 32,
          parameter int NUM_COL = 4
        )(
          input  logic                 clk,
          input  logic                 rst_n,
          input  logic                 en,
          input  logic                 clr,
        
          input  logic [ACC_W-1:0]     pprod [0:NUM_COL-1],
          output logic [ACC_W-1:0]     c_out [0:NUM_COL-1]
        );
        
          genvar j;
          generate
            for (j = 0; j < NUM_COL; j++) begin : GEN_ACC
              always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)      c_out[j] <= '0;
                else if (clr)    c_out[j] <= '0;
                else if (en)     c_out[j] <= c_out[j] + pprod[j];
              end
            end
          endgenerate
        
        endmodule

- gemv_1x4_extacc_top.sv
        module gemv_1x4_ws_top #(
          parameter int DATA_W  = 8,
          parameter int ACC_W   = 32,
          parameter int NUM_COL = 4
        )(
          input  logic                 clk,
          input  logic                 rst_n,
          input  logic                 en,
          input  logic                 clr,
        
          input  logic [DATA_W-1:0]    a_in,
        
          input  logic                 w_load [0:NUM_COL-1],
          input  logic [DATA_W-1:0]    w_in   [0:NUM_COL-1],
        
          output logic [ACC_W-1:0]     c_out  [0:NUM_COL-1],
        
          // monitors
          output logic [ACC_W-1:0]     pprod  [0:NUM_COL-1],
          output logic [DATA_W-1:0]    a_pipe [0:NUM_COL-1],
          output logic [DATA_W-1:0]    w_reg_mon[0:NUM_COL-1]
        );
        
          systolic_1x4_chain #(
            .DATA_W (DATA_W),
            .ACC_W  (ACC_W),
            .NUM_COL(NUM_COL)
          ) u_chain (
            .clk       (clk),
            .rst_n     (rst_n),
            .en        (en),
            .a_in      (a_in),
            .w_load    (w_load),
            .w_in      (w_in),
            .a_pipe    (a_pipe),
            .w_reg_mon (w_reg_mon),
            .pprod     (pprod)
          );
        
          col_accumulators #(
            .ACC_W  (ACC_W),
            .NUM_COL(NUM_COL)
          ) u_acc (
            .clk   (clk),
            .rst_n (rst_n),
            .en    (en),
            .clr   (clr),
            .pprod (pprod),
            .c_out (c_out)
          );
        
        endmodule

### Testbench
        `timescale 1ns/1ps
        module tb_gemv_1x4_ws;
        
          localparam int DATA_W  = 8;
          localparam int ACC_W   = 32;
          localparam int NUM_COL = 4;
          localparam int K_DIM   = 5;
        
          // input + skew fill + drain
          // (pprod가 register로 1-cycle latency이므로 여유 포함)
          localparam int TMAX = K_DIM + (NUM_COL-1) + 2;
        
          logic clk, rst_n;
          logic en, clr;
        
          logic [DATA_W-1:0] a_in;
          logic              w_load [0:NUM_COL-1];
          logic [DATA_W-1:0] w_in   [0:NUM_COL-1];
        
          logic [ACC_W-1:0]  c_out     [0:NUM_COL-1];
        
          // monitors
          logic [ACC_W-1:0]  pprod     [0:NUM_COL-1];
          logic [DATA_W-1:0] a_pipe    [0:NUM_COL-1];
          logic [DATA_W-1:0] w_reg_mon [0:NUM_COL-1];
        
          gemv_1x4_ws_top #(
            .DATA_W (DATA_W),
            .ACC_W  (ACC_W),
            .NUM_COL(NUM_COL)
          ) dut (
            .clk       (clk),
            .rst_n     (rst_n),
            .en        (en),
            .clr       (clr),
            .a_in      (a_in),
            .w_load    (w_load),
            .w_in      (w_in),
            .c_out     (c_out),
            .pprod     (pprod),
            .a_pipe    (a_pipe),
            .w_reg_mon (w_reg_mon)
          );
        
          // clock
          initial clk = 0;
          always #5 clk = ~clk;
        
          // -----------------------------
          // test vectors
          // -----------------------------
          int A [0:K_DIM-1] = '{1, 3, 5, 7, 9};
        
          // W[k][col]
          int W [0:K_DIM-1][0:NUM_COL-1] = '{
            '{10,  1,  2,  3},
            '{20,  4,  5,  6},
            '{30,  7,  8,  9},
            '{40, 10, 11, 12},
            '{50, 13, 14, 15}
          };
        
          int EXP [0:NUM_COL-1];
        
          // -----------------------------
          // main
          // -----------------------------
          initial begin
            // expected GEMV: C[col] = sum_k A[k]*W[k][col]
            for (int col=0; col<NUM_COL; col++) begin
              EXP[col] = 0;
              for (int k=0; k<K_DIM; k++) begin
                EXP[col] += A[k] * W[k][col];
              end
            end
        
            // init
            rst_n = 0;
            en    = 0;
            clr   = 0;
            a_in  = '0;
        
            for (int j=0; j<NUM_COL; j++) begin
              w_load[j] = 0;
              w_in[j]   = '0;
            end
        
            // reset release
            repeat(3) @(posedge clk);
            rst_n = 1;
        
            // clear column accumulators
            clr = 1;
            @(posedge clk);
            clr = 0;
        
            // run
            en = 1;
        
            // ---------------------------------------------------------
            // WS skew rule:
            //   activation: A[t] enters at t (t< K_DIM)
            //   weight skew per column:
            //     k = t - col
            //     if (0<=k<K_DIM) -> feed W[k][col] with w_load=1
            // ---------------------------------------------------------
            for (int t=0; t<TMAX; t++) begin
              // activation stream
              if (t < K_DIM) a_in = DATA_W'(A[t]);
              else           a_in = '0;
        
              // weight stream (skewed)
              for (int col=0; col<NUM_COL; col++) begin
                int k;
                k = t - col;
                if (k >= 0 && k < K_DIM) begin
                  w_load[col] = 1;
                  w_in[col]   = DATA_W'(W[k][col]);
                end else begin
                  w_load[col] = 0;
                  w_in[col]   = '0;
                end
              end
        
              @(posedge clk);
            end
        
            en = 0;
        
            $display("Final result from DUT accumulators:");
            $display("C = %0d %0d %0d %0d",
              c_out[0], c_out[1], c_out[2], c_out[3]);
        
            for (int col=0; col<NUM_COL; col++) begin
              if (c_out[col] !== ACC_W'(EXP[col])) begin
                $display("[FAIL] col%0d got=%0d exp=%0d", col, c_out[col], EXP[col]);
                $fatal;
              end
            end
        
            $display("[PASS] C matches expected.");
            $display("EXP = %0d %0d %0d %0d", EXP[0], EXP[1], EXP[2], EXP[3]);
        
            $finish;
          end
        
        endmodule

### Result

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_62.png" width="400"/>

<div align="left">

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_63.png" width="400"/>

<div align="left">

Simulation 결과 Result Report를 통해 연산 결과가 golden과 일치함을 확인하였습니다.

### Weight-Stationary의 한계와 Output-Stationary로의 설계 전환

기존 Weight-Stationary (WS) dataflow에서는 activation이 PE 안에 들어온 뒤부터 같은 입력(sample)에 해당하는 데이터들이 맞게 흘러가고 있는지를 계속 관리해야 합니다.

​이러한 특성은 1-D systolic array에서는 관리 가능하지만, 2-D 배열에서는 여러 방향으로 데이터를 동시에 보내면서 같은 입력끼리 맞추는 일이 점점 어려워집니다.

​특히 activation이 한 칸씩 cycle 차이를 두고 이동하는 구조에서는, 모든 PE가 같은 입력에 대한 계산을 하고 있는지 확인하기 위한 추가적인 control logic이 필요해집니다. 이로 인해 array가 커질수록 control path가 복잡해집니다.

​이에 따라 다음 글부터는 Output-Stationary (OS) dataflow를 사용하여, 각 PE는 하나의 output에 대한 계산 결과를 지속적으로 accumulate하는 역할만 수행하고, input(sample)의 전환 시점과 accumulator initialization은 upper-level control logic에서 일괄적으로 관리하는 구조를 중심으로 설계를 이어갑니다.​


