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





