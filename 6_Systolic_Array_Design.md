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
