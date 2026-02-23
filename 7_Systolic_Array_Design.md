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
