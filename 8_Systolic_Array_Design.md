## Systolic Array 설계_(OS) 2-D Array

이어서 2-D Systolic Array와 Controller, Activation을 정리해보겠습니다.

## 3. Systolic Array

### DUT

    module pe_systolic_cell #(
      parameter int DATA_W = 8,
      parameter int ACC_W  = 2*DATA_W
    )(
      input  logic                 clk,
      input  logic                 rst_n,
      input  logic                 clr,
      input  logic                 en,
    
      input  logic signed[DATA_W-1:0]    a_in,
      input  logic signed[DATA_W-1:0]    b_in,
      output logic signed[DATA_W-1:0]    a_out,
      output logic signed[DATA_W-1:0]    b_out,
    
      output logic signed[ACC_W-1:0]     mul,
      output logic signed[ACC_W-1:0]     acc_sum
    );
    
      // a, b를 한 번 레지스터에 잡고 옆/아래로 전달
      logic signed[DATA_W-1:0] a_reg, b_reg;
      assign a_out = a_reg;
      assign b_out = b_reg;
    
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          a_reg <= '0;
          b_reg <= '0;
        //end else if (clr) begin
        //clr은 연산기(mac_pe)의 누적값 초기화에만 관여하고, 데이터 전달(a_reg, b_reg)은 건드리지 않는 것이 일반적인 설계
        end else if (en) begin //debugging point
          a_reg <= a_in;
          b_reg <= b_in;
        end
        // en=0 이면 값 유지 (Stall)
      end
    
      // 내부 MAC 동작: a_reg * b_reg 누산
      mac_pe #(
        .DATA_W (DATA_W),
        .ACC_W  (ACC_W)
      ) u_mac_pe (
        .clk     (clk),
        .rst_n   (rst_n),
        .clr     (clr),
        .en      (en),
        .a       (a_reg),
        .b       (b_reg),
        .mul     (mul),
        .acc_sum (acc_sum)
      );
    
    endmodule

        //============================================================
        // 3단계: 2D Systolic PE Array
        // - 2x2 타일 예제 (ROWS, COLS 파라미터로 확장 가능)
        //============================================================
        module systolic_array_2d #(
          parameter int DATA_W = 8,
          parameter int ACC_W  = 2*DATA_W,
          parameter int ROWS   = 2,
          parameter int COLS   = 2
        )(
          input  logic                      clk,
          input  logic                      rst_n,
        
          input  logic                      clr,
          input  logic                      en,
        
          // 왼쪽에서 들어오는 A 행렬 스트림 (각 행마다 입력 하나)
          input  logic signed [DATA_W-1:0]         a_in_row [0:ROWS-1],
          // 위쪽에서 들어오는 B 행렬 스트림 (각 열마다 입력 하나)
          input  logic signed [DATA_W-1:0]         b_in_col [0:COLS-1],
        
          // Watchpoint용 출력 (각 PE의 mul / acc_sum)
          output logic signed [ACC_W-1:0]          pe_mul    [0:ROWS-1][0:COLS-1],
          output logic signed [ACC_W-1:0]          pe_acc_sum[0:ROWS-1][0:COLS-1]
        );
        
          // 내부 a/b 전달 신호
          logic signed [DATA_W-1:0] a_sig [0:ROWS-1][0:COLS-1];
          logic signed [DATA_W-1:0] b_sig [0:ROWS-1][0:COLS-1];
        
        /*  genvar r, c;
        //  generate
        //    for (r = 0; r < ROWS; r++) begin : G_ROW
        //      for (c = 0; c < COLS; c++) begin : G_COL
        //
        //        logic [DATA_W-1:0] a_in_cell;
        //        logic [DATA_W-1:0] b_in_cell;
        //
        //        // 왼쪽에서 들어오는 a
        //        assign a_in_cell = (c == 0) ? a_in_row[r] : a_sig[r][c-1];
        //        //r0c0 assign a_in_cell = a_in_row[0]
        //        //r0c1 assign a_in_cell = a_sig[0][0]
        //        //r1c0 assign a_in_cell = a_in_row[1]
        //        //r1c1 assign a_in_cell = a_sig[1][0];
        //
        //        // 위쪽에서 들어오는 b
        //        assign b_in_cell = (r == 0) ? b_in_col[c] : b_sig[r-1][c];
        //
        //        pe_systolic_cell #(
        //          .DATA_W (DATA_W),
        //          .ACC_W  (ACC_W)
        //        ) u_cell (
        //          .clk     (clk),
        //          .rst_n   (rst_n),
        //          .clr     (clr),
        //          .en      (en),
        //
        //          .a_in    (a_in_cell),
        //          .b_in    (b_in_cell),
        //          .a_out   (a_sig[r][c]),
        //          .b_out   (b_sig[r][c]),
        //
        //          .mul     (pe_mul[r][c]),
        //          .acc_sum (pe_acc_sum[r][c])
        //        );
        //
        //      end
        //    end
        //  endgenerate
        */
        
        // [개선 포인트 1] 연결 와이어를 (개수 + 1)만큼 선언
          // a_conn[r][c] : r행 c열 "앞"에 있는 와이어 (입력)
          // a_conn[r][c+1] : r행 c열 "뒤"에 있는 와이어 (출력)
          logic signed [DATA_W-1:0] a_conn [0:ROWS-1][0:COLS];   // 가로 방향: 0 ~ COLS
          logic signed [DATA_W-1:0] b_conn [0:ROWS][0:COLS-1];   // 세로 방향: 0 ~ ROWS
        
        genvar r, c;
        
          // [개선 포인트 2] 외부 입력을 0번 인덱스 와이어에 연결
          generate
            for (r = 0; r < ROWS; r++) begin : A_INPUT_BIND
              assign a_conn[r][0] = a_in_row[r];
            end
        
            for (c = 0; c < COLS; c++) begin : B_INPUT_BIND
              assign b_conn[0][c] = b_in_col[c];
            end
          endgenerate
        
        // [개선 포인트 3] 반복문 내부가 아주 단순해짐 (조건문 삭제)
          generate
            for (r = 0; r < ROWS; r++) begin : G_ROW
              for (c = 0; c < COLS; c++) begin : G_COL
        
                pe_systolic_cell #(
                  .DATA_W (DATA_W),
                  .ACC_W  (ACC_W)
                ) u_cell (
                  .clk     (clk),
                  .rst_n   (rst_n),
                  .clr     (clr),
                  .en      (en),
        
                  // 현재 위치(c)에서 받아서 -> 다음 위치(c+1)로 넘김
                  .a_in    (a_conn[r][c]),
                  .a_out   (a_conn[r][c+1]),
        
                  // 현재 위치(r)에서 받아서 -> 다음 위치(r+1)로 넘김
                  .b_in    (b_conn[r][c]),
                  .b_out   (b_conn[r+1][c]),
        
                  .mul     (pe_mul[r][c]),
                  .acc_sum (pe_acc_sum[r][c])
                );
        
              end
            end
          endgenerate
        
        endmodule
        
### TB

        //[3] systolic_array_2d
        
        `timescale 1ns/1ps
        
        module tb_systolic_array_2d;
        
          localparam int K_DIM  = 2;        // A,B의 공통 차원 (2x2 행렬 곱, 이전 코드 개선: 누산 시 발생할 수 있는 Overflow 방지
          localparam int DATA_W = 8;
        //  localparam int DATA_W = 4;        // debugging point
          localparam int ACC_W  = 2*DATA_W+$clog2(K_DIM);
          localparam int ROWS   = 2;
          localparam int COLS   = 2;
          localparam int TOTAL_CYCLES = 10;
        
          //==========================================================
          // DUT 포트
          //==========================================================
          logic                   clk;
          logic                   rst_n;
          logic                   clr; //unused
          logic                   en;
        
          logic signed [DATA_W-1:0]      a_in_row [0:ROWS-1];
          logic signed [DATA_W-1:0]      b_in_col [0:COLS-1];
        
          logic signed [ACC_W-1:0]       pe_mul     [0:ROWS-1][0:COLS-1];
          logic signed [ACC_W-1:0]       pe_acc_sum [0:ROWS-1][0:COLS-1];
        
          systolic_array_2d #(
            .DATA_W (DATA_W),
            .ACC_W  (ACC_W),
            .ROWS   (ROWS),
            .COLS   (COLS)
          ) dut (
            .clk        (clk),
            .rst_n      (rst_n),
            .clr        (clr),
            .en         (en),
            .a_in_row   (a_in_row),
            .b_in_col   (b_in_col),
            .pe_mul     (pe_mul),
            .pe_acc_sum (pe_acc_sum)
          );
        
          //==========================================================
          // Clock / Reset
          //==========================================================
          initial begin
            clk = 1'b0;
            forever #5 clk = ~clk;  // 100MHz
          end
        
        
          int A [0:ROWS-1][0:K_DIM-1];
          int B [0:K_DIM-1][0:COLS-1];
          int C_ref [0:ROWS-1][0:COLS-1];
          int err_cnt;
        
        
        //==========================================================
        // Pilot Test
        //==========================================================
        // Reference Matrices (2x2)
        //   A = [ 1  2 ]
        //       [ 3  4 ]
        //
        //   B = [ 5  6 ]
        //       [ 7  8 ]
        //
        //   C = A * B = [ 1*5+2*7   1*6+2*8 ] = [ 19  22 ]
        //               [ 3*5+4*7   3*6+4*8 ]   [ 43  50 ]
        //==========================================================  
        //
        //initial begin
        //  // A
        //  A[0][0] = 1; A[0][1] = 2;
        //  A[1][0] = 3; A[1][1] = 4;
        //
        //  // B
        //  B[0][0] = 5; B[0][1] = 6;
        //  B[1][0] = 7; B[1][1] = 8;
        
          initial begin
              // 1. 초기화
              rst_n = 0; en = 0;
              a_in_row = '{default:0}; b_in_col = '{default:0};
        
              #20 rst_n = 1;
              #10 en = 1; 
        
              $display("=== [Step 3.5] Random Verification Start (10 Iterations) ===");
        
              // ---------------------------------------------------------
              // 10번 반복 테스트 (Random)
              // ---------------------------------------------------------
              for (int iter = 0; iter < TOTAL_CYCLES; iter++) begin
        
                  en = $urandom_range(0, 1); //debugging point
        
              // (1) 입력 데이터 랜덤 생성 & 정답 미리 계산
              foreach (A[r, c]) A[r][c] = $urandom_range(-(1 << (DATA_W-1)), (1 << (DATA_W-1)) - 1);
              foreach (B[r, c]) B[r][c] = $urandom_range(-(1 << (DATA_W-1)), (1 << (DATA_W-1)) - 1);
        
           
        
                  for(int r=0; r<ROWS; r++) begin
                      for(int c=0; c<COLS; c++) begin
                          C_ref[r][c] = 0;
                          for(int k=0; k<K_DIM; k++) begin
                              C_ref[r][c] += A[r][k] * B[k][c];
                          end
                      end
                  end
        
                  // (2) 데이터 주입 (Skew 적용)
                  // Cycle 0
                  @(posedge clk); 
                  a_in_row[0] <= A[0][0]; b_in_col[0] <= B[0][0];
                  a_in_row[1] <= 0;       b_in_col[1] <= 0;
        
                  // Cycle 1
                  @(posedge clk);
                  a_in_row[0] <= A[0][1]; b_in_col[0] <= B[1][0];
                  a_in_row[1] <= A[1][0]; b_in_col[1] <= B[0][1];
        
                  // Cycle 2
                  @(posedge clk);
                  a_in_row[0] <= 0;       b_in_col[0] <= 0;
                  a_in_row[1] <= A[1][1]; b_in_col[1] <= B[1][1];
        
                  // Cycle 3 (Flush)
                  @(posedge clk);
                  a_in_row[1] <= 0;       b_in_col[1] <= 0;
        
                  // (3) 연산 대기
                  repeat(5) @(posedge clk);
        
                  // (4) 결과 체크 (요청하신 포맷 적용)
                  // -------------------------------------------------------
                  err_cnt = 0;
                  $display("==================================================");
                  $display("[TB] Iteration %0d Checking C = A*B result", iter);
        
                  if (en) begin //debugging point
                    for (int r = 0; r < ROWS; r++) begin
                      for (int c = 0; c < COLS; c++) begin
                        if (pe_acc_sum[r][c] !== C_ref[r][c]) begin
                          err_cnt++;
                          $display("ERROR! C[%0d][%0d] mismatch: dut=%0d, ref=%0d, time=%0t",
                                   r, c, pe_acc_sum[r][c], C_ref[r][c], $time);
                        end else begin
                          $display("PASS!  C[%0d][%0d] match   : dut=%0d, ref=%0d, time=%0t",
                                   r, c, pe_acc_sum[r][c], C_ref[r][c], $time);
                        end
                      end
                    end

                    if (err_cnt == 0)
                      $display("[TB] RESULT: PASS");
                    else
                      $display("[TB] RESULT: FAIL (err_cnt=%0d)", err_cnt);
        
                    $display("==================================================");
                  end
                  // -------------------------------------------------------
        
                  // (5) 리셋 및 다음 루프 준비
                  rst_n = 0; 
                  repeat(2) @(posedge clk);
                  rst_n = 1;
                  @(posedge clk);
        
              end // End of Loop
        
              $display("=== All 10 Random Tests Finished! ===");
              $finish;
          end
        
        endmodule
        
### Simulation Result

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_74.png" width="400"/>

<div align="left">

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_75.png" width="400"/>

<div align="left">

<mark>Check Point !!</mark>

- pe_systolic_cell.sv : OS(Output-Stationary) dataflow에서 각 PE는 좌측/상단에서 입력된 a_in, b_in을 내부 register(a_reg, b_reg)에 capture한 뒤, 다음 cycle에 인접 PE(우측 a_out, 하단 b_out)로 전달합니다. 즉, PE 간에는 1-stage pipeline register가 존재하며, 각 cycle의 입력이 en=1일 때 register에 저장되고 다음 cycle에 출력으로 전달됩니다. en=0이면 a_reg/b_reg가 유지되어 데이터 propagation이 정지(stall)합니다.

    ​동시에 PE 내부에서는 a_reg * b_reg 곱(mul)을 계산하고, mac_pe에서 출력 누산값(acc_sum)을 갱신합니다. 이때 clr은 누산기 상태(예: acc_sum) 초기화에만 관여하며, 데이터 전달 레지스터(a_reg/b_reg)는 초기화하지 않습니다.

- ROWS / CONS : 2-D systolic array는 행(row)과 열(column)로 PE가 배치되므로, ROWS, COLS parameter로 array의 크기(PE 개수)를 일반화합니다. 해당 module은 ROWS × COLS 개의 pe_systolic_cell을 generate로 instance화하며, 입력 stream은 a_in_row[0:ROWS-1], b_in_col[0:COLS-1]로 정의합니다.

- packed / unpacked array : SystemVerilog에서 신호의 bit-width는 packed dimension으로, 구성 개수(배열 차원) 는 unpacked dimension으로 표현합니다.

예를 들어 

        logic signed [DATA_W-1:0] a_in_row [0:ROWS-1];

는 각 요소가 DATA_W-bit signed인 1-D 배열(ROWS개)이며, 

        logic signed [ACC_W-1:0] pe_acc_sum [0:ROWS-1][0:COLS-1];
        
는 각 PE의 누산 결과를 저장하는 2-D 배열(ROWS×COLS개)입니다.

- PE Interconnect 구성 방식: 내부 조건 분기 vs. 외부 Bus 확장

        assign a_in_cell = (c == 0) ? a_in_row[r] : a_sig[r][c-1];
        assign b_in_cell = (r == 0) ? b_in_col[c] : b_sig[r-1][c];

    기존 코드에서는 각 PE 인스턴스 내부(중첩된 generate 반복문 내부)에서 경계 조건을 직접 판단하여 입력을 선택합니다. 즉, 각 PE가 자신이 배열의 경계에 위치하는지 여부(c==0, r==0) 를 판단한 뒤, 외부 입력(a_in_row, b_in_col) 또는 인접 PE의 출력(a_sig, b_sig) 중 하나를 선택하는 구조입니다.

​    반면, 개선된 구조에서는 inter-PE 연결용 버스를 경계까지 포함한 (N+1) 크기로 선언합니다.

        logic signed [DATA_W-1:0] a_conn [0:ROWS-1][0:COLS];   // COLS+1 개
        logic signed [DATA_W-1:0] b_conn [0:ROWS][0:COLS-1];   // ROWS+1 개
        
        assign a_conn[r][0] = a_in_row[r];
        assign b_conn[0][c] = b_in_col[c];
        
        .a_in  (a_conn[r][c]),
        .a_out (a_conn[r][c+1]),
        .b_in  (b_conn[r][c]),
        .b_out (b_conn[r+1][c]),
        
외부 입력은 이 확장된 버스의 첫 번째 경계(a_conn[r][0], b_conn[0][c])에 사전에 binding되며, 이후 각 PE는 항상 동일한 규칙으로 데이터를 전달합니다.
- a_conn[r][c] → a_conn[r][c+1]
- b_conn[r][c] → b_conn[r+1][c]
이 방식에서는 경계 처리가 generate 반복문 외부에서 완료되므로, 반복문 내부에서는 조건 분기 없이 동일한 인덱싱 규칙만 적용하면 됩니다.

​    그 결과,
- generate 내부 코드가 단순화되고,
- ROWS/COLS 확장 시 인덱스 관련 실수 가능성이 감소하며,
- interconnection 구조와 PE 기능 로직이 명확히 분리되어 가독성과 유지보수성이 향상됩니다.

### Error / Warning

- Error

    첫 번째 시도 당시, report에서 결과값이 일치하지 않아 ERROR가 발생했었습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_73.png" width="400"/>

<div align="left">

    Error의 원인은 DUT 및 하위 모듈에서는 입력 Port를 signed로 선언하였으나, tb_top에서 랜덤 입력을 생성할 때 다음과 같이 값을 생성하여

        foreach(A[r,c]) A[r][c] = $urandom_range(0, (1 << DATA_W) - 1);
        foreach(B[r,c]) B[r][c] = $urandom_range(0, (1 << DATA_W) - 1);     

    실제 입력 값이 0 ~ 255 범위로 생성되었기 때문입니다. 

​    이 값들은 TB 내부에서는 정상적인 정수로 보이지만, DUT의 logic signed [DATA_W-1:0] 입력 Port로 전달되는 과정에서 bid-width에 따라 truncation이 발생하며, DATA_W=8 기준으로는 128~255 구간의 값이 음수로 해석됩니다.

​이로 인해 TB에서 계산한 reference 결과와 DUT에서 실제로 연산한 값 사이에 불일치가 발생하였습니다. 이를 해결하기 위해 랜덤 입력 생성 범위를 다음과 같이 DATA_W-bit signed 표현 범위로 제한하였고,

        foreach (A[r, c]) A[r][c] = $urandom_range(-(1 << (DATA_W-1)), (1 << (DATA_W-1)) - 1);
        foreach (B[r, c]) B[r][c] = $urandom_range(-(1 << (DATA_W-1)), (1 << (DATA_W-1)) - 1);
      
그 결과 TB에서 계산한 reference 값과 DUT 입력 값이 동일하게 유지되어 정상적으로 동작함을 확인할 수 있었습니다.

- Warning

Simulation은 정상적으로 수행되었으나, compile 과정에서 warning 메시지가 검출되었습니다.

해당 warning을 분석한 결과, 

systolic_array_2d에 instance화된 pe_systolic_cell module서 발생한 문제임을 확인할 수 있었습니다.


관련 코드를 확인한 결과, a_out 신호의 bit-width 선언이 잘못 작성되어 있음을 확인하였습니다.


구체적으로, a_out은 DATA_W(8-bit)로 선언되어야 하나, 실제 코드에서는 16-bit로 선언되어 있었으며, 

이로 인해 상위 module과의 port 연결 과정에서 bit-width 불일치에 따른 warning이 발생하였습니다.

​

즉, 기능적인 동작에는 직접적인 영향을 주지는 않았으나, 

interface 정의와 실제 신호 폭이 일치하지 않는 상태였음을 의미하며, 

향후 확장 또는 합성 단계에서 잠재적인 오류로 이어질 수 있는 부분이므로 [DATA_W-1:0]으로 수정하였습니다.
