## Systolic Array 설계_(OS) Systolic Array Controller + Activation

## 4. Systolic Array + Controller

### DUT
    
    module systolic_controller #(
      parameter int DATA_W = 8,
      parameter int K_DIM  = 2, //Common dimension
      parameter int ROWS   = 2,
      parameter int COLS   = 2,
      // 내부 파라미터
      parameter int ACC_W  = 2*DATA_W + $clog2(K_DIM)
    )(
      input  logic                      clk,
      input  logic                      rst_n,
    
      // --- Control Interface ---
      input  logic                      i_start,
      output logic                      o_done,
      output logic                      o_busy,
    
      // --- Data Interface ---
      input  logic signed [DATA_W-1:0]         i_mat_a [0:ROWS-1][0:K_DIM-1],
      input  logic signed [DATA_W-1:0]         i_mat_b [0:K_DIM-1][0:COLS-1],
      output logic signed [ACC_W-1:0]          o_mat_c [0:ROWS-1][0:COLS-1]
    );
    
      //==========================================================
      // 1. 내부 상태 및 버퍼 정의
      //==========================================================
      // FSM 상태 정의
      typedef enum logic [1:0] {
        IDLE,
        RUN,
        DONE_STATE
      } state_t;
    
      state_t state, next_state;
      logic [7:0] cnt; 
    
      // Input Buffer: 입력 데이터를 저장(Latch)해두는 공간
      // 연산 도중 입력값이 바뀌어도 내부 연산이 꼬이지 않도록 capture
      logic signed [DATA_W-1:0] latched_mat_a [0:ROWS-1][0:K_DIM-1];
      logic signed [DATA_W-1:0] latched_mat_b [0:K_DIM-1][0:COLS-1];
    
      // Array Interface: 실제 Systolic Array로 들어가는 신호
      // 각 row/col 단위로 데이터가 들어감
      logic signed [DATA_W-1:0] array_a_in [0:ROWS-1];
      logic signed [DATA_W-1:0] array_b_in [0:COLS-1];
      logic              array_en;
      // Clear 신호: Accumulator 초기화 용도
      logic              array_clr; 
    
      // 연산 완료까지 걸리는 시간 계산
      // 데이터 주입 완료(ROWS + K_DIM) + 파이프라인 통과(COLS) + 여유
      localparam int CALC_CYCLES = ROWS + COLS + K_DIM + 2;
    
      //==========================================================
      // 2. FSM & Counter
      //==========================================================
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          state <= IDLE;
          cnt   <= 0;
        end else begin
          state <= next_state;
          if (state == RUN) cnt <= cnt + 1;
          else              cnt <= 0;
        end
      end
    
      always_comb begin
        next_state = state;
        case (state)
          IDLE: begin
            if (i_start) next_state = RUN;
          end
          RUN: begin
            if (cnt >= CALC_CYCLES) next_state = DONE_STATE;
          end
          DONE_STATE: begin
            if (!i_start) next_state = IDLE;
          end
        endcase
      end
    
      // Output Assignments
      // 각 state에 따라 done, busy 신호 제어
      assign o_busy   = (state == RUN);
      assign o_done   = (state == DONE_STATE);
    
      // Array Control Signals
      assign array_en = (state == RUN);
    
      // IDLE 상태에서 start가 들어오는 순간 Clear 수행 (Accumulator 초기화)
      // 이렇게 하면 PE 내부의 acc_sum이 0으로 리셋되고 새 연산을 시작합니다.
      assign array_clr = (state == IDLE) && i_start;
    
      //==========================================================
      // 3. Input Buffer Latching
      //==========================================================
      // 연산 도중 입력값이 바뀌어도 내부 연산이 꼬이지 않도록 캡처
      always_ff @(posedge clk) begin
        if (state == IDLE && i_start) begin
          latched_mat_a <= i_mat_a;
          latched_mat_b <= i_mat_b;
        end
      end
    
      //==========================================================
      // 4. Data Skewing Logic
      //==========================================================
      genvar r, c;
      generate
        // Row Input Control (Matrix A -> Row Input)
        for (r = 0; r < ROWS; r++) begin : GEN_SKEW_A
          always_comb begin
            array_a_in[r] = '0; // Default 0 padding
            if (state == RUN) begin
              // Timing Logic: time(cnt) - row_index(r) = col_index(k)
              int k;
              k = cnt - r;
              if (k >= 0 && k < K_DIM) begin
                array_a_in[r] = latched_mat_a[r][k];
              end
            end
          end
        end
    
        // Column Input Control (Matrix B -> Col Input)
        for (c = 0; c < COLS; c++) begin : GEN_SKEW_B
          always_comb begin
            array_b_in[c] = '0; // Default 0 padding
            if (state == RUN) begin
              // Timing Logic: time(cnt) - col_index(c) = row_index(k)
              int k;
              k = cnt - c;
              if (k >= 0 && k < K_DIM) begin
                array_b_in[c] = latched_mat_b[k][c];
              end
            end
          end
        end
      endgenerate
    
      //==========================================================
      // 5. Systolic Array Instantiation
      //==========================================================
      systolic_array_2d #(
        .DATA_W (DATA_W),
        .ACC_W  (ACC_W),
        .ROWS   (ROWS),
        .COLS   (COLS)
      ) u_core_array (
        .clk        (clk),
        .rst_n      (rst_n),
        .clr        (array_clr),
        .en         (array_en),
        .a_in_row   (array_a_in),
        .b_in_col   (array_b_in),
        .pe_mul     (),        // Unused monitor port
        .pe_acc_sum (o_mat_c)  // Final Result
      );
    
    endmodule

### TB

        //[4] systolic_array_2d + controller
        
        `timescale 1ns/1ps
        
        module tb_systolic_controller;
        
          parameter int DATA_W = 8;
          parameter int K_DIM  = 2;
          parameter int ROWS   = 2;
          parameter int COLS   = 2;
          parameter int ACC_W  = 2*DATA_W + $clog2(K_DIM);
        
          logic clk, rst_n;
        
          logic start, done, busy;
          logic signed [DATA_W-1:0] tb_mat_a [0:ROWS-1][0:K_DIM-1];
          logic signed [DATA_W-1:0] tb_mat_b [0:K_DIM-1][0:COLS-1];
          logic signed [ACC_W-1:0]  tb_mat_c [0:ROWS-1][0:COLS-1];
        
          // Reference Model
          int signed C_ref [0:ROWS-1][0:COLS-1];
          int err_cnt;
        
          // DUT Instantiation
          systolic_controller #(
            .DATA_W(DATA_W), 
            .K_DIM(K_DIM), 
            .ROWS(ROWS), 
            .COLS(COLS)
          ) dut (
            .clk        (clk),
            .rst_n      (rst_n),
        
            // Control Interface
            .i_start    (start),
            .o_done     (done),
            .o_busy     (busy),
        
            // Data Interface
            .i_mat_a    (tb_mat_a),
            .i_mat_b    (tb_mat_b),
            .o_mat_c    (tb_mat_c)
          );
        
          // Clock Gen
          initial begin
            clk = 0;
            forever #5 clk = ~clk;
          end
        
          initial begin
            rst_n = 0;
            start = 0;
            tb_mat_a  = '{default:0};
            tb_mat_b  = '{default:0};
        
            #20 rst_n = 1;
        
            $display("=== Controller-based Verification Start ===");
        
            for (int iter = 0; iter < 5; iter++) begin
        
            // 1. Data Setup (Random, signed)
            //    TB에서는 그냥 평범한 행렬 모양으로 넣으면 됨 (Skew 불필요)
            for (int r = 0; r < ROWS; r++)
              for (int k = 0; k < K_DIM; k++)
                tb_mat_a[r][k] =
                  $urandom_range(-(1 << (DATA_W-1)), (1 << (DATA_W-1)) - 1);
        
            for (int k = 0; k < K_DIM; k++)
              for (int c = 0; c < COLS; c++)
                tb_mat_b[k][c] =
                  $urandom_range(-(1 << (DATA_W-1)), (1 << (DATA_W-1)) - 1);
        
            // 2. Calculate Reference (signed)
            for (int r = 0; r < ROWS; r++) begin
              for (int c = 0; c < COLS; c++) begin
                C_ref[r][c] = 0;
                for (int k = 0; k < K_DIM; k++)
                  C_ref[r][c] += tb_mat_a[r][k] * tb_mat_b[k][c];
              end
            end
        
        
              // 3. Start Operation
              @(posedge clk);
              start = 1;
              @(posedge clk);
              start = 0;
        
              // 4. Wait for Done
              wait(done == 1);
        
              // 5. Check Result
              err_cnt = 0;
              $display("--------------------------------------------------");
              $display("[Iter %0d] Checking Result...", iter);
              for(int r=0; r<ROWS; r++) begin
                for(int c=0; c<COLS; c++) begin
                  if (tb_mat_c[r][c] !== C_ref[r][c]) begin
                     $display("ERROR! [%0d][%0d]: DUT=%0d, REF=%0d", r, c, tb_mat_c[r][c], C_ref[r][c]);
                     err_cnt++;
                  end else begin
                     $display("PASS! at [%0d][%0d]: DUT=%0d, REF=%0d", r, c, tb_mat_c[r][c], C_ref[r][c]);
                  end
                end
              end
        
              if (err_cnt == 0) $display("[Iter %0d] FINAL RESULT: PASS!", iter);
              else              $display("[Iter %0d] FINAL RESULT: FAIL! (%0d errors)", iter, err_cnt);
              $display("--------------------------------------------------");
        
              // 6. Wait a bit before next run
              @(posedge clk);
            end
        
            $display("=== All Tests Finished ===");
            $finish;
          end
        
        endmodule

### Simulation Result

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_80.png" width="400"/>

<div align="left">

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_81.png" width="400"/>

<div align="left">

<mark>Check Point !!</mark>

- K_DIM / ACC_W : K_DIM은 행렬 곱 연산에서 A와 B가 공유하는 공통 차원(common dimension) 을 의미합니다. 즉, 각 출력 원소 C[r][c]는 k = 0 … K_DIM-1에 대해 A[r][k] * B[k][c]를 누적하여 계산되며, K_DIM은 이 누산 횟수를 결정합니다. ACC_W는 MAC 연산 과정에서 발생할 수 있는 bit growth를 고려하여 정의된 누산기(accumulator) bit width입니다. 
    각 PE에서는 DATA_W-bit signed 입력 두 개를 곱하므로 곱셈 결과는 최대 2*DATA_W 비트를 요구하며, 여기에 K_DIM번의 누산이 수행되므로 추가적인 bit growth이 발생합니다.

        ACC_W = 2*DATA_W + log2(K_DIM)

- FSM : typedef enum을 사용하여 컨트롤러의 상태(state)를 정의하고, 현재 상태(state)와 다음 상태(next_state)를 선언합니다. 본 FSM은 IDLE, RUN, DONE_STATE의 세 가지 상태로 구성됩니다. 이후 case 문을 통해 각 상태에서의 전이 조건을 정의합니다.

    - IDLE : 연산 대기 상태로, i_start 신호가 asserted되면 RUN 상태로 전이합니다.
    
    - RUN : systolic array에 대한 데이터 주입 및 연산이 진행되며, 내부 counter가 설정된 연산 사이클 수(CALC_CYCLES)에 도달하면 DONE_STATE로 전이합니다.(CALC_CYCLES = ROWS + COLS + K_DIM + 2는 정확한 최소 cycle 수를 수학적으로 계산한 값이라기보다는,데이터 주입 구간(K_DIM), row/column 방향 데이터 propagataion delay(ROWS, COLS), 그리고 PE 내부 pipeline 및 flush 구간을 고려한 보수적인 완료 기준)
    
    - DONE_STATE : 연산이 완료된 상태를 유지하며, i_start가 deasserted되면 다음 연산을 위해 다시 IDLE 상태로 복귀합니다.

- Input buffer Latching:

    연산 도중 상위 모듈 또는 TB에서 i_mat_a, i_mat_b 값이 변경되더라도 내부 연산이 영향을 받지 않도록, controller는 연산 시작 시점에 입력 행렬을 내부 buffer(latched_mat_a, latched_mat_b)로 캡처합니다. 

          always_ff @(posedge clk) begin
            if (state == IDLE && i_start) begin
              latched_mat_a <= i_mat_a;
              latched_mat_b <= i_mat_b;
            end
          end
  
구체적으로 IDLE 상태에서 i_start가 asserted되는 순간 입력을 1회 latch하며, 이후 RUN 상태에서는 외부 입력이 아니라 latch된 buffer만을 참조하여 데이터 stream을 생성합니다. 이 방식으로 입력 데이터의 일관성을 보장하고, transaction 단위로 안정적인 연산을 수행할 수 있도록 합니다.

​- Data Skewing : 2-D systolic array는 각 PE에서 A[r][k]와 B[k][c]가 동일한 cycle에 도달하도록 입력 stream에 skew를 적용해야 합니다. 이를 위해 controller 내부 counter와 row/con 인덱스를 이용하여, 각 cycle에 array에 주입할 원소를 선택합니다.

    - A 주입: k = cnt - r로 계산하여, k가 유효 범위(0 ≤ k < K_DIM)일 때 latched_mat_a[r][k]를 array_a_in[r]로 출력합니다. 유효 범위를 벗어나면 0을 주입하여 padding을 수행합니다.
    
    - B 주입: k = cnt - c로 계산하여, k가 유효 범위일 때 latched_mat_b[k][c]를 array_b_in[c]로 출력합니다. 

        마찬가지로 유효 범위를 벗어나면 0을 주입합니다.

//==========================================================
// Data Skewing Logic
//==========================================================
genvar r, c;
generate
  // Row Input Control (Matrix A -> Row Stream)
  for (r = 0; r < ROWS; r++) begin : GEN_SKEW_A
    always_comb begin
      array_a_in[r] = '0;   // default padding
      if (state == RUN) begin
        int k;
        k = cnt - r;
        if (k >= 0 && k < K_DIM) begin
          array_a_in[r] = latched_mat_a[r][k];
        end
      end
    end
  end

  // Column Input Control (Matrix B -> Column Stream)
  for (c = 0; c < COLS; c++) begin : GEN_SKEW_B
    always_comb begin
      array_b_in[c] = '0;   // default padding
      if (state == RUN) begin
        int k;
        k = cnt - c;
        if (k >= 0 && k < K_DIM) begin
          array_b_in[c] = latched_mat_b[k][c];
        end
      end
    end
  end
endgenerate
이와 같은 skewing을 통해, 

배열 내에서 A는 우측, B는 하단으로 전파되는 동안 각 PE에서 동일한 k에 해당하는 데이터가 정렬되어 MAC이 정상적으로 수행됩니다.

5. Systolic Array + Controller + Activation Function

DUT

module systolic_controller_relu #(
  parameter int DATA_W = 8,
  parameter int K_DIM  = 2, //Common dimension
  parameter int ROWS   = 2,
  parameter int COLS   = 2,

  // [추가] 활성화 함수용 파라미터 (Threshold/Bias)
  // 이 값보다 작은 연산 결과는 노이즈로 간주하고 0으로 만듭니다.
  parameter int BIAS   = 50, 

  // 내부 파라미터
  parameter int ACC_W  = 2*DATA_W + $clog2(K_DIM)
)(
  input  logic                      clk,
  input  logic                      rst_n,

  // --- Control Interface ---
  input  logic                      i_start,
  output logic                      o_done,
  output logic                      o_busy,

  // --- Data Interface ---
  input  logic signed [DATA_W-1:0]         i_mat_a [0:ROWS-1][0:K_DIM-1],
  input  logic signed [DATA_W-1:0]         i_mat_b [0:K_DIM-1][0:COLS-1],
  output logic signed [ACC_W-1:0]          o_mat_c [0:ROWS-1][0:COLS-1]
);

  //==========================================================
  // 1. 내부 상태 및 버퍼 정의
  //==========================================================
  typedef enum logic [1:0] {
    IDLE,
    RUN,
    DONE_STATE
  } state_t;

  state_t state, next_state;
  logic [7:0] cnt; 

  // Input Buffer: 입력 데이터를 저장(Latch)해두는 공간
  logic signed [DATA_W-1:0] latched_mat_a [0:ROWS-1][0:K_DIM-1];
  logic signed [DATA_W-1:0] latched_mat_b [0:K_DIM-1][0:COLS-1];

  // Array Interface: 실제 Systolic Array로 들어가는 신호
  logic signed [DATA_W-1:0] array_a_in [0:ROWS-1];
  logic signed [DATA_W-1:0] array_b_in [0:COLS-1];
  logic              array_en;
  logic              array_clr; 

  // [핵심] Array의 순수 결과값 (PPU 입력용 내부 와이어)
  logic signed [ACC_W-1:0]  raw_acc_sum [0:ROWS-1][0:COLS-1];

  // 연산 완료까지 걸리는 시간 계산
  // 데이터 주입 완료(ROWS + K_DIM) + 파이프라인 통과(COLS) + 여유
  localparam int CALC_CYCLES = ROWS + COLS + K_DIM + 2;

  //==========================================================
  // 2. FSM & Counter
  //==========================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      cnt   <= 0;
    end else begin
      state <= next_state;
      if (state == RUN) cnt <= cnt + 1;
      else              cnt <= 0;
    end
  end

  always_comb begin
    next_state = state;
    case (state)
      IDLE: begin
        if (i_start) next_state = RUN;
      end
      RUN: begin
        if (cnt >= CALC_CYCLES) next_state = DONE_STATE;
      end
      DONE_STATE: begin
        if (!i_start) next_state = IDLE;
      end
    endcase
  end

  // Output Assignments
  assign o_busy   = (state == RUN);
  assign o_done   = (state == DONE_STATE);

  // Array Control Signals
  assign array_en = (state == RUN);

  // IDLE 상태에서 start가 들어오는 순간 Clear 수행 (Accumulator 초기화)
  // 이렇게 하면 PE 내부의 acc_sum이 0으로 리셋되고 새 연산을 시작합니다.
  assign array_clr = (state == IDLE) && i_start;

  //==========================================================
  // 3. Input Buffer Latching
  //==========================================================
  // 연산 도중 입력값이 바뀌어도 내부 연산이 꼬이지 않도록 캡처
  always_ff @(posedge clk) begin
    if (state == IDLE && i_start) begin
      latched_mat_a <= i_mat_a;
      latched_mat_b <= i_mat_b;
    end
  end

  //==========================================================
  // 4. Data Skewing Logic
  //==========================================================
  genvar r, c;
  generate
    // Row Input Control (Matrix A -> Row Input)
    for (r = 0; r < ROWS; r++) begin : GEN_SKEW_A
      always_comb begin
        array_a_in[r] = '0; // Default 0 padding
        if (state == RUN) begin
          // Timing Logic: time(cnt) - row_index(r) = col_index(k)
          int k;
          k = cnt - r;
          if (k >= 0 && k < K_DIM) begin
            array_a_in[r] = latched_mat_a[r][k];
          end
        end
      end
    end

    // Column Input Control (Matrix B -> Col Input)
    for (c = 0; c < COLS; c++) begin : GEN_SKEW_B
      always_comb begin
        array_b_in[c] = '0; // Default 0 padding
        if (state == RUN) begin
          // Timing Logic: time(cnt) - col_index(c) = row_index(k)
          int k;
          k = cnt - c;
          if (k >= 0 && k < K_DIM) begin
            array_b_in[c] = latched_mat_b[k][c];
          end
        end
      end
    end
  endgenerate

  //==========================================================
  // 5. Systolic Array Instantiation
  //==========================================================
  systolic_array_2d #(
    .DATA_W (DATA_W),
    .ACC_W  (ACC_W),
    .ROWS   (ROWS),
    .COLS   (COLS)
  ) u_core_array (
    .clk        (clk),
    .rst_n      (rst_n),
    .clr        (array_clr),
    .en         (array_en),
    .a_in_row   (array_a_in),
    .b_in_col   (array_b_in),
    .pe_mul     (),
    .pe_acc_sum (raw_acc_sum) // [주의] 최종 출력이 아니라 내부 와이어로 연결
  );
  //==========================================================
  // [NEW] 6. Post-Processing Unit (ReLU + Bias)
  //==========================================================
  generate
    for (r = 0; r < ROWS; r++) begin : GEN_PPU_ROW
      for (c = 0; c < COLS; c++) begin : GEN_PPU_COL

        always_comb begin
          // 1. Bias Subtraction (Signed 연산)
          //    값이 작으면 음수가 될 수 있으므로 int(32bit)로 변환하여 계산
          int temp_val;

          temp_val = int'(raw_acc_sum[r][c]) - BIAS;

          // 2. ReLU (Rectified Linear Unit)
          //    0보다 작으면 0(Dead), 아니면 그대로 통과
          if (temp_val < 0) begin
            o_mat_c[r][c] = '0; // Deactivate (Output 0)
          end else begin
            o_mat_c[r][c] = temp_val[ACC_W-1:0]; // Activate (Pass)
          end
        end

      end
    end
  endgenerate

endmodule
TB

//[5] systolic_array_2d + controller + Activation(ReLu)

`timescale 1ns/1ps

module tb_systolic_controller_relu;

  parameter int DATA_W = 8;
  parameter int K_DIM  = 2;
  parameter int ROWS   = 2;
  parameter int COLS   = 2;

  // [설정] Bias를 50으로 설정 (행렬 곱 결과가 50보다 작으면 0이 됨)
  // 랜덤 입력이 보통 15*15=225 근처이므로, 일부 값은 0이 될 수 있음
  parameter int BIAS   = 50; 

  parameter int ACC_W  = 2*DATA_W + $clog2(K_DIM);

  logic clk, rst_n;

  logic start, done, busy;
  logic signed [DATA_W-1:0] tb_mat_a [0:ROWS-1][0:K_DIM-1];
  logic signed [DATA_W-1:0] tb_mat_b [0:K_DIM-1][0:COLS-1];
  logic signed [ACC_W-1:0]  tb_mat_c [0:ROWS-1][0:COLS-1];

  // Reference Model
  int signed C_ref [0:ROWS-1][0:COLS-1];
  int err_cnt;

  // DUT Instantiation
  systolic_controller_relu #(
    .DATA_W(DATA_W), 
    .K_DIM(K_DIM), 
    .ROWS(ROWS), 
    .COLS(COLS),
    .BIAS(BIAS) // Bias 파라미터 전달
  ) dut (
    .clk        (clk),
    .rst_n      (rst_n),

    // Control Interface
    .i_start    (start),
    .o_done     (done),
    .o_busy     (busy),

    // Data Interface
    .i_mat_a    (tb_mat_a),
    .i_mat_b    (tb_mat_b),
    .o_mat_c    (tb_mat_c)
  );

  // Clock Gen
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    rst_n = 0;
    start = 0;
    tb_mat_a  = '{default:0};
    tb_mat_b  = '{default:0};

    #20 rst_n = 1;

    $display("=== Controller Verification with ReLU (Bias=%0d) ===", BIAS);

    for (int iter = 0; iter < 5; iter++) begin

      // 1. Data Setup (Random)
      //    TB에서는 그냥 평범한 행렬 모양으로 넣으면 됨 (Skew 불필요)
      for(int r=0; r<ROWS; r++)
        for(int k=0; k<K_DIM; k++) 
        tb_mat_a[r][k] = $urandom_range(-(1 << (DATA_W-1)), (1 << (DATA_W-1)) - 1);


      for(int k=0; k<K_DIM; k++)
        for(int c=0; c<COLS; c++) 
          tb_mat_b[k][c] = $urandom_range(-(1 << (DATA_W-1)), (1 << (DATA_W-1)) - 1);

      // 2. Calculate Reference (ReLU Logic 적용)
      for(int r=0; r<ROWS; r++) begin
        for(int c=0; c<COLS; c++) begin
          int temp_sum;
          temp_sum = 0;

          // (1) MAC Calculation
          for(int k=0; k<K_DIM; k++) 
            temp_sum += tb_mat_a[r][k] * tb_mat_b[k][c];

          // (2) Bias Subtraction
          temp_sum = temp_sum - BIAS;

          // (3) ReLU (Activation)
          if (temp_sum < 0) C_ref[r][c] = 0;
          else              C_ref[r][c] = temp_sum;
        end
      end

      // 3. Start Operation
      @(posedge clk);
      start = 1;
      @(posedge clk);
      start = 0;

      // 4. Wait for Done
      wait(done == 1);

      // 5. Check Result
      err_cnt = 0;
      $display("--------------------------------------------------");
      $display("[Iter %0d] Checking Result...", iter);
      for(int r=0; r<ROWS; r++) begin
        for(int c=0; c<COLS; c++) begin
          if (tb_mat_c[r][c] !== C_ref[r][c]) begin
             $display("ERROR! [%0d][%0d]: DUT=%0d, REF=%0d", r, c, tb_mat_c[r][c], C_ref[r][c]);
             err_cnt++;
          end else begin
             // 0이 나오면 ReLU가 동작하여 값을 죽인 것
             if (tb_mat_c[r][c] == 0)
               $display("PASS!  [%0d][%0d]: DUT=%0d (ReLU Activated)", r, c, tb_mat_c[r][c]);
             else
               $display("PASS!  [%0d][%0d]: DUT=%0d", r, c, tb_mat_c[r][c]);
          end
        end
      end

      if (err_cnt == 0) $display("[Iter %0d] FINAL RESULT: PASS!", iter);
      else              $display("[Iter %0d] FINAL RESULT: FAIL! (%0d errors)", iter, err_cnt);
      $display("--------------------------------------------------");

      // 6. Wait a bit before next run
      @(posedge clk);
    end

    $display("=== All Tests Finished ===");
    $finish;
  end

endmodule
Simulation Result



Check Point !!

- Bias Subtraction & ReLU:

Systolic array의 MAC 결과(raw_acc_sum)에 대해 bias subtraction 및 ReLU를 적용합니다.

BIAS는 activation 이전에 적용되는 threshold(문턱값) 역할을 하며, MAC 결과에서 해당 값을 subtract한 뒤 ReLU를 수행합니다.

​

이때 subtraction 결과는 음수가 될 수 있고 bit growth가 발생할 수 있으므로, 중간 연산은 int로 확장하여 signed 산술을 안정적으로 처리합니다.

​

즉, 최종 출력은 다음과 같은 형태로 계산됩니다.

y=max(0, raw_acc_sum−BIAS)

이를 통해 연산 결과가 BIAS보다 작은 경우에는 noise로 간주하여 0으로 제거하고, 충분히 큰 값만 활성화되도록 합니다. 

또한 ReLU 이후 양수 값은 ACC_W 비트 폭으로 출력되며(temp_val[ACC_W-1:0]), ACC_W를 초과하는 상위 비트는 절단됩니다. 

​

본 방식은 학습된 bias를 더하는 전통적인 형태와는 달리, 출력 분포의 하한을 제어하는 threshold-based ReLU로 동작합니다.

이와 같이 OS(Output-Stationary) 기반 Systolic Array를 구현해 보았습니다.

​

본 테스트에서는 2×2 구조를 기준으로 검증을 수행하였으나, 

설계는 parameterized 되어 있으므로 4×4, 5×5, 10×10 등 다양한 크기로 확장이 가능합니다.

​

그렇다면 어디까지 확장하는 것이 적절한가에 대한 고민이 필요합니다.

​

현재 스터디에서는 4×4 array를 표준으로 사용하고 있으나, 이는 절대적인 기준은 아니며 다양한 대안을 함께 고려할 수 있습니다.

​

아래는 array size를 결정할 때 고려해볼 수 있는 주요 관점들입니다.

​

- 각 Layer 에서 연산할때의 효율은?

기존 Reference Model에서는 각 layer의 연산 차원이 Layer1 = 30, Layer2 = 20, Layer3 = 10으로 정의되어 있습니다.

​

이때 weight를 tile 단위로 분할하여 systolic array에 mapping 한다고 가정합니다.

​

먼저 Layer1(30) 을 기준으로 살펴보면,

4×4 array를 사용할 경우 한 tile에서 4개씩 처리하므로 총 7개의 full tile과 2개의 잔여(weight) 가 발생합니다. 

이로 인해 마지막 tile에서는 일부 PE만 활성화되고, 나머지 PE는 IDLE 상태에 들어가게 됩니다.

또한 input activation을 4개 단위로 처리한다고 가정하면, sample 100개에 대해 25개의 tile 이 필요합니다.

​

따라서 전체 연산은 대략적으로

8 (weight tile 수, idle 포함) × 25 (activation tile 수) 로, 총 200번​의 tile 연산 단계가 소요됩니다.

​

반면 5×5 array를 사용할 경우, Layer1 기준으로 weight는 6개의 tile로 정확히 분할되며 idle PE가 발생하지 않습니다. 

또한 activation 역시 5개 단위로 처리되므로 sample 100개에 대해 20개의 tile 이 필요합니다.

이 경우 전체 연산은 6 × 20 = 120번의 tile 연산 단계로 줄어들게 됩니다.

​

즉, 동일한 연산을 수행하더라도 array 크기가 커질수록 tile 분할 효율이 개선되고, 

idle PE가 감소하여 전체 실행 cycle 관점에서 더 높은 효율을 기대할 수 있습니다

​

다만 이러한 효율 향상은 performance 관점에서의 비교이며,

실제 설계 시, data supply rate, interface bandwidth, PPA constraints 등을 함께 고려하여 최종 array size를 결정해야 합니다.

​

​

- 외부 인터페이스 관점 (AXI / PCIe)

실제 동작 환경에서 systolic array는 외부 interface를 통해 지속적으로 데이터를 공급받습니다. 

​

일반적으로 AXI-Lite는 제어 신호 전달을 목적으로 32-bit 폭을 사용하며, 

대용량 데이터 전송을 담당하는 AXI-Full(또는 AXI-Stream)는 32/64/128/256/512-bit와 같이 

2의 거듭제곱 형태의 bus width를 사용하는 것이 일반적입니다.

​

또한 Google TPU v1 역시  PCIe 기반 interface를 사용하며, 

PCIe 역시 lane aggregation 구조상 datawidth가 2의 거듭제곱 단위로 확장되는 특성을 가집니다.

​

이러한 interface 특성을 고려하면, systolic array의 크기 역시 2의 거듭제곱(예: 4×4, 8×8)으로 구성하는 것이 

데이터 packing 및 전송 효율 측면에서 구조적으로 유리할 수 있습니다.

​

정리하자면

입력 activation 및 weight를 bus width에 맞게 정렬(pack)하기 용이

부분 tile 처리로 인한 padding 및 wasted bandwidth 감소

DMA 및 burst 전송 구성 시 address 계산 및 제어 로직 단순화

array와 인터페이스 간 데이터 스트리밍이 보다 일정한 cadence로 유지됨

​

반면, 5×5와 같은 비-2의 거듭제곱 형태의 array는 연산 관점에서는 PE 활용률이 높을 수 있으나, 

인터페이스 측면에서는 bus utilization이 떨어지거나 추가적인 packing/unpacking 로직이 필요해질 수 있습니다.

​

따라서 array size 선택 시에는 단순히 연산 효율(PE utilization)만이 아니라, 

외부 interface의 bus width, 데이터 전송 패턴, 시스템 전체 throughput을 함께 고려하는 것이 중요합니다. 

​

- PPA(Performance / Power / Area) 관점

Systolic array의 크기는 PPA trade-off를 직접적으로 결정합니다. 

일반적으로 array를 키우면 병렬도가 증가하여 성능은 개선되지만, 그 대가로 area과 power 소모가 증가합니다. 

​

따라서 “얼마나 크게 확장할 것인가”는 목표 성능과 시스템 제약(Area/Power budget)에 의해 제한됩니다.

​

- 시스템 전체 Throughput 관점 (PE 효율 vs Bandwidth)

4×4 array는 일부 layer에서 PE utilization이 낮아질 수 있습니다. 

그렇다고 항상 더 큰 array가 유리한 것은 아닙니다.

​

만약 데이터 공급 시간이 SA 연산 시간보다 길다면, 아무리 PE 효율이 높아도 NPU는 입력을 기다리며 idle 상태가 됩니다. 

​

이 경우에는 bus bandwidth와 궁합이 좋은 4×4 구조를 사용하여, 

끊김 없이 데이터를 공급하는 것이 시스템 전체 throughput을 높이는 데 유리할 수 있습니다.

​

반대로, 데이터는 충분히 빠르게 들어오지만 PE 수가 부족하여 연산이 병목이 되는 경우라면, 

5×5와 같은 더 큰 array가 오히려 성능 면에서 유리할 수 있습니다.

이제 지금까지 설계한 systolic array를 실제 layer 구성에 적용하여, mini-TPU 형태의 가속기를 설계해보도록 하겠습니다.
