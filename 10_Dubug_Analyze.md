## Reference Model Golden 확정 및 Intermediate Trace 기반 Debug 분석

앞서 Reference Model에 대해 100개의 sample을 대상으로 simulation을 수행한 결과, 99.0%의 accuracy를 확보하였음을 확인하였습니다.



이제부터는 해당 Reference Model을 Golden Model로 간주하고, 각 layer(내부 neuron 포함)에서의 intermediate result를 추출하고자 합니다.

​추출된 intermediate result는 향후 systolic array가 적용된 Proposed Model의 연산 결과와 비교하기 위한 기준(reference value)으로 활용됩니다.

​이를 위해 기존 Reference Model의 testbench 하단에 Debug Monitor 코드를 추가하였습니다.

        //`ifdef DEBUG_LAYER_OUTPUT
        // --- Layer Output Monitors ---
        integer m, n, p;
        reg [`dataWidth-1:0] l1_unpacked [29:0];
        reg [`dataWidth-1:0] l2_unpacked [19:0];
        reg [`dataWidth-1:0] l3_unpacked [9:0];
        // layer별 output을 한 번만 출력하기 위한 flag
        reg l1_printed = 0, l2_printed = 0, l3_printed = 0; 
    
        always @(posedge clock) begin
            if (dut.o1_valid[0] == 1 && l1_printed == 0) begin                      // Layer1 output valid 시점 검출
                $display("\n============================================================");
                $display("[Ref] Layer 1 Calculation Complete! (Img 0 Output Matrix [1x30])");
                $display("============================================================");
                $write("[ ");
                for (m=0; m<30; m=m+1) begin                                        // packed bus 형태의 x1_out을 neuron 단위로 unpack
                    l1_unpacked[m] = dut.x1_out[m*`dataWidth +: `dataWidth];
                    $write("%d ", $signed(l1_unpacked[m]));                         // signed 기준으로 출력
                    if ((m+1)%10 == 0 && m != 29) $write("\n  ");                   // 가독성을 위한 line break
                end
                $write(" ]\n");
                $display("============================================================\n");
                l1_printed <= 1;                                                    // 동일 layer에 대해 중복 출력 방지
            end
    
            if (dut.o2_valid[0] == 1 && l2_printed == 0) begin                      // Layer2 output valid 시점 검출
                $display("\n============================================================");
                $display("[Ref] Layer 2 Calculation Complete! (Img 0 Output Matrix [1x20])");
                $display("============================================================");
                $write("[ ");
                for (n=0; n<20; n=n+1) begin                                        // packed bus 형태의 x2_out을 neuron 단위로 unpack
                    l2_unpacked[n] = dut.x2_out[n*`dataWidth +: `dataWidth];
                    $write("%d ", $signed(l2_unpacked[n]));
                    if ((n+1)%10 == 0 && n != 19) $write("\n  ");
                end
                $write(" ]\n");
                $display("============================================================\n");
                l2_printed <= 1;                                                    // one-shot dump
            end
    
            if (dut.o3_valid[0] == 1 && l3_printed == 0) begin                      // Layer3 output valid 시점 검출
                $display("\n============================================================");
                $display("[Ref] Layer 3 Calculation Complete! (Img 0 Output Matrix [1x10])");
                $display("============================================================");
                $write("[ ");
                for (n=0; n<10; n=n+1) begin
                    l3_unpacked[n] = dut.x3_out[n*`dataWidth +: `dataWidth];
                    $write("%d ", $signed(l3_unpacked[n]));
                    if ((n+1)%10 == 0 && n != 9) $write("\n  ");
                end
                $write(" ]\n");
                $display("============================================================\n");
                l3_printed <= 1; 
            end
        end
        //`endif
    
        // --- [Added] Debug Monitor for Neuron 0 & 4 (L1 & L2) ---
        //`ifdef DEBUG_PIXEL
        always @(posedge clock) begin
            // Layer 1 Monitor (Neuron 0)
            // inputValid 또는 mult_valid 발생 시 MAC 내부 동작 추적용 log 출력
            if (dut.l1.n_0.myinputValid || dut.l1.n_0.mult_valid) begin
                $display("[Ref_DEBUG_L1_N0] Time=%0t | k=%d | Input=%d | Weight=%d | Mul=%d | Sum=%d",
                         $time,
                         dut.l1.n_0.r_addr,                                         // 현재 read address (input index)
                         $signed(dut.l1.n_0.myinputd),                              // input activation
                         $signed(dut.l1.n_0.w_out),                                 // weight
                         $signed(dut.l1.n_0.mul),                                   // multiplication result
                         $signed(dut.l1.n_0.sum));                                  // accumulated sum
            end                             
    
            // Layer 2 Monitor (Neuron 0)
            // Layer1 output이 Layer2 MAC에 반영되는 흐름 확인용
            if (dut.l2.n_0.myinputValid || dut.l2.n_0.mult_valid) begin
                $display("[Ref_DEBUG_L2_N0] Time=%0t | k=%d | Input=%d | Weight=%d | Mul=%d | Sum=%d",
                         $time,
                         dut.l2.n_0.r_addr,
                         $signed(dut.l2.n_0.myinputd),
                         $signed(dut.l2.n_0.w_out),
                         $signed(dut.l2.n_0.mul),
                         $signed(dut.l2.n_0.sum));
            end
        end
        //`endif

추가한 Debug Monitor는 두 가지 기능으로 구성됩니다.

1. Layer Output Monitor

    각 layer의 output이 valid 되는 시점을 기준으로 해당 layer의 전체 output vector를 simulation 전체 기준 one-shot으로 출력하도록 구성하였습니다.
    Layer1~Layer3의 packed bus 형태 출력(x1_out, x2_out, x3_out)을 neuron 단위로 unpack한 뒤 $signed() 기준으로 출력함으로써,
  - layer별 activation 분포
  - 연산 결과의 정상 여부
  - layer 간 데이터 전달 상태
    를 확인할 수 있습니다.

​

해당 출력은 이후 Proposed Model과의 비교를 위한 Golden reference로 활용됩니다.

​

2. Neuron MAC Debug Monitor

특정 neuron에 대해서는 MAC(Multiply-Accumulate) 동작을 cycle 단위로 추적할 수 있도록 하였습니다.

​

출력 항목은 다음과 같습니다.

k : 현재 input index

Input : activation 값

Weight : 해당 index의 weight

Mul : multiplier 결과

Sum : accumulated 값

​

이를 통해 다음 항목을 직접 검증할 수 있습니다.

input과 weight indexing이 올바르게 매칭되는지

multiplier 결과가 정상적으로 생성되는지

pipeline 지연이 존재하는지

accumulation이 의도한 순서대로 반영되는지

​

즉, 최종 classification 결과만 비교하는 것이 아니라, 내부 연산 흐름을 단계별로 검증할 수 있습니다.

​

Simulation Log 확인

simulation log를 통해 test_data_0000.txt가 input으로 주입된 이후, 

Layer 1 neuron에서 MAC 연산이 cycle 단위로 수행되는 과정을 확인할 수 있습니다.

Time resolution is 1 ps
Configuration completed                      0 ns
Filename:     test_data_0000.txt
[Ref_DEBUG_L1_N0] Time=285000 | k=   0 | Input=   x | Weight=   x | Mul=     x | Sum=     0
[Ref_DEBUG_L1_N0] Time=295000 | k=   1 | Input=   0 | Weight=   0 | Mul=     x | Sum=     0

...

1. Accuracy: 100.000000, Detected number: 7, Expected: 07
보다 상세한 scope에서 연산 흐름을 확인해보면, 

다음과 같이 (Name / Time / k / Input / Weight / Mul / Sum) 형식의 debug log가 출력됩니다.

[Ref_DEBUG_L2_N0] Time=8195000 | k= 0 | Input=   x | Weight=   x | Mul=     x | Sum=     0
[Ref_DEBUG_L2_N0] Time=8205000 | k= 1 | Input=  72 | Weight=  -9 | Mul=     x | Sum=     0
[Ref_DEBUG_L2_N0] Time=8215000 | k= 2 | Input=  17 | Weight= -17 | Mul=  -648 | Sum=     0
[Ref_DEBUG_L2_N0] Time=8225000 | k= 3 | Input=  95 | Weight= -27 | Mul=  -289 | Sum=  -648
[Ref_DEBUG_L2_N0] Time=8235000 | k= 4 | Input=  30 | Weight=  26 | Mul= -2565 | Sum=  -937
[Ref_DEBUG_L2_N0] Time=8245000 | k= 5 | Input=   0 | Weight=   0 | Mul=   780 | Sum= -3502
k=0 시점에서는 input과 weight가 아직 유효하지 않기 때문에 x로 출력되며, 

k=1부터 실제 input과 weight 값이 입력되는 것을 확인할 수 있습니다.

​

다만 해당 Reference Model은 data_pipe 구조를 사용하고 있으므로, 

초기 몇 cycle 동안은 이전 cycle의 미유효 입력이 multiplier 경로에 반영됩니다.

그 결과 k=1 시점의 multiplier 출력 역시 x로 나타납니다.

​

k=2부터는 multiplier 연산이 정상적으로 수행되며, 해당 결과는 pipeline 지연으로 인해 k=3 시점부터 sum에 반영됩니다.

​

즉, Input이 유효해진 시점을 기준으로 multiplier 결과가 sum에 반영되기까지 2 cycle latency가 존재함을 확인할 수 있습니다.

​

최종적으로 test_data_0000(숫자 7에 해당하는 입력 데이터)에 대해 각 layer의 연산 결과를 정상적으로 추출할 수 있었습니다.
