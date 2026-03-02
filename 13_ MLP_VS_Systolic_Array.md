## MLP (Reference) vs Systolic Array (Proposed)_(1): Sample 10,000개 비교

앞서 Systolic Array가 적용된 Proposed Model (Ver.2) 구현을 완료한 후, 100개 sample에 대해 inference Accuracy를 측정하고 Reference Model (MLP)과 비교하였습니다.

​그 결과: Reference Model: 99.0%, Proposed Model (Ver.2): 98.0%의 Accuracy를 확인하였습니다.

​이번에는 실제 MNIST test dataset의 10,000개 sample로 확장하여, 보다 정량적인 성능 비교를 수행합니다.

### MNIST Test Sample 10,000개 생성

두 모델 비교에 앞서, inference에 사용될 10,000개 sample을 생성합니다.

기존에 사용한 genTestData.py를 수정하여, 생성 경로를 다음과 같이 지정합니다.

    outputPath = "C:/Users/x/Desktop/NPU/Sample_Data/"
    headerFilePath = "C:/Users/x/Desktop/NPU/Sample_Data/"
    
이후 sample 개수를 10,000개로 설정하고, file name zero-padding을 5-digit 기준으로 수정합니다.

        for i in range(10000):
            if i < 10:
                ext = "0000" + str(i)
            elif i < 100:
                ext = "000" + str(i)
            elif i < 1000:
                ext = "00" + str(i)
            elif i < 10000:
                ext = "0" + str(i)
            else:
                ext = str(i)
    
            fileName = "test_data_" + ext + ".txt"
            f = open(outputPath + fileName, "w")
    
            for j in range(0, x):
                dInDec = DtoB(test_inputs[i][0][j], dataWidth, d)
                myData = bin(dInDec)[2:]
                f.write(myData + "\n")
    
            f.write(bin(DtoB(te_d[1][i], dataWidth, 0))[2:])
            f.close()
            
Windows PowerShell에서 다음 명령어를 실행하여 파일 생성을 확인합니다.

    py genTestData.py
이를 통해 MNIST test dataset 10,000개에 대한 inference 입력 파일이 생성됩니다.

### Reference Model (MLP) – 10,000 Samples Inference

Reference Model TB에서 다음과 같이 sample 개수를 확장합니다.

    `define MaxTestSamples 10000
    
Reference TB는 sample index로부터 fileName을 생성할 때 5-digit zero-padding을 전제로 동작합니다.

                for (i = 0; i < 5; i = i + 1) 
                begin
                    fileNum[i] = 8'b0; // Initialize each register to 0
                end
    
                i=0;
                while(testDataCount_int != 0)
                begin
                    fileNum[i] = (testDataCount_int%10);
                    testDataCount_int = testDataCount_int/10;
                    i=i+1;
                end
                
                for (i = 5; i > 0; i = i-1)
                begin
                    if (fileNum[i - 1] == 0)
                    begin
                        fileName = {fileName, "0"};
                    end
                    else
                    begin
                        fileName = {fileName, to_ascii(fileNum[i - 1])};
                    end
                end
                
- 초기 Issue: Zero-Padding Overflow : 초기에는 MaxTestSamples만 10,000으로 수정하였고, filename mapping 과정에서 4-digit 기준 변수/로직이 그대로 유지되어 있었습니다.

​- 그 결과:
  - 5-digit sample index 처리 중 overflow 발생
  - fileName mismatch 발생
  - 잘못된 입력 데이터 load

​- Accuracy가 100-sample 실행 결과와 동일하게 출력되었으나,log 확인 결과 정상적인 10,000-sample inference가 아님을 확인했습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_116.png" width="400"/>

Overflow 발생 (4-digit zero padding 미반영 → fileName mismatch)​

<div align="left">

- filename mapping 로직에서 5-digit zero-padding이 정확히 반영되도록 수정한 뒤 재측정하였습니다.

​- 그 결과:
  - Accuracy: 96.32%
  - PASS: 9,632
  - FAIL: 368

- 정상적인 10,000-sample inference 결과를 확인하였습니다.

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_117.png" width="400"/>

<div align="left">

2. Proposed Model (Systolic Array, Ver.2) – 10,000 Samples Inference

Proposed TB에서도 동일하게 sample 개수를 확장합니다.

localparam int NUM_SAMPLES = 10000;
fileName mapping은 다음과 같이 5-digit zero-padding을 적용합니다.

f0 = $sformatf("test_data_%05d.txt", base_idx + 0);
Inference 수행 결과:

Accuracy: 94.99%

PASS: 9,499

FAIL: 501


3. Reference vs Proposed 비교 분석

3.1 Accuracy

Reference Model: 96.32%

Proposed Model: 94.99%

Accuracy Degradation: −1.33%p (Ref 대비)

Proposed Model에서 추가적인 분류 오차가 발생하였습니다.

​

3.2 Inference Time (Simulation-based)

Reference: 87,300,235 ns (≈ 87.30 ms)

Proposed: 173,350,145 ns (≈ 173.35 ms)

1회 inference(done 1회) 기준으로 Proposed Model의 시간이 더 길었습니다.

​

3.3 Throughput (samples/s)

Systolic Array는 1회 inference에서 4 samples를 동시에 처리합니다.

따라서 throughput은 inferences/s가 아닌 samples/s 기준으로 환산해야 합니다.

Reference: 11.45 samples/s

Proposed: 23.07 samples/s

1회 inference 시간은 더 길지만, 병렬 처리로 인해 samples/s 기준 throughput은 Proposed가 더 높습니다.

​

MLP (Reference)

Systolic Array (Proposed)

Accuracy

96.32%

94.99% (Ref 대비 −1.33%p)

inference time

87,300,235 ns (≈ 87.30 ms)

173,350,145 ns (≈ 173.35 ms)

Throughput (samples/s)

11.45

23.07

4. 결과 의의

4.1 Inference Time 측정 한계

현재 Inference Time은 Simulation-based measurement입니다.

​

해당 값에는 다음이 포함될 수 있습니다:

TB reset 구간

idle cycle

wait overhead

따라서 pure Compute Time을 대표한다고 보기 어렵습니다.

​

정확한 연산 시간 측정을 위해서는:

start_inference → (실제 compute 시작 signal) → done_interrupt
구간에 대해 Compute Window 기반 Time Flag를 설정하여 재측정할 필요가 있습니다.

​

4.2 Accuracy Degradation 원인 분석

이론적으로 동일한 MAC 연산 / Accumulation 방식 / Q-format / Activation Function 을 유지했다면 

Reference와 Proposed는 동일 Accuracy를 보여야 합니다.

​

그러나 −1.33%p 차이가 발생하였으므로, 다음 두 관점에서 원인 분석이 필요합니다.

(1) Numerical Precision Loss 가능성

Rounding/Truncation 위치 차이

Saturation 타이밍 차이

Accumulation bit-growth 처리 차이

Quantization error 누적

(2) Functional / Timing Issue 가능성

Input/Weight skew misalignment

Flush cycle 부족

Done 시점과 output 안정화 타이밍 불일치

Lane별 data mapping 오류

​

해당 항목은 설계 과정에서 검토되었으나,

Accuracy Degradation 원인 배제를 위해 Reference 대비 비교 분석을 추가로 수행이 필요합니다.

5. 추가 검증 항목

5.1 Compute Window 기반 Latency 재측정

(추후 기재 예정)

​

5.2 Accuracy Degradation 원인 분리 분석

(추후 기재 예정)
