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

### Proposed Model (Systolic Array, Ver.2) – 10,000 Samples Inference

Proposed TB에서도 동일하게 sample 개수를 확장합니다.

        localparam int NUM_SAMPLES = 10000;
        
fileName mapping은 다음과 같이 5-digit zero-padding을 적용합니다.

        f0 = $sformatf("test_data_%05d.txt", base_idx + 0);

- Inference 수행 결과:
    - Accuracy: 96.32%
    - PASS: 9,632
    - FAIL: 368

<div align="center"><img src="https://github.com/yakgwa/Mini_NPU/blob/main/Picture_Data/image_123.png" width="400"/>

Proposed Model 10,000-sample inference 결과

<div align="left">

### Reference vs Proposed 비교 분석

- Accuracy
    - Reference Model & Proposed Model : 96.32%

​- Inference Time (Simulation-based)
    - Reference: 87,300,235 ns (≈ 87.30 ms)
    - Proposed: 174,150,145 ns (≈ 174.15 ms)

- Proposed Model은 4-lane 병렬 구조로, 1회 inference(done 1회)에 4개 sample을 동시에 처리합니다. 따라서 단순 실행 시간만으로는 두 구조를 직접 비교하기 어려우며, sample당 평균 latency 기준으로 비교하는 것이 적절합니다.

|Metric|MLP (Reference)|Systolic Array (Proposed)|
|------|---|---|
|Accuracy|96.32%|96.32%|
|Total Inference time(sample 10000)|87,300,235 ns (≈ 87.30 ms)|174,150,145 ns (≈ 174.15 ms)|
|Average latency per sample|8,730.02 ns (≈ 8.73 μs)|17,415.01 ns (≈ 17.42 μs)|
|Throughput (samples/s)|114,547|57,422|
|Relative speed|1.00×|0.50×|
|Accuracy difference|baseline|0.00%p|

### 결과 의의
본 실험에서는 기존 MLP 구조와 4×4 Systolic Array 구조를 비교하였습니다.

두 구조는 동일한 정확도(96.32%)를 보였으므로, Systolic Array를 적용하더라도 모델의 기능적 정확도는 유지됨을 확인하였습니다. 다만 전체 실행 시간은 Proposed 구조가 더 길게 측정되었습니다. 

​Proposed Model은 4-lane 병렬 구조로 1회에 4개 sample을 동시에 처리하지만, flush, BUFFER_WR 구간, FSM 제어와 같은 시스템 수준 오버헤드가 병렬 처리의 이점을 상쇄하였습니다.

또한 본 설계에서는 4×4 규모의 소규모 Systolic Array를 사용하였기 때문에, 이러한 오버헤드의 상대적 비중이 더욱 크게 나타났습니다. 

더 큰 규모의 Systolic Array를 사용하는 경우 연산량 대비 오버헤드의 상대적 비중이 감소하여, 병렬 계산의 이점이 실제 성능 향상으로 이어질 가능성이 높습니다.

정리하면 다음과 같습니다.

- 정확도는 기존 구조와 동일하게 유지되었습니다. Systolic Array 기반 병렬 계산 구조가 정상적으로 동작함을 확인하였습니다.

- 다만 flush, BUFFER_WR 구간, FSM 제어와 같은 시스템 수준 오버헤드로 인해 전체 throughput 향상은 나타나지 않았습니다. 이는 소규모 SA 설계에서 나타나는 구조적 특성으로, 규모 확장 시 개선될 수 있습니다.

- 추후 flush cycle 감소, BUFFER_WR 구간 최적화, dataflow 개선을 통해 Systolic Array의 병렬 처리 이점을 실제 throughput 향상으로 연결할 수 있을 것입니다.


