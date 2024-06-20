import math
import traceback

from copy import copy
from serial import Serial  # pip install pySerial
from threading import Thread, Lock
from typing import Dict


class LidarX2:
    def __init__(self, port: str = "COM3", chunkSize: int = 2000):
        self._port = port
        self._chunkSize = chunkSize
        self._is_connected = False
        self._min_distance = 100
        self._max_distance = 8000
        self._LOCK = Lock()
        self._results_polar = {str(key): 0 for key in range(360)}
        self._results_cartesian = list()

    def __enter__(self):
        self.__open__()
        self._startScan()
        return self

    def __open__(self):
        '''
        Open the serial port and prepare for the scan
        '''
        if self._is_connected:
            return True
        try:
            self.serial = Serial(
                port=self._port,
                baudrate=115200,
                timeout=1
            )
            self._is_connected = True
        except Exception as e:
            print(e)
            self._is_connected = False
        return self._is_connected

    def _startScan(self):
        '''
        Start the scan thread
        '''
        if self._is_connected:
            self._last_chunk = None
            self._scanThread = Thread(target=self._scan, args=(), daemon=False, name="ScanThread")
            self._scanThread.start()
            return True
        return False

    def _endScan(self):
        '''
        End the scan thread
        '''
        if self._is_connected:
            self._scanThread.join()  # 스캔 스레드 종료 기다림
            self._scanThread = None  # 스캔스레드가 종료되었으면 이후 스캔이 다시 실행될 수 있도록 None 설정
            self._last_chunk = None  # 다음 스캔에서 이전 데이터가 사용되지 않도록 None 설정
            return True
        return False

    def __exit__(self, exc_type, exc_val, exc_tb: traceback):
        '''
        Close the serial port and end the scan thread
        '''
        print("YDLidarX2.__exit__(...) is called by: ", exc_type, exc_val)
        traceback.print_tb(exc_tb)
        self._endScan()
        self.serial.close()
        self._is_connected = False

    # 거리에 따른 각도 보정
    def _getAngleCorrection(self, dist):
        if dist != 0:
            return math.atan(21.8 * ((155.3 - dist) / (155.3 * dist))) # 어디서 나온 상수값들?
        return 0

    # 각도 차이
    def _getAngleDiff(self, startAngle, endAngle):
        angleDiff = endAngle - startAngle
        if startAngle > endAngle:
            return 360 - angleDiff
        else:
            return angleDiff

    # 라이다 센서로부터 데이터 읽기
    def _scan(self):
        while True:
            # 시리얼 포트에서 _chunkSize만큼의 값을 읽어와서 '\xaa\x55'를 기준으로 분할하여 dataBlocks에 저장
            dataBlocks = self.serial.read(self._chunkSize).split(b'\xaa\x55')
            # 이전 데이터가 있다면 현재 읽은 값과 합치기
            if self._last_chunk is not None:
                dataBlocks[0] = self._last_chunk + dataBlocks[0]
            self._last_chunk = dataBlocks.pop()

            # 데이터 블록값 분석
            for dataBlock in dataBlocks:
                self._analysisDataBlock(dataBlock)
            # self._cleanResults()

    # Analysis each data block
    def _analysisDataBlock(self, dataBlock):
        lenOfDataBlock = len(dataBlock)
        # print(lenOfDataBlock)
        if lenOfDataBlock < 10:
            # Reasonable length of the data slice?
            return list()

        # Get sample count and start and end angle
        sampleCount = dataBlock[1]
        if sampleCount == 0 or sampleCount == 1:
            # If sample count is 0 return list which length == 0
            return list()

        # Distance analysis
        distances = list()
        # 거리정보 9번째 바이트부터 시작, 두 바이트씩 끊어서 처리
        for i in range(8, lenOfDataBlock - 1, 2):
            # 바이너리 값 10진수 변환
            dist = dataBlock[i] + 256 * dataBlock[i + 1]
            # 최대, 최소 거리 범위 확인
            if dist > self._max_distance:
                dist = 0
            if dist < self._min_distance:
                dist = 0
            # 유효 거리값이면 distances 리스트에 추가
            distances.append(dist)

        # First-level analysis
        startAngle = ((dataBlock[2] + 256 * dataBlock[3]) >> 1) / 64
        endAngle = ((dataBlock[4] + 256 * dataBlock[5]) >> 1) / 64
        angleDiff = self._getAngleDiff(startAngle, endAngle)
        firstAnalysis = list()
        for i in range(2, sampleCount + 2, 1):
            angle_i = ((angleDiff * (i - 1)) / (sampleCount - 1)) + startAngle
            firstAnalysis.append(angle_i)

        # Second-level analysis
        angleCorrections = list()
        for distance in distances:
            angleCorrection = self._getAngleCorrection(distance)
            angleCorrections.append(angleCorrection)

        # Intergrate analysis
        for angle_i, angleCorrection_i, distance_i in zip(firstAnalysis, angleCorrections, distances):
            angle = round(angle_i + angleCorrection_i)
            distance = distance_i
            if angle >= 360:
                angle -= 360

            if angle >= 0 and angle < 360:
                self._results_polar[str(angle)] = distance

    # 각도 범위 벗어나는 키-값 쌍 제거
    def _cleanResults(self):
        self._LOCK.acquire()
        for key in list(self._results_polar.keys()):
            key = float(key)
            if 0 > key or key < 360:
                print("Delete key: ", key, " value: ", self._results_polar[key])
                del self._results_polar[key]
        self._LOCK.release()

    # 각도:거리 딕셔너리 반환
    def getPolarResults(self) -> Dict[str, int]:
        '''
        Get the polar coordinate results
        Dictionalry: {angle(Radian): distance(mm)}
        '''
        self._LOCK.acquire()
        results = copy(self._results_polar)
        self._LOCK.release()
        return results


### 사용법 참고하세요 ###

# Lidar = LidarX2()  # 객체 만들고
# with Lidar as lidar:  # with문으로 열어서 사용 => 자동으로 스캔 시작.
#     # with문을 벗어나면 자동으로 Serial 연결 끊어지고, Thread 종료함
#     while True:
#         result = lidar.getPolarResults()  # 극좌표계 결과 받아오기
#         print(result)