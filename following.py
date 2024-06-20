
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

import PoseDetector as pd
from YDLidarX2 import LidarX2
import cv2
import serial
import threading
import time

detector = pd.poseDetector()
serial = serial.Serial('COM3', 115200)
target_distance = 1700   # 목표 거리
move_ready = False
lock = threading.Lock()  # 쓰레드 간 동기화를 위한 락 객체

cap = cv2.VideoCapture(0)
def locationBodyCenter():
    global move_ready
    while True:
        success, img = cap.read()
        if not success:
            print("카메라 프레임 읽기 실패")
            cap.release()
            time.sleep(0.1)  # 잠시 대기 후 다시 시도
            continue

        img = detector.findPose(img, False)
        lmList = detector.getPosition(img, False)
        h, w, c = img.shape
        #몸 중앙이 들어와야 하는 화면 가로 범위
        start_x = w * 4 // 9
        end_x = w * 5 // 9

        center_x, center_y = None, None
        if len(lmList) != 0:
            center_x, center_y = detector.getBodycenter(img)

        if center_x is not None:
            if center_x < start_x:
                with lock:
                    move_ready = False
                serial.write('R'.encode('utf-8'))
                print('R')
            elif center_x > end_x:
                with lock:
                    move_ready = False
                serial.write('L'.encode('utf-8'))
                print('L')
            else:
                with lock:
                    move_ready = True
                    print("사용자 화면 중앙 위치 완료")

        cv2.imshow("image", img)
        if cv2.waitKey(1) == ord('q'):
            break

def forwardToUser():
    global move_ready
    with LidarX2() as lidar:
        while True:
            with lock:
                if move_ready:
                    result = lidar.getPolarResults()
                    filtered_result = {key: value for key, value in result.items()
                                       if int(key) in range(0, 20) or int(key) in range(340, 360)}
                    print(filtered_result)
                    filtered_values = [value for value in filtered_result.values() if value != 0]  # 0이 아닌 값만 필터링

                    if filtered_values:  # 리스트가 비어있지 않다면
                        min_value = min(filtered_values)  # 최소값 계산
                        print(min_value)
                        if min_value >= target_distance:
                            serial.write('F'.encode('utf-8'))
                            print('F')
                        elif min_value < target_distance:
                            serial.write('B'.encode('utf-8'))
                            print('B')
                    else:
                        print("리스트 비어있음")
                        break

# 각각의 작업을 병렬로 실행
camera_thread = threading.Thread(target=locationBodyCenter)
lidar_thread = threading.Thread(target=forwardToUser)
try:
    camera_thread.start()
    lidar_thread.start()
except Exception as e:
    print(f"스레드 시작 오류: {e}")

camera_thread.join()
lidar_thread.join()

cap.release()
cv2.destroyAllWindows()

# --------------------
# def locationBodyCenter():
#     global move_ready
#     while True:
#         success, img = cap.read()  # 카메라에서 새로운 프레임을 읽어옴
#         if not success:
#             break
#
#         if img is None:
#             continue
#
#         img = detector.findPose(img, False)
#         lmlist = detector.getPosition(img, False)
#
#         h, w, c = img.shape
#
#         # 몸 중앙이 들어와야 하는 화면 가로 범위
#         start_x = w * 10 // 21
#         end_x = w * 11 // 21
#
#         center_x, center_y = None, None
#         if len(lmlist) != 0:
#             center_x, center_y = detector.getBodycenter(img)
#
#         if center_x is not None:
#             if center_x < start_x:
#                 serial.write('R'.encode('utf-8'))
#                 print('R')
#             elif center_x > end_x:
#                 serial.write('L'.encode('utf-8'))
#                 print('L')
#             else:
#                 with lock:
#                     move_ready = True
#                     print("사용자 화면 중앙 위치 완료")
#                     break
#
#         cv2.imshow("image", img)
#         cv2.waitKey(1)
#
#
# def forwardToUser():
#     global move_ready
#     with LidarX2() as lidar:
#         while True:
#             with lock:
#                 if move_ready:
#                     result = lidar.getPolarResults()
#                     filtered_result = {key: value for key, value in result.items()
#                                        if int(key) in range(0, 20) or int(key) in range(340, 360)}
#                     print(filtered_result)
#                     filtered_values = [value for value in filtered_result.values() if value != 0]  # 0이 아닌 값만 필터링
#
#                     if filtered_values:  # 리스트가 비어있지 않다면
#                         min_value = min(filtered_values)  # 최소값 계산
#                         print(min_value)
#                         if min_value >= target_distance:
#                             serial.write('F 40'.encode('utf-8'))
#                             print('F')
#                         elif min_value < target_distance:
#                             serial.write('S'.encode('utf-8'))
#                             print('S')
#                             break
#                     else:
#                         print("리스트 비어있음")
#                         break