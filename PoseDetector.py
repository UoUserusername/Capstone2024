import cv2
import math
import mediapipe as mp

class poseDetector:
    # mediapipe에서 제공하는 기본값 사용
    def __init__(self, mode=False, mComp=1, smoothLM=True, enableSeg=False,
                 smoothSeg=True, detectionConf=0.5, trackingConf=0.5):
        self.mode = mode
        self.mComp = mComp
        self.smoothLM = smoothLM
        self.enableSeg = enableSeg
        self.smoothSeg = smoothSeg
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.mComp, self.smoothLM, self.enableSeg, self.smoothSeg,
                                      self.detectionConf, self.trackingConf)
        self.mp_draw = mp.solutions.drawing_utils

    # 랜드마크, 선 그리기 (선택 가능, 기본 True)
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return img

    # 관절 위치 얻기
    def getPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape

                # z좌표는 사용하지 않고 2D로 판단 진행
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        return self.lmList

    def getAngle(self, img, p1, p2):
        # 원하는 관절 좌표 얻기
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]

        # 각도 계산
        slope = (y1 - y2)/(x1 - x2 + 0.1)  # 두 관절 사이 기울기
        angle = abs(math.degrees(math.atan(slope)))  # 바닥과의 각도
        # 예각 사용
        if angle > 90:
            angle = angle - 90

        return angle

    # 몸 중심 좌표
    def getBodycenter(self, img):
        # 왼쪽어깨(11), 오른쪽어깨(12), 왼쪽 골반(23), 오른쪽 골반(24) 좌표 평균값
        x1, y1 = self.lmList[11][1:]
        x2, y2 = self.lmList[12][1:]
        x3, y3 = self.lmList[23][1:]
        x4, y4 = self.lmList[24][1:]

        center_x, center_y = (x1 + x2 + x3 + x4) / 4, (y1 + y2 + y3 + y4) / 4

        return center_x, center_y