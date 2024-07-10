import cv2
import dlib
import numpy as np
from scaled_image_test import display_resized_image

# dlib의 얼굴 감지기와 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 이미지 파일 로드
image_path = 'skensense_file\\black\\blackskin (10).png'
image0 = cv2.imread(image_path)
image = display_resized_image(image0)



def analyze_skin(face_region):
    # HSV 색공간으로 변환
    hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
    
    # 피부 톤 분석
    hue = np.mean(hsv[:,:,0])
    saturation = np.mean(hsv[:,:,1])
    value = np.mean(hsv[:,:,2])
    
    # 밝기 분석
    brightness = np.mean(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY))
    
    # 균일성 분석 (표준편차 사용)
    uniformity = np.std(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY))
    
    return hue, saturation, value, brightness, uniformity

def recommend_products(hue, saturation, value, brightness, uniformity):
    recommendations = []
    
    # 보습 제품
    if saturation < 0.37:
        recommendations.append("보습 크림 (건조한 피부용)")
    
    # 지성 피부용 제품
    if saturation > 0.43:
        recommendations.append("오일 프리 모이스처라이저 (지성 피부용)")
    
    # 미백 제품
    if brightness < 126.91:
        recommendations.append("비타민 C 세럼 (피부 톤 개선용)")
    
    # 자외선 차단제
    if brightness > 166.75:
        recommendations.append("자외선 차단제 (높은 SPF)")
    
    # 피부 톤 개선 제품
    if 3.84 <= hue < 12.30:
        recommendations.append("그린 컬러 코렉터 (붉은 기 개선용)")
    elif 10.19 <= hue < 20.65:
        recommendations.append("퍼플 컬러 코렉터 (노란 기 개선용)")
    
    return recommendations



if image is None:
    print(f"Error: Could not load image from path: {image_path}")
    exit()

# 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 감지
faces = detector(gray)
for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 랜드마크 감지
    landmarks = predictor(gray, face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
        
        
    # 얼굴 영역의 마스크 생성
    mask = np.zeros_like(image)
    points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        points.append((x, y))
    
    points = np.array(points, dtype=np.int32)
    cv2.fillConvexPoly(mask, points, (255, 255, 255))
    
    # 얼굴 영역 추출
    face_region = cv2.bitwise_and(image, mask)


for face in faces:
    
    # 피부 분석
    hue, saturation, value, brightness, uniformity = analyze_skin(face_region)
    
    # 제품 추천
    recommendations = recommend_products(hue, saturation, value, brightness, uniformity)
    
    # 결과 출력
    print("피부 분석 결과:")
    print(f"색조(Hue): {hue:.2f}")
    print(f"채도(Saturation): {saturation:.2f}")
    print(f"명도(Value): {value:.2f}")
    print(f"밝기: {brightness:.2f}")
    print(f"균일성: {uniformity:.2f}")
    
    print("\n추천 제품:")
    for product in recommendations:
        print(f"- {product}")

    # 랜드마크 표시 (시각화를 위해)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

# 결과 이미지 표시
cv2.imshow("Face Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
