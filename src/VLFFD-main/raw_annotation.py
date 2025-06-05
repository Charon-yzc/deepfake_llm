import cv2
import numpy as np
import dlib
import os
import random
from tqdm import tqdm
from glob import glob
import skimage
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pdb
from skimage.metrics import structural_similarity
from scipy.ndimage import binary_erosion, binary_dilation
# 常量定义
PREDICTOR_PATH = "/shape_predictor_68_face_landmarks.dat"

# 人脸关键点定义
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
BROW_POINTS = RIGHT_BROW_POINTS + LEFT_BROW_POINTS
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
EYE_POINTS = RIGHT_EYE_POINTS + LEFT_EYE_POINTS
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))
def dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:

    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)



class BlendingDetector:
    def __init__(self):
        self.gradient_threshold = 0.5
        self.edge_threshold = 0.4
        self.boundary_width = 10  # 边界检测的宽度

    def get_boundary_region(self, mask: np.ndarray, width: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取mask的内外边界区域
        Args:
            mask: 输入mask
            width: 边界宽度
        Returns:
            inner_boundary: 内边界区域
            outer_boundary: 外边界区域
        """
        # 确保mask是二值的
        binary_mask = (mask > 0).astype(np.uint8)
        
        # 获取内部收缩区域和外部扩张区域
        kernel = np.ones((3,3), np.uint8)
        inner_mask = cv2.erode(binary_mask, kernel, iterations=width)
        outer_mask = cv2.dilate(binary_mask, kernel, iterations=width)
        
        # 获取内外边界
        inner_boundary = binary_mask - inner_mask  # 内边界
        outer_boundary = outer_mask - binary_mask  # 外边界
        
        return inner_boundary, outer_boundary

    def detect_edge_artifacts(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        检测边界处的边缘伪影
        """
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
            
        if len(mask.shape) == 3:
            mask = mask[..., 0]
            
        # 获取边界区域
        inner_boundary, outer_boundary = self.get_boundary_region(mask, self.boundary_width)
        total_boundary = inner_boundary + outer_boundary
        
        # 在边界区域进行边缘检测
        edges = cv2.Canny(gray_image, 100, 200)
        boundary_edges = edges * total_boundary
        
        # 计算边界处的边缘密度
        edge_density = np.sum(boundary_edges) / (np.sum(total_boundary) + 1e-6)
        return edge_density

    def detect_gradient_discontinuity(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        检测边界处的梯度不连续性
        """
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
            
        if len(mask.shape) == 3:
            mask = mask[..., 0]
            
        # 获取边界区域
        inner_boundary, outer_boundary = self.get_boundary_region(mask, self.boundary_width)
        
        # 计算梯度
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # 计算内外边界的梯度差异
        inner_gradient = np.mean(gradient_magnitude[inner_boundary > 0])
        outer_gradient = np.mean(gradient_magnitude[outer_boundary > 0])
        
        gradient_diff = abs(inner_gradient - outer_gradient) / (outer_gradient + 1e-6)
        return gradient_diff

    def detect_frequency_artifacts(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        检测边界处的频率特征
        """
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
            
        if len(mask.shape) == 3:
            mask = mask[..., 0]
            
        # 获取边界区域
        inner_boundary, outer_boundary = self.get_boundary_region(mask, self.boundary_width)
        total_boundary = inner_boundary + outer_boundary
        
        # 在边界区域进行DCT变换
        boundary_image = gray_image * total_boundary
        dct = cv2.dct(np.float32(boundary_image))
        
        # 分析高频成分
        h, w = dct.shape
        high_freq = dct[h//2:, w//2:]
        low_freq = dct[:h//2, :w//2]
        
        freq_ratio = np.sum(np.abs(high_freq)) / (np.sum(np.abs(low_freq)) + 1e-6)
        return freq_ratio

    def detect_blending_artifacts(self, image: np.ndarray, mask: np.ndarray) -> Tuple[bool, List[str]]:
        """
        综合检测blending artifacts
        """
        evidence = []
        
        # 检测梯度不连续
        gradient_score = self.detect_gradient_discontinuity(image, mask)
        if gradient_score > self.gradient_threshold:
            evidence.append("sharp changes in image gradients at the boundaries")
        
        # 检测边缘伪影
        edge_score = self.detect_edge_artifacts(image, mask)
        if edge_score > self.edge_threshold:
            evidence.append("unnatural edge patterns")
        
        # 检测频率域异常
        freq_score = self.detect_frequency_artifacts(image, mask)
        if freq_score > 0.5:
            evidence.append("unusual frequency patterns at the boundaries")
        
        # 判断是否存在足够的证据
        has_artifacts = len(evidence) >= 2
        
        return has_artifacts, evidence

@dataclass
class ForgeryFeatures:
    """伪造特征定义"""
    # 基础视觉特征
    BLUR = "blurry"
    NOISE = "noisy"
    ARTIFACT = "has digital artifacts"
    
    # 颜色相关
    COLOR_MISMATCH = "has inconsistent colors"
    SKIN_TONE = "has unnatural skin tone"
    LIGHTING = "has inconsistent lighting"
    
    # 结构相关
    DEFORMATION = "is deformed"
    ASYMMETRIC = "is asymmetric"
    PROPORTION = "has unnatural proportions"
    
    # 纹理相关
    TEXTURE_LOSS = "lacks natural texture"
    TEXTURE_ARTIFACTS = "has unnatural texture"
    
    # 边界相关
    BLENDING = "has visible blending artifacts"
    BOUNDARY = "has unnatural boundaries"

class ForgeryDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)
        self.blending_detector = BlendingDetector()
        self.mask_threshold = 55  # 添加mask判断的阈值
        
        # 阈值设置
        self.BLUR_THRESHOLD = 100
        self.COLOR_THRESHOLD = 2.0
        self.TEXTURE_THRESHOLD = 0.7
        self.STRUCTURE_THRESHOLD = 0.97
        
    def get_landmarks(self, image):
        """获取人脸关键点"""
        rects = self.detector(image, 1)
        if len(rects) == 0:
            return []
        
        landmarks_list = []
        for rect in rects:
            points = self.predictor(image, rect).parts()
            landmarks = np.array([[p.x, p.y] for p in points])
            landmarks_list.append(landmarks)
        return landmarks_list

    def generate_mask_from_diff(self, real_image: np.ndarray, fake_image: np.ndarray) -> np.ndarray:

        # 确保图片大小一致
        if real_image.shape != fake_image.shape:
            # 处理尺寸不一致的情况
            diff_h = real_image.shape[0] - fake_image.shape[0]
            diff_w = real_image.shape[1] - fake_image.shape[1]
            
            if diff_h > 0:
                m = diff_h // 2
                real_image = real_image[m:-m,:]
            elif diff_h < 0:
                m = -diff_h // 2
                fake_image = fake_image[m:-m,:]
                
            if diff_w > 0:
                m = diff_w // 2
                real_image = real_image[:,m:-m]
            elif diff_w < 0:
                m = -diff_w // 2
                fake_image = fake_image[:,m:-m]
        # if real_image.shape != fake_image.shape:
        #     real_image = cv2.resize(real_image, (fake_image.shape[1], fake_image.shape[0]))
        
        # 使用structural_similarity计算差异
        _, diff = structural_similarity(
            real_image, 
            fake_image, 
            multichannel=True, 
            full=True
        )
        
        # 转换差异图为mask
        diff = 1 - diff  # 将相似度转换为差异度
        mask = (diff * 255).astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        return mask

    
    def detect_blur(self, real_part: np.ndarray, fake_part: np.ndarray) -> bool:
        """检测模糊"""
        real_var = cv2.Laplacian(real_part, cv2.CV_64F).var()
        fake_var = cv2.Laplacian(fake_part, cv2.CV_64F).var()
        return (real_var > fake_var) and (real_var - fake_var) > self.BLUR_THRESHOLD
    
    def detect_color_inconsistency(self, real_part: np.ndarray, fake_part: np.ndarray) -> float:
        """检测颜色不一致"""
        real_lab = cv2.cvtColor(real_part, cv2.COLOR_RGB2LAB)
        fake_lab = cv2.cvtColor(fake_part, cv2.COLOR_RGB2LAB)
        
        # 计算Lab空间的均值和方差差异
        real_mean = np.mean(real_lab, axis=(0,1))
        fake_mean = np.mean(fake_lab, axis=(0,1))
        real_std = np.std(real_lab, axis=(0,1))
        fake_std = np.std(fake_lab, axis=(0,1))
        
        mean_diff = np.mean(np.abs(real_mean - fake_mean))
        std_diff = np.mean(np.abs(real_std - fake_std))
        
        return (mean_diff + std_diff) / 2

    def detect_texture_anomaly(self, real_part: np.ndarray, fake_part: np.ndarray) -> bool:
        """检测纹理异常"""
        def get_glcm_features(img):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            glcm = skimage.feature.graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
            contrast = skimage.feature.graycoprops(glcm, 'contrast')[0, 0]
            return contrast
        
        real_contrast = get_glcm_features(real_part)
        fake_contrast = get_glcm_features(fake_part)
        return abs(real_contrast - fake_contrast) > self.TEXTURE_THRESHOLD

    def detect_structural_deformation(self, real_part: np.ndarray, fake_part: np.ndarray) -> bool:
        """检测结构变形"""
        ssim = skimage.metrics.structural_similarity(
            real_part, fake_part, multichannel=True, data_range=255
        )
        return ssim < self.STRUCTURE_THRESHOLD
    
    def get_five_key(self, landmarks_68):
        """获取五个关键点"""
        leye_center = (landmarks_68[36] + landmarks_68[39])*0.5
        reye_center = (landmarks_68[42] + landmarks_68[45])*0.5
        nose = landmarks_68[33]
        lmouth = landmarks_68[48]
        rmouth = landmarks_68[54]
        leye_left = landmarks_68[36]
        leye_right = landmarks_68[39]
        reye_left = landmarks_68[42]
        reye_right = landmarks_68[45]
        out = [tuple(x.astype('int32')) for x in [
            leye_center,reye_center,nose,lmouth,rmouth,leye_left,leye_right,reye_left,reye_right
        ]]
        return out

    def remove_eyes(self, image, landmarks, opt):
        if opt == 'l':
            (x1, y1), (x2, y2) = landmarks[5:7]
        elif opt == 'r':
            (x1, y1), (x2, y2) = landmarks[7:9]
        elif opt == 'b':
            (x1, y1), (x2, y2) = landmarks[:2]
        else:
            print('wrong region')
            
        mask = np.zeros_like(image[..., 0])
        line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
        w = dist((x1, y1), (x2, y2))
        dilation = int(w // 4)
        if opt != 'b':
            dilation *= 4
        line = binary_dilation(line, iterations=dilation)
        return line

    def remove_face(self, image, landmarks):
        mask = np.zeros_like(image[..., 0])
        points = cv2.convexHull(landmarks[FACE_POINTS])
        line = cv2.fillConvexPoly(mask, points, color=(1, 1, 1))
        return line

    def remove_nose(self, image, landmarks):
        (x1, y1), (x2, y2) = landmarks[:2]
        x3, y3 = landmarks[2]
        mask = np.zeros_like(image[..., 0])
        x4 = int((x1 + x2) / 2)
        y4 = int((y1 + y2) / 2)
        line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
        w = dist((x1, y1), (x2, y2))
        dilation = int(w // 4)
        line = binary_dilation(line, iterations=dilation)
        return line

    def remove_mouth(self, image, landmarks):
        (x1, y1), (x2, y2) = landmarks[3:5]
        mask = np.zeros_like(image[..., 0])
        line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
        w = dist((x1, y1), (x2, y2))
        dilation = int(w // 3)
        line = binary_dilation(line, iterations=dilation)
        return line

    def extract_region(self, image: np.ndarray, landmarks: np.ndarray, 
                      forgery_mask: np.ndarray, region: str) -> Tuple[np.ndarray, bool]:
        """
        基于mask提取特定区域
        """
        # 获取五个关键点
        five_key = self.get_five_key(landmarks)
        
        # 根据区域生成对应的mask
        if region == "face":
            # pdb.set_trace()
            region_mask = self.remove_face(image, landmarks)
        elif region == "eyes":
            region_mask = self.remove_eyes(image, five_key, 'b')
        elif region == "nose":
            region_mask = self.remove_nose(image, five_key)
        elif region == "mouth":
            region_mask = self.remove_mouth(image, five_key)
        else:
            raise ValueError(f"Unknown region: {region}")
        
        # 计算该区域内的伪造程度
        area = np.count_nonzero(region_mask)
        if forgery_mask.shape != region_mask.shape:
            forgery_mask = cv2.resize(forgery_mask, region_mask.shape[::-1])
        process_mask = region_mask * forgery_mask
        avg_forgery = np.sum(process_mask) / area
        
        # 判断该区域是否被伪造
        is_forged = avg_forgery > self.mask_threshold
        
        # 提取图像区域
        extracted_region = image * np.expand_dims(region_mask, -1)
        
        return extracted_region, is_forged


class ForgeryDescriptionGenerator:
    def __init__(self, detector: ForgeryDetector):
        self.detector = detector
        
    def analyze_region(self, real_region: np.ndarray, fake_region: np.ndarray, region: str,fake_image) -> Tuple[List[str], List[str]]:
        """分析特定区域的伪造特征"""
        features = []
        reasons = []
        
        # 检测模糊
        if self.detector.detect_blur(real_region, fake_region):
            features.append(ForgeryFeatures.BLUR)
            reasons.append(f"the {region} appears blurry compared to natural faces")
            
        # 检测颜色不一致
        color_score = self.detector.detect_color_inconsistency(real_region, fake_region)
        if color_score > self.detector.COLOR_THRESHOLD:
            features.append(ForgeryFeatures.COLOR_MISMATCH)
            reasons.append(f"the {region} shows unnatural color variations")
            
        # 检测纹理异常
        if self.detector.detect_texture_anomaly(real_region, fake_region):
            features.append(ForgeryFeatures.TEXTURE_ARTIFACTS)
            reasons.append(f"the {region} exhibits unusual texture patterns")
            
        # 检测结构变形
        if self.detector.detect_structural_deformation(real_region, fake_region):
            features.append(ForgeryFeatures.DEFORMATION)
            reasons.append(f"the {region} appears structurally distorted")

        # 检测blending artifacts
        has_artifacts, evidence = self.detector.blending_detector.detect_blending_artifacts(fake_image,fake_region)
        if has_artifacts:
            features.append(ForgeryFeatures.BLENDING)
            # 将证据转换为自然语言描述
            evidence_text = ", ".join(evidence)
            reasons.append(f"the {region} shows blending artifacts characterized by {evidence_text}")
        
            
        return features, reasons
    
    def generate_description(self, features: List[str], reasons: List[str], region: str) -> str:
        """
        生成描述文本
        Args:
            features: 特征列表
            reasons: 原因列表
            region: 伪造区域
        Returns:
            生成的描述文本
        """
        # 第一句话描述基本判断
        description = "This image appears to be manipulated. "
        
        # 直接描述区域和原因，不重复提及manipulation
        if region == "face":
            feature_desc = f"The entire face "
        else:
            feature_desc = f"The {region} region "
            
        # 添加具体原因
        if reasons:
            feature_desc += "shows "
            for i, reason in enumerate(reasons):
                if i == len(reasons) - 1:
                    feature_desc += f"{reason}."
                else:
                    feature_desc += f"{reason}, and "
        
        description += feature_desc
        return description
    
    def validate_description(self, description: str) -> bool:
        """验证描述质量"""
        required_elements = ["manipulated", "appears", "region"]
        return all(element in description.lower() for element in required_elements)

class PFIG:
    def __init__(self, image_root: str, mask_root: str):
        self.image_root = image_root
        self.mask_root = mask_root
        self.detector = ForgeryDetector()
        self.generator = ForgeryDescriptionGenerator(self.detector)
        
    def get_opposite_path(self, fake_path: str) -> str:
        """获取对应的真实图片路径"""
        video_id = fake_path.split("/")[-2]
        opposite_path = fake_path.replace("manipulated_sequences", "original_sequences")
        opposite_path = opposite_path.split("/")
        opposite_path[-2] = video_id.split("_")[0]
        opposite_path[-4] = "youtube"
        opposite_path = "/".join(opposite_path)
        
        if not os.path.exists(opposite_path):
            original_folder = '/'.join(opposite_path.split('/')[:-1])+'/'
            face_paths = glob(os.path.join(original_folder, '*.jpg'))
            opposite_path = random.choice(face_paths)
        return opposite_path
    def sample_frames(self, video_folder: str, num_frames: int = 10, strategy: str = 'uniform') -> List[str]:
        """
        从视频文件夹中采样指定数量的帧，并确保稳定的采样结果
        """
        frame_paths = sorted(glob(os.path.join(video_folder, '*.jpg')))
        total_frames = len(frame_paths)
        
        if total_frames <= num_frames:
            return frame_paths
            
        if strategy == 'uniform':
            # 均匀采样
            interval = total_frames // num_frames
            indices = [i * interval for i in range(num_frames)]
            return [frame_paths[i] for i in indices]
        else:
            # 使用固定种子的随机采样，确保可重复性
            random.seed(hash(video_folder))  # 使用文件夹路径作为随机种子
            sampled = random.sample(frame_paths, num_frames)
            random.seed()  # 重置随机种子
            return sorted(sampled)  # 返回排序后的结果确保顺序稳定

    
    def process_image_pair(self, fake_path: str) -> Optional[str]:
        """处理图片对并生成描述"""
        # 读取图片和mask
        fake_image = cv2.imread(fake_path)
        real_path = self.get_opposite_path(fake_path)
        real_image = cv2.imread(real_path)
        if real_image.shape!=fake_image.shape:
            real_image = cv2.resize(real_image, (fake_image.shape[1], fake_image.shape[0]))
        # 获取mask路径
        method = fake_path.split("/")[-4]
        idx = fake_path.split("/")[-2]
        mask_path = fake_path.replace("c23", "masks")
        
        # 尝试读取已有mask，如果不存在则生成
        if os.path.exists(mask_path):
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_image = cv2.resize(mask_image, (fake_image.shape[1], fake_image.shape[0]))
        else:
            mask_paths = glob(f"{self.mask_root}/{method}/masks/{idx}/*.jpg")
            if mask_paths:
                mask_path = random.choice(mask_paths)
                mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask_image = cv2.resize(mask_image, (fake_image.shape[1], fake_image.shape[0]))
            else:
                # 如果没有找到mask文件，则通过图片差异生成
                print(f"generate mask {mask_path}")
                mask_image = self.detector.generate_mask_from_diff(real_image, fake_image)
                mask_image = (mask_image).astype(np.uint8)
                mask_dir = os.path.dirname(mask_path)
                if not os.path.exists(mask_dir):
                    os.makedirs(mask_dir)
                cv2.imwrite(mask_path, mask_image)
                mask_image = cv2.resize(mask_image, (fake_image.shape[1], fake_image.shape[0]))
            
        # mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # mask_image = cv2.resize(mask_image, (fake_image.shape[1], fake_image.shape[0]))
        
        if fake_image is None or real_image is None or mask_image is None:
            return None
            
        # 获取关键点
        fake_landmarks = self.detector.get_landmarks(fake_image)
        real_landmarks = self.detector.get_landmarks(real_image)
        
        if not fake_landmarks or not real_landmarks:
            return None
            
        descriptions = []
        regions = ["face", "eyes", "nose", "mouth"]
        
        for region in regions:
            # 提取区域并判断是否伪造
            fake_region, is_forged = self.detector.extract_region(
                fake_image, fake_landmarks[0], mask_image, region)
            
            if is_forged:
                real_region, _ = self.detector.extract_region(
                    real_image, real_landmarks[0], mask_image, region)
                
                # 分析特征
                features, reasons = self.generator.analyze_region(real_region, fake_region, region,fake_image)
                
                if features:
                    desc = self.generator.generate_description(features, reasons, region)
                    if self.generator.validate_description(desc):
                        descriptions.append(desc)
        
        if descriptions:
            return " ".join(descriptions)
        return None
    
    def generate_dataset(self, methods: List[str] = None, frames_per_video: int = 10, 
                    sample_strategy: str = 'uniform'):
        """
        生成数据集（增量式存储）
        Args:
            methods: 处理方法列表
            frames_per_video: 每个视频采样的帧数
            sample_strategy: 采样策略
        """
        if methods is None:
            methods = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

            
        for method in methods:
            print(f"Processing {method}...")
            output_file = f'./output/{method}_descriptions.txt'
            
            # 读取已有的记录
            processed_paths = set()
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    for line in f:
                        path = line.split('*')[0].strip()
                        processed_paths.add(path)
            
            # 获取所有视频文件夹
            video_folders = sorted(glob(f"{self.image_root}/{method}/c23/*/"))
            
            # 使用追加模式打开文件
            with open(output_file, 'a') as f:
                for video_folder in tqdm(video_folders):
                    # 对每个视频文件夹进行采样
                    sampled_frames = self.sample_frames(
                        video_folder, 
                        num_frames=frames_per_video,
                        strategy=sample_strategy
                    )
                    
                    # 处理采样的帧
                    for frame_path in sampled_frames:
                        # 检查是否已处理过
                        if frame_path in processed_paths:
                            continue
                            
                        description = self.process_image_pair(frame_path)
                        if description:
                            content = f"{frame_path}*{description}\n"
                            f.write(content)
                            f.flush()  # 立即写入磁盘
                            processed_paths.add(frame_path)


def main():
    image_root = "../manipulated_sequences"
    mask_root = "../manipulated_sequences"
    
    pfig = PFIG(image_root, mask_root)
    
    pfig.generate_dataset()

if __name__ == "__main__":
    main()