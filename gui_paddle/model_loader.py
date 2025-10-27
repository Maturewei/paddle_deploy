"""
安全的PaddleDetection模型加载器
支持加载PaddleDetection导出的推理模型
"""

import os
import sys
import tempfile
import uuid
import shutil
import streamlit as st
import numpy as np
import yaml

# 添加PaddleDetection路径
PADDLE_DETECTION_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PaddleDetection")
sys.path.insert(0, os.path.join(PADDLE_DETECTION_ROOT, "deploy", "python"))

from infer import Detector, PredictConfig

class PaddleDetectorWrapper:
    """PaddleDetection检测器包装类，模拟YOLO接口"""
    
    def __init__(self, model_dir, device='CPU', threshold=0.5):
        """
        初始化检测器
        
        Args:
            model_dir: 模型目录路径
            device: 运行设备 (CPU/GPU)
            threshold: 检测阈值
        """
        self.model_dir = model_dir
        self.device = device
        self.threshold = threshold
        
        # 加载模型配置
        infer_cfg_path = os.path.join(model_dir, 'infer_cfg.yml')
        if not os.path.exists(infer_cfg_path):
            raise ValueError(f"Cannot find infer_cfg.yml in {model_dir}")
        
        with open(infer_cfg_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        
        # 获取标签列表
        self.names = {i: name for i, name in enumerate(self.cfg.get('label_list', []))}
        
        # 初始化检测器
        self.detector = Detector(
            model_dir=model_dir,
            device=device,
            threshold=threshold,
            output_dir='output'
        )
    
    def predict(self, image, verbose=False, save=False, show=False, conf=0.5, iou=0.5, imgsz=640, half=False):
        """
        预测图像，模拟YOLO的predict接口
        
        Args:
            image: 输入图像 (numpy array, BGR格式)
            conf: 置信度阈值
            其他参数为兼容性参数
            
        Returns:
            预测结果列表 (模拟ultralytics Results对象)
        """
        # 确保图像是numpy数组
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be numpy array")
        
        # 更新阈值
        self.detector.threshold = conf
        
        try:
            # 预处理
            inputs = self.detector.preprocess([image])
            
            # 预测
            result = self.detector.predict()
            
            # 后处理
            result = self.detector.postprocess(inputs, result)
            
            # 过滤低置信度的框
            result = self.detector.filter_box(result, conf)
            
            # 构造结果对象
            results = [PaddleDetectionResult(image, result, self.names, conf)]
            
            return results
        except Exception as e:
            # 如果预测失败，返回空结果
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            empty_result = {'boxes': np.array([]), 'boxes_num': np.array([0])}
            return [PaddleDetectionResult(image, empty_result, self.names, conf)]


class PaddleDetectionResult:
    """模拟ultralytics Results对象"""
    
    def __init__(self, orig_img, detection_result, names, conf_threshold):
        """
        初始化结果对象
        
        Args:
            orig_img: 原始图像 (numpy array)
            detection_result: PaddleDetection的检测结果
            names: 类别名称字典
            conf_threshold: 置信度阈值
        """
        self.orig_img = orig_img
        self.names = names
        self.conf_threshold = conf_threshold
        
        # 解析检测结果
        self.boxes = PaddleDetectionBoxes(detection_result, names)
    
    def plot(self, line_width=2, font_size=20, conf=True, labels=True):
        """
        绘制检测结果
        
        Args:
            line_width: 边框线宽
            font_size: 字体大小
            conf: 是否显示置信度
            labels: 是否显示标签
            
        Returns:
            绘制后的图像 (numpy array, BGR格式)
        """
        import cv2
        
        img = self.orig_img.copy()
        
        if self.boxes is None or len(self.boxes) == 0:
            return img
        
        # 定义颜色映射
        colors = {}
        for i in range(len(self.names)):
            np.random.seed(i)
            colors[i] = tuple(map(int, np.random.randint(0, 255, 3)))
        
        # 绘制每个检测框
        for box in self.boxes:
            cls_id = int(box.cls)
            conf_score = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # 绘制边框（适中粗细）
            color = colors.get(cls_id, (0, 255, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)
            
            # 绘制标签和置信度
            if labels:
                label = self.names[cls_id]
                if conf:
                    # 格式化置信度为百分比，更清晰
                    label = f"{label} {conf_score:.1%}"
                
                # 使用适中的字体大小，确保清晰但不突兀
                font_scale = font_size / 40  # 适中的字体比例
                thickness = max(int(line_width * 0.5), 1)  # 适中的字体粗细
                
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, thickness
                )
                
                # 适中的padding，避免标签过大
                padding_x = 4
                padding_y = 4
                
                # 绘制标签背景
                bg_y1 = y1 - label_h - baseline - padding_y * 2
                bg_y2 = y1
                bg_x1 = x1
                bg_x2 = x1 + label_w + padding_x * 2
                
                # 确保背景不超出图像边界
                bg_y1 = max(0, bg_y1)
                
                # 绘制半透明背景，提高可读性
                overlay = img.copy()
                cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
                cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
                
                # 绘制标签边框，使标签更清晰
                cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
                
                # 绘制标签文字（白色，清晰可读）
                text_y = y1 - baseline - padding_y
                text_x = x1 + padding_x
                cv2.putText(img, label, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                          (255, 255, 255), thickness, cv2.LINE_AA)
        
        return img


class PaddleDetectionBoxes:
    """模拟ultralytics Boxes对象"""
    
    def __init__(self, detection_result, names):
        """
        初始化boxes对象
        
        Args:
            detection_result: PaddleDetection的检测结果
            names: 类别名称字典
        """
        self.names = names
        
        # 解析检测框
        # detection_result['boxes'] 格式: [class_id, score, x1, y1, x2, y2]
        boxes = detection_result.get('boxes', np.array([]))
        
        if len(boxes) == 0:
            self._boxes = []
        else:
            self._boxes = [PaddleDetectionBox(box, names) for box in boxes]
    
    def __len__(self):
        return len(self._boxes)
    
    def __iter__(self):
        return iter(self._boxes)
    
    def __getitem__(self, idx):
        return self._boxes[idx]


class PaddleDetectionBox:
    """模拟ultralytics Box对象"""
    
    def __init__(self, box_data, names):
        """
        初始化单个检测框
        
        Args:
            box_data: 检测框数据 [class_id, score, x1, y1, x2, y2]
            names: 类别名称字典
        """
        self.names = names
        self.cls = int(box_data[0])  # 类别ID
        self.conf = float(box_data[1])  # 置信度
        self.xyxy = np.array([[box_data[2], box_data[3], box_data[4], box_data[5]]])  # 坐标


class SafeModelLoader:
    """安全的模型加载器"""
    
    def __init__(self):
        self.loaded_models = {}
        self.temp_dirs = []
    
    def load_model_from_upload(self, uploaded_file, model_dir=None):
        """
        从上传的文件或指定目录加载模型
        
        Args:
            uploaded_file: Streamlit上传的文件对象（可以为None）
            model_dir: 模型目录路径（如果提供，优先使用）
        
        Returns:
            PaddleDetectorWrapper对象
        """
        if model_dir is not None and os.path.exists(model_dir):
            # 直接使用指定的模型目录
            model_path = model_dir
            cache_key = f"dir_{model_dir}"
        elif uploaded_file is not None:
            # 从上传文件加载（这里假设上传的是压缩包或模型文件）
            file_hash = str(hash(uploaded_file.read()))
            uploaded_file.seek(0)  # 重置文件指针
            cache_key = f"upload_{file_hash}"
            
            if cache_key in self.loaded_models:
                st.info("🔄 使用已缓存的模型")
                return self.loaded_models[cache_key]
            
            # 创建临时目录
            temp_dir = tempfile.mkdtemp(prefix="paddle_model_")
            self.temp_dirs.append(temp_dir)
            
            # 保存上传的文件（这里简化处理，实际可能需要解压等）
            temp_file = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file, 'wb') as f:
                f.write(uploaded_file.read())
            
            model_path = temp_dir
        else:
            st.error("❌ 请提供模型文件或模型目录")
            return None
        
        try:
            # 检查模型文件是否存在
            infer_cfg = os.path.join(model_path, 'infer_cfg.yml')
            model_file = os.path.join(model_path, 'model.pdmodel')
            params_file = os.path.join(model_path, 'model.pdiparams')
            
            if not os.path.exists(infer_cfg):
                st.error(f"❌ 找不到配置文件: {infer_cfg}")
                return None
            
            if not os.path.exists(model_file):
                st.error(f"❌ 找不到模型文件: {model_file}")
                return None
            
            if not os.path.exists(params_file):
                st.error(f"❌ 找不到参数文件: {params_file}")
                return None
            
            # 加载模型
            model = PaddleDetectorWrapper(model_path, device='CPU', threshold=0.5)
            
            # 缓存模型
            self.loaded_models[cache_key] = model
            
            # 保存到session state
            if 'safe_model_loader' not in st.session_state:
                st.session_state.safe_model_loader = self
            
            st.success("✅ 模型加载成功!")
            return model
            
        except Exception as e:
            st.error(f"❌ 模型加载失败: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None
    
    def cleanup(self):
        """清理临时文件"""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except:
                pass
        self.temp_dirs.clear()
        self.loaded_models.clear()


# 全局模型加载器实例
@st.cache_resource
def get_model_loader():
    return SafeModelLoader()


def load_model_safe(uploaded_file=None, model_dir=None):
    """
    安全的模型加载函数
    
    Args:
        uploaded_file: Streamlit上传的文件对象
        model_dir: 模型目录路径
    
    Returns:
        模型对象
    """
    loader = get_model_loader()
    return loader.load_model_from_upload(uploaded_file, model_dir)

