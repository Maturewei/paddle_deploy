import os
import warnings
import logging

# 在导入streamlit之前设置环境变量和日志级别
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
# 禁用XSRF保护以避免CORS配置冲突警告
os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'

# 设置日志级别来抑制警告
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)

# 忽略所有警告
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import pandas as pd

# 设置页面配置
st.set_page_config(
    page_title="基于飞桨PaddleDetection的智能零售系统",
    page_icon="https://paddlepaddle-org-cn.cdn.bcebos.com/paddle-site-front/static/media/paddlelogo.0b483fa7.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

# 导入安全模型加载器
from model_loader import load_model_safe, get_model_loader

# 导入多人拿取检测器
from multi_person_detector import MultiPersonDetector

# ========================================
# 📁 模型路径配置区
# ========================================
# 请在下方修改模型文件夹的路径
# 模型文件夹必须包含以下3个文件：
# - infer_cfg.yml
# - model.pdmodel
# - model.pdiparams

# 方式1：使用相对路径（推荐）
MODEL_PATH = "../output_inference"

# 方式2：使用绝对路径
#MODEL_PATH = r"D:\output_inference"


# 验证并设置模型路径
def validate_and_set_model_path(path):
    """验证模型路径并返回标准化的路径"""
    # 标准化路径
    normalized_path = os.path.normpath(os.path.abspath(path))
    
    # 检查路径是否存在
    if not os.path.exists(normalized_path):
        return None, f"路径不存在: {normalized_path}"
    
    # 检查必需文件
    required_files = ['infer_cfg.yml', 'model.pdmodel', 'model.pdiparams']
    missing_files = []
    
    for file_name in required_files:
        file_path = os.path.join(normalized_path, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
    
    if missing_files:
        return None, f"缺少必需文件: {', '.join(missing_files)}"
    
    return normalized_path, None

# 验证模型路径
model_path, error_msg = validate_and_set_model_path(MODEL_PATH)

# 主标题
st.markdown('<h1 class="main-header">PaddlePickUp: 让智能零售"看得清，数得明"</h1>', unsafe_allow_html=True)

# 侧边栏配置
st.sidebar.markdown("## ⚙️ 系统配置")

# 显示模型路径信息
st.sidebar.markdown("### 📁 模型配置")
if model_path:
    st.sidebar.success("✅ 模型路径已配置")
else:
    st.sidebar.error("❌ 模型路径配置错误")
    st.sidebar.error(error_msg)
    st.sidebar.warning("⚠️ 请在代码中正确配置 MODEL_PATH")
    st.sidebar.code("MODEL_PATH = '../output_inference'", language="python")

# 模型参数配置
st.sidebar.markdown("## 🎛️ 预测参数")
confidence_threshold = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.5, 0.01)

# 视频检测参数（统一使用多人检测算法，兼容单人场景）
st.sidebar.markdown("### 🎯 视频检测参数")
multi_action_window = st.sidebar.slider("动作窗口大小", 10, 100, 80, help="检测动作的时间窗口（帧数）")
multi_arm_thresh = st.sidebar.slider("手臂检测阈值", 1, 10, 3, help="触发检测所需的手臂出现次数")
multi_handGoods_thresh = st.sidebar.slider("手部商品检测阈值", 1, 10, 2, help="确认拿取所需的手部商品出现次数")
multi_arm_miss_limit = st.sidebar.slider("手臂消失限制", 1, 10, 4, help="允许手臂暂时消失的帧数")
multi_cooldown_limit = st.sidebar.slider("冷却限制", 1, 20, 5, help="拿取动作后的冷却帧数")
multi_iou_threshold = st.sidebar.slider("IoU匹配阈值", 0.1, 0.5, 0.25, 0.01, help="检测框匹配的IoU阈值")

# 行列统计参数
st.sidebar.markdown("### 🗄️ 行列统计参数")
position_confidence_threshold = st.sidebar.slider(
    "位置识别置信度阈值", 
    0.3, 0.9, 0.6, 0.05, 
    help="只保存置信度高于此阈值的行列位置数据，确保统计准确性"
)

# 性能优化选项
st.sidebar.markdown("## ⚡ 性能设置")
resize_display = st.sidebar.checkbox("启用显示优化", value=True, help="降低显示分辨率以提高性能")
video_quality = st.sidebar.slider("视频质量", 70, 100, 95, help="数值越高视频质量越好，但文件越大")
realtime_display = st.sidebar.checkbox("实时显示检测过程", value=True, help="启用后会实时显示每一帧的检测结果")

# 导入字体工具
from font_utils import safe_draw_text

# 颜色位置映射表（参考智能柜系统）
COLOR_POSITION_MAP = {
    "argent": (1, 1), "yellow": (2, 1), "green": (3, 1),
    "grey": (4, 1), "orange": (5, 1), "cyan": (6, 1),
    "black": (1, 2), "scarlet": (2, 2), "red": (3, 2),
    "blue": (4, 2), "purple": (5, 2), "darkBlue": (6, 2)
}

def draw_chinese_text(img, text, position, font_path="SimHei.ttf", font_size=32, color=(0, 255, 0)):
    """在图像上绘制中文文字"""
    return safe_draw_text(img, text, position, font_size, color)

def format_timestamp(frame_idx, fps):
    """格式化时间戳"""
    total_millis = int((frame_idx / fps) * 1000)
    seconds = total_millis // 1000
    millis = total_millis % 1000
    minutes = seconds // 60
    hours = minutes // 60
    return f"{hours:02d}.{minutes%60:02d}.{seconds%60:02d}.{millis:03d}"

def analyze_drawer_colors(boxes_dict, arm_boxes, hand_boxes):
    """
    分析检测框中的抽屉颜色，判断拿取的是哪一行哪一列的商品
    
    Args:
        boxes_dict: 字典，key为标签名，value为检测框列表
        arm_boxes: 手臂检测框列表
        hand_boxes: 手持商品检测框列表
    
    Returns:
        detected_positions: 检测到的位置列表 [(row, col, color, score), ...]
    """
    detected_positions = []
    
    # 检查是否有手臂或手持商品
    has_interaction = len(arm_boxes) > 0 or len(hand_boxes) > 0
    
    if has_interaction:
        # 遍历所有检测到的颜色
        for label, box_list in boxes_dict.items():
            if label in COLOR_POSITION_MAP:
                row, col = COLOR_POSITION_MAP[label]
                # 找到该颜色框中置信度最高的
                if box_list:
                    best_box = max(box_list, key=lambda x: x['conf'])
                    detected_positions.append((row, col, label, best_box['conf']))
    
    return detected_positions

# 主界面 - 视频预测
st.markdown('<h2 class="sub-header">📹 视频智能预测演示</h2>', unsafe_allow_html=True)

if model_path is None:
    st.warning("⚠️ 请先设置模型路径")
else:
    model = load_model_safe(model_dir=model_path)
    
    if model is not None:
        # 视频上传
        uploaded_video = st.file_uploader("选择视频文件", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_video is not None:
            # 保存上传的视频到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name
            
            # 显示视频信息
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            st.info(f"📹 视频信息: {width}x{height} @ {fps:.1f}fps, 总帧数: {total_frames}")
            # st.info(f"🎯 当前质量设置: {video_quality}% | 实时显示: {'开启' if realtime_display else '关闭'}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("🚀 开始视频预测", type="primary"):
                    # 创建进度条
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 视频处理
                    cap = cv2.VideoCapture(video_path)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # 创建输出视频（优化质量设置）
                    output_video_path = "output_prediction.mp4"
                    
                    # 根据质量设置选择合适的编码器
                    encoding_options = [
                        ('H264', cv2.VideoWriter_fourcc(*'H264')),  # H.264编码（最兼容）
                        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # MPEG-4编码
                        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # XVID编码
                        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),  # Motion JPEG编码
                    ]
                    
                    # 尝试不同的编码器
                    out = None
                    for codec_name, fourcc in encoding_options:
                        try:
                            test_out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                            if test_out.isOpened():
                                out = test_out
                                break
                            else:
                                test_out.release()
                        except:
                            continue
                    
                    if out is None:
                        st.error("❌ 无法初始化视频编码器，请检查系统支持")
                        st.stop()
                    
                    # 设置视频质量参数（使用安全的属性设置）
                    try:
                        # 尝试设置质量参数，如果属性不存在则忽略
                        if hasattr(cv2, 'VIDEOWRITER_PROP_QUALITY'):
                            out.set(cv2.VIDEOWRITER_PROP_QUALITY, video_quality)
                    except:
                        pass  # 忽略所有相关错误
                    
                    # 预测状态变量
                    count = 0
                    frame_idx = 0
                    pickup_frame_map = {}
                    log_data = []
                    
                    # 行列统计数据
                    row_col_stats = {}  # {(row, col): count}
                    pickup_details = []  # 记录每次拿取的详细信息
                    
                    # 初始化多人检测器（统一使用，兼容单人和多人场景）
                    multi_detector = MultiPersonDetector(
                        action_window=multi_action_window,
                        arm_thresh=multi_arm_thresh,
                        handGoods_thresh=multi_handGoods_thresh,
                        arm_miss_limit=multi_arm_miss_limit,
                        cooldown_limit=multi_cooldown_limit,
                        iou_threshold=multi_iou_threshold
                    )
                    
                    # 创建视频显示区域
                    video_placeholder = st.empty()
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_idx += 1
                        
                        # 更新进度条
                        progress = frame_idx / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"处理进度: {frame_idx}/{total_frames} 帧 ({progress:.1%})")
                        
                        # PaddleDetection预测
                        results = model.predict(frame, verbose=False, save=False, show=False, 
                                               conf=confidence_threshold, iou=0.5, 
                                               imgsz=640, half=False)
                        frame_labels = set()
                        scores = {}
                        arm_boxes = []
                        hand_boxes = []
                        color_boxes = {}  # 存储各颜色的检测框
                        
                        for result in results:
                            for box in result.boxes:
                                label = result.names[int(box.cls)]
                                conf = float(box.conf)
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                
                                if conf >= confidence_threshold:
                                    if label not in scores or conf > scores[label]:
                                        scores[label] = conf
                                    
                                    if label == "arm":
                                        arm_boxes.append((x1, y1, x2, y2))
                                    elif label == "handGoods":
                                        hand_boxes.append((x1, y1, x2, y2))
                                    
                                # 收集颜色框信息
                                if label in COLOR_POSITION_MAP:
                                    if label not in color_boxes:
                                        color_boxes[label] = []
                                    color_boxes[label].append({'conf': conf, 'box': (x1, y1, x2, y2)})
                                
                                frame_labels.add(label)
                        
                        # 使用多人检测器处理帧（统一处理单人和多人场景）
                        prev_count = count
                        detection_result = multi_detector.process_frame(frame_idx, arm_boxes, hand_boxes)
                        count = detection_result["total_count"]
                        pickup_frame_map[frame_idx] = count
                        
                        # 检测到新的拿取动作时，分析颜色位置
                        if count > prev_count:
                            detected_positions = analyze_drawer_colors(color_boxes, arm_boxes, hand_boxes)
                            if detected_positions:
                                # 暂存位置识别数据（仅保存置信度高于阈值的记录，等待后端修正）
                                for row, col, color, conf_score in detected_positions:
                                    # 应用置信度过滤
                                    if conf_score >= position_confidence_threshold:
                                        pickup_details.append({
                                            'frame': frame_idx,
                                            'timestamp': format_timestamp(frame_idx, fps),
                                            'row': row,
                                            'col': col,
                                            'color': color,
                                            'confidence': round(conf_score, 2)
                                        })
                        
                        # 绘制预测结果
                        rendered_img = None
                        try:
                            for result in results:
                                # 使用标准绘制（适中线宽和字体，清晰但不突兀）
                                rendered_img = result.plot(line_width=3, font_size=24, 
                                                         conf=True, labels=True)
                                
                                # 检查图像是否有效
                                if rendered_img is not None and hasattr(rendered_img, 'shape') and len(rendered_img.shape) >= 2:
                                    try:
                                        # 计算文字位置（水平居中）
                                        img_height, img_width = rendered_img.shape[:2]
                                        
                                        # 根据图像宽度动态调整文字位置（水平居中）
                                        font_size = 28
                                        text_content = f"该顾客拿取总商品数为: {count}"
                                        
                                        # 更精确的文字宽度估算
                                        chinese_chars = len([c for c in text_content if '\u4e00' <= c <= '\u9fff'])
                                        other_chars = len(text_content) - chinese_chars
                                        estimated_text_width = chinese_chars * font_size + other_chars * font_size * 0.6
                                        
                                        # 计算水平居中位置
                                        text_x = (img_width - int(estimated_text_width)) // 2
                                        text_y = 50
                                        
                                        # 确保文字不超出图像边界
                                        text_x = max(10, min(text_x, img_width - int(estimated_text_width)))
                                        text_y = max(30, min(text_y, img_height - 10))
                                        
                                        # 绘制总拿取数量（单人和多人检测统一显示）
                                        rendered_img = draw_chinese_text(rendered_img,
                                                                         f"顾客拿取总商品数为: {count}",
                                                                         position=(text_x, text_y),
                                                                         font_size=28,
                                                                         color=(0, 255, 0))
                                    except Exception as e:
                                        pass
                                
                            # 确保有图像可以写入
                            if rendered_img is not None:
                                out.write(rendered_img)
                            else:
                                out.write(frame)
                        except Exception as e:
                            # 如果整个绘制过程失败，使用原始帧
                            out.write(frame)
                        
                        # 实时显示当前帧
                        if realtime_display:
                            try:
                                display_img = None
                                if rendered_img is not None and hasattr(rendered_img, 'shape'):
                                    display_img = rendered_img.copy()
                                else:
                                    display_img = frame.copy()
                            
                                if resize_display and hasattr(display_img, 'shape') and len(display_img.shape) >= 2:
                                    height, width = display_img.shape[:2]
                                    if width > 800:
                                        ratio = 800 / width
                                        new_width = 800
                                        new_height = int(height * ratio)
                                        display_img = cv2.resize(display_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                                
                                # 确保颜色格式正确
                                if len(display_img.shape) == 3 and display_img.shape[2] == 3:
                                    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                                
                                # 显示当前处理状态（增强的信息显示）
                                fsm1_count = detection_result.get("fsm1_count", 0)
                                fsm2_count = detection_result.get("fsm2_count", 0)
                                active_persons = len(detection_result.get("active_persons", []))
                                caption_text = f"🎯 视频检测 | 帧: {frame_idx}/{total_frames} | 总拿取数: {count} | 活跃人数: {active_persons}"
                                
                                video_placeholder.image(display_img, caption=caption_text, 
                                                      use_container_width=True, channels="RGB")
                            except Exception as e:
                                pass
                        
                        # 记录日志数据
                        timestamp = format_timestamp(frame_idx, fps)
                        log_data.append([
                            timestamp,
                            round(scores.get("arm", 0), 2) if "arm" in scores else 0,
                            round(scores.get("handGoods", 0), 2) if "handGoods" in scores else 0,
                            pickup_frame_map.get(frame_idx, 0)
                        ])
                    
                    cap.release()
                    out.release()
                    
                    # ========== 后端数据修正：确保位置识别总数=拿取总数 ==========
                    # 修正策略：拿取总数不变，调整位置识别数据使其等于拿取总数
                    
                    initial_position_count = len(pickup_details)
                    
                    # 情况1：位置识别数 > 拿取总数 → 删除多余的（保留置信度最高的）
                    if initial_position_count > count > 0:
                        # 按置信度从高到低排序，只保留前N条（N = 拿取总数）
                        pickup_details = sorted(
                            pickup_details, 
                            key=lambda x: x['confidence'], 
                            reverse=True
                        )[:count]
                    
                    # 情况2：位置识别数 < 拿取总数 → 无法创造数据，保持原样
                    # （这种情况说明部分拿取动作未识别到位置，属于正常现象）
                    
                    # 情况3：位置识别数 = 拿取总数 → 不需要修正
                    
                    # 基于修正后的pickup_details重新计算row_col_stats
                    row_col_stats = {}
                    for detail in pickup_details:
                        key = (detail['row'], detail['col'])
                        row_col_stats[key] = row_col_stats.get(key, 0) + 1
                    
                    # 后端验证
                    final_position_count = len(pickup_details)
                    # print(f"[后端修正] 拿取总数: {count}, 原始位置识别: {initial_position_count}, 修正后: {final_position_count}, 等于拿取总数: {final_position_count == count or final_position_count < count}")
                    
                    # ========== 后端修正完成 ==========
                    
                    # 显示完成信息
                    status_text.text("✅ 视频处理完成!")
                    progress_bar.progress(1.0)
                    
                    # 显示处理结果摘要
                    st.success("🎉 视频处理完成！")
                    st.info(f"📊 处理摘要: 共处理 {frame_idx} 帧，检测到 {count} 次拿取动作")
                    
                    # 显示统计信息
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总帧数", f"{total_frames}")
                    with col2:
                        st.metric("检测到的拿取次数", f"{count}")
                    with col3:
                        st.metric("处理时长", f"{frame_idx/fps:.1f}秒")
                    
                    # 显示行列统计信息（后端已修正）
                    if row_col_stats or pickup_details:
                        st.markdown("---")
                        st.markdown("### 🗄️ 智能柜行列统计")
                        
                        # 显示颜色映射说明
                        with st.expander("📖 颜色位置映射表", expanded=False):
                            st.markdown("""
                            **智能柜布局: 6行 × 2列**
                            
                            | 行号 | 第1列 | 第2列 |
                            |------|-------|-------|
                            | 1 | argent | black |
                            | 2 | yellow | scarlet |
                            | 3 | green | red |
                            | 4 | grey | blue |
                            | 5 | orange | purple |
                            | 6 | cyan | darkBlue |
                            """)
                        
                        col_left, col_right = st.columns([1, 1])
                        
                        with col_left:
                            st.markdown("#### 📍 各行列拿取统计")
                            if row_col_stats:
                                # 创建6行2列的统计表格
                                stats_data = []
                                for row in range(1, 7):
                                    row_data = {'行号': row}
                                    for col in range(1, 3):
                                        key = (row, col)
                                        row_data[f'第{col}列'] = row_col_stats.get(key, 0)
                                    stats_data.append(row_data)
                                
                                df_stats = pd.DataFrame(stats_data)
                                st.dataframe(df_stats, use_container_width=True, hide_index=True)
                                
                                # 总计
                                total_by_position = sum(row_col_stats.values())
                                st.info(f"📦 各位置总拿取次数: {total_by_position}")
                                
                                # 绘制简单的热力图
                                try:
                                    import matplotlib.pyplot as plt
                                    import matplotlib
                                    matplotlib.use('Agg')
                                    
                                    # 创建热力图数据矩阵
                                    heatmap_data = np.zeros((6, 2))
                                    for (row, col), count in row_col_stats.items():
                                        heatmap_data[row-1, col-1] = count
                                    
                                    # 绘制热力图
                                    fig, ax = plt.subplots(figsize=(4, 6))
                                    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
                                    
                                    # 设置坐标轴
                                    ax.set_xticks([0, 1])
                                    ax.set_xticklabels(['第1列', '第2列'])
                                    ax.set_yticks(range(6))
                                    ax.set_yticklabels([f'第{i+1}行' for i in range(6)])
                                    
                                    # 在每个格子中显示数值
                                    for i in range(6):
                                        for j in range(2):
                                            text = ax.text(j, i, int(heatmap_data[i, j]),
                                                         ha="center", va="center", color="black", fontsize=12)
                                    
                                    ax.set_title("拿取次数热力图")
                                    plt.colorbar(im, ax=ax, label='拿取次数')
                                    plt.tight_layout()
                                    
                                    st.pyplot(fig)
                                    plt.close(fig)
                                except Exception as e:
                                    pass  # 如果绘图失败，不影响其他功能
                            else:
                                st.info("未检测到具体的行列位置信息")
                        
                        with col_right:
                            st.markdown("#### 📋 拿取详情记录")
                            if pickup_details:
                                df_details = pd.DataFrame(pickup_details)
                                df_details = df_details[['timestamp', 'row', 'col', 'color', 'confidence']]
                                df_details.columns = ['时间戳', '行', '列', '颜色', '置信度']
                                st.dataframe(df_details, use_container_width=True, hide_index=True)
                                
                                # 按颜色统计
                                st.markdown("##### 🎨 颜色分布")
                                color_counts = df_details['颜色'].value_counts()
                                for color, count in color_counts.items():
                                    row, col = COLOR_POSITION_MAP.get(color, (0, 0))
                                    st.write(f"• {color} (第{row}行第{col}列): {count}次")
                            else:
                                st.info("未检测到拿取详情")
                    
                    # 显示和下载处理后的视频
                    if os.path.exists(output_video_path):
                        st.markdown("---")
                        st.markdown("### 🎬 预测视频展示")
                        
                        try:
                            # 显示视频
                            with open(output_video_path, "rb") as file:
                                video_bytes = file.read()
                                
                            # 使用更兼容的视频显示方式
                            st.video(video_bytes, format="video/mp4")
                            
                            # 下载按钮
                            st.download_button(
                                label="📥 下载处理后的视频",
                                data=video_bytes,
                                file_name="processed_video.mp4",
                                mime="video/mp4"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ 视频显示失败: {str(e)}")
                            st.info("💡 请尝试下载视频文件到本地播放")
                            
                            # 提供下载按钮
                            try:
                                with open(output_video_path, "rb") as file:
                                    video_bytes = file.read()
                                st.download_button(
                                    label="📥 下载处理后的视频",
                                    data=video_bytes,
                                    file_name="processed_video.mp4",
                                    mime="video/mp4"
                                )
                            except Exception as download_error:
                                st.error(f"❌ 下载也失败了: {str(download_error)}")
                    else:
                        st.error("❌ 输出视频文件不存在")
                    
                    # 显示检测日志（在下载视频之后）
                    if log_data:
                        st.markdown("---")
                        st.markdown("### 📊 检测日志")
                        df = pd.DataFrame(log_data, columns=["时间戳", "手臂置信度", "手部商品置信度", "拿取次数"])
                        st.dataframe(df.head(100), use_container_width=True)
            
            # 清理临时文件
            if os.path.exists(video_path):
                os.unlink(video_path)

# 页脚信息
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>基于PaddleDetection模型的智能零售检测系统 | 让智能零售"看得清，数得明" | 支持实时视频检测 + 智能柜行列统计</p>
    <p>系统版本: 3.1 (PaddlePaddle) | 更新时间: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
</div>
""", unsafe_allow_html=True)

