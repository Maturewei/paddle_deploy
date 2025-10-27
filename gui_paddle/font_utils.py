"""
字体工具模块
处理中文字体显示问题
"""

import os
import platform
import requests
from PIL import ImageFont

def get_system_fonts():
    """获取系统可用字体"""
    system = platform.system()
    fonts = []
    
    if system == "Windows":
        font_dirs = [
            "C:/Windows/Fonts/",
            "C:/WINDOWS/Fonts/"
        ]
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for file in os.listdir(font_dir):
                    if file.lower().endswith(('.ttf', '.ttc', '.otf')):
                        fonts.append(os.path.join(font_dir, file))
    elif system == "Darwin":  # macOS
        font_dirs = [
            "/System/Library/Fonts/",
            "/Library/Fonts/",
            "~/Library/Fonts/"
        ]
        for font_dir in font_dirs:
            expanded_dir = os.path.expanduser(font_dir)
            if os.path.exists(expanded_dir):
                for file in os.listdir(expanded_dir):
                    if file.lower().endswith(('.ttf', '.ttc', '.otf')):
                        fonts.append(os.path.join(expanded_dir, file))
    elif system == "Linux":
        font_dirs = [
            "/usr/share/fonts/",
            "/usr/local/share/fonts/",
            "~/.fonts/"
        ]
        for font_dir in font_dirs:
            expanded_dir = os.path.expanduser(font_dir)
            if os.path.exists(expanded_dir):
                for file in os.listdir(expanded_dir):
                    if file.lower().endswith(('.ttf', '.ttc', '.otf')):
                        fonts.append(os.path.join(expanded_dir, file))
    
    return fonts

def find_chinese_font():
    """查找支持中文的字体"""
    # 优先查找的字体名称（按优先级排序）
    preferred_fonts = [
        "simhei.ttf",  # 黑体
        "simsun.ttc",  # 宋体
        "msyh.ttf",    # 微软雅黑
        "msyhbd.ttf",  # 微软雅黑粗体
        "simkai.ttf",  # 楷体
        "simfang.ttf", # 仿宋
        "notosanscjk.ttc",  # Noto Sans CJK
        "sourcehansansc.ttf",  # 思源黑体
    ]
    
    system_fonts = get_system_fonts()
    
    # 首先尝试找到优先字体
    for preferred in preferred_fonts:
        for font_path in system_fonts:
            if preferred.lower() in os.path.basename(font_path).lower():
                try:
                    # 测试字体是否可以加载
                    ImageFont.truetype(font_path, 24)
                    return font_path
                except:
                    continue
    
    # 如果没有找到优先字体，尝试其他字体
    for font_path in system_fonts:
        try:
            font = ImageFont.truetype(font_path, 24)
            # 简单测试是否支持中文（通过尝试渲染一个中文字符）
            # 这里只是测试字体是否可以加载，实际的中文支持需要更复杂的测试
            return font_path
        except:
            continue
    
    return None

def download_chinese_font():
    """下载中文字体（如果系统没有）"""
    font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJK-Regular.otf"
    font_path = "NotoSansCJK-Regular.otf"
    
    if os.path.exists(font_path):
        return font_path
    
    try:
        response = requests.get(font_url, timeout=10)
        if response.status_code == 200:
            with open(font_path, 'wb') as f:
                f.write(response.content)
            return font_path
    except:
        pass
    
    return None

def get_best_font_path():
    """获取最佳字体路径"""
    # 首先尝试系统字体
    system_font = find_chinese_font()
    if system_font:
        return system_font
    
    # 尝试下载字体
    downloaded_font = download_chinese_font()
    if downloaded_font:
        return downloaded_font
    
    # 返回None，使用默认字体
    return None

def safe_draw_text(img, text, position, font_size=24, color=(0, 255, 0)):
    """安全地绘制文字，自动处理字体问题"""
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    import numpy as np
    
    try:
        # 转换图像格式
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 获取最佳字体
        font_path = get_best_font_path()
        
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
        
        # 转换颜色格式
        rgb_color = (color[2], color[1], color[0])  # BGR转RGB
        
        # 绘制文字
        draw.text(position, text, font=font, fill=rgb_color)
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        # 如果PIL失败，使用OpenCV的英文文字
        try:
            # 提取数字部分作为英文显示
            if ":" in text:
                english_text = "Count: " + text.split(":")[-1].strip()
            else:
                english_text = text
            
            cv2.putText(img, english_text, position, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        except:
            pass
        
        return img

