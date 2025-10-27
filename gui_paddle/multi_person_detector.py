"""
多人拿取检测模块
基于双FSM状态机的多人拿取动作检测
"""

import cv2
import numpy as np
import datetime
from typing import List, Tuple, Dict, Optional


class MultiPersonDetector:
    """多人拿取检测器"""
    
    def __init__(self, 
                 action_window: int = 80,
                 arm_thresh: int = 3,
                 handGoods_thresh: int = 2,
                 arm_miss_limit: int = 4,
                 cooldown_limit: int = 5,
                 iou_threshold: float = 0.25):
        """
        初始化多人拿取检测器
        
        Args:
            action_window: 动作窗口大小
            arm_thresh: 手臂检测阈值
            handGoods_thresh: 手部商品检测阈值
            arm_miss_limit: 手臂消失限制
            cooldown_limit: 冷却限制
            iou_threshold: IoU匹配阈值
        """
        self.action_window = action_window
        self.arm_thresh = arm_thresh
        self.handGoods_thresh = handGoods_thresh
        self.arm_miss_limit = arm_miss_limit
        self.cooldown_limit = cooldown_limit
        self.iou_threshold = iou_threshold
        
        # 初始化两个FSM
        self.fsm1 = self._init_fsm()
        self.fsm2 = self._init_fsm()
        
        # 总计数
        self.total_count = 0
        
        # 检测日志
        self.detection_log = []
        
        # 人员颜色映射（用于可视化）
        self.person_colors = {
            'FSM1': (0, 255, 0),    # 绿色
            'FSM2': (255, 0, 0),    # 蓝色
        }
    
    def _init_fsm(self) -> Dict:
        """初始化FSM状态机"""
        return {
            "state": "idle",
            "arm_box": None,
            "arm_buffer": [],
            "handGoods_buffer": [],
            "arm_missing_count": 0,
            "cooldown_counter": 0,
            "count": 0,
            "last_detection_frame": 0
        }
    
    def _iou(self, boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        """计算两个框的IoU"""
        ax1, ay1, ax2, ay2 = boxA
        bx1, by1, bx2, by2 = boxB
        
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        areaA = (ax2 - ax1) * (ay2 - ay1)
        areaB = (bx2 - bx1) * (by2 - by1)
        union_area = areaA + areaB - inter_area
        
        if union_area == 0:
            return 0
        return inter_area / union_area
    
    def _update_fsm(self, fsm: Dict, fsm_name: str, frame_idx: int, 
                   arm_boxes: List[Tuple], hand_boxes: List[Tuple]) -> bool:
        """
        更新FSM状态机
        
        Args:
            fsm: FSM状态机
            fsm_name: FSM名称
            frame_idx: 当前帧索引
            arm_boxes: 手臂检测框列表
            hand_boxes: 手部商品检测框列表
            
        Returns:
            bool: 是否检测到拿取动作
        """
        detection_occurred = False
        
        # 更新手臂框
        if fsm_name == "FSM1" and len(arm_boxes) >= 1:
            fsm["arm_box"] = arm_boxes[0]
            fsm["arm_missing_count"] = 0
        elif fsm_name == "FSM2" and len(arm_boxes) >= 2:
            fsm["arm_box"] = arm_boxes[1]
            fsm["arm_missing_count"] = 0
        else:
            fsm["arm_missing_count"] += 1
        
        # 如果没有手臂框，跳过处理
        if fsm["arm_box"] is None:
            return False
        
        # 匹配手部商品框
        matched_hand = False
        for hbox in hand_boxes:
            if self._iou(fsm["arm_box"], hbox) > self.iou_threshold:
                matched_hand = True
                break
        
        # FSM状态流转
        if fsm["state"] == "idle":
            if fsm["arm_box"] is not None:
                fsm["arm_buffer"].append(frame_idx)
                fsm["arm_buffer"] = [f for f in fsm["arm_buffer"] 
                                   if frame_idx - f <= self.action_window]
                if len(fsm["arm_buffer"]) >= self.arm_thresh:
                    fsm["state"] = "waiting"
                    fsm["handGoods_buffer"].clear()
        
        elif fsm["state"] == "waiting":
            if matched_hand:
                fsm["handGoods_buffer"].append(frame_idx)
                fsm["handGoods_buffer"] = [f for f in fsm["handGoods_buffer"] 
                                         if frame_idx - f <= self.action_window]
            else:
                fsm["handGoods_buffer"] = [f for f in fsm["handGoods_buffer"] 
                                         if frame_idx - f <= self.action_window]
            
            if len(fsm["handGoods_buffer"]) >= self.handGoods_thresh:
                self.total_count += 1
                fsm["count"] += 1
                fsm["last_detection_frame"] = frame_idx
                detection_occurred = True
                
                # 记录检测日志
                now = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                log_entry = f"[{time_str}] ✅ {fsm_name} 拿取成功，总数={self.total_count}"
                self.detection_log.append(log_entry)
                
                fsm["state"] = "cooldown"
                fsm["cooldown_counter"] = 0
                fsm["arm_buffer"].clear()
                fsm["handGoods_buffer"].clear()
        
        elif fsm["state"] == "cooldown":
            if matched_hand:
                fsm["cooldown_counter"] = 0
            else:
                fsm["cooldown_counter"] += 1
                if fsm["cooldown_counter"] >= self.cooldown_limit:
                    fsm["state"] = "idle"
                    fsm["cooldown_counter"] = 0
        
        return detection_occurred
    
    def process_frame(self, frame_idx: int, arm_boxes: List[Tuple], 
                     hand_boxes: List[Tuple]) -> Dict:
        """
        处理单帧检测结果
        
        Args:
            frame_idx: 当前帧索引
            arm_boxes: 手臂检测框列表
            hand_boxes: 手部商品检测框列表
            
        Returns:
            Dict: 检测结果信息
        """
        # 更新两个FSM
        detection1 = self._update_fsm(self.fsm1, "FSM1", frame_idx, arm_boxes, hand_boxes)
        detection2 = self._update_fsm(self.fsm2, "FSM2", frame_idx, arm_boxes, hand_boxes)
        
        return {
            "total_count": self.total_count,
            "fsm1_count": self.fsm1["count"],
            "fsm2_count": self.fsm2["count"],
            "fsm1_state": self.fsm1["state"],
            "fsm2_state": self.fsm2["state"],
            "detection_occurred": detection1 or detection2,
            "active_persons": self._get_active_persons()
        }
    
    def _get_active_persons(self) -> List[str]:
        """获取当前活跃的人员"""
        active = []
        if self.fsm1["state"] != "idle":
            active.append("FSM1")
        if self.fsm2["state"] != "idle":
            active.append("FSM2")
        return active
    
    def get_detection_summary(self) -> Dict:
        """获取检测摘要"""
        return {
            "total_pickups": self.total_count,
            "person1_pickups": self.fsm1["count"],
            "person2_pickups": self.fsm2["count"],
            "detection_log": self.detection_log.copy(),
            "active_persons": self._get_active_persons()
        }
    
    def reset(self):
        """重置检测器"""
        self.fsm1 = self._init_fsm()
        self.fsm2 = self._init_fsm()
        self.total_count = 0
        self.detection_log.clear()
    
    def draw_detection_info(self, frame: np.ndarray, detection_info: Dict) -> np.ndarray:
        """
        在帧上绘制检测信息
        
        Args:
            frame: 输入帧
            detection_info: 检测信息
            
        Returns:
            np.ndarray: 绘制后的帧
        """
        # 绘制总计数
        total_text = f"总拿取数量: {detection_info['total_count']}"
        
        # 绘制各人员计数
        person1_text = f"人员1: {detection_info['fsm1_count']}"
        person2_text = f"人员2: {detection_info['fsm2_count']}"
        
        # 绘制状态信息
        state1_text = f"状态1: {detection_info['fsm1_state']}"
        state2_text = f"状态2: {detection_info['fsm2_state']}"
        
        # 使用OpenCV绘制文字（简化版本）
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # 绘制总计数（绿色）
        cv2.putText(frame, total_text, (50, 50), font, font_scale, (0, 255, 0), thickness)
        
        # 绘制人员计数
        cv2.putText(frame, person1_text, (50, 80), font, font_scale, (0, 255, 0), thickness)
        cv2.putText(frame, person2_text, (50, 110), font, font_scale, (255, 0, 0), thickness)
        
        # 绘制状态信息
        cv2.putText(frame, state1_text, (50, 140), font, font_scale * 0.5, (0, 255, 0), thickness)
        cv2.putText(frame, state2_text, (50, 160), font, font_scale * 0.5, (255, 0, 0), thickness)
        
        return frame

