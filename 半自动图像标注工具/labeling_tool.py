#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visual Tension 图像标注工具
用于半自动化图像标注，训练 Stable Diffusion LoRA 模型
"""

import os
import json
import base64
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from io import BytesIO
from datetime import datetime

import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 设置 customtkinter 外观
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

COLORS = {
    "app_bg": "#F3F5F9",
    "surface": "#FFFFFF",
    "surface_alt": "#F8FAFC",
    "canvas": "#111827",
    "border": "#DDE3EC",
    "divider": "#E7EBF1",
    "text": "#172033",
    "muted": "#667085",
    "primary": "#2563EB",
    "primary_hover": "#1D4ED8",
    "primary_soft": "#EFF6FF",
    "ai": "#0F766E",
    "ai_hover": "#115E59",
    "ai_soft": "#ECFDF5",
    "warning": "#D97706",
}


# ==================== 配置管理函数 ====================

APP_DIR = Path(__file__).resolve().parent
ALLOWED_AI_TAGS = [
    "flat_facade", "facade_setback", "overhang_canopy", "street_spillout", "spatial_folding",
    "low_transparency", "medium_transparency", "high_transparency",
    "limited_entrance", "moderate_entrance", "multiple_entrance"
]

def load_api_config():
    """加载API配置"""
    config_file = APP_DIR / "api_config.json"
    if config_file.exists():
        with config_file.open('r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "api_key": "",
        "api_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o",
        "quality": "auto",
        "timeout": 60
    }


def save_api_config(api_key, api_url, model, quality, timeout):
    """保存API配置"""
    config = {
        "api_key": api_key,
        "api_url": api_url,
        "model": model,
        "quality": quality,
        "timeout": timeout
    }
    with (APP_DIR / "api_config.json").open('w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_prompt_config():
    """加载Prompt配置"""
    config_file = APP_DIR / "prompt_config.json"
    if config_file.exists():
        with config_file.open('r', encoding='utf-8') as f:
            return json.load(f)
    # 默认prompt（如果文件不存在）
    return {
        "enabled": True,
        "system_prompt": "You are a precise architectural image annotation agent.",
        "default_prompt": "Select only visibly supported labels from the configured taxonomy and return one comma-separated line."
    }


# ==================== GPT-4V API调用函数 ====================

def encode_image_to_base64(image_path_or_pil):
    """编码图片为base64

    Args:
        image_path_or_pil: 图片文件路径(str)或PIL Image对象

    Returns:
        str: base64编码的图片字符串
    """
    if isinstance(image_path_or_pil, str):
        with open(image_path_or_pil, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    else:
        # PIL Image对象
        buffered = BytesIO()
        image_path_or_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def call_gpt4v_api(image, system_prompt, prompt, api_key, api_url, model="gpt-4o", quality="auto", timeout=60):
    """调用GPT-4V API进行图片打标

    Args:
        image: PIL Image对象或图片路径
        prompt: 提示词
        api_key: OpenAI API密钥
        api_url: API URL
        quality: 图片质量 ('auto', 'high', 'low')
        timeout: 超时时间（秒）

    Returns:
        tuple: (caption, error) - caption为标签字符串，error为错误信息
    """
    try:
        # 编码图片
        image_base64 = encode_image_to_base64(image)

        # 构建请求数据
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": quality
                        }}
                    ]
                }
            ],
            "max_tokens": 200
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # 配置重试策略
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )

        session = requests.Session()
        session.mount('https://', HTTPAdapter(max_retries=retries))

        # 发送请求
        response = session.post(api_url, headers=headers, json=data, timeout=timeout)
        response.raise_for_status()

        # 解析响应
        response_data = response.json()
        if 'error' in response_data:
            return None, f"API错误: {response_data['error']['message']}"

        choices = response_data.get("choices") or []
        if not choices:
            return None, "API响应中没有可用结果"
        caption = choices[0].get("message", {}).get("content", "")
        if not isinstance(caption, str):
            return None, "API返回内容格式异常"
        return caption, None

    except requests.exceptions.Timeout:
        return None, "请求超时，请检查网络连接或增加超时时间"
    except requests.exceptions.ConnectionError:
        return None, "连接错误，请检查网络和API URL"
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("error", {}).get("message", "")
        except Exception:
            pass
        return None, f"HTTP错误: {e}" + (f"；{detail}" if detail else "")
    except Exception as e:
        return None, f"未知错误: {str(e)}"


# ==================== 工具类 ====================

class ToolTip:
    """工具提示类，用于在鼠标悬停时显示说明"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert") if hasattr(self.widget, "bbox") else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                        background="#2b2b2b", foreground="white",
                        relief=tk.SOLID, borderwidth=1,
                        font=("Arial", 10), padx=8, pady=6)
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class CustomDialog(ctk.CTkToplevel):
    """自定义对话框，支持自定义按钮文本"""

    def __init__(self, parent, title, message, button1_text="按钮1", button2_text="按钮2", button3_text="按钮3"):
        super().__init__(parent)

        self.result = None

        # 窗口设置
        self.title(title)
        self.geometry("500x300")
        self.resizable(False, False)

        # 居中显示
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.winfo_screenheight() // 2) - (300 // 2)
        self.geometry(f"500x300+{x}+{y}")

        # 设置为模态对话框
        self.transient(parent)
        self.grab_set()

        # 主容器
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # 消息文本
        message_label = ctk.CTkLabel(
            main_frame,
            text=message,
            font=ctk.CTkFont(size=13),
            justify="left",
            wraplength=460
        )
        message_label.pack(pady=(10, 30))

        # 按钮容器
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(side="bottom", fill="x", pady=(20, 0))

        # 按钮1 - 继续应用（绿色，主要操作）
        self.button1 = ctk.CTkButton(
            button_frame,
            text=button1_text,
            command=lambda: self.on_button_click(True),
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2d8659",
            hover_color="#1e5f3f"
        )
        self.button1.pack(side="left", expand=True, fill="x", padx=(0, 5))

        # 按钮3 - 重新裁剪（灰色，次要操作）
        self.button3 = ctk.CTkButton(
            button_frame,
            text=button3_text,
            command=lambda: self.on_button_click(None),
            height=40,
            font=ctk.CTkFont(size=13),
            fg_color="#666666",
            hover_color="#555555"
        )
        self.button3.pack(side="left", expand=True, fill="x", padx=5)

        # 按钮2 - 下一张（橙色，跳过操作）
        self.button2 = ctk.CTkButton(
            button_frame,
            text=button2_text,
            command=lambda: self.on_button_click(False),
            height=40,
            font=ctk.CTkFont(size=13),
            fg_color="#d97706",
            hover_color="#b45309"
        )
        self.button2.pack(side="left", expand=True, fill="x", padx=(5, 0))

        # 绑定关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 等待窗口关闭
        self.wait_window()

    def on_button_click(self, value):
        """按钮点击处理"""
        self.result = value
        self.destroy()

    def on_closing(self):
        """关闭窗口"""
        self.result = None
        self.destroy()

    def get_result(self):
        """获取结果"""
        return self.result


class ImagePanel(ctk.CTkFrame):
    """左侧图片面板：图片显示、裁剪和导航功能"""

    def __init__(self, master, app):
        super().__init__(
            master,
            fg_color=COLORS["surface"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"]
        )
        self.app = app

        # 裁剪相关状态
        self.crop_mode = "4:3"  # 默认比例
        self.crop_ratio = 4/3  # 当前裁剪比例
        self.crop_box = None  # 裁剪框位置 (x, y, width, height) 在显示图片上的坐标
        self.dragging = False  # 是否正在拖动
        self.drag_start = None  # 拖动起始点
        self.resize_handle = None  # 当前调整的手柄（None, 'tl', 'tr', 'bl', 'br', 't', 'b', 'l', 'r'）
        self.handle_size = 10  # 调整手柄的大小
        self.image_offset = (0, 0)  # 图片在frame中的偏移量（用于居中显示）
        self.image_display_size = (0, 0)  # 图片实际显示尺寸

        self.setup_ui()

    def setup_ui(self):
        """设置 UI 组件"""
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=18, pady=(18, 10))
        ctk.CTkLabel(
            header, text="01  数据预处理",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLORS["text"]
        ).pack(anchor="w")
        ctk.CTkLabel(
            header, text="导入、预览并统一训练图像比例",
            font=ctk.CTkFont(size=11), text_color=COLORS["muted"]
        ).pack(anchor="w", pady=(3, 0))

        # 顶部按钮区域
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.pack(fill="x", padx=18, pady=(0, 8))

        self.select_folder_btn = ctk.CTkButton(
            top_frame, text="📁 选择图片文件夹",
            command=self.select_folder,
            height=40, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"],
            corner_radius=8
        )
        self.select_folder_btn.pack(fill="x")

        # 进度显示
        self.progress_label = ctk.CTkLabel(
            top_frame, text="未加载图片",
            font=ctk.CTkFont(size=11), text_color=COLORS["muted"]
        )
        self.progress_label.pack(pady=(10, 0))

        # 图片显示区域（固定尺寸，调整为500x500）
        self.image_frame = ctk.CTkFrame(
            self, fg_color=COLORS["canvas"], width=500, height=500,
            corner_radius=10, border_width=1, border_color="#273449"
        )
        self.image_frame.pack(fill="none", expand=False, padx=18, pady=10)
        self.image_frame.pack_propagate(False)  # 防止内容改变frame尺寸

        self.image_label = ctk.CTkLabel(
            self.image_frame, text="请选择图片文件夹",
            font=ctk.CTkFont(size=14), text_color="#E5E7EB"
        )
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")  # 使用place居中

        # 绑定鼠标事件用于裁剪
        self.image_label.bind("<ButtonPress-1>", self.on_crop_start)
        self.image_label.bind("<B1-Motion>", self.on_crop_drag)
        self.image_label.bind("<ButtonRelease-1>", self.on_crop_end)

        # 裁剪控制区域
        crop_frame = ctk.CTkFrame(
            self, fg_color=COLORS["surface_alt"], corner_radius=10,
            border_width=1, border_color=COLORS["border"]
        )
        crop_frame.pack(fill="x", padx=18, pady=10)

        ctk.CTkLabel(
            crop_frame, text="裁剪比例:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=5, pady=(5, 2))

        # 裁剪比例选择
        self.crop_mode_menu = ctk.CTkOptionMenu(
            crop_frame,
            values=["原始", "4:3", "16:9", "1:1", "3:2", "2:3", "自由裁剪", "自定义比例"],
            command=self.on_crop_mode_change,
            height=32
        )
        self.crop_mode_menu.set("4:3")  # 默认选项
        self.crop_mode_menu.pack(fill="x", padx=5, pady=5)

        # 自定义比例输入区域
        self.custom_ratio_frame = ctk.CTkFrame(crop_frame, fg_color="transparent")

        ratio_input_frame = ctk.CTkFrame(self.custom_ratio_frame, fg_color="transparent")
        ratio_input_frame.pack(fill="x")

        ctk.CTkLabel(ratio_input_frame, text="宽:", font=ctk.CTkFont(size=11)).pack(side="left", padx=(0, 5))
        self.ratio_width_entry = ctk.CTkEntry(ratio_input_frame, width=70, height=28, placeholder_text="4")
        self.ratio_width_entry.pack(side="left", padx=(0, 5))
        self.ratio_width_entry.insert(0, "4")

        ctk.CTkLabel(ratio_input_frame, text=":", font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=5)

        ctk.CTkLabel(ratio_input_frame, text="高:", font=ctk.CTkFont(size=11)).pack(side="left", padx=(5, 5))
        self.ratio_height_entry = ctk.CTkEntry(ratio_input_frame, width=70, height=28, placeholder_text="3")
        self.ratio_height_entry.pack(side="left")
        self.ratio_height_entry.insert(0, "3")

        self.apply_ratio_btn = ctk.CTkButton(
            self.custom_ratio_frame, text="应用比例",
            command=self.apply_custom_ratio,
            height=28, fg_color="#4a9eff"
        )
        self.apply_ratio_btn.pack(fill="x", pady=(5, 0))

        # 裁剪提示（动态更新）
        self.crop_info_label = ctk.CTkLabel(
            crop_frame, text="💡 拖动裁剪框调整位置和大小",
            font=ctk.CTkFont(size=10),
            text_color=COLORS["muted"]
        )
        self.crop_info_label.pack(pady=5)

        # 裁剪状态提示（醒目提示）
        self.crop_status_label = ctk.CTkLabel(
            crop_frame, text="",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["warning"]
        )
        self.crop_status_label.pack(pady=(0, 5))

        # 按钮区域
        crop_btn_frame = ctk.CTkFrame(crop_frame, fg_color="transparent")
        crop_btn_frame.pack(fill="x", padx=5, pady=5)

        self.apply_crop_btn = ctk.CTkButton(
            crop_btn_frame, text="✂️ 应用裁剪",
            command=self.apply_crop,
            height=34, fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"]
        )
        self.apply_crop_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.reset_crop_btn = ctk.CTkButton(
            crop_btn_frame, text="🔄 重置",
            command=self.reset_crop,
            height=34, fg_color="#E7ECF3", hover_color="#D7DEE9",
            text_color=COLORS["text"]
        )
        self.reset_crop_btn.pack(side="left", expand=True, fill="x")

        # 导航按钮
        nav_frame = ctk.CTkFrame(self, fg_color="transparent")
        nav_frame.pack(fill="x", padx=18, pady=(0, 18))

        self.prev_btn = ctk.CTkButton(
            nav_frame, text="← 上一张",
            command=self.app.prev_image,
            height=35, state="disabled"
        )
        self.prev_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.next_btn = ctk.CTkButton(
            nav_frame, text="下一张 →",
            command=self.app.next_image,
            height=35, state="disabled"
        )
        self.next_btn.pack(side="left", expand=True, fill="x")

    def select_folder(self):
        """选择图片文件夹"""
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if folder:
            self.app.load_image_folder(folder)

    def on_crop_mode_change(self, choice):
        """裁剪比例改变"""
        self.crop_mode = choice

        # 显示/隐藏自定义比例输入
        if choice == "自定义比例":
            self.custom_ratio_frame.pack(fill="x", padx=5, pady=5, before=self.crop_info_label)
        else:
            self.custom_ratio_frame.pack_forget()

        # 更新裁剪比例
        ratio_map = {
            "4:3": 4/3,
            "16:9": 16/9,
            "1:1": 1.0,
            "3:2": 3/2,
            "2:3": 2/3
        }

        if choice in ratio_map:
            self.crop_ratio = ratio_map[choice]
        elif choice == "自由裁剪":
            self.crop_ratio = None  # 自由裁剪不限制比例

        # 如果有图片，重新初始化裁剪框
        if self.app.app_state["current_image"]:
            img = self.app.app_state.get("cropped_image") or self.app.app_state["current_image"]

            if choice == "原始":
                self.crop_box = None
                # 清除状态提示
                self.crop_status_label.configure(text="")
                self.display_image(img)
            else:
                # 先显示图片，再初始化裁剪框
                self.display_image(img)
                self._init_crop_box()
                self.display_image(img, show_crop_rect=True)

    def apply_custom_ratio(self):
        """应用自定义比例"""
        try:
            width_ratio = float(self.ratio_width_entry.get())
            height_ratio = float(self.ratio_height_entry.get())

            if width_ratio <= 0 or height_ratio <= 0:
                messagebox.showwarning("警告", "比例必须大于 0")
                return

            self.crop_ratio = width_ratio / height_ratio

            if self.app.app_state["current_image"]:
                img = self.app.app_state.get("cropped_image") or self.app.app_state["current_image"]
                # 先显示图片，再初始化裁剪框
                self.display_image(img)
                self._init_crop_box()
                self.display_image(img, show_crop_rect=True)

            messagebox.showinfo("成功", f"已应用比例: {width_ratio}:{height_ratio}")

        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")

    def _get_resize_handle(self, event_x, event_y):
        """检测鼠标是否在调整手柄上"""
        if not self.crop_box:
            return None

        x, y, w, h = self.crop_box
        hs = self.handle_size  # 手柄尺寸

        # 检查四个角
        if abs(event_x - x) <= hs and abs(event_y - y) <= hs:
            return 'tl'  # 左上角
        if abs(event_x - (x + w)) <= hs and abs(event_y - y) <= hs:
            return 'tr'  # 右上角
        if abs(event_x - x) <= hs and abs(event_y - (y + h)) <= hs:
            return 'bl'  # 左下角
        if abs(event_x - (x + w)) <= hs and abs(event_y - (y + h)) <= hs:
            return 'br'  # 右下角

        # 检查四条边
        if abs(event_x - x) <= hs and y < event_y < y + h:
            return 'l'  # 左边
        if abs(event_x - (x + w)) <= hs and y < event_y < y + h:
            return 'r'  # 右边
        if abs(event_y - y) <= hs and x < event_x < x + w:
            return 't'  # 上边
        if abs(event_y - (y + h)) <= hs and x < event_x < x + w:
            return 'b'  # 下边

        # 检查是否在框内（用于移动）
        if x < event_x < x + w and y < event_y < y + h:
            return 'move'

        return None

    def _init_crop_box(self):
        """初始化裁剪框（居中显示，按比例）"""
        img = self.app.app_state.get("cropped_image") or self.app.app_state["current_image"]
        if not img:
            return

        # 获取实际显示尺寸和偏移
        display_width, display_height = self.image_display_size
        if display_width == 0 or display_height == 0:
            return

        if self.crop_mode == "自由裁剪":
            # 自由裁剪：初始裁剪框为图片 70% 大小
            crop_display_width = int(display_width * 0.7)
            crop_display_height = int(display_height * 0.7)
        elif self.crop_ratio is not None:
            # 按比例裁剪：根据图片尺寸和目标比例计算裁剪框
            # 目标：初始裁剪框尽可能大，至少有一边填满图片
            img_ratio = display_width / display_height

            if img_ratio > self.crop_ratio:
                # 图片更宽，高度填满
                crop_display_height = display_height  # 使用 100% 高度
                crop_display_width = int(crop_display_height * self.crop_ratio)
            else:
                # 图片更高或相等，宽度填满
                crop_display_width = display_width  # 使用 100% 宽度
                crop_display_height = int(crop_display_width / self.crop_ratio)
        else:
            return

        # 确保裁剪框不超出图片显示区域
        crop_display_width = min(crop_display_width, display_width)
        crop_display_height = min(crop_display_height, display_height)

        # 居中放置（相对于图片左上角）
        x = (display_width - crop_display_width) // 2
        y = (display_height - crop_display_height) // 2

        self.crop_box = (x, y, crop_display_width, crop_display_height)

        # 更新状态提示
        self.crop_status_label.configure(text="⚠️ 请先点击'应用裁剪'")
        self.crop_status_label.pack(pady=(0, 5))

    def on_crop_start(self, event):
        """开始拖动/绘制/缩放裁剪框"""
        if self.crop_mode == "原始":
            return

        if self.crop_mode == "自由裁剪" and not self.crop_box:
            # 自由裁剪：开始绘制新的裁剪框
            self.drag_start = (event.x, event.y)
            self.resize_handle = 'draw'
        elif self.crop_box:
            # 检测点击位置
            handle = self._get_resize_handle(event.x, event.y)
            if handle:
                self.resize_handle = handle
                self.drag_start = (event.x, event.y)
                self.dragging = (handle == 'move')

    def on_crop_drag(self, event):
        """拖动/绘制/缩放裁剪框"""
        if not self.drag_start or not self.resize_handle:
            return

        img = self.app.app_state.get("cropped_image") or self.app.app_state["current_image"]
        if not img:
            return

        display_size = self._get_display_size(img.size)
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]

        if self.resize_handle == 'draw':
            # 绘制新裁剪框（自由裁剪）
            x1 = min(self.drag_start[0], event.x)
            y1 = min(self.drag_start[1], event.y)
            x2 = max(self.drag_start[0], event.x)
            y2 = max(self.drag_start[1], event.y)

            w = x2 - x1
            h = y2 - y1

            if w > 10 and h > 10:
                self.crop_box = (x1, y1, w, h)
                self.display_image(img, show_crop_rect=True)

        elif self.resize_handle == 'move':
            # 移动裁剪框
            x, y, w, h = self.crop_box
            new_x = x + dx
            new_y = y + dy

            # 获取实际图片边界
            display_width, display_height = self.image_display_size

            # 限制在图片范围内
            new_x = max(0, min(new_x, display_width - w))
            new_y = max(0, min(new_y, display_height - h))

            self.crop_box = (new_x, new_y, w, h)
            self.drag_start = (event.x, event.y)
            self.display_image(img, show_crop_rect=True)

        else:
            # 缩放裁剪框
            x, y, w, h = self.crop_box
            display_width, display_height = self.image_display_size

            # 有比例限制且不是自由裁剪：只能等比例缩放
            if self.crop_ratio and self.crop_mode != "自由裁剪":
                # 计算主要调整方向
                if self.resize_handle in ['tl', 'tr', 'bl', 'br']:
                    # 角点：根据对角线距离等比例缩放
                    # 计算从中心点的缩放
                    center_x = x + w / 2
                    center_y = y + h / 2

                    # 计算新的尺寸（保持比例）
                    if 'r' in self.resize_handle:
                        new_w = w + dx
                    else:  # 'l'
                        new_w = w - dx

                    new_h = int(new_w / self.crop_ratio)

                    # 根据调整的角更新位置
                    if 't' in self.resize_handle:
                        new_y = y + h - new_h
                        new_x = x if 'l' not in self.resize_handle else x + w - new_w
                    else:  # 'b'
                        new_y = y
                        new_x = x if 'l' not in self.resize_handle else x + w - new_w
                else:
                    # 边缘：根据调整方向保持比例
                    if 'l' in self.resize_handle or 'r' in self.resize_handle:
                        # 水平调整
                        new_w = w + dx if 'r' in self.resize_handle else w - dx
                        new_h = int(new_w / self.crop_ratio)
                        new_x = x if 'r' in self.resize_handle else x + w - new_w
                        # 保持垂直居中
                        new_y = y + (h - new_h) / 2
                    else:
                        # 垂直调整
                        new_h = h + dy if 'b' in self.resize_handle else h - dy
                        new_w = int(new_h * self.crop_ratio)
                        new_y = y if 'b' in self.resize_handle else y + h - new_h
                        # 保持水平居中
                        new_x = x + (w - new_w) / 2
            else:
                # 自由裁剪：可以任意调整
                new_x, new_y, new_w, new_h = x, y, w, h

                if 'l' in self.resize_handle:
                    new_x = x + dx
                    new_w = w - dx
                if 'r' in self.resize_handle:
                    new_w = w + dx
                if 't' in self.resize_handle:
                    new_y = y + dy
                    new_h = h - dy
                if 'b' in self.resize_handle:
                    new_h = h + dy

            # 确保最小尺寸
            if new_w < 20:
                new_w = 20
            if new_h < 20:
                new_h = 20

            # 限制在图片范围内
            new_x = max(0, new_x)
            new_y = max(0, new_y)
            new_x = min(new_x, display_width - new_w)
            new_y = min(new_y, display_height - new_h)
            new_w = min(new_w, display_width - new_x)
            new_h = min(new_h, display_height - new_y)

            # 如果是比例模式，边界限制后重新调整以保持精确比例
            if self.crop_ratio and self.crop_mode != "自由裁剪":
                # 根据限制后的尺寸，重新计算以保持比例
                # 选择能够容纳的最大尺寸
                if new_w / self.crop_ratio > new_h:
                    # 高度是限制因素，根据高度重新计算宽度
                    new_w = new_h * self.crop_ratio
                else:
                    # 宽度是限制因素，根据宽度重新计算高度
                    new_h = new_w / self.crop_ratio

                # 再次确保不超出边界
                if new_x + new_w > display_width:
                    new_w = display_width - new_x
                    new_h = new_w / self.crop_ratio
                if new_y + new_h > display_height:
                    new_h = display_height - new_y
                    new_w = new_h * self.crop_ratio

            self.crop_box = (int(new_x), int(new_y), int(new_w), int(new_h))
            self.drag_start = (event.x, event.y)
            self.display_image(img, show_crop_rect=True)

    def on_crop_end(self, event):
        """结束拖动/绘制/缩放"""
        self.dragging = False
        self.drag_start = None
        self.resize_handle = None

    def apply_crop(self):
        """应用裁剪"""
        img = self.app.app_state.get("cropped_image") or self.app.app_state["current_image"]
        if not img:
            return

        if self.crop_mode == "原始":
            messagebox.showinfo("提示", "当前为原始模式，无需裁剪")
            return

        if not self.crop_box:
            messagebox.showwarning("警告", "请先初始化裁剪框")
            return

        # 获取显示尺寸和实际尺寸的比例
        display_width, display_height = self.image_display_size
        if display_width == 0 or display_height == 0:
            messagebox.showerror("错误", "图片显示异常")
            return

        scale_x = img.width / display_width
        scale_y = img.height / display_height

        # 转换裁剪框坐标到原图（crop_box是相对于显示图片的坐标）
        x, y, w, h = self.crop_box
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)

        # 确保坐标在图片范围内
        x1 = max(0, min(x1, img.width))
        y1 = max(0, min(y1, img.height))
        x2 = max(0, min(x2, img.width))
        y2 = max(0, min(y2, img.height))

        # 如果是比例模式，调整坐标以确保精确比例
        if self.crop_ratio and self.crop_mode != "自由裁剪":
            crop_w = x2 - x1
            crop_h = y2 - y1
            actual_ratio = crop_w / crop_h

            # 如果比例不匹配（允许0.5%的误差），调整裁剪区域
            if abs(actual_ratio - self.crop_ratio) / self.crop_ratio > 0.005:
                if actual_ratio > self.crop_ratio:
                    # 宽度太大，缩小宽度
                    target_w = int(crop_h * self.crop_ratio)
                    # 保持中心位置
                    center_x = (x1 + x2) // 2
                    x1 = max(0, center_x - target_w // 2)
                    x2 = min(img.width, x1 + target_w)
                    # 如果右边界超出，调整左边界
                    if x2 == img.width:
                        x1 = x2 - target_w
                else:
                    # 高度太大，缩小高度
                    target_h = int(crop_w / self.crop_ratio)
                    # 保持中心位置
                    center_y = (y1 + y2) // 2
                    y1 = max(0, center_y - target_h // 2)
                    y2 = min(img.height, y1 + target_h)
                    # 如果底边界超出，调整顶边界
                    if y2 == img.height:
                        y1 = y2 - target_h

        # 裁剪图片
        cropped = img.crop((x1, y1, x2, y2))

        # 检查分辨率（短边是否低于768像素）
        short_edge = min(cropped.width, cropped.height)
        if short_edge < 768:
            # 分辨率过低，询问用户是否继续
            crop_ratio = cropped.width / cropped.height
            message = (
                f"⚠️ 裁剪后的图片分辨率较低：\n\n"
                f"📐 尺寸: {cropped.width} x {cropped.height} 像素\n"
                f"📏 短边: {short_edge} 像素 (建议 ≥ 768)\n"
                f"📊 比例: {crop_ratio:.2f}:1\n\n"
                f"分辨率过低可能影响训练效果。"
            )

            # 使用自定义对话框
            dialog = CustomDialog(
                self.app,
                "分辨率过低警告",
                message,
                button1_text="✅ 继续应用",
                button2_text="⏭️ 下一张",
                button3_text="🔄 重新裁剪"
            )
            result = dialog.get_result()

            if result is True:
                # 用户选择继续使用
                pass  # 继续执行后面的代码
            elif result is False:
                # 用户选择跳到下一张
                messagebox.showinfo("提示", "已跳过当前图片")
                self.app.next_image()
                return
            else:
                # 用户选择取消（重新调整）
                # 不保存裁剪结果，保持裁剪框显示
                self.display_image(img, show_crop_rect=True)
                return

        # 应用裁剪
        self.app.app_state["cropped_image"] = cropped
        self.crop_box = None  # 清空裁剪框
        self.display_image(cropped)

        # 更新状态提示为成功状态
        self.crop_status_label.configure(text="✅ 裁剪已应用", text_color="#4caf50")

        # 显示裁剪信息
        crop_ratio = cropped.width / cropped.height
        messagebox.showinfo(
            "裁剪成功",
            f"✅ 裁剪尺寸: {cropped.width}x{cropped.height} 像素\n"
            f"✅ 短边: {short_edge} 像素\n"
            f"✅ 裁剪比例: {crop_ratio:.2f}:1"
        )

    def reset_crop(self):
        """重置裁剪"""
        self.app.app_state["cropped_image"] = None
        self.crop_box = None
        self.crop_mode_menu.set("4:3")
        self.crop_mode = "4:3"
        self.crop_ratio = 4/3

        # 清除状态提示
        self.crop_status_label.configure(text="", text_color="#ff9800")

        if self.app.app_state["current_image"]:
            self.display_image(self.app.app_state["current_image"])

    def display_image(self, img: Image.Image, show_crop_rect: bool = False):
        """显示图片"""
        if not img:
            return

        # 使用固定的显示区域大小（调整为480x480，留出20px边距）
        frame_width, frame_height = 480, 480

        # 计算缩放比例
        display_size = self._get_display_size(img.size, (frame_width, frame_height))

        # 创建副本用于显示
        display_img = img.copy()
        display_img.thumbnail(display_size, Image.Resampling.LANCZOS)

        # 记录实际显示尺寸
        self.image_display_size = display_img.size

        # 计算图片在frame中的偏移量（居中显示）
        offset_x = (frame_width - display_img.width) // 2
        offset_y = (frame_height - display_img.height) // 2
        self.image_offset = (offset_x, offset_y)

        # 如果有裁剪框，显示裁剪框
        if self.crop_box and self.crop_mode != "原始":
            display_img = display_img.convert("RGBA")
            overlay = Image.new("RGBA", display_img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)

            x, y, w, h = self.crop_box

            # 确保裁剪框坐标在有效范围内
            x = max(0, min(x, display_img.width))
            y = max(0, min(y, display_img.height))
            w = max(0, min(w, display_img.width - x))
            h = max(0, min(h, display_img.height - y))

            # 绘制半透明遮罩（裁剪框外的区域）
            # 上方（只在有高度时绘制）
            if y > 0:
                draw.rectangle([0, 0, display_img.width, y], fill=(0, 0, 0, 100))
            # 下方（只在有空间时绘制）
            if y + h < display_img.height:
                draw.rectangle([0, y + h, display_img.width, display_img.height], fill=(0, 0, 0, 100))
            # 左侧（只在有宽度时绘制）
            if x > 0:
                draw.rectangle([0, y, x, y + h], fill=(0, 0, 0, 100))
            # 右侧（只在有空间时绘制）
            if x + w < display_img.width:
                draw.rectangle([x + w, y, display_img.width, y + h], fill=(0, 0, 0, 100))

            # 绘制裁剪框边框（绿色）
            draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0, 255), width=2)

            # 绘制调整手柄（白色圆点，绿色边框）
            handle_r = 6  # 手柄半径

            # 四个角的手柄
            handles = [
                (x, y),  # 左上
                (x + w, y),  # 右上
                (x, y + h),  # 左下
                (x + w, y + h),  # 右下
                (x + w//2, y),  # 上中
                (x + w//2, y + h),  # 下中
                (x, y + h//2),  # 左中
                (x + w, y + h//2),  # 右中
            ]

            for hx, hy in handles:
                # 绘制白色圆圈，绿色边框
                draw.ellipse(
                    [hx - handle_r, hy - handle_r, hx + handle_r, hy + handle_r],
                    fill=(255, 255, 255, 255),
                    outline=(0, 255, 0, 255),
                    width=2
                )

            # 辅助线（加在裁剪框上，细直线）
            line_color = (255, 255, 0, 180)
            # 右边1/8处竖线
            x_1_8 = x + int(w * 7 / 8)
            draw.line([(x_1_8, y), (x_1_8, y + h)], fill=line_color, width=1)
            # 中间1/2处横线
            y_half = y + int(h / 2)
            draw.line([(x, y_half), (x + w, y_half)], fill=line_color, width=1)
            # 从上往下3/4处横线
            y_3_4 = y + int(h * 3 / 4)
            draw.line([(x, y_3_4), (x + w, y_3_4)], fill=line_color, width=1)

            display_img = Image.alpha_composite(display_img, overlay)

        # 更新显示
        photo = ctk.CTkImage(light_image=display_img, dark_image=display_img,
                            size=display_img.size)
        self.image_label.configure(image=photo, text="", width=display_img.width, height=display_img.height)
        self.image_label.image = photo  # 保持引用

        # 确保label居中
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

    def _get_display_size(self, img_size: Tuple[int, int],
                         frame_size: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """计算显示尺寸"""
        if frame_size is None:
            frame_size = (400, 400)

        img_ratio = img_size[0] / img_size[1]
        frame_ratio = frame_size[0] / frame_size[1]

        if img_ratio > frame_ratio:
            # 图片更宽
            width = frame_size[0]
            height = int(width / img_ratio)
        else:
            # 图片更高
            height = frame_size[1]
            width = int(height * img_ratio)

        return (width, height)


class LabelingPanel(ctk.CTkScrollableFrame):
    """中间标签选择面板：动态生成标签复选框"""

    def __init__(self, master, app):
        super().__init__(
            master, fg_color=COLORS["surface"], corner_radius=14,
            border_width=1, border_color=COLORS["border"],
            scrollbar_button_color="#CBD5E1",
            scrollbar_button_hover_color="#94A3B8"
        )
        self.app = app
        self.checkboxes = {}  # 存储所有复选框 {英文key: CTkCheckBox}

        self.setup_ui()

    def setup_ui(self):
        """根据 JSON 配置动态生成 UI"""
        # 标题
        title_label = ctk.CTkLabel(
            self, text="02  人机协同标注",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLORS["text"]
        )
        title_label.pack(anchor="w", padx=18, pady=(18, 0))
        ctk.CTkLabel(
            self, text="AI 初标后由人工复核，保证标签可靠",
            font=ctk.CTkFont(size=11), text_color=COLORS["muted"]
        ).pack(anchor="w", padx=18, pady=(3, 12))

        # 加载配置
        spatial_dict = self.app.config.get("spatial_perception", {})
        interface_dict = self.app.config.get("interface_perception", {})

        # 每个配置文件包含一个一级概念及其二级分类、三级标签。
        if spatial_dict:
            # 获取第一个（也是唯一一个）原则的数据
            for principle_name, principle_data in spatial_dict.items():
                self._build_principle_section(principle_data)
                break  # 只处理第一个

        if interface_dict:
            # 添加分隔线
            separator = ctk.CTkFrame(self, height=1, fg_color=COLORS["divider"])
            separator.pack(fill="x", padx=18, pady=14)

            # 获取第一个（也是唯一一个）原则的数据
            for principle_name, principle_data in interface_dict.items():
                self._build_principle_section(principle_data)
                break  # 只处理第一个

    def _build_principle_section(self, principle_dict: Dict):
        """构建一个原则区域"""
        # 原则标题
        principle_cn = principle_dict.get("principle", {}).get("cn", "")
        principle_frame = ctk.CTkFrame(
            self, fg_color=COLORS["surface_alt"], corner_radius=10,
            border_width=1, border_color=COLORS["border"]
        )
        principle_frame.pack(fill="x", padx=12, pady=8)

        principle_label = ctk.CTkLabel(
            principle_frame,
            text=principle_cn,
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=COLORS["primary"]
        )
        principle_label.pack(anchor="w", padx=15, pady=(15, 10))

        # 遍历指标
        indicators = principle_dict.get("indicators", {})
        for indicator_key, indicator_data in indicators.items():
            self._build_indicator_section(principle_frame, indicator_key, indicator_data)

    def _build_indicator_section(self, parent, indicator_key: str, indicator_data: Dict):
        """构建指标区域"""
        indicator_cn = indicator_data.get("name", {}).get("cn", "")

        # 指标标题
        indicator_frame = ctk.CTkFrame(parent, fg_color="transparent")
        indicator_frame.pack(fill="x", padx=10, pady=(10, 5))

        indicator_label = ctk.CTkLabel(
            indicator_frame,
            text=f"▸ {indicator_cn}",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["primary"]
        )
        indicator_label.pack(anchor="w", padx=10)

        # 标签复选框
        tags = indicator_data.get("tags", [])
        for tag in tags:
            self._create_tag_checkbox(parent, tag)

    def _create_tag_checkbox(self, parent, tag_data: Dict):
        """创建标签复选框"""
        tag_en = tag_data.get("tag_en", "")
        tag_cn = tag_data.get("tag_cn", "")
        description = tag_data.get("description", "")
        # 复选框，点击文字和按钮都能切换（CTkCheckBox默认支持）
        checkbox = ctk.CTkCheckBox(
            parent,
            text=tag_cn,
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text"],
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            border_color="#98A2B3",
            command=lambda: self.app.on_tag_changed()
        )
        checkbox.pack(anchor="w", padx=30, pady=3)
        # 添加工具提示
        if description:
            ToolTip(checkbox, description)
        # 存储复选框
        self.checkboxes[tag_en] = checkbox

    def get_selected_tags(self) -> List[str]:
        """获取所有选中的标签（英文key）"""
        selected = []
        for tag_en, checkbox in self.checkboxes.items():
            if checkbox.get():
                selected.append(tag_en)
        return selected

    def clear_selection(self):
        """清空所有选择"""
        for checkbox in self.checkboxes.values():
            checkbox.deselect()


class OutputPanel(ctk.CTkFrame):
    """右侧输出面板：API配置、触发词、AI标签、最终标签编辑和保存"""

    def __init__(self, master, app):
        super().__init__(
            master, fg_color=COLORS["surface"], corner_radius=14,
            border_width=1, border_color=COLORS["border"]
        )
        self.app = app
        self.api_config = load_api_config()
        self.prompt_config = load_prompt_config()
        self.setup_ui()

    def setup_ui(self):
        """设置 UI 组件"""
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(18, 10))
        ctk.CTkLabel(
            header, text="03  AI 辅助与导出",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLORS["text"]
        ).pack(anchor="w")
        ctk.CTkLabel(
            header, text="生成建议、校对标签并保存训练对",
            font=ctk.CTkFont(size=11), text_color=COLORS["muted"]
        ).pack(anchor="w", pady=(3, 0))

        # ===== 1. 触发词区域（置顶、固定、醒目） =====
        trigger_container = ctk.CTkFrame(
            self,
            fg_color=COLORS["primary_soft"],
            corner_radius=10,
            border_width=1,
            border_color="#BFDBFE"
        )
        trigger_container.pack(fill="x", padx=16, pady=(0, 10))

        # 内层padding frame
        trigger_inner = ctk.CTkFrame(trigger_container, fg_color="transparent")
        trigger_inner.pack(fill="x", padx=10, pady=8)

        # 标题行
        trigger_header = ctk.CTkFrame(trigger_inner, fg_color="transparent")
        trigger_header.pack(fill="x", pady=(0, 6))

        ctk.CTkLabel(
            trigger_header,
            text="触发词  Trigger Word",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["primary"]
        ).pack(side="left")

        # 固定标识
        ctk.CTkLabel(
            trigger_header,
            text="固定",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=COLORS["primary"],
            fg_color="#DBEAFE",
            corner_radius=5,
            padx=6,
            pady=1
        ).pack(side="right")

        # 输入框
        self.trigger_entry = ctk.CTkEntry(
            trigger_inner,
            placeholder_text="输入触发词",
            height=32,
            font=ctk.CTkFont(size=12),
            fg_color="#ffffff",
            text_color=COLORS["text"],
            border_width=1,
            border_color="#BFDBFE"
        )
        self.trigger_entry.pack(fill="x")
        self.trigger_entry.insert(0, "street_vitality")

        # 分隔线
        ctk.CTkFrame(self, height=1, fg_color=COLORS["divider"]).pack(fill="x", padx=16, pady=(0, 10))

        # ===== 2. API配置区域 =====
        api_frame = ctk.CTkFrame(self, fg_color="transparent")
        api_frame.pack(fill="x", padx=16, pady=(0, 10))

        # API配置标题
        api_header = ctk.CTkFrame(api_frame, fg_color=COLORS["surface_alt"], corner_radius=7)
        api_header.pack(fill="x", pady=(0, 4))

        ctk.CTkLabel(
            api_header,
            text="🔧 API 配置",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(side="left", padx=8, pady=5)

        # API Key输入
        ctk.CTkLabel(api_frame, text="API Key:", anchor="w", font=ctk.CTkFont(size=10)).pack(fill="x", pady=(3, 1))
        self.api_key_entry = ctk.CTkEntry(
            api_frame,
            placeholder_text="输入OpenAI API Key",
            show="●",
            height=28
        )
        self.api_key_entry.pack(fill="x", pady=(0, 5))
        self.api_key_entry.insert(0, self.api_config.get("api_key", ""))

        # API URL输入
        ctk.CTkLabel(api_frame, text="API URL:", anchor="w", font=ctk.CTkFont(size=10)).pack(fill="x", pady=(3, 1))
        self.api_url_entry = ctk.CTkEntry(
            api_frame,
            placeholder_text="https://api.openai.com/v1/chat/completions",
            height=28
        )
        self.api_url_entry.pack(fill="x", pady=(0, 5))
        self.api_url_entry.insert(0, self.api_config.get("api_url", ""))

        # 模型名称
        ctk.CTkLabel(api_frame, text="模型:", anchor="w", font=ctk.CTkFont(size=10)).pack(fill="x", pady=(3, 1))
        self.model_entry = ctk.CTkEntry(
            api_frame,
            placeholder_text="gpt-4o",
            height=28
        )
        self.model_entry.pack(fill="x", pady=(0, 5))
        self.model_entry.insert(0, self.api_config.get("model", "gpt-4o"))

        # Quality和Timeout设置（并排）
        settings_frame = ctk.CTkFrame(api_frame, fg_color="transparent")
        settings_frame.pack(fill="x", pady=(0, 5))

        # Quality选择
        quality_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        quality_frame.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkLabel(quality_frame, text="质量:", anchor="w", font=ctk.CTkFont(size=10)).pack(fill="x")
        self.quality_menu = ctk.CTkOptionMenu(
            quality_frame,
            values=["auto", "high", "low"],
            width=80,
            height=26
        )
        self.quality_menu.pack(fill="x")
        self.quality_menu.set(self.api_config.get("quality", "auto"))

        # Timeout设置
        timeout_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        timeout_frame.pack(side="right", fill="x", expand=True, padx=(4, 0))
        ctk.CTkLabel(timeout_frame, text="超时(秒):", anchor="w", font=ctk.CTkFont(size=10)).pack(fill="x")
        self.timeout_entry = ctk.CTkEntry(timeout_frame, width=80, height=26)
        self.timeout_entry.pack(fill="x")
        self.timeout_entry.insert(0, str(self.api_config.get("timeout", 60)))

        # 保存配置按钮
        ctk.CTkButton(
            api_frame,
            text="💾 保存配置",
            command=self.save_api_config_action,
            height=28,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            font=ctk.CTkFont(size=11)
        ).pack(fill="x")

        # 分隔线
        ctk.CTkFrame(self, height=1, fg_color=COLORS["divider"]).pack(fill="x", padx=16, pady=(0, 10))

        # ===== 3. AI标签生成区域 =====
        ai_frame = ctk.CTkFrame(
            self, fg_color=COLORS["ai_soft"], corner_radius=10,
            border_width=1, border_color="#A7F3D0"
        )
        ai_frame.pack(fill="x", padx=16, pady=(0, 10))

        # AI标签标题
        ai_header = ctk.CTkFrame(ai_frame, fg_color="transparent")
        ai_header.pack(fill="x", padx=10, pady=(10, 8))

        ctk.CTkLabel(
            ai_header,
            text="AI 自动标签",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["ai"]
        ).pack(side="left")

        # 生成按钮（醒目的绿色）
        self.generate_btn = ctk.CTkButton(
            ai_header,
            text="✨ 生成",
            command=self.generate_ai_tags,
            width=100,
            height=30,
            fg_color=COLORS["ai"],
            hover_color=COLORS["ai_hover"],
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self.generate_btn.pack(side="right")

        # AI标签显示区域
        self.ai_tags_text = ctk.CTkTextbox(
            ai_frame,
            height=60,
            font=ctk.CTkFont(size=10),
            fg_color=COLORS["surface"],
            text_color=COLORS["text"],
            border_color="#A7F3D0",
            border_width=1
        )
        self.ai_tags_text.pack(fill="x", padx=10, pady=(0, 10))

        # ===== 4. 最终标签区域 =====
        final_frame = ctk.CTkFrame(self, fg_color="transparent")
        final_frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        ctk.CTkLabel(
            final_frame,
            text="📝 最终标签（可编辑）",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", pady=(0, 4))

        self.final_tags_text = ctk.CTkTextbox(
            final_frame,
            height=100,
            font=ctk.CTkFont(size=10),
            wrap="word",
            fg_color=COLORS["surface_alt"],
            text_color=COLORS["text"],
            border_width=1,
            border_color=COLORS["border"]
        )
        self.final_tags_text.pack(fill="both", expand=True)
        # 绑定失去焦点和回车事件，实现内容同步到人工标签
        self.final_tags_text.bind("<FocusOut>", lambda e: self.sync_tags_from_final())
        self.final_tags_text.bind("<Return>", lambda e: self.sync_tags_from_final())

        # ===== 5. 保存控制区域 =====
        save_frame = ctk.CTkFrame(self, fg_color="transparent")
        save_frame.pack(fill="x", padx=16, pady=(0, 16))

        # 输出文件夹选择
        self.select_output_btn = ctk.CTkButton(
            save_frame,
            text="📁 选择输出文件夹",
            command=self.select_output_folder,
            height=30,
            font=ctk.CTkFont(size=11)
        )
        self.select_output_btn.pack(fill="x", pady=(0, 4))

        self.output_path_label = ctk.CTkLabel(
            save_frame,
            text="未选择输出文件夹",
            font=ctk.CTkFont(size=9),
            text_color=COLORS["muted"]
        )
        self.output_path_label.pack(pady=(2, 6))

        # 保存按钮组
        button_frame = ctk.CTkFrame(save_frame, fg_color="transparent")
        button_frame.pack(fill="x")

        self.save_only_btn = ctk.CTkButton(
            button_frame,
            text="💾 仅保存",
            command=self.save_only,
            height=36,
            width=100,
            font=ctk.CTkFont(size=11),
            fg_color="#E7ECF3", hover_color="#D7DEE9",
            text_color=COLORS["text"]
        )
        self.save_only_btn.pack(side="left", padx=(0, 6))

        self.save_next_btn = ctk.CTkButton(
            button_frame,
            text="💾 保存 & 下一张 →",
            command=self.save_and_next,
            height=36,
            fg_color=COLORS["ai"],
            hover_color=COLORS["ai_hover"],
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self.save_next_btn.pack(side="left", fill="x", expand=True)

    def save_api_config_action(self):
        """保存API配置"""
        try:
            timeout = int(self.timeout_entry.get())
        except ValueError:
            timeout = 60

        save_api_config(
            self.api_key_entry.get(),
            self.api_url_entry.get(),
            self.model_entry.get(),
            self.quality_menu.get(),
            timeout
        )
        messagebox.showinfo("成功", "API配置已保存")

    def generate_ai_tags(self):
        """生成AI标签"""
        if not self.prompt_config.get("enabled", True):
            messagebox.showwarning("智能体未启用", "请在 prompt_config.json 中将 enabled 设置为 true")
            return
        # 检查是否有未应用的裁剪框
        if self.app.image_panel.crop_box is not None:
            result = messagebox.askyesno(
                "提示",
                "⚠️ 检测到未应用的裁剪框\n\n建议先点击'应用裁剪'按钮\n\n是否继续生成AI标签？",
                icon='warning'
            )
            if not result:
                return

        # 获取当前图片
        img = self.app.app_state.get("cropped_image") or self.app.app_state.get("current_image")
        if not img:
            messagebox.showwarning("警告", "请先加载图片")
            return

        # 获取API配置
        api_key = self.api_key_entry.get().strip()
        api_url = self.api_url_entry.get().strip()
        model = self.model_entry.get().strip() or "gpt-4o"

        if not api_key:
            messagebox.showwarning("警告", "请先输入API Key")
            return
        if not api_url:
            messagebox.showwarning("警告", "请填写 API URL")
            return

        # 生成前同步保存当前配置，避免下次启动丢失。
        try:
            timeout = max(10, int(self.timeout_entry.get()))
        except ValueError:
            timeout = 60
        save_api_config(api_key, api_url, model, self.quality_menu.get(), timeout)

        # 禁用按钮并显示加载状态
        self.generate_btn.configure(state="disabled", text="⏳ 生成中")
        self.ai_tags_text.delete("1.0", "end")
        self.ai_tags_text.insert("1.0", "正在调用GPT-4o API，请稍候...")

        # 异步调用API（避免UI冻结）
        def api_call_thread():
            prompt = self.prompt_config.get("default_prompt", "")
            system_prompt = self.prompt_config.get("system_prompt", "You are a precise architectural image annotation agent.")
            quality = self.quality_menu.get()

            caption, error = call_gpt4v_api(
                img, system_prompt, prompt, api_key, api_url, model, quality, timeout
            )

            # 更新UI（需要在主线程）
            self.after(0, lambda: self.update_ai_tags_result(caption, error))

        threading.Thread(target=api_call_thread, daemon=True).start()

    def update_ai_tags_result(self, caption, error):
        """更新AI标签结果"""
        self.generate_btn.configure(state="normal", text="✨ 生成")

        if error:
            self.ai_tags_text.delete("1.0", "end")
            self.ai_tags_text.insert("1.0", f"❌ 错误: {error}")
            messagebox.showerror("AI标签生成失败", error)
        else:
            # 清洗并执行受控标签白名单，防止模型产生体系外标签。
            raw_tags = [
                tag.strip().lower().replace(" ", "_")
                for tag in (caption or "").replace("\n", ",").split(",")
                if tag.strip()
            ]
            valid_tags = []
            for tag in raw_tags:
                tag = tag.strip("'\"*` .;:-")
                if tag in ALLOWED_AI_TAGS and tag not in valid_tags:
                    valid_tags.append(tag)

            # 通透程度、开口密度各自只能保留一个结果。
            exclusive_groups = [
                {"low_transparency", "medium_transparency", "high_transparency"},
                {"limited_entrance", "moderate_entrance", "multiple_entrance"}
            ]
            for group in exclusive_groups:
                matches = [tag for tag in valid_tags if tag in group]
                for duplicate in matches[1:]:
                    valid_tags.remove(duplicate)

            boundary_tags = {"facade_setback", "overhang_canopy", "street_spillout", "spatial_folding"}
            if "flat_facade" in valid_tags and any(tag in valid_tags for tag in boundary_tags):
                valid_tags.remove("flat_facade")

            caption = ", ".join(tag for tag in ALLOWED_AI_TAGS if tag in valid_tags)

            self.ai_tags_text.delete("1.0", "end")
            self.ai_tags_text.insert("1.0", caption)
            # 自动更新最终标签
            self.update_final_tags()

    def update_final_tags(self):
        """更新最终标签（合并触发词、人工标签、AI标签，按指定顺序）"""
        trigger_word = self.trigger_entry.get().strip()
        human_tags = self.app.labeling_panel.get_selected_tags()
        ai_tags_raw = self.ai_tags_text.get("1.0", "end").strip()
        ai_tags = ai_tags_raw if not ai_tags_raw.startswith("❌") and not ai_tags_raw.startswith("正在") else ""
        # 指定顺序
        order = [
            trigger_word,
            # 空间感知
            "flat_facade", "facade_setback", "overhang_canopy", "street_spillout", "spatial_folding",
            # 通透程度
            "low_transparency", "medium_transparency", "high_transparency",
            # 开口密度
            "limited_entrance", "moderate_entrance", "multiple_entrance"
        ]
        # 解析AI标签
        ai_tag_list = [t.strip() for t in ai_tags.split(',') if t.strip()] if ai_tags else []
        # 合并人工和AI标签，去重
        tag_set = set(human_tags)
        tag_set.update(ai_tag_list)
        # 按顺序排列
        final_tags = []
        if trigger_word:
            final_tags.append(trigger_word)
        for tag in order[1:]:
            if tag in tag_set:
                final_tags.append(tag)
                tag_set.remove(tag)
        # 剩余其它标签
        for tag in sorted(tag_set):
            if tag and tag != trigger_word:
                final_tags.append(tag)
        final_tags_str = ", ".join(final_tags)
        self.final_tags_text.delete("1.0", "end")
        self.final_tags_text.insert("1.0", final_tags_str)

    def sync_tags_from_final(self):
        """将最终标签文本框内容同步到人工标签勾选"""
        content = self.final_tags_text.get("1.0", "end").strip()
        tag_list = [t.strip() for t in content.split(',') if t.strip()]
        # 先清空所有勾选
        self.app.labeling_panel.clear_selection()
        # 逐个勾选
        for tag in tag_list:
            if tag in self.app.labeling_panel.checkboxes:
                self.app.labeling_panel.checkboxes[tag].select()
        # 触发最终标签更新，防止重复
        self.update_final_tags()

    def select_output_folder(self):
        """选择输出文件夹"""
        folder = filedialog.askdirectory(title="选择输出文件夹")
        if folder:
            self.app.app_state["output_folder"] = folder
            # 显示简短路径
            short_path = folder if len(folder) <= 40 else "..." + folder[-37:]
            self.output_path_label.configure(text=short_path)

            # 提示用户统计报告的保存位置
            messagebox.showinfo(
                "输出文件夹已设置",
                f"✅ 输出路径: {folder}\n\n"
                f"📁 图片和标签将保存到:\n"
                f"   - spatial_perception_only/\n"
                f"   - interface_perception_only/\n"
                f"   - spatial_interface_both/\n"
                f"   - other/\n"
                f"   - no_tags/\n\n"
                f"📊 统计报告将保存到:\n"
                f"   - statistics/labeling_report.txt\n"
                f"   - statistics/labeling_statistics.json"
            )

    def save_only(self):
        """仅保存当前标签"""
        self.app.save_current_tags()

    def save_and_next(self):
        """保存并切换到下一张"""
        if self.app.save_current_tags():
            self.app.next_image()


class App(ctk.CTk):
    """主应用程序"""

    def __init__(self):
        super().__init__()

        # 应用状态
        self.app_state = {
            "image_folder": None,
            "image_files": [],
            "current_index": -1,
            "output_folder": None,
            "current_image": None,
            "cropped_image": None,
        }

        # 配置字典
        self.config = {}

        # 统计数据
        self.statistics = {
            "total_processed": 0,
            "tag_counts": {},
            "category_counts": {
                "spatial_perception_only": 0,
                "interface_perception_only": 0,
                "spatial_interface_both": 0,
                "other": 0,
                "no_tags": 0
            },
            "session_start_time": None,
            "last_update_time": None
        }

        # 窗口配置
        self.title("Street Vitality · AI 数据标注工作台")
        self.geometry("1500x900")
        self.minsize(1180, 720)
        self.configure(fg_color=COLORS["app_bg"])

        # 加载配置
        self.load_config()

        # 初始化统计
        self.init_statistics()

        # 设置 UI
        self.setup_ui()

        # 绑定快捷键
        self.bind("<Left>", lambda e: self.prev_image())
        self.bind("<Right>", lambda e: self.next_image())
        self.bind("<Control-s>", lambda e: self.save_current_tags())
        self.bind("<Control-Return>", lambda e: self.output_panel.save_and_next())

        # 关闭窗口事件
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_config(self):
        """加载 JSON 配置文件"""
        try:
            # 获取脚本所在目录
            script_dir = Path(__file__).parent

            # 加载空间感知标签配置
            spatial_path = script_dir / "spatial_perception_dict.json"
            if spatial_path.exists():
                with open(spatial_path, "r", encoding="utf-8") as f:
                    self.config["spatial_perception"] = json.load(f)
            else:
                messagebox.showerror("错误", f"未找到配置文件: {spatial_path}")

            # 加载界面感知标签配置
            interface_path = script_dir / "interface_perception_dict.json"
            if interface_path.exists():
                with open(interface_path, "r", encoding="utf-8") as f:
                    self.config["interface_perception"] = json.load(f)
            else:
                messagebox.showerror("错误", f"未找到配置文件: {interface_path}")

            if not self.config:
                messagebox.showerror("错误", "无法加载任何配置文件，应用将退出")
                self.quit()

        except Exception as e:
            messagebox.showerror("错误", f"加载配置文件时出错: {str(e)}")
            self.quit()

    def init_statistics(self):
        """初始化统计数据"""
        # 从配置中提取所有标签并初始化计数
        for concept_config in self.config.values():
            for principle_name, principle_data in concept_config.items():
                indicators = principle_data.get("indicators", {})
                for indicator_data in indicators.values():
                    for tag in indicator_data.get("tags", []):
                        self.statistics["tag_counts"][tag["tag_en"]] = 0

        # 设置会话开始时间（每次运行程序都重新开始统计）
        self.statistics["session_start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.statistics["last_update_time"] = self.statistics["session_start_time"]

    def save_statistics(self):
        """保存统计数据到文件"""
        # 检查是否已选择输出文件夹
        if not self.app_state.get("output_folder"):
            return

        # 创建统计文件夹
        stats_folder = os.path.join(self.app_state["output_folder"], "statistics")
        try:
            os.makedirs(stats_folder, exist_ok=True)
        except Exception as e:
            print(f"创建统计文件夹失败: {e}")
            return

        # 保存统计数据
        stats_file = os.path.join(stats_folder, "labeling_statistics.json")
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.statistics, f, indent=2, ensure_ascii=False)

            # 同时生成可读的报告
            self.generate_statistics_report()
        except Exception as e:
            print(f"保存统计数据失败: {e}")

    def generate_statistics_report(self):
        """生成可读的统计报告"""
        # 检查是否已选择输出文件夹
        if not self.app_state.get("output_folder"):
            return

        # 统计文件夹路径
        stats_folder = os.path.join(self.app_state["output_folder"], "statistics")
        report_file = os.path.join(stats_folder, "labeling_report.txt")

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("图像标注工具 - 运行统计报告\n")
                f.write("=" * 70 + "\n\n")

                # 基本统计
                f.write(f"会话开始时间: {self.statistics['session_start_time']}\n")
                f.write(f"最后更新时间: {self.statistics['last_update_time']}\n")
                f.write(f"已处理图片总数: {self.statistics['total_processed']}\n\n")

                # 分类统计
                f.write("-" * 70 + "\n")
                f.write("分类统计\n")
                f.write("-" * 70 + "\n")
                for category in ["spatial_perception_only", "interface_perception_only", "spatial_interface_both", "other", "no_tags"]:
                    count = self.statistics["category_counts"][category]
                    percentage = (count / self.statistics['total_processed'] * 100) if self.statistics['total_processed'] > 0 else 0
                    f.write(f"  {category:20s}: {count:4d} 张 ({percentage:5.1f}%)\n")
                f.write("\n")

                # 空间感知标签统计
                f.write("-" * 70 + "\n")
                f.write("空间感知（人可停留）标签使用统计\n")
                f.write("-" * 70 + "\n")
                spatial_tags = []
                if "spatial_perception" in self.config:
                    for principle_name, principle_data in self.config["spatial_perception"].items():
                        indicators = principle_data.get("indicators", {})
                        for indicator_key, indicator_data in indicators.items():
                            indicator_name = indicator_data.get("name", {}).get("cn", indicator_key)
                            f.write(f"\n【{indicator_name}】\n")
                            for tag in indicator_data.get("tags", []):
                                tag_en = tag["tag_en"]
                                tag_cn = tag["tag_cn"]
                                count = self.statistics["tag_counts"].get(tag_en, 0)

                                percentage = (count / self.statistics['total_processed'] * 100) if self.statistics['total_processed'] > 0 else 0

                                f.write(f"  {tag_cn:25s} ({tag_en:35s}): {count:4d} 次 ({percentage:5.1f}%)\n")
                                spatial_tags.append((tag_en, count))

                spatial_total = sum(count for _, count in spatial_tags)
                f.write(f"\n空间感知标签总使用次数: {spatial_total}\n")

                # 界面感知标签统计
                f.write("\n" + "-" * 70 + "\n")
                f.write("界面感知（人可看见）标签使用统计\n")
                f.write("-" * 70 + "\n")
                interface_tags = []
                if "interface_perception" in self.config:
                    for principle_name, principle_data in self.config["interface_perception"].items():
                        indicators = principle_data.get("indicators", {})
                        for indicator_key, indicator_data in indicators.items():
                            indicator_name = indicator_data.get("name", {}).get("cn", indicator_key)
                            f.write(f"\n【{indicator_name}】\n")
                            for tag in indicator_data.get("tags", []):
                                tag_en = tag["tag_en"]
                                tag_cn = tag["tag_cn"]
                                count = self.statistics["tag_counts"].get(tag_en, 0)
                                percentage = (count / self.statistics['total_processed'] * 100) if self.statistics['total_processed'] > 0 else 0
                                f.write(f"  {tag_cn:25s} ({tag_en:35s}): {count:4d} 次 ({percentage:5.1f}%)\n")
                                interface_tags.append((tag_en, count))

                interface_total = sum(count for _, count in interface_tags)
                f.write(f"\n界面感知标签总使用次数: {interface_total}\n")

                # 底部
                f.write("\n" + "=" * 70 + "\n")
                f.write("报告生成完毕\n")
                f.write("=" * 70 + "\n")

        except Exception as e:
            print(f"生成统计报告失败: {e}")

    def update_statistics(self, selected_tags: List[str], category: str):
        """更新统计数据"""
        # 增加已处理图片数
        self.statistics["total_processed"] += 1

        # 更新标签计数
        for tag in selected_tags:
            if tag in self.statistics["tag_counts"]:
                self.statistics["tag_counts"][tag] += 1

        # 更新分类计数
        if category in self.statistics["category_counts"]:
            self.statistics["category_counts"][category] += 1

        # 更新最后更新时间
        self.statistics["last_update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 保存统计数据
        self.save_statistics()

    def setup_ui(self):
        """设置主界面"""
        # 创建主容器
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=16, pady=16)

        # 三列布局
        # 左侧：图片面板（40%）
        self.image_panel = ImagePanel(main_container, self)
        self.image_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        # 中间：标签选择面板（30%）
        self.labeling_panel = LabelingPanel(main_container, self)
        self.labeling_panel.grid(row=0, column=1, sticky="nsew", padx=6)

        # 右侧：输出面板（30%）
        self.output_panel = OutputPanel(main_container, self)
        self.output_panel.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

        # 配置列权重
        main_container.grid_columnconfigure(0, weight=5, minsize=540)
        main_container.grid_columnconfigure(1, weight=3, minsize=330)
        main_container.grid_columnconfigure(2, weight=3, minsize=360)
        main_container.grid_rowconfigure(0, weight=1)

    def load_image_folder(self, folder: str):
        """加载图片文件夹"""
        # 获取所有图片文件
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
        image_files = []

        for file in sorted(os.listdir(folder)):
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(folder, file))

        if not image_files:
            messagebox.showwarning("警告", "该文件夹中没有找到图片文件")
            return

        self.app_state["image_folder"] = folder
        self.app_state["image_files"] = image_files
        self.app_state["current_index"] = 0

        # 启用导航按钮
        self.image_panel.prev_btn.configure(state="normal")
        self.image_panel.next_btn.configure(state="normal")

        # 清空标签选择（新文件夹，重新开始）
        self.labeling_panel.clear_selection()

        # 显示第一张图片
        self.load_current_image()

        # 更新最终标签预览（只保留触发词）
        self.output_panel.update_final_tags()

    def load_current_image(self):
        """加载当前图片"""
        if not self.app_state["image_files"] or self.app_state["current_index"] < 0:
            return
        try:
            image_path = self.app_state["image_files"][self.app_state["current_index"]]
            img = Image.open(image_path)
            # 处理 RGBA 转 RGB
            if img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
            self.app_state["current_image"] = img
            self.app_state["cropped_image"] = None
            # 更新进度
            self.update_progress()
            # 重置裁剪状态
            self.image_panel.crop_box = None
            # 先显示图片（这会更新 image_display_size）
            self.image_panel.display_image(img)
            # 然后初始化裁剪框（如果不是原始模式）
            if self.image_panel.crop_mode != "原始":
                self.image_panel._init_crop_box()
                # 重新显示图片以显示裁剪框
                self.image_panel.display_image(img, show_crop_rect=True)
            # ===== 新增：自动读取同名txt标签并自动勾选人工标签，不写入AI标签区域 =====
            txt_path = os.path.splitext(image_path)[0] + '.txt'
            txt_content = ""
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        txt_content = f.read().strip()
                except Exception as e:
                    txt_content = f"读取标签文件出错: {e}"
            # 不再写入AI标签区域
            # 自动勾选人工标签
            if txt_content:
                tag_list = [t.strip() for t in txt_content.split(',') if t.strip()]
                self.labeling_panel.clear_selection()
                for tag in tag_list:
                    if tag in self.labeling_panel.checkboxes:
                        self.labeling_panel.checkboxes[tag].select()
            else:
                self.labeling_panel.clear_selection()
            self.output_panel.update_final_tags()
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {str(e)}")

    def update_progress(self):
        """更新进度显示"""
        if not self.app_state["image_files"]:
            return

        current = self.app_state["current_index"] + 1
        total = len(self.app_state["image_files"])
        filename = os.path.basename(self.app_state["image_files"][self.app_state["current_index"]])

        progress_text = f"图片 {current}/{total}: {filename}"
        self.image_panel.progress_label.configure(text=progress_text)

    def next_image(self):
        """下一张图片"""
        if not self.app_state["image_files"]:
            return
        if self.app_state["current_index"] < len(self.app_state["image_files"]) - 1:
            self.app_state["current_index"] += 1
            self.load_current_image()
            # 不再清空人工标签和AI标签，由load_current_image自动处理
        else:
            messagebox.showinfo("提示", "已经是最后一张图片了")

    def prev_image(self):
        """上一张图片"""
        if not self.app_state["image_files"]:
            return
        if self.app_state["current_index"] > 0:
            self.app_state["current_index"] -= 1
            self.load_current_image()
            # 不再清空人工标签和AI标签，由load_current_image自动处理
        else:
            messagebox.showinfo("提示", "已经是第一张图片了")

    def on_tag_changed(self):
        """标签选择改变时"""
        self.output_panel.update_final_tags()

    def _detect_tag_category(self) -> str:
        """检测标签类型，返回子文件夹名称"""
        selected_tags = self.labeling_panel.get_selected_tags()

        if not selected_tags:
            return "no_tags"  # 无标签

        # 检查标签所属的原则
        has_spatial = False
        has_interface = False

        # 从配置中获取空间感知和界面感知的全部标签
        spatial_tags = set()
        interface_tags = set()

        # 正确的配置访问方式：先获取原则层级，再获取indicators
        if "spatial_perception" in self.config:
            for principle_name, principle_data in self.config["spatial_perception"].items():
                indicators = principle_data.get("indicators", {})
                for indicator_data in indicators.values():
                    for tag in indicator_data.get("tags", []):
                        spatial_tags.add(tag["tag_en"])

        if "interface_perception" in self.config:
            for principle_name, principle_data in self.config["interface_perception"].items():
                indicators = principle_data.get("indicators", {})
                for indicator_data in indicators.values():
                    for tag in indicator_data.get("tags", []):
                        interface_tags.add(tag["tag_en"])

        # 检查选中的标签
        for tag in selected_tags:
            if tag in spatial_tags:
                has_spatial = True
            if tag in interface_tags:
                has_interface = True

        # 根据标签类型返回文件夹名称
        if has_spatial and has_interface:
            return "spatial_interface_both"
        elif has_spatial:
            return "spatial_perception_only"
        elif has_interface:
            return "interface_perception_only"
        else:
            return "other"

    def save_current_tags(self) -> bool:
        """保存当前标签（带自动分类）"""
        # 检查是否有未应用的裁剪框
        if self.image_panel.crop_box is not None:
            result = messagebox.askyesno(
                "提示",
                "⚠️ 检测到未应用的裁剪框\n\n建议先点击'应用裁剪'按钮\n\n是否继续保存？",
                icon='warning'
            )
            if not result:
                return False

        # 检查是否选择了输出文件夹
        if not self.app_state.get("output_folder"):
            messagebox.showwarning("警告", "请先选择输出文件夹")
            return False

        # 检查是否有当前图片
        if not self.app_state["image_files"] or self.app_state["current_index"] < 0:
            messagebox.showwarning("警告", "没有可保存的图片")
            return False

        # 获取最终标签
        final_tags = self.output_panel.final_tags_text.get("1.0", "end-1c").strip()

        if not final_tags:
            result = messagebox.askyesno("确认", "标签为空，确定要保存空标签吗？")
            if not result:
                return False

        # 获取当前图片文件名
        image_path = self.app_state["image_files"][self.app_state["current_index"]]
        image_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(image_name)[0]

        # 检测标签类型并确定子文件夹
        category = self._detect_tag_category()
        category_folder = os.path.join(self.app_state["output_folder"], category)

        # 获取选中的人工标签（用于显示调试信息）
        selected_human_tags = self.labeling_panel.get_selected_tags()

        # 创建子文件夹（如果不存在）
        try:
            os.makedirs(category_folder, exist_ok=True)
        except Exception as e:
            messagebox.showerror("错误", f"创建文件夹失败: {str(e)}")
            return False

        # 生成输出文件路径
        output_filename = f"{name_without_ext}.txt"
        output_path = os.path.join(category_folder, output_filename)

        # 检查文件是否已存在
        if os.path.exists(output_path):
            result = messagebox.askyesno("确认", f"文件 {output_filename} 已存在于 {category} 文件夹中，是否覆盖？")
            if not result:
                return False

        # 保存标签和图片
        try:
            # 保存txt标签
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_tags)

            # 保存裁剪后的图片（或原图）
            img_to_save = self.app_state.get("cropped_image") or self.app_state.get("current_image")
            if img_to_save:
                image_output_filename = f"{name_without_ext}.png"
                image_output_path = os.path.join(category_folder, image_output_filename)

                # 确保图片是RGB模式（PNG需要）
                if img_to_save.mode == "RGBA":
                    # 保持RGBA
                    img_to_save.save(image_output_path, "PNG")
                elif img_to_save.mode != "RGB":
                    img_to_save.convert("RGB").save(image_output_path, "PNG")
                else:
                    img_to_save.save(image_output_path, "PNG")

                # 构建详细的保存信息
                tag_info = f"人工标签: {', '.join(selected_human_tags)}" if selected_human_tags else "人工标签: 无"
                messagebox.showinfo(
                    "保存成功",
                    f"✅ 文件: {output_filename}\n"
                    f"✅ 图片: {image_output_filename}\n"
                    f"✅ {tag_info}\n"
                    f"✅ 分类: {category}\n"
                    f"📁 路径: {category_folder}"
                )
            else:
                tag_info = f"人工标签: {', '.join(selected_human_tags)}" if selected_human_tags else "人工标签: 无"
                messagebox.showinfo(
                    "保存成功",
                    f"✅ 文件: {output_filename}\n"
                    f"✅ {tag_info}\n"
                    f"✅ 分类: {category}\n"
                    f"📁 路径: {category_folder}"
                )

            # 更新统计数据
            self.update_statistics(selected_human_tags, category)

            return True

        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
            return False

    def on_closing(self):
        """关闭应用"""
        if messagebox.askokcancel("退出", "确定要退出吗？"):
            # 保存最终统计数据
            self.save_statistics()
            self.quit()


def main():
    """主函数"""
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
