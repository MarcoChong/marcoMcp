# File: roblox_agent.py
import os
import winreg
import time
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from mss import mss
from PIL import Image
import pyautogui
from pynput import mouse, keyboard
import configparser
import logging

# 配置参数
config = configparser.ConfigParser()
config.read('config.ini')
CONFIG = {
    "screen_region": tuple(map(int, config['DEFAULT']['screen_region'].split(','))),  # 截屏区域
    "model_path": config['DEFAULT']['model_path'],
    "action_delay": tuple(map(float, config['DEFAULT']['action_delay'].split(','))),  # 操作随机延迟
    "action_types": {                    # 动作类型映射
        0: "click",
        1: "press_space",
        2: "press_w",
        3: "press_s",
        4: "press_a",
        5: "press_d",
        6: "press_ctrl"
    }
}

logging.basicConfig(filename='roblox_agent.log', level=logging.INFO)

class VisionSystem:
    def __init__(self):
        self.sct = mss()
        self.monitor = {
            "top": CONFIG["screen_region"][1],
            "left": CONFIG["screen_region"][0],
            "width": CONFIG["screen_region"][2] - CONFIG["screen_region"][0],
            "height": CONFIG["screen_region"][3] - CONFIG["screen_region"][1]
        }

    def capture(self):
        """捕获当前屏幕图像"""
        img = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

class ActionEngine:
    @staticmethod
    def human_click(x, y):
        """拟人化点击操作"""
        # 坐标安全限制
        screen_w, screen_h = 3072, 1920  # 调整为新分辨率
        x = max(0, min(x, screen_w))
        y = max(0, min(y, screen_h))
        
        pyautogui.moveTo(x, y, duration=random.uniform(0.1, 0.3))
        pyautogui.click(duration=random.uniform(0.05, 0.1))
        time.sleep(random.uniform(*CONFIG["action_delay"]))

    @staticmethod
    def press_key(key, duration=0.1):
        """按键操作"""
        pyautogui.keyDown(key)
        time.sleep(duration + random.uniform(-0.05, 0.05))
        pyautogui.keyUp(key)
        time.sleep(random.uniform(*CONFIG["action_delay"]))

class ImitationModel(nn.Module):
    def __init__(self, input_size=(3, 1920, 3072)):  # 调整为新分辨率
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_size[0], 16, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.fc = nn.Linear(32 * (input_size[1] // 4 - 1) * (input_size[2] // 4 - 1), len(CONFIG["action_types"]))

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)

class RobloxAgent:
    def __init__(self):
        self.vision = VisionSystem()
        self.model = self.load_model()
        self.listener = None

    def load_model(self):
        """加载预训练模型"""
        model = ImitationModel(input_size=(3, CONFIG["screen_region"][3] - CONFIG["screen_region"][1], CONFIG["screen_region"][2] - CONFIG["screen_region"][0]))
        if os.path.exists(CONFIG["model_path"]):
            try:
                model.load_state_dict(torch.load(CONFIG["model_path"]))
            except Exception as e:
                logging.error(f"加载模型失败: {str(e)}")
        return model.eval()

    def start_roblox(self):
        """启动Roblox客户端"""
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, config['ROBLOX']['install_path'])
            path = winreg.QueryValueEx(key, "InstallPath")[0]
            os.startfile(os.path.join(path, "RobloxPlayerLauncher.exe"))
            time.sleep(5)  # 等待客户端启动
        except Exception as e:
            logging.error(f"启动失败: {str(e)}")

    def record_demo(self):
        """录制演示操作"""
        recordings = []
        stop_recording = False

        def on_click(x, y, button, pressed):
            nonlocal stop_recording
            if pressed:
                frame = self.vision.capture()
                recordings.append((frame, ("click", x, y)))
            if not pressed and button == mouse.Button.right:
                stop_recording = True

        def on_press(key):
            try:
                frame = self.vision.capture()
                recordings.append((frame, ("press", key.char)))
            except AttributeError:
                pass

        # 同时监听鼠标和键盘
        with mouse.Listener(on_click=on_click) as m_listener, \
             keyboard.Listener(on_press=on_press) as k_listener:
            while not stop_recording:
                time.sleep(0.1)
        
        return recordings

    def train(self, dataset):
        """训练模仿学习模型"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0] * len(CONFIG["action_types"])))

        # 转换演示数据为训练格式
        frames, labels = [], []
        for frame, action in dataset:
            frames.append(frame)
            # 将动作转换为数字标签
            if action[0] == "click":
                labels.append(0)
            else:
                key_mapping = {'space': 1, 'w': 2, 'a': 3, 'd': 4, 's': 5, 'ctrl': 6}
                labels.append(key_mapping.get(action[1], -1))
        
        # 过滤无效标签
        valid_indices = [i for i, lbl in enumerate(labels) if lbl != -1]
        frames = [frames[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]

        # 转换为张量
        frames_tensor = torch.stack([
            torch.tensor(f).permute(2, 0, 1).float() / 255.0 
            for f in frames
        ])
        labels_tensor = torch.LongTensor(labels)

        # 训练循环
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = self.model(frames_tensor)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        torch.save(self.model.state_dict(), CONFIG["model_path"])

    def execute_action(self, action_type):
        """执行预测动作"""
        action_name = CONFIG["action_types"].get(action_type, "invalid")
        
        if action_name == "click":
            # 点击屏幕中心区域（可替换为实际坐标预测）
            x = CONFIG["screen_region"][0] + (CONFIG["screen_region"][2] - CONFIG["screen_region"][0]) // 2
            y = CONFIG["screen_region"][1] + (CONFIG["screen_region"][3] - CONFIG["screen_region"][1]) // 2
            ActionEngine.human_click(x, y)
        elif action_name == "press_space":
            ActionEngine.press_key('space')
        elif action_name == "press_w":
            ActionEngine.press_key('w', duration=0.3)
        elif action_name == "press_a":
            ActionEngine.press_key('a', duration=0.2)
        elif action_name == "press_d":
            ActionEngine.press_key('d', duration=0.2)
        elif action_name == "press_s":
            ActionEngine.press_key('s', duration=0.2)
        elif action_name == "press_ctrl":
            ActionEngine.press_key('ctrl', duration=0.2)
        else:
            logging.warning(f"无效的动作类型: {action_type}")

    def run(self):
        """运行代理主循环"""
        print("代理开始运行，按Ctrl+C停止")
        try:
            while True:
                frame = self.vision.capture()
                tensor_frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
                
                with torch.no_grad():
                    output = self.model(tensor_frame.unsqueeze(0))
                
                action_type = torch.argmax(output).item()
                self.execute_action(action_type)
                
        except KeyboardInterrupt:
            print("代理已停止")

if __name__ == "__main__":
    agent = RobloxAgent()
    agent.start_roblox()
    
    # 训练模式
    if not os.path.exists(CONFIG["model_path"]):
        input("按回车开始录制演示操作...右键结束录制...")
        dataset = agent.record_demo()
        agent.train(dataset)
    
    # 执行模式
    agent.run()