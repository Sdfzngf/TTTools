from PIL import Image as image
import numpy as np
import random
import string
import cv2
import math
from tqdm import tqdm
from reedsolo import RSCodec
from numba import njit
import time
import sys
import ctypes
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

#@njit
def logistic_map(x0, length):
    mu = 1  # 混沌参数（取4时处于完全混沌状态）
    x = np.zeros(length,dtype=np.float64)
    x[0] = x0
    print("正在生成混沌序列...")
    for i in range(1, length):
        x[i] = mu * x[i-1] * (1 - x[i-1])
        # 添加抗扰动处理
        if x[i] < 1e-9 or x[i] > 1 - 1e-9:
            x[i] = (x[i-1] + 0.5) % 1
        # 不使用tqdm进度条
        # 每隔1000个像素更新进度
        #if i % 1000 == 0:
        #    print(f"生成混沌序列进度：{i}/{length}\r")
    return x


if __name__ == "__main__":
    img_path = input("输入图片路径（如 in.jpg）：")
    img = image.open(img_path)
    img_array = np.array(img).astype(np.uint8)  # 转换为numpy数组（0-255像素值）
    height, width = img_array.shape[:2]  # 获取图像尺寸
    pixels_total = height * width  # 总像素数
    channels = img_array.shape[2] if len(img_array.shape) == 3 else 1  # 处理单通道图像（如灰度图）

    way = input("输入处理方式（1加密/2解密）：")

    # 将种子转换为混沌初始值x0（0 < x0 < 1）
    seed = input("输入密钥（格式：[文件名称]{原始数据长度}{原始宽度}{原始高度}，留空随机生成）：")

    if seed == "":
        # 随机生成种子+自动计算原始数据长度（加密场景）
        # 计算原始数据长度（仅加密时有效）
        img = image.open(img_path)
        img_array = np.array(img).astype(np.uint8)
        total_bytes = img_array.flatten().shape[0]
        seed_content = f"{img_path.split('.')[0]}_{total_bytes}.{img_path.split('.')[-1]}" #''.join(random.choices((string.ascii_letters + string.digits).replace("{",'%').replace('}','%'), k=64))
        seed = f"[{seed_content}]{{{total_bytes}}}{{{width}}}{{{height}}}"
        print(f"生成的密钥为 {seed} 请复制完整内容")
    elif seed.startswith("[") and seed.count("{") == 3 and seed.endswith("}"):  # 扩展格式需3个{和结尾}
        # 解析种子（解密场景，扩展格式：[文件名称]{原始数据长度}{原始宽度}{原始高度}
        first_brace = seed.index("{")
        second_brace = seed.index("{", first_brace + 1)
        third_brace = seed.index("{", second_brace + 1)
        seed_content = seed[1:first_brace-1]  # 提取64位种子内容
        total_bytes = int(seed[first_brace+1:second_brace-1])  # 提取原始数据长度
        original_width = int(seed[second_brace+1:third_brace-1])  # 提取原始宽度
        original_height = int(seed[third_brace+1:-1])  # 提取原始高度（去掉末尾}）
        print(f"解析的密钥为 {seed}")
        print(f"原文件名称为 {seed_content}")
        print(f"原始数据长度：{total_bytes}")
        print(f"原始宽度：{original_width}")
        print(f"原始高度：{original_height}")
    else:
        print("密钥格式错误：需为[文件名称]{原始数据长度}{原始宽度}{原始高度}格式（例：[abc123]{1555200}{720}{720}）")
        exit()

    # 将种子转换为混沌初始值x0（0 < x0 < 1）
    seed_hash = sum([ord(c) for c in seed]) % 10000
    x0 = 0.1 + (seed_hash % 8900) / 10000  # 确保x0在(0.1, 0.99)范围内（避免混沌边界）
    rssize=40
    chunksz=100
    if way == "1":
        rs = RSCodec(rssize)
        flat_data = img_array.flatten()
        total_bytes = len(flat_data)
        
        chunk_size = chunksz
        num_chunks = (total_bytes + chunk_size - 1) // chunk_size
        padded_data = np.pad(flat_data, (0, num_chunks * chunk_size - total_bytes), mode='constant')
        
        encoded_data = []
        print("正在编码")
        def encode_chunk():
            for i in tqdm(range(0, len(padded_data), chunk_size)):
                chunk = padded_data[i:i+chunk_size]
                chunk_bytes = bytes(chunk.tolist())
                encoded_chunk = rs.encode(chunk_bytes)
                encoded_data.extend(encoded_chunk)
        encode_chunk()
        encoded_data = np.array(encoded_data, dtype=np.uint8)
        
        #chaos_seq = logistic_map(x0, len(encoded_data))
        #chaos_seq = (chaos_seq * 255).astype(np.uint8)
        encrypted_data = encoded_data #^ chaos_seq

        # 填充数据以确保长度能被通道数整除
        padding = (channels - (len(encrypted_data) % channels)) % channels
        encrypted_data = np.pad(encrypted_data, (0, padding), mode='constant')
        
        # 计算总像素数（保持通道数不变）
        new_pixels = len(encrypted_data) // channels  # 必须整除（因加密数据长度是通道数的倍数）
        # 保持原始宽高比（width/height）
        aspect_ratio = width / height
        # 计算近似高度（取平方根后取整）
        new_height = int(np.sqrt(new_pixels / aspect_ratio))
        # 调整宽度以确保new_height × new_width = new_pixels
        new_width = int(new_pixels / new_height)
        # 验证并微调（避免整除误差）
        print("开始加密...")
        while new_height * new_width != new_pixels:
            new_height += 1
            new_width = int(new_pixels / new_height)
        
        encrypted_img_array = encrypted_data.reshape((new_height, new_width, channels))
        encrypted_img = image.fromarray(encrypted_img_array)
        encrypted_filename = f"encrypted_{seed_content}"
        encrypted_img.save(encrypted_filename)
        print(f"加密完成，新尺寸：{new_height}x{new_width}，已保存为 {encrypted_filename}")

    elif way == "2":
        rs = RSCodec(rssize)
        # 验证种子格式并解析信息（修改）
        if not (seed.startswith("[") and "{" in seed and seed.count("{") == 3 and seed.endswith("}")):
            print("密钥格式错误：需为[64位字符]{原始数据长度}{原始宽度}{原始高度}格式（例：[abc123]{1555200}{720}{720}）")
            exit()
        
        # 解密核心逻辑（保持）
        encrypted_flat_data = img_array.flatten()
        #chaos_seq = logistic_map(x0, len(encrypted_flat_data))
        #chaos_seq = (chaos_seq * 255).astype(np.uint8)
        decoded_data = encrypted_flat_data #^ chaos_seq
        
        chunk_size_encoded = chunksz+rssize
        decoded_data_clean = []
        print("开始解密...")
        def decode_chunk():
            for i in tqdm(range(0, len(decoded_data), chunk_size_encoded)):
                chunk = decoded_data[i:i+chunk_size_encoded]
                chunk_bytes = bytes(chunk.tolist())
                try:
                    decoded_chunk = rs.decode(chunk_bytes)[0]
                    decoded_data_clean.extend(decoded_chunk)
                except:
                    decoded_data_clean.extend(chunk[:-rssize])
        decode_chunk()
        # 恢复原始尺寸（修改）
        decoded_data_clean = np.array(decoded_data_clean[:total_bytes], dtype=np.uint8)
        decrypted_img_array = decoded_data_clean.reshape((original_height, original_width, channels))
        decrypted_img = image.fromarray(decrypted_img_array)
        decrypted_img.save("decrypted_" + seed_content)
        print("解密完成，已恢复原始尺寸：{}x{}".format(original_width, original_height))
        print("已保存为 decrypted_" + seed_content)