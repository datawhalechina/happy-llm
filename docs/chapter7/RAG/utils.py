#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2025/06/20 13:50:47
@Author  :   不要葱姜蒜
@Version :   1.1
@Desc    :   None
'''

import os
from typing import Dict, List, Optional, Tuple, Union

import PyPDF2
import markdown
import json
from tqdm import tqdm
import tiktoken
from bs4 import BeautifulSoup
import re

enc = tiktoken.get_encoding("cl100k_base")


class ReadFiles:
    """
    class to read files
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self.file_list = self.get_files()

    def get_files(self):
        # args：dir_path，目标文件夹路径
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            # os.walk 函数将递归遍历指定文件夹
            for filename in filenames:
                # 通过后缀名判断文件类型是否满足要求
                if filename.endswith(".md"):
                    # 如果满足要求，将其绝对路径加入到结果列表
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".pdf"):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        docs = []
        # 读取文件内容
        for file in self.file_list:
            content = self.read_file_content(file)
            chunk_content = self.get_chunk(
                content, max_token_len=max_token_len, cover_content=cover_content)
            docs.extend(chunk_content)
        return docs

    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        chunk_text = []

        curr_len = 0
        curr_chunk = ''

        token_len = max_token_len - cover_content
        lines = text.splitlines()  # 假设以换行符分割文本为行

        for line in lines:
            # 保留空格，只移除行首行尾空格
            line = line.strip()
            line_len = len(enc.encode(line))
            
            if line_len > max_token_len:
                # 如果单行长度就超过限制，则将其分割成多个块
                # 先保存当前块（如果有内容）
                if curr_chunk:
                    chunk_text.append(curr_chunk)
                    curr_chunk = ''
                    curr_len = 0
                
                # 将长行按token长度分割
                line_tokens = enc.encode(line)
                num_chunks = (len(line_tokens) + token_len - 1) // token_len
                
                for i in range(num_chunks):
                    start_token = i * token_len
                    end_token = min(start_token + token_len, len(line_tokens))
                    
                    # 解码token片段回文本
                    chunk_tokens = line_tokens[start_token:end_token]
                    chunk_part = enc.decode(chunk_tokens)
                    
                    # 添加覆盖内容（除了第一个块）
                    if i > 0 and chunk_text:
                        prev_chunk = chunk_text[-1]
                        cover_part = prev_chunk[-cover_content:] if len(prev_chunk) > cover_content else prev_chunk
                        chunk_part = cover_part + chunk_part
                    
                    chunk_text.append(chunk_part)
                
                # 重置当前块状态
                curr_chunk = ''
                curr_len = 0
                
            elif curr_len + line_len + 1 <= token_len:  # +1 for newline
                # 当前行可以加入当前块
                if curr_chunk:
                    curr_chunk += '\n'
                    curr_len += 1
                curr_chunk += line
                curr_len += line_len
            else:
                # 当前行无法加入当前块，开始新块
                if curr_chunk:
                    chunk_text.append(curr_chunk)
                
                # 开始新块，添加覆盖内容
                if chunk_text:
                    prev_chunk = chunk_text[-1]
                    cover_part = prev_chunk[-cover_content:] if len(prev_chunk) > cover_content else prev_chunk
                    curr_chunk = cover_part + '\n' + line
                    curr_len = len(enc.encode(cover_part)) + 1 + line_len
                else:
                    curr_chunk = line
                    curr_len = line_len

        # 添加最后一个块（如果有内容）
        if curr_chunk:
            chunk_text.append(curr_chunk)

        return chunk_text

    @classmethod
    def read_file_content(cls, file_path: str):
        # 根据文件扩展名选择读取方法
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        else:
            raise ValueError("Unsupported file type")

    @classmethod
    def read_pdf(cls, file_path: str):
        # 读取PDF文件
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            return text

    @classmethod
    def read_markdown(cls, file_path: str):
        # 读取Markdown文件
        with open(file_path, 'r', encoding='utf-8') as file:
            md_text = file.read()
            html_text = markdown.markdown(md_text)
            # 使用BeautifulSoup从HTML中提取纯文本
            soup = BeautifulSoup(html_text, 'html.parser')
            plain_text = soup.get_text()
            # 使用正则表达式移除网址链接
            text = re.sub(r'http\S+', '', plain_text) 
            return text

    @classmethod
    def read_text(cls, file_path: str):
        # 读取文本文件
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()


class Documents:
    """
        获取已分好类的json格式文档
    """
    def __init__(self, path: str = '') -> None:
        self.path = path
    
    def get_content(self):
        with open(self.path, mode='r', encoding='utf-8') as f:
            content = json.load(f)
        return content
