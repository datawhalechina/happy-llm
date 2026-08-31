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
        if max_token_len <= 0:
            raise ValueError("max_token_len must be greater than 0")
        if cover_content < 0 or cover_content >= max_token_len:
            raise ValueError(
                "cover_content must be greater than or equal to 0 "
                "and less than max_token_len"
            )

        chunk_text = []
        token_len = max_token_len - cover_content

        def split_line(line: str) -> list[str]:
            """按完整 Unicode 字符切分，并优先遵守新增内容预算。"""
            parts = []
            start = 0
            while start < len(line):
                best_end = None
                for end in range(start + 1, len(line) + 1):
                    if len(enc.encode(line[start:end])) <= token_len:
                        best_end = end
                    else:
                        break
                if best_end is None:
                    for end in range(start + 1, len(line) + 1):
                        if len(enc.encode(line[start:end])) <= max_token_len:
                            best_end = end
                        else:
                            break
                if best_end is None:
                    raise ValueError(
                        "单个 Unicode 字符的 Token 数超过 max_token_len"
                    )
                parts.append(line[start:best_end])
                start = best_end
            return parts

        def add_overlap(previous: str, payload: str, separator: str) -> str:
            """从完整字符边界提取重叠，并确保最终片段不超过总预算。"""
            if not cover_content:
                return payload
            overlap = ""
            for start in range(len(previous) - 1, -1, -1):
                candidate = previous[start:]
                if len(enc.encode(candidate)) <= cover_content:
                    overlap = candidate
                else:
                    break
            while overlap:
                candidate = overlap + separator + payload
                if len(enc.encode(candidate)) <= max_token_len:
                    return candidate
                overlap = overlap[1:]
            return payload

        curr_len = 0
        curr_chunk = ''
        lines = text.splitlines()  # 假设以换行符分割文本为行

        for line in lines:
            # 保留空格，只移除行首行尾空格
            line = line.strip()
            line_len = len(enc.encode(line))

            if line_len > token_len:
                # 如果单行长度就超过新增内容预算，则按完整字符切分
                if curr_chunk:
                    chunk_text.append(curr_chunk)
                    curr_chunk = ''
                    curr_len = 0

                for i, chunk_part in enumerate(split_line(line)):
                    if i > 0 and chunk_text:
                        chunk_part = add_overlap(
                            chunk_text[-1], chunk_part, separator=""
                        )
                    chunk_text.append(chunk_part)

                curr_chunk = ''
                curr_len = 0

            elif curr_len + line_len + (1 if curr_chunk else 0) <= token_len:
                if curr_chunk:
                    curr_chunk += '\n'
                    curr_len += 1
                curr_chunk += line
                curr_len += line_len
            else:
                if curr_chunk:
                    chunk_text.append(curr_chunk)

                if chunk_text:
                    curr_chunk = add_overlap(
                        chunk_text[-1], line, separator='\n'
                    )
                    curr_len = line_len + (1 if curr_chunk != line else 0)
                else:
                    curr_chunk = line
                    curr_len = line_len

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
