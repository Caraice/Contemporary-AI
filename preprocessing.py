"""清洗帖子头部：去掉发帖人等信息，留下主题和正文。"""
import re


def clean_metadata(text):
    """用在清洗对照实验中，不改变其他特征方案。"""
    # 邮件格式的帖子通常用第一个空行分隔头部与正文。
    header, separator, body = text.partition("\n\n")
    if not separator:
        # 没找到分隔位置时保留原文，只转小写。
        return text.lower()
    # 从头部取出 Subject，避免把讨论主题一起删掉。
    subjects = re.findall(r"^Subject:\s*(.*)$", header, flags=re.MULTILINE | re.IGNORECASE)
    return (" ".join(subjects) + "\n" + body).lower()
