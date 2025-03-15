import os
import requests
from bs4 import BeautifulSoup
import shutil

baseurl = "https://hf-mirror.com/internlm/internlm-xcomposer2d5-7b/resolve/main/"
tree_url = "https://hf-mirror.com/internlm/internlm-xcomposer2d5-7b/tree/main"
dataset_path = "/hd2/tangzhenchen/model/Internlm-xcomposer2d5-7B"

# 创建本地目录
os.makedirs(dataset_path, exist_ok=True)

# 获取网页内容
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'
}
response = requests.get(tree_url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# 解析所有文件名（href 中包含 '/resolve/main/' 的链接）
file_list = []
for link in soup.find_all('a'):
    href = link.get('href')
    if href and '/resolve/main/' in href:
        filename = href.split('/resolve/main/')[-1].split('?')[0]
        file_list.append(filename)

print(f"[i] 共找到 {len(file_list)} 个文件")

# 下载所有文件
for filename in file_list:
    download_url = baseurl + filename
    local_path = os.path.join(dataset_path, filename)

    # 如果已存在就跳过
    if os.path.exists(local_path):
        print(f"[✓] 已存在，跳过：{filename}")
        continue

    print(f"[↓] 正在下载：{filename}")
    try:
        with requests.get(download_url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print(f"[✓] 下载完成：{filename}")
    except Exception as e:
        print(f"[!] 下载失败：{filename}, 错误：{e}")
