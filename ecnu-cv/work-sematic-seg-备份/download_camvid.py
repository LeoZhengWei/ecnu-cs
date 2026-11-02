"""下载 CamVid 语义分割数据集（fast.ai 公开镜像，不是课堂数据）。

来源：
  Brostow, Fauqueur, Cipolla. Semantic object classes in video:
  A high-definition ground truth database. Pattern Recognition Letters, 2009.
镜像：
  https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz
  md5 648371e4f3a833682afb39b08a3ce2aa
"""

import hashlib
import os
import tarfile
import urllib.request

URL = "https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz"
# 文件大小与镜像 Content-Length 598913237 一致，tar 可完整列出。
# fastai download_checks 里旧的 md5 与当前镜像文件不一致，这里记录实际校验值。
MD5 = "2e796d442fe723192014ace89a1515b1"
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
ARCHIVE = os.path.join(DATA_DIR, "camvid.tgz")
TARGET = os.path.join(DATA_DIR, "camvid")


def md5sum(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.isdir(os.path.join(TARGET, "images")) and os.path.isdir(os.path.join(TARGET, "labels")):
        print(f"already extracted: {TARGET}")
        return
    if not (os.path.isfile(ARCHIVE) and md5sum(ARCHIVE) == MD5):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE)
    digest = md5sum(ARCHIVE)
    if digest != MD5:
        raise RuntimeError(f"md5 mismatch: {digest} != {MD5}")
    print(f"extracting {ARCHIVE}")
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    print(f"done: {TARGET}")


if __name__ == "__main__":
    download()
