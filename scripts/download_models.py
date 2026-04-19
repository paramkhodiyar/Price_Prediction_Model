"""
Download LFS-tracked model files via GitHub's LFS batch API.
Used in Render's build step instead of `git lfs pull` (which requires git-lfs installed).
"""
import json
import os
import sys
import urllib.request

REPO = "paramkhodiyar/Price_Prediction_Model"
LFS_BATCH_URL = f"https://github.com/{REPO}.git/info/lfs/objects/batch"

MODELS = [
    {
        "oid": "603b269f51533795485024aa84a81ae81a7d66da39eade07d2f20f2cf3f7bc36",
        "size": 172182817,
        "path": "models/mayaai_sale_rf_model.pkl",
    },
    {
        "oid": "d7ee124c903b385bafb010c95222c19cb3e5bc6d8cf3df0378efd47fb5f64aa7",
        "size": 27826,
        "path": "models/mayaai_sale_features.pkl",
    },
    {
        "oid": "815ad449548b59b61f7c8472ec1b4bdf5ce1577e94bd1d442a86eb3ac6bfeea3",
        "size": 66401,
        "path": "models/mayaai_sale_lr_pipeline.pkl",
    },
]

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(44) == LFS_POINTER_PREFIX
    except FileNotFoundError:
        return True


def get_download_urls(objects: list) -> dict:
    payload = json.dumps({
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{"oid": o["oid"], "size": o["size"]} for o in objects],
    }).encode()

    req = urllib.request.Request(
        LFS_BATCH_URL,
        data=payload,
        headers={
            "Content-Type": "application/vnd.git-lfs+json",
            "Accept": "application/vnd.git-lfs+json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    return {obj["oid"]: obj["actions"]["download"]["href"]
            for obj in data["objects"] if "actions" in obj}


def download(url: str, dest: str, size: int):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading {os.path.basename(dest)} ({size // 1_000_000} MB)...")
    urllib.request.urlretrieve(url, dest)
    print(f"  ✓ {dest}")


def main():
    needed = [m for m in MODELS if is_lfs_pointer(m["path"])]
    if not needed:
        print("All model files already present — nothing to download.")
        return

    print(f"Fetching LFS download URLs for {len(needed)} file(s)...")
    urls = get_download_urls(needed)

    for model in needed:
        url = urls.get(model["oid"])
        if not url:
            print(f"ERROR: No download URL returned for {model['path']}", file=sys.stderr)
            sys.exit(1)
        download(url, model["path"], model["size"])

    print("All model files downloaded successfully.")


if __name__ == "__main__":
    main()
