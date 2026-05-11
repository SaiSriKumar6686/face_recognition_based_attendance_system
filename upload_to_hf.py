"""
Precise batch upload to Hugging Face Spaces.
Only uploads the exact directories and files needed for deployment.
"""
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

TOKEN = ""  # ENTER_YOUR_HF_TOKEN_HERE
REPO_ID = "Sri6686/face-attendance-demo"
ROOT = Path(".")

# Explicit list of directories to include (relative to project root)
INCLUDE_DIRS = [
    "configs",
    "data/embeddings",
    "data/reference_images",
    "data/seed_images",
    "data/test_inputs",
    "models/onnx",
    "models/pretrained",
    "scripts",
    "src",
    "testing",
    "web",
]

# Explicit root-level files to include
ROOT_FILES = [
    ".gitignore",
    "Dockerfile",
    "README.md",
    "requirements.txt",
]

# Skip patterns inside included directories
SKIP_NAMES = {"__pycache__", ".pyc", "results"}


def collect_files():
    files = []
    
    # Root-level files
    for name in ROOT_FILES:
        p = ROOT / name
        if p.exists():
            files.append(p)
    
    # Files from included directories
    for dir_rel in INCLUDE_DIRS:
        d = ROOT / dir_rel
        if not d.exists():
            continue
        for f in d.rglob("*"):
            if f.is_dir():
                continue
            # Skip __pycache__ and results dirs
            parts = f.relative_to(ROOT).parts
            if any(skip in parts for skip in SKIP_NAMES):
                continue
            files.append(f)
    
    return sorted(files)


def upload_batch(api, batch, batch_num, total_batches):
    ops = []
    for local_path in batch:
        repo_path = local_path.relative_to(ROOT).as_posix()
        ops.append(CommitOperationAdd(
            path_in_repo=repo_path,
            path_or_fileobj=str(local_path),
        ))
    
    msg = f"batch {batch_num}/{total_batches} ({len(ops)} files)"
    print(f"  Committing {msg} ...")
    api.create_commit(
        repo_id=REPO_ID,
        repo_type="space",
        operations=ops,
        commit_message=msg,
    )
    print(f"  [OK] Batch {batch_num} done.")


def main():
    print("=" * 60)
    print("  HF Spaces Uploader (precise mode)")
    print("=" * 60)
    
    api = HfApi(token=TOKEN)
    files = collect_files()
    
    total_size = sum(f.stat().st_size for f in files)
    print(f"\nFiles: {len(files)}  |  Size: {total_size / 1024 / 1024:.1f} MB\n")
    
    for f in files:
        print(f"  {f.relative_to(ROOT).as_posix()}  ({f.stat().st_size / 1024:.1f} KB)")
    
    BATCH_SIZE = 15
    batches = [files[i:i+BATCH_SIZE] for i in range(0, len(files), BATCH_SIZE)]
    print(f"\n{len(batches)} batches to upload.\n")
    
    for i, batch in enumerate(batches, 1):
        try:
            upload_batch(api, batch, i, len(batches))
        except Exception as e:
            print(f"  [FAIL] Batch {i}: {e}")
            print(f"  Retrying...")
            try:
                upload_batch(api, batch, i, len(batches))
            except Exception as e2:
                print(f"  [FAIL] Retry failed: {e2}")
    
    print("\n" + "=" * 60)
    print("  DONE!")
    print(f"  https://huggingface.co/spaces/{REPO_ID}")
    print("=" * 60)


if __name__ == "__main__":
    main()
