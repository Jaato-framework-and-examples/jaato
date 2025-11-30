import os, shutil, zipfile, sys
from pathlib import Path

# Configuration
DIST_DIR = Path('dist')
PACKAGE_DIR = DIST_DIR / 'package'
ZIP_NAME = 'simple_connectivity_dist.zip'

# Files/folders to include (relative paths)
REQUIRED_PATHS = [
    'dist/requirements.txt',
    'dist/.env.example',
    'dist/README.md',
    'simple-connectivity-test/simple-connectivity-test.py',
    'shared/ssl_helper.py',
    'shared/token_accounting.py',
    'zScaler_cert_mgmt/import_root.py',
    'zScaler_cert_mgmt/verify_bundle.py',
]
# Optional: copy cert directory if INCLUDE_CERTS=1
CERT_DIR = 'zScaler_cert_mgmt/certs'

EXCLUDE_NAMES = {'.env', 'sa_keyfile.json', 'token_events_ledger.jsonl'}


def clean_package_dir():
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)


def copy_item(rel_path: str, required: bool = True):
    src = Path(rel_path)
    if not src.exists():
        if required:
            print(f'[ERROR] Missing required path: {rel_path}')
            return False
        print(f'[WARN] Optional path missing: {rel_path}')
        return False
    dst = PACKAGE_DIR / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        if src.name in EXCLUDE_NAMES:
            print(f'[SKIP] Excluding sensitive or runtime artifact: {rel_path}')
            return True
        shutil.copy2(src, dst)
    else:
        # Directory copy (shallow recursive)
        for root, dirs, files in os.walk(src):
            rroot = Path(root)
            for f in files:
                if f in EXCLUDE_NAMES:
                    print(f'[SKIP] Excluding {f} in {rroot}')
                    continue
                rel_file = rroot / f
                target = PACKAGE_DIR / rel_file.relative_to(Path('.'))
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(rel_file, target)
    return True


def build_zip():
    zip_path = DIST_DIR / ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(PACKAGE_DIR):
            for f in files:
                full = Path(root) / f
                rel = full.relative_to(PACKAGE_DIR)
                zf.write(full, rel.as_posix())
    return zip_path


def main():
    repo_root = Path('.').resolve()
    if not (repo_root / 'simple-connectivity-test').exists():
        print('[ERROR] Run from vertex-ai-tests repository root (contains simple-connectivity-test folder).')
        sys.exit(1)
    clean_package_dir()

    copied = []
    for p in REQUIRED_PATHS:
        if copy_item(p, required=True):
            copied.append(p)
        else:
            sys.exit(2)

    if os.environ.get('INCLUDE_CERTS', '') in ('1', 'true', 'True'):
        if copy_item(CERT_DIR, required=False):
            copied.append(CERT_DIR + '/**')
    else:
        print('[INFO] Skipping certs directory (set INCLUDE_CERTS=1 to include).')

    zip_path = build_zip()
    print('[OK] Distribution created:', zip_path)
    print('[INFO] Included paths:')
    for c in copied:
        print('  -', c)
    print('\n[NEXT] To use:')
    print('  1. unzip simple_connectivity_dist.zip')
    print('  2. python -m venv .venv && source .venv/bin/activate (or .venv\\Scripts\\activate)')
    print('  3. pip install -r requirements.txt')
    print('  4. cp .env.example .env and fill values')
    print('  5. python simple-connectivity-test/simple-connectivity-test.py --env-file .env --prompt "Hello"')

if __name__ == '__main__':
    main()
