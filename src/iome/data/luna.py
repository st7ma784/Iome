"""
LUNA (SMB) → MinIO transfer helpers.

Thread-safe: each worker opens its own SMB and MinIO connection so the
ThreadPoolExecutor can saturate the link without lock contention.

Usage:
    python -m iome.data.luna --modality supermag --start 2015-01-01 --end 2015-12-31
"""

import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Iterator, List, Optional, Tuple

from minio import Minio
from smb.SMBConnection import SMBConnection


# ---------------------------------------------------------------------------
# Connection factories
# ---------------------------------------------------------------------------

def _smb(server: str, user: str, password: str) -> SMBConnection:
    conn = SMBConnection(
        user, password,
        f"iome_{threading.current_thread().ident}",
        server,
    )
    conn.connect(server, 445)
    return conn


def _minio(endpoint: str, access_key: str, secret_key: str, secure: bool = False) -> Minio:
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


# ---------------------------------------------------------------------------
# SMB directory walker
# ---------------------------------------------------------------------------

def list_smb_files(
    conn: SMBConnection,
    share: str,
    remote_dir: str,
    suffix: str = "",
) -> List[Tuple[str, int]]:
    """
    Recursively list (remote_path, file_size) pairs under remote_dir.
    Only returns files whose name ends with suffix (e.g. ".nc", ".gz").
    """
    results: List[Tuple[str, int]] = []
    _walk(conn, share, remote_dir, suffix, results)
    return results


def _walk(conn, share, directory, suffix, acc):
    for entry in conn.listPath(share, directory):
        name = entry.filename
        if name in (".", ".."):
            continue
        full = str(PurePosixPath(directory) / name)
        if entry.isDirectory:
            _walk(conn, share, full, suffix, acc)
        elif not suffix or name.endswith(suffix):
            acc.append((full, entry.file_size))


# ---------------------------------------------------------------------------
# Single-file transfer
# ---------------------------------------------------------------------------

def _transfer_one(
    remote_path: str,
    file_size: int,
    minio_bucket: str,
    smb_cfg: dict,
    minio_cfg: dict,
    overwrite: bool = False,
) -> str:
    mc   = _minio(**minio_cfg)
    conn = _smb(**smb_cfg)
    try:
        obj_name = remote_path.lstrip("/")
        if not overwrite:
            try:
                mc.stat_object(minio_bucket, obj_name)
                return f"skip {obj_name}"
            except Exception:
                pass

        with tempfile.NamedTemporaryFile() as tmp:
            conn.retrieveFile(smb_cfg["share"] if "share" in smb_cfg else "fst",
                              remote_path, tmp)
            tmp.seek(0)
            mc.put_object(minio_bucket, obj_name, tmp, file_size)
        return f"ok   {obj_name}"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Batch transfer
# ---------------------------------------------------------------------------

def transfer_directory(
    remote_dir: str,
    minio_bucket: str,
    smb_server: str = "luna",
    smb_share: str = "fst",
    smb_user: Optional[str] = None,
    smb_password: Optional[str] = None,
    minio_endpoint: str = "localhost:9000",
    minio_access_key: str = "minioadmin",
    minio_secret_key: str = "minioadmin",
    suffix: str = "",
    max_workers: int = 8,
    overwrite: bool = False,
) -> None:
    """
    Transfer all files under remote_dir on LUNA to MinIO.
    Credentials fall back to LUNA_USER / LUNA_PASSWORD env vars.
    """
    user     = smb_user     or os.environ["LUNA_USER"]
    password = smb_password or os.environ["LUNA_PASSWORD"]

    smb_cfg = {
        "server":   smb_server,
        "share":    smb_share,
        "user":     user,
        "password": password,
    }
    minio_cfg = {
        "endpoint":   minio_endpoint,
        "access_key": minio_access_key,
        "secret_key": minio_secret_key,
    }

    # List files using a dedicated discovery connection
    disc = _smb(smb_server, user, password)
    try:
        files = list_smb_files(disc, smb_share, remote_dir, suffix)
    finally:
        disc.close()

    print(f"Found {len(files)} files under {remote_dir}")

    # Ensure bucket exists
    mc = _minio(**minio_cfg)
    if not mc.bucket_exists(minio_bucket):
        mc.make_bucket(minio_bucket)

    # Parallel transfer
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(
                _transfer_one, path, size,
                minio_bucket, smb_cfg, minio_cfg, overwrite,
            ): path
            for path, size in files
        }
        for fut in as_completed(futs):
            try:
                print(fut.result())
            except Exception as exc:
                print(f"ERROR {futs[fut]}: {exc}")


# ---------------------------------------------------------------------------
# MinIO object listing (used by datasets)
# ---------------------------------------------------------------------------

def list_minio_objects(
    bucket: str,
    prefix: str,
    endpoint: str = "localhost:9000",
    access_key: str = "minioadmin",
    secret_key: str = "minioadmin",
    suffix: str = "",
) -> List[str]:
    """Return sorted list of object names matching prefix (and optional suffix)."""
    mc = _minio(endpoint, access_key, secret_key)
    objs = mc.list_objects(bucket, prefix=prefix, recursive=True)
    names = [o.object_name for o in objs if not suffix or o.object_name.endswith(suffix)]
    return sorted(names)


def get_minio_bytes(
    bucket: str,
    obj_name: str,
    endpoint: str = "localhost:9000",
    access_key: str = "minioadmin",
    secret_key: str = "minioadmin",
) -> bytes:
    mc = _minio(endpoint, access_key, secret_key)
    resp = mc.get_object(bucket, obj_name)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    MODALITY_DIRS = {
        "supermag": "/data/supermag",
        "tec":      "/data/ionex",
        "superdarn": "/data/superdarn",
    }
    MODALITY_BUCKETS = {
        "supermag":  "supermag-data",
        "tec":       "tec-data",
        "superdarn": "superdarn-data",
    }

    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=list(MODALITY_DIRS), required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    transfer_directory(
        remote_dir=MODALITY_DIRS[args.modality],
        minio_bucket=MODALITY_BUCKETS[args.modality],
        suffix="",
        max_workers=args.workers,
        overwrite=args.overwrite,
    )
