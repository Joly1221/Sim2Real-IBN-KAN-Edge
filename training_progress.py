"""【新增 IBN-REAL-04】进度文件为非关键输出：Windows 文件占用不应中断训练。"""
from __future__ import annotations

import json
from pathlib import Path
import time
from uuid import uuid4


def replace_with_retry(source: Path, target: Path, attempts: int = 6) -> None:
    """【新增】短暂文件占用时重试；关键检查点保存最终失败仍报错。"""
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(.05 * (attempt + 1))


def append_event(path: Path, record: dict) -> bool:
    """【新增】非关键日志写入失败时告警，不使已正常计算的训练失败。"""
    try:
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        return True
    except OSError as error:
        print(f'WARNING: log unavailable ({path.name}): {error}; training continues.', flush=True)
        return False


def write_progress(path: Path, record: dict) -> bool:
    """【修改】原 progress.tmp.replace 无异常保护；现在重试并提供追加日志备份。"""
    temporary = path.with_name(f'.{path.stem}.{uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        replace_with_retry(temporary, path)
        return True
    except OSError as error:
        append_event(path.with_name('progress_events.jsonl'), {**record, 'progress_file_error': str(error)})
        print(f'WARNING: {path.name} update skipped: {error}; training continues.', flush=True)
        return False
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass  # 仅清理本次创建的临时文件；清理失败不影响模型训练。
