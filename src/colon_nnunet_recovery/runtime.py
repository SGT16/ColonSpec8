import errno
import os
import time
import json
import uuid
from pathlib import Path
from fewlabel.runtime import clean
RETRY_ERRNOS = frozenset((errno.EACCES, errno.EBUSY, errno.EPERM))
MAX_REPLACE_WAIT_SECONDS=10.0

def replace_with_backoff(temporary, path, *, replace=None, clock=None, sleep=None):
    replace = replace or os.replace
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    deadline = clock() + MAX_REPLACE_WAIT_SECONDS
    delay = .05
    while True:
        try:
            replace(temporary, path)
            return
        except OSError as exc:


            remaining = deadline - clock()
            if exc.errno not in RETRY_ERRNOS or remaining <= 0:
                raise
            sleep(min(delay, remaining))
            delay = min(.5, delay * 2)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(clean(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_backoff(temporary, path)
