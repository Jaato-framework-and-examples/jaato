#!/usr/bin/env python3
"""Ask the kernel for each syscall family #1503 filters, from inside the payload.

Staged into a confined session's workspace and run by confined_exec.py:

    python confined_exec.py --stage probe_syscalls.py --command 'python3 probe_syscalls.py'
    python confined_exec.py --stage probe_syscalls.py --command 'python3 probe_syscalls.py' --seccomp off

Each probe makes the raw syscall with harmless (mostly invalid) arguments and
prints one JSON line: the call, its return value, errno, and the errno name.
Run it with the filter on and off and compare: a family the filter denies
answers EPERM (clone3: ENOSYS) with it on, and whatever the kernel would
otherwise say with it off (EINVAL, EFAULT, EBADF, success, or EPERM for lack
of a capability — which is why the off/on CONTRAST, plus ``Seccomp: 2`` in
/proc/self/status, is the evidence, not a single errno).

Nothing here changes system state: every call is made with arguments the
kernel rejects or with a request that is undone at once.  x86_64 and aarch64.
"""
import ctypes
import errno
import json
import os
import platform
import threading

libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long

ARCH = platform.machine()
NR = {
    "x86_64": {"bpf": 321, "keyctl": 250, "add_key": 248, "userfaultfd": 323,
               "io_uring_setup": 425, "perf_event_open": 298, "ptrace": 101,
               "process_vm_readv": 310, "unshare": 272, "setns": 308, "clone3": 435,
               "mount": 165, "kexec_load": 246, "init_module": 175, "personality": 135,
               "open_by_handle_at": 304, "fanotify_init": 300, "fsopen": 430,
               "getpid": 39},
    "aarch64": {"bpf": 280, "keyctl": 219, "add_key": 217, "userfaultfd": 282,
                "io_uring_setup": 425, "perf_event_open": 241, "ptrace": 117,
                "process_vm_readv": 270, "unshare": 97, "setns": 268, "clone3": 435,
                "mount": 40, "kexec_load": 104, "init_module": 105, "personality": 92,
                "open_by_handle_at": 265, "fanotify_init": 262, "fsopen": 430,
                "getpid": 172},
}[ARCH]

CLONE_NEWUSER = 0x10000000
CLONE_NEWNS = 0x00020000
PTRACE_PEEKDATA = 2
PER_LINUX32 = 0x0008
PERSONALITY_QUERY = 0xFFFFFFFF


def call(name, *args, family=None, expect_filtered=errno.EPERM):
    ctypes.set_errno(0)
    rc = libc.syscall(ctypes.c_long(NR[name]), *[ctypes.c_long(a) for a in args])
    e = ctypes.get_errno() if rc == -1 else 0
    print(json.dumps({"call": name, "family": family, "rc": rc, "errno": e,
                      "errname": errno.errorcode.get(e, "") if e else "",
                      "filtered_answer": errno.errorcode.get(expect_filtered, "")}))
    return rc, e


def status():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("Seccomp", "NoNewPrivs")):
                k, v = line.split(":", 1)
                print(json.dumps({"status": k, "value": v.strip()}))
    try:
        with open("/proc/self/attr/current") as f:
            print(json.dumps({"status": "attr/current", "value": f.read().strip("\x00\n ")}))
    except OSError as exc:
        print(json.dumps({"status": "attr/current", "value": f"unreadable: {exc}"}))


def controls():
    # Ordinary process and thread creation must be untouched by the filter.
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    _, st = os.waitpid(pid, 0)
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    print(json.dumps({"control": "fork+wait", "ok": os.WIFEXITED(st) and os.WEXITSTATUS(st) == 0}))
    print(json.dumps({"control": "thread", "ok": True}))
    rc, e = call("getpid", family="control", expect_filtered=0)


def in_child(fn):
    import sys
    sys.stdout.flush()
    pid = os.fork()
    if pid == 0:
        try:
            fn()
            sys.stdout.flush()
        finally:
            os._exit(0)
    os.waitpid(pid, 0)


def main():
    print(json.dumps({"arch": ARCH, "kernel": platform.release()}))
    status()
    controls()
    call("bpf", 0, 0, 0, family="bpf")                       # BPF_MAP_CREATE with NULL attr
    call("keyctl", 0, 0, 0, 0, 0, family="keyring")          # KEYCTL_GET_KEYRING_ID(0) -> EINVAL unfiltered
    call("add_key", 0, 0, 0, 0, 0, family="keyring")         # NULL type -> EFAULT unfiltered
    call("userfaultfd", 0, family="userfaultfd")
    call("io_uring_setup", 0, 0, family="io_uring")          # entries=0 -> EINVAL unfiltered
    call("perf_event_open", 0, 0, -1, -1, 0, family="perf")  # NULL attr -> EFAULT unfiltered
    call("ptrace", PTRACE_PEEKDATA, os.getppid(), 0, 0, family="ptrace")
    call("process_vm_readv", os.getpid(), 0, 0, 0, 0, 0, family="ptrace")
    in_child(lambda: call("unshare", CLONE_NEWUSER, family="namespaces"))  # a child: a
    # successful unshare would move THIS process into a new user namespace and
    # change every later answer
    call("setns", -1, 0, family="namespaces")                # bad fd -> EBADF unfiltered
    call("clone3", 0, 0, family="namespaces", expect_filtered=errno.ENOSYS)
    call("mount", 0, 0, 0, 0, 0, family="mount")
    call("fsopen", 0, 0, family="mount")
    call("kexec_load", 0, 0, 0, 0, family="kernel")
    call("init_module", 0, 0, 0, family="kernel")
    call("open_by_handle_at", -1, 0, 0, family="handles")
    call("fanotify_init", 0, 0, family="fanotify")
    call("personality", PERSONALITY_QUERY, family="personality")  # query: the filter allows only persona 0
    call("personality", PER_LINUX32, family="personality")
    call("personality", 0, family="personality", expect_filtered=0)                 # back to default


if __name__ == "__main__":
    main()
