"""阶段 0 · 发现 1：超额申请显存时会不会静默溢出到系统内存。

区分"申请成功"与"真的溢出"：显存内写入是数百 GB/s，被分页到系统内存要走 PCIe，
会掉一个数量级。三档尺寸：舒适容量 / 接近可用 / 超过可用。
"""

from deploy.diagnostics._common import save

import torch


def write_bw(nbytes: int, iters: int = 5) -> tuple[float, float]:
    buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    buf.fill_(1)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for i in range(iters):
        s.record(); buf.fill_(i % 256); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    del buf
    torch.cuda.empty_cache()
    med = sorted(ts)[len(ts) // 2]
    return nbytes / med / 1e9, med


def main() -> None:
    free, total = torch.cuda.mem_get_info()
    rows = []
    for label, nbytes in (("舒适 2 GB", 2 * 1024**3), ("0.85×可用", int(free * 0.85)),
                          ("1.25×可用", int(free * 1.25))):
        try:
            gbs, t = write_bw(nbytes)
            rows.append({"label": label, "gb": round(nbytes / 1024**3, 2),
                         "write_gb_s": round(gbs, 1), "median_ms": round(t * 1e3, 2)})
            print(f"{label:<10} {nbytes / 1024**3:6.2f} GB → {gbs:7.1f} GB/s")
        except torch.OutOfMemoryError:
            rows.append({"label": label, "gb": round(nbytes / 1024**3, 2), "oom": True})
            print(f"{label:<10} {nbytes / 1024**3:6.2f} GB → OOM（未溢出）")
    save("spill_probe", {"free_gb": round(free / 1024**3, 2),
                         "total_gb": round(total / 1024**3, 2), "rows": rows})


if __name__ == "__main__":
    main()
