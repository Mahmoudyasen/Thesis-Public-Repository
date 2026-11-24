"""Harness check on the GPU: forward/backward of the stage-1 detector on a synthetic patch (records peak VRAM for the
configured patch size and for the 192^3 alternative), and forward/backward of stage-2 A/B/C on a synthetic batch."""
import sys, time, torch, yaml
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from lungcad.stage1.model import build_detector, count_params
from lungcad.stage2.models import build_model, FocalLoss
from lungcad.stage2.patches import make_scales
from lungcad.augment import dose_kernel_augment

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/stage1_retinanet.yaml"))


def bench_detector(patch, batch, iters=3):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    det = build_detector(cfg["model"]); det.train(); opt = torch.optim.SGD(det.network.parameters(), lr=1e-3, momentum=0.9); scaler = torch.amp.GradScaler("cuda")
    img = torch.rand(batch, 1, *patch, device="cuda")
    targets = [{"box": torch.tensor([[30., 80., 80., 40., 92., 92.], [10., 20., 20., 16., 28., 28.]], device="cuda"), "label": torch.zeros(2, dtype=torch.long, device="cuda")} for _ in range(batch)]
    targets[-1] = {"box": torch.zeros(0, 6, device="cuda"), "label": torch.zeros(0, dtype=torch.long, device="cuda")}   # an empty patch must also work
    ts = []
    for i in range(iters):
        torch.cuda.synchronize(); t0 = time.time(); img_a = dose_kernel_augment(img)
        with torch.autocast("cuda", dtype=torch.float16):
            losses = det(img_a, targets); loss = losses[det.cls_key] + losses[det.box_reg_key]
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); torch.cuda.synchronize(); ts.append(time.time() - t0)
    print(f"stage1 patch {patch} batch {batch}: params {count_params(det.network)/1e6:.1f}M loss {loss.item():.3f} peak VRAM {torch.cuda.max_memory_allocated()/2**30:.1f} GB  {ts[-1]:.2f} s/iter")
    det.eval(); det.set_sliding_window_inferer(roi_size=list(patch), overlap=0.25, sw_batch_size=2, mode="gaussian", device="cpu", sw_device="cuda")
    vol = torch.rand(1, 1, 200, 300, 300)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = det(vol, use_inferer=True)[0]
    torch.cuda.synchronize(); print(f"  sliding-window inference on 200x300x300: {len(out['box'])} boxes, {time.time()-t0:.1f} s, peak VRAM {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
    del det, opt; torch.cuda.empty_cache()


bench_detector(tuple(cfg["data"]["patch_size"]), cfg["train"]["batch_size"])
try:
    bench_detector((192, 192, 192), 1)
except torch.OutOfMemoryError:
    print("stage1 patch 192^3 batch 1: OOM"); torch.cuda.empty_cache()

s2 = yaml.safe_load(open("configs/stage2.yaml"))
B = s2["train"]["batch_size"]; crit = FocalLoss()
blk = torch.rand(B, 1, 85, 149, 149, device="cuda"); sp = torch.tensor([[1.25, 0.703125, 0.703125]] * B); sub = torch.zeros(B, 3)
x = make_scales(blk, sp, augment=True, subvoxel_offset_zyx=sub); print("multi-scale input", tuple(x.shape))
for name in "ABC":
    torch.cuda.reset_peak_memory_stats(); kw = {k: v for k, v in s2["models"][name].items() if k not in ("lr", "warmup_iters")}
    m = build_model(name, **kw).cuda(); opt = torch.optim.AdamW(m.parameters(), 1e-4); y = (torch.rand(B, device="cuda") < 0.2).long()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = crit(m(x).float(), y)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    print(f"stage2 model {name}: params {sum(p.numel() for p in m.parameters())/1e6:.1f}M loss {loss.item():.3f} peak VRAM {torch.cuda.max_memory_allocated()/2**30:.1f} GB {(time.time()-t0)/3:.2f} s/iter (batch {B})")
    del m, opt; torch.cuda.empty_cache()
mC = build_model("C", fusion="concat").cuda(); print("C ablation fusion=concat ok", tuple(mC(x[:4]).shape))
print("SMOKE PASS")
