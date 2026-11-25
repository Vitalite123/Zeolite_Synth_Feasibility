# -*- coding: utf-8 -*-
"""P2-2 负结果论文：图 F1–F6（确定性、headless、无新读数）

预注册依据：
  图名与内容定义：P2-2_负结果论文_初稿_v0.4_骨架.md Appendix A（F1–F4）
  图表清单：raw/P2-2_投稿要件与JCIM格式_台账.md §4 D6–D9（含数据来源栏）
  表格数值：raw/wp2_p22_supp_v0_骨架.md（S1–S4 结果骨架，预注册先行）

数据来源（逐图）：
  F1 常量：v0.4 §2.1–§2.3（语料构成/标签语义/去重/域与特征数）；
          域读数 24,113 / 19,049 / 5,064 另与
          Documentation/P2-2/data/wp2_p22_supp_v0_metrics.parquet（S3_noise 行 n/n_pos/n_neg）
          及 Data/feasibility_scorer_v1_2_report.txt L11 交叉核对
  F2 常量：三切分构造与审计 v0.4 §2.4；AUROC xtal/pure 六值
          Data/feasibility_scorer_v1_2_report.txt L13–14；
          ECE_post 与 n_test：Data/m10_calib_v1_report.txt L49–51（部署账本）
  F3 常量：§2.5 冻结部署束（共享配置、五臂轮数/温度、12 锚点结构），
          对照 feasibility_scorer_v1_2_report.txt L7–18 与补充锚点脚本 BUNDLE_T（EMT 温度）
  F4 内容：正文 §3.1 声明的诊断链顺序与各工具 target/limit（纯常量，无数据读数）
  F5 数据：Documentation/P2-2/data/wp2_p22_supp_v0_metrics.parquet
          （S2_curve 4 点 mean/sd/n、S4_fit AUC∞/b/c、bootstrap CI 上下界与 n=1000）
  F6 数据：同上 parquet（S3_noise 四口径 + cover_hart_lo + dup/conflict 组计数）

输出：Documentation/P2-2/assets/p22_f{1..6}_*_300dpi.png（JCIM C5：300 dpi，
      双栏宽 ≤7 in、深 ≤9.167 in）；manifest 记录于 Documentation/P2-2/data/p22_figs_manifest.json

排版审计（G1 门，防文字压框/越界/相撞——2026-11-04 修订版规则）：
  R1 每个文本块用 renderer 实测包围盒（get_window_extent），对所属矩形输出
     左/右/下/上四向内边距（px），任一 < MIN_PAD 判越界；语料框三行文字同样登记；
  R2 所有文本两两做包围盒相交检测（正面积相交判相撞，零宽/零高贴边不算）；
  R3 文本与线/箭头相交判相撞，白底垫层的刻意覆盖须在 overlays 白名单显式登记；
  R4 框与框的可见边界（含 FancyBboxPatch 的 PAD 外扩）不得互压；
  R5 文本不得压到非所属框，专门捕获语料/读数压相邻框；
  R6 审计不过则 main() fail，不落盘 manifest；
  F4–F6 版式沿用同六条规则（柱顶标注、CI 带注记先登记后绘制）。
排版间距按 PAD 实扣：横排框 pitch = w + gap，可见间隙 = gap − 2×PAD，
  F1/F2 的 gap 取 0.070（可见 0.030），避免外扩边框互压。
画布：7.0 in 满幅（subplots_adjust 0/1/0/1），savefig 不再 bbox_inches="tight"，
  保证输出严格 7.00 in 宽、坐标留白不被裁掉。
确定性：无随机数、无交互；输入门 = 文件存在 + 关键读数断言（对照已登记值，容差 0）。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.transforms
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]          # ...\FASE_Zeolites
P22 = ROOT / "Documentation" / "P2-2"
DATA_SUPP = P22 / "data" / "wp2_p22_supp_v0_metrics.parquet"
DATA_L2 = ROOT / "Data" / "feasibility_scorer_v1_2_report.txt"
DATA_M10 = ROOT / "Data" / "m10_calib_v1_report.txt"
ASSETS = P22 / "assets"
MANIFEST = P22 / "data" / "p22_figs_manifest.json"

DPI = 300
MIN_PAD = 10          # px @300dpi（≈0.033 in）内边距下限
MIN_FS = 6.0          # 正文字号下限（脚注元信息沿用已定稿的 5.8，不属"为塞下而缩"）
PAD = 0.02            # FancyBboxPatch 外扩量（数据坐标）；排版间距必须扣除 2×PAD

# 与 P2-1 组图同一色系（p21_supp_figs.py），保证两文观感互洽
BLUE = "#2f5d8a"
BLUE_L = "#e3eef7"
GREEN_L = "#eaf1e4"
ORANGE_L = "#f6ece2"
PURPLE_L = "#f0eef5"
GREY_T = "#5c6b73"
FOOT = "#6b7b84"


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_metrics() -> pd.DataFrame:
    assert DATA_L2.exists() and DATA_M10.exists() and DATA_SUPP.exists(), "input missing"
    df = pd.read_parquet(DATA_SUPP)

    def val(exp, setting, metric):
        r = df[(df["exp"] == exp) & (df["setting"] == setting) & (df["metric"] == metric)]
        assert len(r) == 1, f"anchor not unique: {exp}/{setting}/{metric}"
        return float(r["value"].iloc[0])

    # 输入门：对照 raw/wp2_p22_supp_v0_骨架.md 已登记读数（逐位，容差 0）
    assert abs(val("S0_anchor", "random", "AUROC") - 0.630710) < 5e-7
    assert abs(val("S1_LOSO", "weighted_mean", "AUROC") - 0.645890) < 5e-7
    assert abs(val("S3_noise", "eps_dup", "noise_rate") - 0.017999) < 5e-7
    assert abs(val("S3_noise", "eps_1nn", "noise_rate") - 0.159375) < 5e-7
    assert abs(val("S3_noise", "eps_1nn_xdoi", "noise_rate") - 0.248580) < 5e-7
    assert abs(val("S4_fit", "point", "AUC_inf") - 0.653685) < 5e-7
    # 域读数交叉核对（v0.4 §2.3 / feasibility_scorer_v1_2_report.txt L11；
    # eps_model 行 n 为 random test 2,298，与 m10_calib_v1_report.txt L49 一致）
    s3_dom = df[(df["exp"] == "S3_noise") & (df["setting"] != "eps_model")]
    assert (s3_dom["n"] == 24113).all() and (s3_dom["n_pos"] == 19049).all() \
        and (s3_dom["n_neg"] == 5064).all()
    s3_model = df[(df["exp"] == "S3_noise") & (df["setting"] == "eps_model")]
    assert len(s3_model) == 1 and int(s3_model["n"].iloc[0]) == 2298
    print("G0 input gate: 3 files, 8 readback assertions PASS")
    return df


def get_row(df: pd.DataFrame, exp: str, setting: str, metric: str) -> pd.Series:
    r = df[(df["exp"] == exp) & (df["setting"] == setting) & (df["metric"] == metric)]
    assert len(r) == 1, f"row not unique: {exp}/{setting}/{metric}"
    return r.iloc[0]


def kv(extra: str, key: str) -> float:
    for part in str(extra).split(";"):
        if part.startswith(f"{key}="):
            return float(part.split("=", 1)[1])
    raise AssertionError(f"missing {key} in extra: {extra}")


# ------------------------------------------------------------------ 审计基建
def new_reg() -> dict:
    return {"contain": [],   # (label, text_artist, rect_patch)  文本须装在矩形内
            "texts": [],     # (label, text_artist)             参与两两相交检测
            "lines": [],     # (label, artist)                  线/箭头
            "rects": []}     # (label, rect_patch)              框与框不得互压


def box(ax, reg, xy, w, h, text=None, fc=BLUE_L, ec=BLUE, fs=8.0, bold=False,
        style="normal", color="#1c2b33", linespacing=1.35, label="?"):
    rect = FancyBboxPatch(xy, w, h, boxstyle=f"round,pad={PAD},rounding_size=0.015",
                          linewidth=1.0, edgecolor=ec, facecolor=fc)
    ax.add_patch(rect)
    reg["rects"].append((label, rect))
    if text is not None:
        t = ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
                    fontsize=fs, fontweight="bold" if bold else "normal", style=style,
                    color=color, linespacing=linespacing)
        reg["contain"].append((f"{label}:text-in-rect", t, rect))
        reg["texts"].append((f"{label}:text", t))
    return rect


def txt(ax, reg, x, y, s, label, fs=7.0, **kw):
    t = ax.text(x, y, s, fontsize=fs, **kw)
    reg["texts"].append((label, t))
    return t


def arrow(ax, reg, label, p0, p1):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                        linewidth=1.0, color=GREY_T, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    reg["lines"].append((label, a))
    return a


def line(ax, reg, label, xs, ys):
    (ln,) = ax.plot(xs, ys, color=GREY_T, linewidth=1.0)
    reg["lines"].append((label, ln))
    return ln


def audit(fig, reg, name: str, overlays: set = frozenset()) -> bool:
    """G1 排版审计：R1 内边距 / R2 文本两两相交 / R3 文本×线相交 /
    R4 框×框互压 / R5 文本×外来框。"""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    ok = True
    print(f"--- G1 audit [{name}] ---")
    own = {id(t): id(rect) for _, t, rect in reg["contain"]}
    for label, t, rect in reg["contain"]:
        tb = t.get_window_extent(renderer=ren)
        rb = rect.get_window_extent(renderer=ren)
        pl, pr = tb.x0 - rb.x0, rb.x1 - tb.x1
        pb, pt = tb.y0 - rb.y0, rb.y1 - tb.y1
        good = min(pl, pr, pb, pt) >= MIN_PAD
        ok &= good
        print(f"  R1 {label:<36s} pad L{pl:5.0f} R{pr:5.0f} B{pb:5.0f} T{pt:5.0f}px "
              f"{'OK' if good else '** OOB **'}")
    ts = reg["texts"]
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            l1, t1 = ts[i]
            l2, t2 = ts[j]
            if frozenset((l1, l2)) in overlays:
                continue
            b1 = t1.get_window_extent(renderer=ren)
            b2 = t2.get_window_extent(renderer=ren)
            ib = matplotlib.transforms.Bbox.intersection(b1, b2)
            if ib is not None and ib.width > 0.5 and ib.height > 0.5:
                ok = False
                print(f"  R2 TEXT-TEXT COLLISION: {l1}  ×  {l2}  ({ib.width:.0f}x{ib.height:.0f}px)")
    for lbl, t in ts:
        tb = t.get_window_extent(renderer=ren)
        for lnl, art in reg["lines"]:
            if frozenset((lbl, lnl)) in overlays:
                continue
            lb = art.get_window_extent(renderer=ren)
            ib = matplotlib.transforms.Bbox.intersection(tb, lb)
            if ib is not None and ib.width > 0.5 and ib.height > 0.5:
                ok = False
                print(f"  R3 TEXT-LINE COLLISION: {lbl}  ×  {lnl}  ({ib.width:.0f}x{ib.height:.0f}px)")
    # R4 框×框：外扩后的可见边界不得互压（横排框互压专项）
    rs = reg["rects"]
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            l1, r1 = rs[i]
            l2, r2 = rs[j]
            ib = matplotlib.transforms.Bbox.intersection(r1.get_window_extent(renderer=ren),
                                                         r2.get_window_extent(renderer=ren))
            if ib is not None and ib.width > 0.5 and ib.height > 0.5:
                ok = False
                print(f"  R4 RECT-RECT OVERLAP: {l1}  ×  {l2}  ({ib.width:.0f}x{ib.height:.0f}px)")
    # R5 文本×外来框：文本不得压到非所属框（语料/读数压框专项）
    for lbl, t in ts:
        tb = t.get_window_extent(renderer=ren)
        for rlbl, rect in rs:
            if own.get(id(t)) == id(rect):
                continue
            ib = matplotlib.transforms.Bbox.intersection(tb, rect.get_window_extent(renderer=ren))
            if ib is not None and ib.width > 0.5 and ib.height > 0.5:
                ok = False
                print(f"  R5 TEXT-FRAME COLLISION: {lbl}  ×  {rlbl}  "
                      f"({ib.width:.0f}x{ib.height:.0f}px)")
    print(f"  => {name}: {len(reg['contain'])} containment, {len(ts)} texts, "
          f"{len(reg['lines'])} lines, {len(rs)} boxes — {'ALL PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- F1
def fig_f1(out: Path) -> bool:
    """数据流：语料 → 标签派生 → 可行性域。
    与 P2-1 F1 的分工（台账 F4）：P2-1 侧重双轨分流，本图侧重标签派生与任务定义。"""
    fig, ax = plt.subplots(figsize=(7.0, 4.3), dpi=DPI)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    reg = new_reg()

    corpus = box(ax, reg, (0.10, 0.815), 0.80, 0.160, label="corpus")
    t = txt(ax, reg, 0.5, 0.935, "Hydrothermal dataset — 24,706 records × 149 fields",
            "corpus:title", fs=8.6, ha="center", va="center", fontweight="bold")
    reg["contain"].append(("corpus:title-in-rect", t, corpus))
    t = txt(ax, reg, 0.5, 0.878, "provenance:  source 23,961 · lit. increments 428",
            "corpus:prov1", fs=7.0, ha="center", va="center", color="#33454f")
    reg["contain"].append(("corpus:prov1-in-rect", t, corpus))
    t = txt(ax, reg, 0.5, 0.846, "rolling extraction 307 · R6 upgrades 10   (sum = 24,706)",
            "corpus:prov2", fs=7.0, ha="center", va="center", color="#33454f")
    reg["contain"].append(("corpus:prov2-in-rect", t, corpus))

    labels = [("positive\n15,187\ntarget only", BLUE_L, "pos"),
              ("relative_neg\n4,478\ntarget + secondary", GREEN_L, "rel"),
              ("absolute_neg\n4,993\namorphous / dense /\nlayered / unknown", ORANGE_L, "abs"),
              ("no_product\n48", PURPLE_L, "noprod")]
    w2, gap2, total2 = 0.185, 0.070, 0.95
    x0 = (1 - total2) / 2
    centers = [x0 + i * (w2 + gap2) + w2 / 2 for i in range(4)]
    for i, (t, c, lb) in enumerate(labels):
        box(ax, reg, (x0 + i * (w2 + gap2), 0.459), w2, 0.165, t, fc=c, fs=7.2,
            label=f"lab_{lb}")

    # 漏斗：单箭头直落水平分配线 → 短竖头到标签框外沿（不在框内停笔）
    arrow(ax, reg, "funnel_arrow", (0.5, 0.815 - PAD), (0.5, 0.672))
    line(ax, reg, "dist_line", [centers[0], centers[-1]], [0.672, 0.672])
    for i, cx_ in enumerate(centers):
        line(ax, reg, f"stub_{i}", [cx_, cx_], [0.672, 0.459 + 0.165 + PAD])

    box(ax, reg, (0.13, 0.318), 0.74, 0.088,
        "task  xtal = positive ∪ relative_neg        task  pure = positive only",
        fc="#f4f6f8", fs=7.8, label="task")

    txt(ax, reg, 0.5, 0.283, "remove 901 true-duplicate records",
        "dedup_note", fs=7.0, style="italic", ha="center", va="center", color=GREY_T,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    arrow(ax, reg, "dedup_arrow", (0.5, 0.318 - PAD), (0.5, 0.088 + 0.160 + PAD))

    box(ax, reg, (0.10, 0.088), 0.80, 0.160,
        "Feasibility domain\n24,113 rows = 19,049 xtal-positive + 5,064 negative\n"
        "490 features = 36 composition / conditions + 9 extended ratios\n"
        "+ 1 OSDA flag + 444 OSDA descriptors",
        fc="#e8f0e8", fs=7.4, linespacing=1.28, label="domain")

    txt(ax, reg, 0.5, 0.022,
        "Sources: P2-2 v0.4 §2.1–§2.3 · wp2_p22_supp_v0_metrics.parquet "
        "(S3_noise n/n_pos/n_neg) · feasibility_scorer_v1_2_report.txt L11",
        "footnote", fs=5.8, ha="center", va="center", color=FOOT)

    ok = audit(fig, reg, "F1", overlays={frozenset(("dedup_note", "dedup_arrow"))})
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out.name}")
    return ok


# ---------------------------------------------------------------- F2
def fig_f2(out: Path) -> bool:
    """三切分示意 + 泄漏审计。构造规则：v0.4 §2.4；
    读数：xtal/pure AUROC = feasibility_scorer_v1_2_report.txt L13–14；
    ECE_post / n_test = m10_calib_v1_report.txt L49–51（部署账本，与 P2-1 §3.3 同源）。
    版式：无穿框箭头（上中下三段直接纵排），避免文本×线相撞。"""
    fig, ax = plt.subplots(figsize=(7.0, 3.9), dpi=DPI)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    reg = new_reg()

    cols = [
        ("random", "80 / 10 / 10 split by route", BLUE_L,
         "recipes resembling the\ntraining distribution",
         "AUROC  xtal 0.6307 · pure 0.6210\nECE (post-calib.) 0.0293  ✓\ntest n = 2,298"),
        ("heldout", "6 frameworks excluded:\nLTL · SAS · ITG · RTH · SZR · LTA", GREEN_L,
         "transfer to unseen topologies\n(near-chance discrimination)",
         "AUROC  xtal 0.5158 · pure 0.5432\nECE (post-calib.) 0.0561  ✗\ntest n = 2,992"),
        ("time_2018", "train on records\npublished in or before 2018", ORANGE_L,
         "transfer forward in\npublication time",
         "AUROC  xtal 0.6650 · pure 0.6271\nECE (post-calib.) 0.0291  ✓\ntest n = 2,006"),
    ]
    w, gap, total = 0.270, 0.070, 0.95
    x0 = (1 - total) / 2
    for i, (name, rule, c, probe, readings) in enumerate(cols):
        x = x0 + i * (w + gap)
        lb = f"col_{name}"
        box(ax, reg, (x, 0.755), w, 0.165, f"{name}\n{rule}", fc=c, fs=7.4, label=f"{lb}:head")
        txt(ax, reg, x + w / 2, 0.665, probe, f"{lb}:probe", fs=6.8, ha="center",
            va="center", style="italic", color=GREY_T, linespacing=1.3)
        box(ax, reg, (x, 0.425), w, 0.185, readings, fc="#f7f9fa", ec="#9fb2bd",
            fs=7.4, label=f"{lb}:read")

    box(ax, reg, (x0, 0.225), total, 0.105,
        "leakage audits over DOI / framework / time, run in CI · six audits PASS\n"
        "DOI sets of the main dataset and the literature-failure table are disjoint (0 overlap)",
        fc="#f4f6f8", fs=7.2, label="audit_band")

    txt(ax, reg, 0.5, 0.145,
        "ECE threshold 0.05 (deployment ledger): random ✓ · heldout ✗ · time_2018 ✓",
        "ece_note", fs=6.8, ha="center", va="center", style="italic", color="#8a4a3a")
    txt(ax, reg, 0.5, 0.055,
        "Sources: split design P2-2 v0.4 §2.4 · AUROC feasibility_scorer_v1_2_report.txt "
        "L13–14 · ECE & n_test m10_calib_v1_report.txt L49–51",
        "footnote", fs=5.8, ha="center", va="center", color=FOOT)

    ok = audit(fig, reg, "F2")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out.name}")
    return ok


# ---------------------------------------------------------------- F3
P21_T2 = ROOT / "p21_supp_t2.py"


def assert_bundle_gate() -> None:
    """F3 输入门：部署束常量与登记报告逐 token 对照（容差 0）。"""
    assert DATA_L2.exists() and P21_T2.exists(), "bundle input missing"
    rep = DATA_L2.read_text(encoding="utf-8", errors="replace")
    for token in ("T_L1=1.2458", "轮 36, T=0.9434", "早停轮 = 59", "T=1.1836",
                  "早停@128", "T_L3=1.2845", "12 项锚点 assert"):
        assert token in rep, f"bundle token missing: {token}"
    t2 = P21_T2.read_text(encoding="utf-8", errors="replace")
    assert '"L3emt": 1.2845' in t2, "EMT temperature missing in anchor script"


def fig_f3(out: Path) -> bool:
    """冻结束：一个共享配置 → 五个臂 → 十二个锚点（EMT 臂在锚点集外）。
    内容为 §2.5 已登记常量（配置、轮数、温度、headline 锚点），无新读数。"""
    assert_bundle_gate()
    fig, ax = plt.subplots(figsize=(7.0, 3.9), dpi=DPI)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    reg = new_reg()

    txt(ax, reg, 0.5, 0.955, "one configuration  ·  five arms  ·  twelve anchor readings",
        "header", fs=7.4, ha="center", va="center", style="italic", color=GREY_T)

    box(ax, reg, (0.025, 0.300), 0.210, 0.440,
        "Shared configuration\n\nLightGBM\nlearning rate 0.05\n31 leaves\nmin_child_samples 100\n"
        "L2 regularisation 1.0\nmax_delta_step 1.0\nseed 42",
        fc="#f4f6f8", fs=6.9, linespacing=1.5, label="config")

    arms = [
        ("L2 · binary feasibility · xtal labels (headline)\n36 rounds · T = 0.9434",
         BLUE_L, "arm_l2xt"),
        ("L2 · binary feasibility · pure labels\n59 rounds · T = 1.1836", "#f7f9fa", "arm_l2pu"),
        ("L1 · phase classification · 198 classes\n118 rounds · T = 1.2458", "#f7f9fa", "arm_l1"),
        ("L3 · purity / competition · 200 classes\n128 rounds · T = 1.2845", "#f7f9fa", "arm_l3mc"),
        ("L3 · binary EMT variant (FAU stacking twin)\nT = 1.2845 · outside the anchor set",
         ORANGE_L, "arm_emt"),
    ]
    aw, ah, pitch, y0 = 0.355, 0.115, 0.165, 0.105
    for i, (t, c, lb) in enumerate(reversed(arms)):
        box(ax, reg, (0.335, y0 + i * pitch), aw, ah, t, fc=c, fs=6.8,
            linespacing=1.4, label=lb)

    box(ax, reg, (0.745, 0.300), 0.230, 0.440,
        "Twelve anchors\nL1 · L2 xtal · L2 pure · L3\nrandom · heldout · time_2018\n\n"
        "headline anchor\nrandom AUROC 0.6307095094\n\nsupplementary script\n"
        "recomputes them; aborts\non any deviation",
        fc="#e8f0e8", fs=6.8, linespacing=1.5, label="anchors")

    arrow(ax, reg, "a_cfg", (0.025 + 0.210 + PAD, 0.520), (0.335 - PAD, 0.520))
    arrow(ax, reg, "a_anc", (0.335 + aw + PAD, 0.520), (0.745 - PAD, 0.520))

    txt(ax, reg, 0.5, 0.030,
        "Sources: §2.5 frozen deployment bundle · feasibility_scorer_v1_2_report.txt L7–18 "
        "· supplementary anchor script (EMT temperature)",
        "footnote", fs=5.8, ha="center", va="center", color=FOOT)

    ok = audit(fig, reg, "F3")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out.name}")
    return ok


# ---------------------------------------------------------------- F4
def fig_f4(out: Path) -> bool:
    """研究框架：诊断链六工具（蛇形 1→6）+ 裁决/报告/边界汇入条。
    内容为正文 §3.1 声明的顺序与各工具 target/limit，纯常量，无数据读数。"""
    fig, ax = plt.subplots(figsize=(7.0, 3.9), dpi=DPI)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    reg = new_reg()

    W, H, G = 0.26, 0.20, 0.055
    xs = [0.030, 0.360, 0.690]
    top_y, bot_y = 0.665, 0.345
    mid_top, mid_bot = top_y + H / 2, bot_y + H / 2

    steps = [
        ("1 · Headline + calibration", "target: the score to explain", "limit: single seed", BLUE_L),
        ("2 · Source holdout", "target: same-paper bias", "limit: ten sources", GREEN_L),
        ("3 · Saturation probe", "target: data volume", "limit: four fractions", ORANGE_L),
        ("4 · Noise decomposition", "target: label channels", "limit: ordering only", PURPLE_L),
        ("5 · Cross-family control", "target: model-family route", "limit: extended variant", BLUE_L),
        ("6 · Gap diagnostics", "L1 · L3 · conformal", "limit: same features", GREEN_L),
    ]
    # 蛇形顺序：上排 1→2→3（左→右），下排 4→5→6（右→左）
    pos = [(xs[0], top_y), (xs[1], top_y), (xs[2], top_y),
           (xs[2], bot_y), (xs[1], bot_y), (xs[0], bot_y)]
    for (x, y), (t1, t2, t3, c), i in zip(pos, steps, range(6)):
        box(ax, reg, (x, y), W, H,
            f"{t1}\n{t2}\n{t3}", fc=c, fs=7.0, linespacing=1.35, label=f"step{i + 1}")

    arrow(ax, reg, "a12", (xs[0] + W + PAD, mid_top), (xs[1] - PAD, mid_top))
    arrow(ax, reg, "a23", (xs[1] + W + PAD, mid_top), (xs[2] - PAD, mid_top))
    arrow(ax, reg, "a34", (xs[2] + W / 2, top_y - PAD), (xs[2] + W / 2, bot_y + H + PAD))
    arrow(ax, reg, "a45", (xs[2] - PAD, mid_bot), (xs[1] + W + PAD, mid_bot))
    arrow(ax, reg, "a56", (xs[1] - PAD, mid_bot), (xs[0] + W + PAD, mid_bot))

    box(ax, reg, (0.030, 0.135), 0.92, 0.105,
        "Adjudication of the three candidate explanations  →  reporting requirements  "
        "→  boundaries of every claim",
        fc="#f4f6f8", fs=7.2, label="closing")
    arrow(ax, reg, "a6c", (xs[0] + W / 2, bot_y - PAD), (xs[0] + W / 2, 0.135 + 0.105 + PAD))

    txt(ax, reg, 0.5, 0.045,
        "Sources: manuscript §3.1 (framework) · instrument limitations as declared in the text",
        "footnote", fs=5.8, ha="center", va="center", color=FOOT)

    ok = audit(fig, reg, "F4")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out.name}")
    return ok


# ---------------------------------------------------------------- F5
def fig_f5(out: Path) -> bool:
    """学习曲线 + 预注册饱和拟合 + AUC∞ 的 bootstrap 区间。
    数据：S2_curve（4 点 mean/sd/n）、S4_fit（AUC∞/b/c 与 1,000 次自助 CI）。
    版式：对数横轴、误差棒、拟合实线、渐近虚线与 CI 带；不引入新读数。"""
    df = load_metrics()
    fracs = ["f=0.10", "f=0.25", "f=0.50", "f=1.00"]
    ns, mus, sds = [], [], []
    for f in fracs:
        r = get_row(df, "S2_curve", f, "AUROC_mean")
        ns.append(int(r["n"]))
        mus.append(float(r["value"]))
        sds.append(kv(r["extra"], "sd"))
    exp_mu = [0.591352, 0.617845, 0.647288, 0.630710]
    exp_sd = [0.035296, 0.019559, 0.004536, 0.000000]
    for m, e in zip(mus, exp_mu):
        assert abs(m - e) < 5e-7
    for s, e in zip(sds, exp_sd):
        assert abs(s - e) < 5e-7
    assert ns == [1945, 4863, 9727, 19454]

    fit = get_row(df, "S4_fit", "point", "AUC_inf")
    auc_inf, b, c = float(fit["value"]), kv(fit["extra"], "b"), kv(fit["extra"], "c")
    boot = get_row(df, "S4_fit", "boot", "AUC_inf_median")
    ci_lo, ci_hi = kv(boot["extra"], "ci_lo"), kv(boot["extra"], "ci_hi")
    assert abs(auc_inf - 0.653685) < 5e-7 and abs(b - 10.0) < 5e-7 and abs(c - 0.6733) < 5e-7
    assert abs(ci_lo - 0.621098) < 5e-7 and abs(ci_hi - 0.687924) < 5e-7
    assert abs(float(boot["value"]) - 0.654001) < 5e-7

    fig, ax = plt.subplots(figsize=(7.0, 4.3), dpi=DPI)
    fig.subplots_adjust(left=0.095, right=0.975, bottom=0.155, top=0.950)
    reg = new_reg()

    ax.set_xscale("log")
    ax.set_xlim(1600, 26000)
    ax.set_ylim(0.54, 0.70)
    ax.set_xlabel("Training rows (log scale)", fontsize=7.5)
    ax.set_ylabel("AUROC (random test, n = 2,298)", fontsize=7.5)
    ax.tick_params(labelsize=7.0)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", color="#dce3e8", linewidth=0.6)

    ax.axhspan(ci_lo, ci_hi, color=BLUE, alpha=0.07, zorder=0)
    ax.axhline(auc_inf, color=GREY_T, linestyle="--", linewidth=1.0, zorder=1)
    xg = np.logspace(np.log10(1600), np.log10(26000), 400)
    ax.plot(xg, auc_inf - b * xg ** (-c), color=BLUE, linewidth=1.6, zorder=2)
    ax.errorbar(ns, mus, yerr=sds, fmt="o", markersize=4.6, color=BLUE,
                ecolor="#9fb2bd", elinewidth=1.0, capsize=3.0, zorder=3)

    for n_, m_, s_ in zip(ns, mus, sds):
        t = ax.text(n_, m_ + s_ + 0.004, f"{m_:.4f} ± {s_:.4f}", fontsize=6.2, ha="center",
                    va="bottom", color="#1c2b33")
        reg["texts"].append((f"val_{n_}", t))

    t = ax.text(1680, 0.697,
                "observed plateau 0.63–0.65\n"
                f"AUC∞ {auc_inf:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]\n"
                f"b pinned at upper bound ({b:.1f}) → weakly identified",
                fontsize=6.8, ha="left", va="top", color="#1c2b33", linespacing=1.3,
                bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=2.0))
    reg["texts"].append(("annot_fit", t))
    t = ax.text(25200, 0.548,
                "−1.66 pp vs 50% (≈ ±1.7 pp design noise)",
                fontsize=6.4, ha="right", va="bottom", color=GREY_T, linespacing=1.25)
    reg["texts"].append(("annot_drop", t))

    t = fig.text(0.5, 0.018,
                 "Sources: S2 learning curve · S4 saturation fit "
                 "(AUC∞ CI = 1,000 test-row bootstrap) · wp2_p22_supp_v0_metrics.parquet",
                 fontsize=5.8, ha="center", va="center", color=FOOT)
    reg["texts"].append(("footnote", t))

    ok = audit(fig, reg, "F5")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out.name}")
    return ok


# ---------------------------------------------------------------- F6
def fig_f6(out: Path) -> bool:
    """噪声分解：ε_dup / ε_1nn / ε_1nn_xdoi / ε_model 四口径柱图。
    数据：S3_noise（noise_rate、dup/conflict 组计数、cover_hart_lo）。
    版式：柱顶数值、Cover–Hart 下界虚线、顶部两条读数注记。"""
    df = load_metrics()
    keys = ["eps_dup", "eps_1nn", "eps_1nn_xdoi", "eps_model"]
    vals = [float(get_row(df, "S3_noise", k, "noise_rate")["value"]) for k in keys]
    exp_vals = [0.017999, 0.159375, 0.248580, 0.170366]
    for v, e in zip(vals, exp_vals):
        assert abs(v - e) < 5e-7

    dup = get_row(df, "S3_noise", "eps_dup", "noise_rate")
    dup_groups = kv(dup["extra"], "dup_groups")
    conflict_groups = kv(dup["extra"], "conflict_groups")
    conflict_rows = kv(dup["extra"], "conflict_rows")
    cover = kv(get_row(df, "S3_noise", "eps_1nn", "noise_rate")["extra"], "cover_hart_lo")
    assert int(dup_groups) == 1471 and int(conflict_groups) == 386 \
        and int(conflict_rows) == 1148 and abs(cover - 0.079687) < 5e-7

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=DPI)
    fig.subplots_adjust(left=0.095, right=0.975, bottom=0.205, top=0.945)
    reg = new_reg()

    colors = [PURPLE_L, GREEN_L, ORANGE_L, BLUE_L]
    edges = ["#7d8493", "#7f9a68", "#b56b3c", BLUE]
    bars = ax.bar(range(4), vals, width=0.56, color=colors, edgecolor=edges,
                  linewidth=0.9, zorder=2)
    ax.set_ylim(0, 0.32)
    ax.set_yticks(np.arange(0, 0.301, 0.05))
    ax.set_ylabel("Estimated error rate (fraction of rows)", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7.0)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["ε_dup\nexact duplicates", "ε_1nn\nall neighbours",
                        "ε_1nn_xdoi\ncross-source", "ε_model\ncalibrated"], fontsize=6.8)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#dce3e8", linewidth=0.6)

    for i, v in enumerate(vals):
        t = ax.text(i, v + 0.007, f"{v:.4f}", fontsize=6.8, ha="center", va="bottom",
                    fontweight="bold" if i == 2 else "normal", color="#1c2b33")
        reg["texts"].append((f"barval_{i}", t))

    ln, = ax.plot([0.62, 1.62], [cover, cover], color=GREY_T, linestyle="--",
                  linewidth=1.0, zorder=1)
    reg["lines"].append(("cover_line", ln))
    t = ax.text(0.58, cover + 0.010, f"Cover–Hart lower bound {cover:.4f}",
                fontsize=6.2, ha="right", va="bottom", color=GREY_T)
    reg["texts"].append(("cover_label", t))

    t = ax.text(-0.30, 0.312,
                f"ε_dup: {int(dup_groups):,} duplicate groups · "
                f"{int(conflict_groups)} conflict ({conflict_groups / dup_groups:.2%}) · "
                f"{int(conflict_rows):,} rows — lower bound over 1.8% of the domain\n"
                f"ε_1nn_xdoi − ε_1nn = +{(vals[2] - vals[1]) * 100:.1f} pp: "
                "same-paper neighbours are disproportionately label-consistent",
                fontsize=6.2, ha="left", va="top", color="#1c2b33", linespacing=1.3,
                bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=2.0))
    reg["texts"].append(("annot_noise", t))

    t = fig.text(0.5, 0.018,
                 "Sources: S3 noise decomposition · wp2_p22_supp_v0_metrics.parquet "
                 "(n = 24,113; eps_model on random test n = 2,298)",
                 fontsize=5.8, ha="center", va="center", color=FOOT)
    reg["texts"].append(("footnote", t))

    ok = audit(fig, reg, "F6")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out.name}")
    return ok


SOURCES = {
    "p22_f1_label_derivation_300dpi.png":
        "P2-2 v0.4 §2.1–§2.3 · wp2_p22_supp_v0_metrics.parquet · feasibility_scorer_v1_2_report.txt",
    "p22_f2_three_splits_300dpi.png":
        "P2-2 v0.4 §2.4 · feasibility_scorer_v1_2_report.txt L13–14 · m10_calib_v1_report.txt L49–51",
    "p22_f3_frozen_bundle_300dpi.png":
        "§2.5 frozen deployment bundle · feasibility_scorer_v1_2_report.txt L7–18 · supplementary anchor script",
    "p22_f4_framework_300dpi.png": "manuscript §3.1 (framework, no data readings)",
    "p22_f5_learning_curve_300dpi.png": "wp2_p22_supp_v0_metrics.parquet (S2_curve / S4_fit)",
    "p22_f6_noise_decomposition_300dpi.png": "wp2_p22_supp_v0_metrics.parquet (S3_noise)",
    "p22_ga_graphical_abstract_300dpi.png":
        "graphical abstract · registered readings only (abstract / §3.2–§3.5 / §4, no new numbers)",
}


# ---------------------------------------------------------------- GA
def fig_ga(out: Path) -> bool:
    """Graphical abstract（Elsevier：≥531×1328 px，h:w≈5:13）。
    顶部问题条 + 三段横幅：冻结束读数 → 三解释逐一排除 → 天花板界与行动项。
    问题条对应正文 xtal / pure 双任务定义，不引入任何数字。
    全部为已登记读数（0.6307 / 0.6459 / 0.65 上界 / 24.9%），无新数字。"""
    fig, ax = plt.subplots(figsize=(6.0, 2.4), dpi=DPI)   # 1800×720 px, h:w = 0.4
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    reg = new_reg()

    txt(ax, reg, 0.5, 0.83,
        "Will a proposed synthesis recipe crystallise the target zeolite, "
        "and will it be phase-pure?",
        "ga_q", fs=9.4, ha="center", va="center", style="italic", color="#1c2b33")

    panels = [
        ("ga_left", (0.03, 0.12), 0.25, 0.56, BLUE_L,
         "Frozen head\non literature data",
         "24,706 records\nrandom AUROC 0.6307"),
        ("ga_mid", (0.34, 0.12), 0.32, 0.56, ORANGE_L,
         "Three explanations tested",
         "same-paper leakage ✗  (LOSO 0.6459)\ndata volume ✗  (no gain at 2×)\n"
         "model family ✗  (neural −7~−16 pp)"),
        ("ga_right", (0.72, 0.12), 0.25, 0.56, GREEN_L,
         "Ceiling bound",
         "≈ 0.65 · upper CI < 0.69\nset by unrecorded variables\n→ log them in benchmarks"),
    ]
    for label, xy, w, h, fc, head, body in panels:
        rect = box(ax, reg, xy, w, h, None, fc=fc, label=label)
        cx, ty, by = xy[0] + w / 2, xy[1] + 0.74 * h, xy[1] + 0.36 * h
        th = ax.text(cx, ty, head, ha="center", va="center", fontsize=7.4,
                     fontweight="bold", color="#1c2b33", linespacing=1.35)
        tb = ax.text(cx, by, body, ha="center", va="center", fontsize=7.4,
                     color="#1c2b33", linespacing=1.35)
        reg["contain"].append((f"{label}:head-in-rect", th, rect))
        reg["contain"].append((f"{label}:body-in-rect", tb, rect))
        reg["texts"].append((f"{label}:head", th))
        reg["texts"].append((f"{label}:body", tb))

    arrow(ax, reg, "ga_a12", (0.28 + PAD, 0.40), (0.34 - PAD, 0.40))
    arrow(ax, reg, "ga_a23", (0.66 + PAD, 0.40), (0.72 - PAD, 0.40))

    ok = audit(fig, reg, "GA")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out.name}")
    return ok


def main() -> None:
    load_metrics()
    which = sys.argv[1:] or ["f1", "f2", "f3", "f4", "f5", "f6"]
    outs = []
    if "f1" in which:
        o = ASSETS / "p22_f1_label_derivation_300dpi.png"
        if fig_f1(o):
            outs.append(o)
    if "f2" in which:
        o = ASSETS / "p22_f2_three_splits_300dpi.png"
        if fig_f2(o):
            outs.append(o)
    if "f3" in which:
        o = ASSETS / "p22_f3_frozen_bundle_300dpi.png"
        if fig_f3(o):
            outs.append(o)
    if "f4" in which:
        o = ASSETS / "p22_f4_framework_300dpi.png"
        if fig_f4(o):
            outs.append(o)
    if "f5" in which:
        o = ASSETS / "p22_f5_learning_curve_300dpi.png"
        if fig_f5(o):
            outs.append(o)
    if "f6" in which:
        o = ASSETS / "p22_f6_noise_decomposition_300dpi.png"
        if fig_f6(o):
            outs.append(o)
    if "ga" in which:
        o = ASSETS / "p22_ga_graphical_abstract_300dpi.png"
        if fig_ga(o):
            outs.append(o)

    from PIL import Image
    figs = {}
    if MANIFEST.exists():
        figs = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for o in outs:
        with Image.open(o) as im:
            w_px, h_px = im.size
        w_in, h_in = w_px / DPI, h_px / DPI
        assert abs(im.info["dpi"][0] - DPI) < 0.01, f"{o.name}: dpi {im.info['dpi']}"
        assert 4.1 <= w_in <= 7.0 + 1e-6, f"{o.name}: width {w_in:.2f} in"
        assert h_in <= 9.1, f"{o.name}: depth {h_in:.2f} in"
        print(f"  {o.name}: {w_px}x{h_px} px = {w_in:.2f} x {h_in:.2f} in @ {DPI} dpi")
        figs[o.name] = {"md5": md5_of(o), "dpi": DPI,
                        "width_in": round(w_in, 3), "height_in": round(h_in, 3),
                        "source": SOURCES[o.name]}
    MANIFEST.write_text(json.dumps(figs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"updated {MANIFEST.name}")


if __name__ == "__main__":
    main()
