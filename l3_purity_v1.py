# -*- coding: utf-8 -*-
"""
WP2/L3 纯度建模 v1 —— EMT-FAU 竞相 d/R 比特征 + 温度校准 + 冻结 scaler
=====================================================================
周计划 P1#5 (主线排产): L3 v1 竞相特征工程, EMT 考题对标 EMT 论文 76%。

v0 首信号根因 (l3_purity_v0_report, 唯一自变量指向竞相几何):
  口径 A time EMT top-1 = 0.4811 (51/106), 错判集中 EMT->FAU 52 行;
  口径 B 三元归并 time = 0.5822 vs 论文 76% 未达; Others->FAU 92/114。

v1 三增量 (params/三切分/早停/域/标签逐位镜像 v0):
  ① COMP 8 维 EMT-FAU 竞相 d/R 比:
     d = OSDA 尺寸两口径 (d_rg = 2×gyration_radius_mean_0; d_eq = volume 等效球径);
     R = IZC 骨架最紧方向自由球直径冻结常数 (EMT 6.54 / FAU 7.35, 输入侧无泄漏;
     惯例 largest_free_sphere 全向口径两骨架同为 6.76 无区分, 故取 a/b 向最紧值);
     特征 = b_od_size + d_rg + d_eq + {d,d_eq}/R_EMT + {d,d_eq}/R_FAU + Δ(d/R)。
     量纲锚点: 18-冠-6 (EMT 经典模板) d_rg = 6.70 Å 落在 EMT 窗口 6.54 与 FAU 7.35 之间。
  ② 温度校准 (temperature scaling): T 于 random val 拟合 NLL 最小, 三模型共用冻结
     (heldout/time 无 val, 锚定主口径惯例); argmax 不变 => 分类读数不变,
     校准的是概率读数 (P(EMT)/竞相分布/ECE/NLL) —— PurityScorer 接口质量。
  ③ 冻结 scaler: A3 各块全库 fit 统计量 (f36/ext/dsc) + COMP 统计量落盘
     feat_frozen.parquet (498 维长表); PurityScorer v1 对任意子集零重 fit
     (修复 v0 局限: predict_proba_df 子集重跑 build_a3 重 fit scaler 的特征漂移)。
     自证: 复刻变换 vs build_a3 全库输出 490 维逐位 allclose (1e-9, equal_nan)。

考题口径同 v0 (A 域内严格 / B 论文对标三元归并; 主切分 time_2018, 辅 heldout)。
产出 (幂等覆盖写):
  Data/l3_purity_v1_report.txt           收口报告 (gitignored)
  Data/l3_purity_v1_pred.parquet         逐行预测 (random 模型全行 + time 模型考题行)
  Data/l3_purity_v1_manifest.json        manifest (数据版本 + 代码 md5 + T + 冻结表 md5)
  Data/l3_purity_v1_feat_frozen.parquet  冻结统计长表 (block/col/mean/std, 498 行)
用法: python l3_purity_v1.py
"""
import os
import json
import time
import hashlib
import subprocess

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from comp_neg_knn import build_features, LOG_MOLE_COLS, RATIO_CAP
from l1_feat_iter import build_ext, zscore_nan, T_COLS
from l1_baseline_v1 import build_a3, SEED
import osda_schema as osd

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, 'Data', 'zeobench_db_v1_8.parquet')
SPLITS = os.path.join(ROOT, 'Data', 'zeobench_splits_v1_8.parquet')
DEDUP = os.path.join(ROOT, 'Data', 'zeosyn_db_dedup_manifest.csv')
OSDA_DESC = os.path.join(ROOT, 'Data', 'osda_descriptors.parquet')
REPORT = os.path.join(ROOT, 'Data', 'l3_purity_v1_report.txt')
PRED = os.path.join(ROOT, 'Data', 'l3_purity_v1_pred.parquet')
MF = os.path.join(ROOT, 'Data', 'l3_purity_v1_manifest.json')
FROZEN = os.path.join(ROOT, 'Data', 'l3_purity_v1_feat_frozen.parquet')

FAU_EMT_TABLES = ['xlsx_2022-03+SI_T_S1', 'SI_T_S2']                  # incr_fau_emt
EMT2026_TABLES = ['EMT_data.csv', 'paper_Table_IV', 'SM_Table_S7']    # incr_emt2026
ANCHOR_EMT = 0.76                                                     # 论文三元有效准确率
R_EMT, R_FAU = 6.54 / 2.0, 7.35 / 2.0    # IZC largest_free_sphere_a/b 最紧方向半径 (Å)
# COMP v2 (L3 v1.2) 源列白名单: 框架无关 OSDA 占位几何
COMP_V2_SRC = ['gyration_radius_mean_0', 'volume_mean_0', 'axes_mean_0', 'axes_mean_1',
               'asphericity_mean_0', 'npr2_mean_0', 'free_sasa_mean_0',
               'num_rot_bonds_mean_0', 'formal_charge']
V0 = {'top1_emt_time': 0.4811, 'tri_time': 0.5822, 'tri_held': 0.3415}  # v0 首信号锚


def topk_acc(y_true, proba, k):
    top = np.argsort(-proba, axis=1)[:, :k]
    return float(np.mean([y_true[i] in top[i] for i in range(len(y_true))]))


def tri_fold(code):
    """三元归并: EMT / FAU / Others (论文对标口径; 域外真实相与非 EMT/FAU 预测均归 Others)"""
    c = str(code)
    return c if c in ('EMT', 'FAU') else 'Others'


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except Exception:
        return 'unavailable'


# ============ 复刻 A3 变换管线 (拆 fit 统计; fit 阶段自证逐位镜像 build_a3) ============

def build_f36(df, fill=None):
    """A3 的 36 维原始 F 矩阵 (comp_neg_knn.build_features 构造段复刻, 无 fit)。
    fill=None 用 df 自身中位 (fit 参照); 传入冻结 dict 保证子集零漂移。"""
    X = {}
    for c in LOG_MOLE_COLS:
        v = pd.to_numeric(df[c], errors='coerce').fillna(0.0).clip(lower=0.0)
        X[f'lg_{c}'] = np.log10(v + 1e-6)
    rSiAl = df['ratio_SiAl'].fillna(RATIO_CAP).clip(upper=RATIO_CAP)
    X['lg_ratio_SiAl'] = np.log10(1 + rSiAl)
    for c in ['ratio_OH_Si', 'ratio_Na_Si', 'ratio_H2O_Si', 'ratio_osda_Si']:
        v = df[c].fillna(0.0).clip(lower=0.0)
        X[f'lg_{c}'] = np.log10(1 + v)
    Tm = df['T_C'].median() if fill is None else fill['T_C_median']
    tm = df['t_h'].median() if fill is None else fill['t_h_median']
    X['T_C_100'] = df['T_C'].fillna(Tm) / 100.0
    X['lg_t_h'] = np.log10(1 + df['t_h'].fillna(tm))
    X['b_osda'] = (df['n_osda1'].fillna(0) > 0).astype(float)
    X['b_seed'] = (df['n_seed'].fillna(0) > 0).astype(float)
    X['b_fluoride'] = (df['n_F'].fillna(0) > 0).astype(float)
    X['b_nonwater'] = (df['solvent'].fillna(0) > 0).astype(float)
    stir = df['stirring'].astype(str).str.lower()
    X['b_stir'] = stir.isin(['yes', 'true', '1']).astype(float)
    F = pd.DataFrame(X, index=df.index)
    return F, list(F.columns)


def build_ext_v1(df, fill=None):
    """A2 扩展比值 9 维 (l1_feat_iter.build_ext 构造段复刻 + 冻结填充)。
    fit 状态两处: ratio_charge_bal 中位填充 + 内部 z 统计 (随后被调用方标准化覆盖)。"""
    def num(c):
        return pd.to_numeric(df[c], errors='coerce')

    def lg_ratio(nr, den):
        v = (nr / den.replace(0, np.nan)).clip(lower=0.0)
        return np.log10(1.0 + v.fillna(0.0))

    T = num('n_Si')
    for c in T_COLS[1:]:
        T = T + num(c)
    X = {}
    X['lg_ratio_OH_T'] = lg_ratio(num('n_OH'), T)
    X['lg_ratio_osda_T'] = lg_ratio(
        num('n_osda1') + num('n_osda2') + num('n_osda3'), T)
    X['lg_ratio_F_T'] = lg_ratio(num('n_F'), T)
    X['lg_ratio_H2O_T'] = lg_ratio(num('n_H2O'), T)
    X['lg_ratio_Ge_T'] = lg_ratio(num('n_Ge'), T)
    X['lg_ratio_seed_T'] = lg_ratio(num('n_seed'), T)
    X['lg_ratio_Na_Al'] = lg_ratio(num('n_Na'), num('n_Al'))
    X['lg_ratio_P_Al'] = lg_ratio(num('n_P'), num('n_Al'))
    cb = pd.to_numeric(df['ratio_charge_bal'], errors='coerce')
    if fill is None:
        cb = cb.fillna(cb.median())
        X['z_ratio_charge_bal'] = (cb - cb.mean()) / max(cb.std(), 1e-12)
    else:
        cb = cb.fillna(fill['rcb_median'])
        X['z_ratio_charge_bal'] = (cb - fill['rcb_in_mean']) / max(fill['rcb_in_std'], 1e-12)
    return pd.DataFrame(X, index=df.index)


def build_dsc_raw(df, osda, desc_cols):
    """A3 的 444 非 BE OSDA 描述符原始矩阵 (build_a3 join 段复刻, 未 zscore)"""
    J = df[['osda1_desc_key']].merge(
        osda[['osda_smiles'] + desc_cols],
        left_on='osda1_desc_key', right_on='osda_smiles',
        how='left', sort=False)
    assert len(J) == len(df), '描述符 join 行数漂移'
    return J[desc_cols].apply(pd.to_numeric, errors='coerce').values.astype(float)


def build_comp_raw(df, osda):
    """COMP 8 维原始值: EMT-FAU 竞相 d/R 比 (R 为冻结常数, 输入侧无泄漏)"""
    J = df[['osda1_desc_key']].merge(
        osda[['osda_smiles', 'gyration_radius_mean_0', 'volume_mean_0']],
        left_on='osda1_desc_key', right_on='osda_smiles',
        how='left', sort=False)
    assert len(J) == len(df), 'COMP join 行数漂移'
    d_rg = pd.to_numeric(J['gyration_radius_mean_0'], errors='coerce')
    d_eq = (6.0 * pd.to_numeric(J['volume_mean_0'], errors='coerce') / np.pi) ** (1.0 / 3.0)
    C = pd.DataFrame({
        'b_od_size': d_rg.notna().astype(float),
        'd_rg': d_rg,
        'd_eq': d_eq,
        'd_rg_o_R_emt': d_rg / R_EMT,
        'd_rg_o_R_fau': d_rg / R_FAU,
        'd_d_rg': d_rg / R_EMT - d_rg / R_FAU,
        'd_eq_o_R_emt': d_eq / R_EMT,
        'd_eq_o_R_fau': d_eq / R_FAU,
    })
    return C.values.astype(float), list(C.columns)


def build_comp_v2(df, osda):
    """COMP v2 8 维: 框架无关 OSDA 占位几何 (尺寸-形状-表面-柔性/电荷耦合)。

    来源见 Documentation/wp2_comp_redesign_v1_骨架.md §0.2; 仅用白名单 OSDA 描述符列,
    不含 framework 级常数 R_EMT/R_FAU。首列 b_od_geom 为可用性指示器 (推理侧恒等)。
    """
    need = COMP_V2_SRC
    missing = [c for c in need if c not in osda.columns]
    assert not missing, 'COMP v2 源列缺失: %s' % missing
    J = df[['osda1_desc_key']].merge(osda[['osda_smiles'] + need],
                                     left_on='osda1_desc_key',
                                     right_on='osda_smiles', how='left', sort=False)
    assert len(J) == len(df), 'COMP v2 join 行数漂移'

    def g(c):
        return pd.to_numeric(J[c], errors='coerce')

    rg = g('gyration_radius_mean_0')
    v = g('volume_mean_0')
    a0 = g('axes_mean_0')
    a1 = g('axes_mean_1')
    npr2 = g('npr2_mean_0')
    sasa = g('free_sasa_mean_0')
    rot = g('num_rot_bonds_mean_0')
    q = g('formal_charge')
    de = (6.0 * v / np.pi) ** (1.0 / 3.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        C = pd.DataFrame({
            'b_od_geom': np.logical_and.reduce(
                [s.notna().values for s in (rg, v, a0, a1, npr2, sasa, rot, q)]
                + [g('asphericity_mean_0').notna().values]).astype(float),
            'g_rg_o_eq': rg / de,
            'g_short_o_eq': a0 / de,
            'g_aspect': a0 / a1,
            'g_sasa_o_eq2': sasa / de ** 2,
            'g_npr2_x_eq': npr2 * de,
            'g_rot_x_short': np.log1p(rot.clip(lower=0.0)) * a0,
            'g_q_o_eq2': q.abs() / de ** 2,
        })
    return C.values.astype(float), list(C.columns)


COMP_BUILDERS = {'v1': build_comp_raw, 'v2': build_comp_v2}


def fit_v1(df, comp='v1'):
    """全库 fit: 各块统计量冻结 + 自证 490 维逐位镜像 build_a3; 返回 (X498, frozen)。

    comp='v1' 为历史默认 (COMP v1 竞相 d/R); comp='v2' 为 L3 v1.2 集成口径。
    """
    X_a3, _ = build_a3(df)
    assert X_a3.shape == (len(df), 490), 'A3 维度漂移'
    osda = osd.load(OSDA_DESC)
    desc_cols = osd.desc_cols(osda)
    assert len(desc_cols) == 444, 'OSDA 非 BE 描述符口径漂移: %d' % len(desc_cols)

    cb_raw = pd.to_numeric(df['ratio_charge_bal'], errors='coerce')
    cb1 = cb_raw.fillna(cb_raw.median())
    fill = {'T_C_median': float(df['T_C'].median()),
            't_h_median': float(df['t_h'].median()),
            'rcb_median': float(cb_raw.median()),
            'rcb_in_mean': float(cb1.mean()), 'rcb_in_std': float(cb1.std())}
    F, cols36 = build_f36(df, fill)
    mu36, sd36 = F.values.mean(0), F.values.std(0)
    ext_df = build_ext_v1(df, fill)
    ext = ext_df.values.astype(float)
    mu_e, sd_e = ext.mean(0), ext.std(0)
    D = build_dsc_raw(df, osda, desc_cols)
    mu_d, sd_d = np.nanmean(D, 0), np.nanstd(D, 0)
    C, cols_c = COMP_BUILDERS[comp](df, osda)
    mu_c, sd_c = np.nanmean(C, 0), np.nanstd(C, 0)
    sd_c = np.where(sd_c < 1e-12, 1.0, sd_c)

    Z36 = (F.values - mu36) / sd36
    Zd = (D - mu_d) / sd_d                       # nan 保留 (zscore_nan 同构)
    Zc = (C - mu_c) / sd_c
    Zc[:, 0] = C[:, 0]                           # b_od_size 指示器不变换 (b_desc 惯例)
    X = np.hstack([Z36, (ext - mu_e) / sd_e,
                   df['osda1_desc_key'].notna().astype(float).values[:, None],
                   Zd, Zc])
    assert X.shape == (len(df), 498), 'v1 498 维漂移'
    ok = np.allclose(X[:, :490], X_a3, atol=1e-9, equal_nan=True)
    mx = float(np.nanmax(np.abs(X[:, :490] - X_a3)))
    assert ok, '复刻变换 vs build_a3 偏差超阈 (max|Δ|=%.2e)' % mx

    rows = []
    for blk, cols_, mu, sd in (('f36', cols36, mu36, sd36),
                               ('ext', list(ext_df.columns), mu_e, sd_e),
                               ('dsc', ['od1_' + c for c in desc_cols], mu_d, sd_d),
                               ('comp', cols_c, mu_c, sd_c)):
        for j, c in enumerate(cols_):
            rows.append({'block': blk, 'col': c, 'mean': float(mu[j]), 'std': float(sd[j])})
    for k, v in fill.items():                        # fill 伪块: 填充标量冻结 (子集零漂移)
        rows.append({'block': 'fill', 'col': k, 'mean': v, 'std': 0.0})
    frozen = pd.DataFrame(rows)
    frozen.loc[frozen['col'] == cols_c[0], ['mean', 'std']] = [0.0, 1.0]
    X_rt = transform_v1(df, frozen, osda, desc_cols)   # 反向自证: 冻结变换 == fit 直产
    assert np.allclose(X_rt, X, atol=1e-9, equal_nan=True), '冻结变换反向自证失败'
    return X, frozen, osda, desc_cols


def transform_v1(df, frozen, osda, desc_cols):
    """冻结统计变换 (推理侧零 fit): 498 维, 子集不变性由统计冻结保证。

    COMP 版本从冻结表 comp 块列名自动识别: v2 首列为 b_od_geom, 否则走 v1。
    """
    comp_cols = frozen.loc[frozen['block'] == 'comp', 'col'].tolist()
    comp = 'v2' if comp_cols and comp_cols[0] == 'b_od_geom' else 'v1'
    fr_f = frozen[frozen['block'] == 'fill']
    fill = dict(zip(fr_f['col'], fr_f['mean']))
    F, _ = build_f36(df, fill)
    ext = build_ext_v1(df, fill).values.astype(float)
    D = build_dsc_raw(df, osda, desc_cols)
    C, cols_c = COMP_BUILDERS[comp](df, osda)
    assert list(cols_c) == comp_cols, 'COMP 冻结列名与构建器不一致 (%s)' % comp
    parts = []
    for M, blk in ((F.values.astype(float), 'f36'), (ext, 'ext'),
                   (D, 'dsc'), (C, 'comp')):
        fr = frozen[frozen['block'] == blk]
        parts.append((M - fr['mean'].values) / fr['std'].values)
    X = np.hstack(parts[:2]                          # A3 列序: f36, ext, b_desc, dsc + comp
                  + [df['osda1_desc_key'].notna().astype(float).values[:, None]]
                  + parts[2:])
    X[:, -8] = C[:, 0]                               # b_od_size 恒等 (冻结 mean=0/std=1)
    assert X.shape == (len(df), 498), '冻结变换维度漂移'
    return X


# ============ 温度校准 (temperature scaling) ============

def proba_with_T(proba, T):
    z = np.log(np.clip(proba, 1e-12, 1.0)) / T
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(proba, y):
    """T 于 val 上 NLL 最小 (粗网格 + 3 轮局部细化)"""
    logits = np.log(np.clip(proba, 1e-12, 1.0))
    tgt = np.zeros_like(proba)
    tgt[np.arange(len(y)), y] = 1.0

    def nll(T):
        z = logits / T
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return float(-np.mean(np.sum(tgt * np.log(np.clip(p, 1e-12, 1.0)), axis=1)))

    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 200))
    T = float(grid[int(np.argmin([nll(t) for t in grid]))])
    for r in (0.5, 0.1, 0.02):
        g = np.linspace(max(1e-3, T * (1 - r)), T * (1 + r), 81)
        T = float(g[int(np.argmin([nll(t) for t in g]))])
    return T, nll(T), nll(1.0)


def ece(proba, y, n_bins=10):
    conf = proba.max(1).astype(float)
    hit = (proba.argmax(1) == y).astype(float)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(e)


def nll_mc(proba, y):
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0))))


class PurityScorerV1:
    """L3 v1 scorer——冻结统计 + 温度 T; 任意子集零重 fit (v0 特征漂移修复)。
    predict_purity 接口与 v0 对齐 (feasibility_scorer 第三层)。"""

    def __init__(self, model, classes, best_iter, frozen, osda, desc_cols, T):
        self.model, self.classes, self.best = model, classes, best_iter
        self.frozen, self.osda, self.desc_cols, self.T = frozen, osda, desc_cols, T
        self.cmap = {c: i for i, c in enumerate(classes)}

    def predict_proba_df(self, df):
        X = transform_v1(df, self.frozen, self.osda, self.desc_cols)
        return proba_with_T(self.model.predict(X, num_iteration=self.best), self.T)

    def predict_purity(self, df, target, top_competing=3):
        proba = self.predict_proba_df(df)
        j = self.cmap.get(str(target))
        target_p = proba[:, j] if j is not None else np.zeros(len(df))
        comp = []
        for row in proba:
            order = np.argsort(-row)
            c = {self.classes[i]: float(row[i]) for i in order[:top_competing + 1]
                 if self.classes[i] != str(target)}
            comp.append(dict(list(c.items())[:top_competing]))
        return {'target_p': target_p.astype(float), 'competing': comp,
                'target_in_classes': j is not None}


def main():
    log = []

    def P(msg=''):
        print(msg, flush=True)
        log.append(str(msg))

    t_start = time.time()
    df = pd.read_parquet(DB)
    sp = pd.read_parquet(SPLITS)
    man = pd.read_csv(DEDUP)
    dup_ids = set(man.loc[man['tier'] == 'A', 'route_id'].dropna())
    drop = df['route_id'].isin(dup_ids)
    P('WP2/L3 纯度建模 v1 —— EMT-FAU 竞相 d/R 比特征 + 温度校准 + 冻结 scaler')
    P('=' * 68)
    P('去重 (Tier A): 剔除 %d 行' % int(drop.sum()))
    df = df[~drop].reset_index(drop=True)

    P('特征 fit: 全库构建 A3 490 + COMP 8 = 498 维 (统计量冻结落盘)...')
    X, frozen, osda, desc_cols = fit_v1(df)
    frozen.to_parquet(FROZEN, engine='pyarrow', compression='zstd', index=False)
    P('  自证 PASS: 复刻变换 vs build_a3 490 维逐位 allclose(1e-9); 冻结表 %d 行 -> %s'
      % (len(frozen), os.path.basename(FROZEN)))
    P('  COMP 块 (R 冻结常数: R_EMT=%.2f / R_FAU=%.2f Å): %s'
      % (R_EMT, R_FAU, [c for c in frozen.loc[frozen.block == 'comp', 'col']]))

    # ---- L3 域 (pos + rel, 与 v0 逐位一致) ----
    dom = ((df['label1'].isin(['positive', 'relative_neg']))
           & df['zeo_desc_key'].notna()).values
    idx = np.where(dom)[0]
    m = df.loc[dom, ['route_id', 'label1', 'framework_code', 'source_table']].merge(
        sp[['route_id', 'fw_key', 'split_random', 'split_heldout', 'split_time_2018']],
        on='route_id', how='left')
    assert len(m) == len(idx) and m['fw_key'].notna().all(), '域行与切分表对不齐'
    y_all = m['fw_key'].astype(str).values
    sr, sh, st = m['split_random'].values, m['split_heldout'].values, m['split_time_2018'].values
    classes = sorted(set(y_all))
    cmap = {c: i for i, c in enumerate(classes)}
    y = np.array([cmap[v] for v in y_all])
    Xm = X[idx]
    P('域: positive + relative_neg & zeo_desc_key 有效 = %d 行, 类别 %d 个'
      ' (与 v0 逐位一致)' % (len(y), len(classes)))
    P('')

    params = dict(objective='multiclass', num_class=len(classes),
                  learning_rate=0.05, num_leaves=31, min_child_samples=100,
                  lambda_l2=1.0, max_delta_step=1.0, seed=SEED, verbosity=-1)

    # ---- 切分一: random 主口径 (val 早停 + 温度拟合) ----
    tr, va, te = sr == 'train', sr == 'val', sr == 'test'
    t0 = time.time()
    dtr = lgb.Dataset(Xm[tr], label=y[tr])
    dva = lgb.Dataset(Xm[va], label=y[va], reference=dtr)
    b_r = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
    best = b_r.best_iteration or -1
    pv = b_r.predict(Xm[va], num_iteration=best)
    pt_r = b_r.predict(Xm[te], num_iteration=best)
    T, nll_T, nll_base = fit_temperature(pv, y[va])
    P('split_random (主口径, val 早停@%d, %.0fs): test top-1/3/5 = %.4f / %.4f / %.4f'
      % (best, time.time() - t0, topk_acc(y[te], pt_r, 1),
         topk_acc(y[te], pt_r, 3), topk_acc(y[te], pt_r, 5)))
    P('  EMT 类支撑: train %d / val %d / test %d 行'
      % (int(((y == cmap['EMT']) & tr).sum()), int(((y == cmap['EMT']) & va).sum()),
         int(((y == cmap['EMT']) & te).sum())))
    P('  温度校准: T* = %.4f (random val 拟合, 三模型共用冻结); val NLL %.4f -> %.4f'
      % (T, nll_base, nll_T))
    P('  random test: ECE %.4f -> %.4f | NLL %.4f -> %.4f (argmax 不变, 校准概率读数)'
      % (ece(pt_r, y[te]), ece(proba_with_T(pt_r, T), y[te]),
         nll_mc(pt_r, y[te]), nll_mc(proba_with_T(pt_r, T), y[te])))
    P('')

    # ---- 切分二: heldout (轮数锚定主口径) ----
    tr_h, te_h = sh == 'train', sh == 'test'
    t0 = time.time()
    b_h = lgb.train(params, lgb.Dataset(Xm[tr_h], label=y[tr_h]), num_boost_round=best)
    pt_h = b_h.predict(Xm[te_h])
    P('split_heldout (固定 %d 轮, %.0fs): test top-1/3/5 = %.4f / %.4f / %.4f'
      % (best, time.time() - t0, topk_acc(y[te_h], pt_h, 1),
         topk_acc(y[te_h], pt_h, 3), topk_acc(y[te_h], pt_h, 5)))
    P('')

    # ---- 切分三: time_2018 (EMT 考题主切分) ----
    tr_t, te_t = st == 'train', st == 'test'
    t0 = time.time()
    b_t = lgb.train(params, lgb.Dataset(Xm[tr_t], label=y[tr_t]), num_boost_round=best)
    pt_t = b_t.predict(Xm[te_t])
    P('split_time_2018 (固定 %d 轮, %.0fs): test top-1/3/5 = %.4f / %.4f / %.4f'
      % (best, time.time() - t0, topk_acc(y[te_t], pt_t, 1),
         topk_acc(y[te_t], pt_t, 3), topk_acc(y[te_t], pt_t, 5)))
    P('')

    # ================= EMT 考题 (口径同 v0) =================
    P('=' * 68)
    P('EMT 考题 (主切分 time_2018, 辅 heldout; 对标论文三元 76.0%):')
    P('  v0 首信号锚: EMT top-1 %.4f | 三元 time %.4f / heldout %.4f'
      % (V0['top1_emt_time'], V0['tri_time'], V0['tri_held']))
    P('')

    src = m['source_table'].values
    is_fau_emt = np.isin(src, FAU_EMT_TABLES)
    is_emt2026 = np.isin(src, EMT2026_TABLES)
    is_quiz = is_fau_emt | is_emt2026

    # ---- 口径 A: 域内严格 (全类 top-1) ----
    quiz_a, acc_a_emt = {}, float('nan')
    for spl_name, te_m, pt_m in (('time_2018', te_t, pt_t), ('heldout', te_h, pt_h)):
        q = is_quiz[te_m]
        if not q.any():
            continue
        y_te = y[te_m]
        pred = pt_m[q].argmax(axis=1)
        pT = proba_with_T(pt_m[q], T)
        v0a = V0['top1_emt_time'] if spl_name == 'time_2018' else None
        P('[口径 A 域内严格 | %s] 两分区 ∩ 域 ∩ test: %d 行, top-1 = %.4f'
          % (spl_name, int(q.sum()), float(np.mean(pred == y_te[q]))))
        emt_q = y_te[q] == cmap['EMT']
        if emt_q.any():
            a1 = float(np.mean(pred[emt_q] == y_te[q][emt_q]))
            P('    真实 EMT %d 行: top-1 = %.4f | P(EMT) mean %.4f -> 校准后 %.4f'
              % (int(emt_q.sum()), a1,
                 float(pt_m[q][emt_q][:, cmap['EMT']].mean()),
                 float(pT[emt_q][:, cmap['EMT']].mean())))
            if v0a is not None:
                P('      vs v0 EMT top-1 %.4f: %+.4f' % (v0a, a1 - v0a))
                acc_a_emt = a1
        fau_q = y_te[q] == cmap['FAU']
        if fau_q.any():
            P('    真实 FAU %d 行: top-1 = %.4f | P(EMT) mean 校准后 = %.4f'
              % (int(fau_q.sum()), float(np.mean(pred[fau_q] == y_te[q][fau_q])),
                 float(pT[fau_q][:, cmap['EMT']].mean())))
        quiz_a[spl_name] = q
    P('')

    # ---- COMP 特征有效性诊断: 考题 EMT vs FAU 行的竞相几何方向 ----
    q = quiz_a.get('time_2018')
    if q is not None:
        df_dom = df.loc[dom].reset_index(drop=True)     # 域内行序 == m (how='left' 保左序)
        Cq = transform_v1(df_dom[te_t][q], frozen, osda, desc_cols)
        j_b = Cq.shape[1] - 8                          # comp 首列 b_od_size (恒等 0/1)
        i_d = Cq.shape[1] - 8 + 5                      # d_d_rg 在 comp 块第 6 列
        y_q = y[te_t][q]
        n_ok = int(Cq[:, j_b].sum())
        P('[COMP 诊断 | time 考题] OSDA 尺寸可用性: %d/%d 行 (缺失 = OSDA 不在 843 行'
          '描述符表——考题冠醚等未入表)' % (n_ok, int(q.sum())))
        m_e = (y_q == cmap['EMT']) & np.isfinite(Cq[:, i_d])
        m_f = (y_q == cmap['FAU']) & np.isfinite(Cq[:, i_d])
        P('    有值真实 EMT %d 行中位 Δd/R = %s | 有值真实 FAU %d 行 = %s'
          ' (18-冠-6 基准 %+.3f)'
          % (int(m_e.sum()), ('%+.3f' % float(np.median(Cq[m_e, i_d]))) if m_e.any() else 'n/a',
             int(m_f.sum()), ('%+.3f' % float(np.median(Cq[m_f, i_d]))) if m_f.any() else 'n/a',
             6.703935 / R_EMT - 6.703935 / R_FAU))
        P('')

    # ---- 口径 B: 论文对标 (全行三元归并; 冻结统计全库 X 直预测) ----
    P('[口径 B 论文对标 | 三元归并 EMT/FAU/Others]')
    quiz_all = df['source_table'].isin(FAU_EMT_TABLES + EMT2026_TABLES).values
    acc_b, acc_h = float('nan'), float('nan')
    for spl_name, split_col, model in (('time_2018', 'split_time_2018', b_t),
                                       ('heldout', 'split_heldout', b_h)):
        spm = df[['route_id']].merge(sp[['route_id', split_col]], on='route_id', how='left')
        qm = quiz_all & (spm[split_col].values == 'test')
        if not qm.any():
            continue
        pq = model.predict(X[qm], num_iteration=best)
        pred_cls = np.array(classes)[pq.argmax(axis=1)]
        true_tri = np.array([tri_fold(v) for v in df['framework_code'].astype(str).values[qm]])
        pred_tri = np.array([tri_fold(v) for v in pred_cls])
        acc = float(np.mean(true_tri == pred_tri))
        if spl_name == 'time_2018':
            acc_b = acc
        else:
            acc_h = acc
        v0a = V0['tri_time'] if spl_name == 'time_2018' else V0['tri_held']
        P('  %s: 全行 %d, 归并 accuracy = %.4f (v0 %.4f %+.4f) — 对标 76.0%% %s'
          % (spl_name, int(qm.sum()), acc, v0a, acc - v0a,
             '达标' if acc >= ANCHOR_EMT else '未达'))
        if spl_name == 'time_2018':
            acc_b = acc
            conf = pd.crosstab(pd.Series(true_tri, name='真实'), pd.Series(pred_tri, name='预测'))
            P('  三元混淆 (time_2018):')
            for r in conf.index:
                P('    %-8s -> %s' % (r, {c: int(conf.loc[r, c]) for c in conf.columns}))
    P('')

    # ---- predict_purity v1 演示 (冻结变换 + T; 子集零漂移断言) ----
    P('predict_purity v1 演示 (time 模型 + 冻结统计 + T=%.4f, 同 v0 三行):' % T)
    demo_m = np.where(is_quiz & te_t & (y_all == 'EMT'))[0][:3]
    pq_d = proba_with_T(b_t.predict(Xm[demo_m], num_iteration=best), T)
    scorer = PurityScorerV1(b_t, classes, best, frozen, osda, desc_cols, T)
    out_s = scorer.predict_purity(df.loc[dom].iloc[demo_m], 'EMT')
    assert np.allclose(out_s['target_p'], pq_d[:, cmap['EMT']], atol=1e-9), \
        '冻结 scorer 子集预测与批量不一致 (漂移未修复)'
    P('  子集零漂移断言 PASS (scorer 子集预测 == 批量预测, v0 该场景会重 fit 漂移)')
    for k, mi in enumerate(demo_m):
        order = np.argsort(-pq_d[k])
        comp = {classes[i]: round(float(pq_d[k][i]), 4) for i in order[:4]
                if classes[i] != 'EMT'}
        P('  route %s (label1=%s): target_p=%.4f, competing=%s'
          % (m['route_id'].values[mi], m['label1'].values[mi],
             pq_d[k][cmap['EMT']], dict(list(comp.items())[:3])))
    P('')

    # ---- 逐行预测落盘 (random 模型全行 + time 模型考题行) ----
    pa = b_r.predict(Xm, num_iteration=best)
    out = pd.DataFrame({
        'route_id': m['route_id'].values, 'label1': m['label1'].values,
        'fw_true': [classes[v] for v in y],
        'fw_pred': [classes[v] for v in pa.argmax(axis=1)],
        'p_max': pa.max(axis=1),
        'p_emt': pa[:, cmap['EMT']] if 'EMT' in cmap else np.full(len(y), np.nan),
        'source_table': m['source_table'].values,
        'split_random': sr, 'split_heldout': sh, 'split_time_2018': st})
    out['p_emt_cal'] = proba_with_T(pa, T)[:, cmap['EMT']] if 'EMT' in cmap else np.nan
    out['quiz'] = is_quiz
    q_te = is_quiz[te_t]
    if q_te.any():
        y_te = y[te_t]
        ptT = proba_with_T(pt_t, T)
        qt = pd.DataFrame({
            'route_id': m['route_id'].values[te_t][q_te],
            'label1': m['label1'].values[te_t][q_te],
            'fw_true': [classes[v] for v in y_te[q_te]],
            'fw_pred': [classes[v] for v in pt_t[q_te].argmax(axis=1)],
            'p_max': pt_t[q_te].max(axis=1),
            'p_emt': pt_t[q_te][:, cmap['EMT']],
            'p_emt_cal': ptT[q_te][:, cmap['EMT']],
            'source_table': m['source_table'].values[te_t][q_te], 'model': 'time'})
        qt['split_random'] = sr[te_t][q_te]; qt['split_heldout'] = sh[te_t][q_te]
        qt['split_time_2018'] = st[te_t][q_te]; qt['quiz'] = True
        out = pd.concat([out, qt], ignore_index=True)
    out.to_parquet(PRED, engine='pyarrow', compression='zstd', index=False)

    # ---- 收口声明 + manifest ----
    P('=' * 68)
    P('收口声明:')
    P('  1. L3 v1 vs v0 (唯一自变量 = COMP 8 维 + 温度校准 + 冻结 scaler):')
    P('     EMT top-1 (time) %.4f -> %.4f; 三元 time %.4f -> %.4f; 三元 heldout %.4f -> %.4f'
      % (V0['top1_emt_time'], acc_a_emt, V0['tri_time'], acc_b,
         V0['tri_held'], acc_h))
    P('  2. 温度校准: T* = %.4f (random val 冻结, argmax 不变——分类读数不受 T 影响,'
      ' 改善的是 ECE/NLL 与概率口径)' % T)
    P('  3. 冻结 scaler: 498 维统计落盘 %s; PurityScorerV1 子集零漂移断言 PASS' %
      os.path.basename(FROZEN))
    P('  4. 局限登记: COMP 的 R 为 EMT/FAU 二骨架冻结常数 (非全骨架泛化孔径特征);'
      ' Others->FAU 92 行的域外吸收问题不属 COMP 攻击面; T 于 random val 拟合'
      ' (time 侧 2026 文献分布差异下外推)')
    P('产出: %s (%d 行)' % (PRED, len(out)))

    manifest = {
        'task': 'WP2/L3 纯度建模 v1 (EMT-FAU 竞相 d/R + 温度校准 + 冻结 scaler)',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'git_head': git_head(), 'code_md5': md5(os.path.abspath(__file__)),
        'data_version': 'zeobench_db_v1_8 + splits_v1_8 (Tier A 去重)',
        'domain': 'label1 in {positive, relative_neg} & zeo_desc_key notna (%d 行 %d 类)'
                  % (len(y), len(classes)),
        'features': 'A3 490 (冻结) + COMP 8 = 498 维; R_EMT=%.2f/R_FAU=%.2f Å (IZC a/b 向'
                    ' 最紧自由球, 冻结常数)' % (R_EMT, R_FAU),
        'frozen_stats': {'path': os.path.basename(FROZEN), 'md5': md5(FROZEN),
                         'rows': int(len(frozen))},
        'temperature': {'T': T, 'fit': 'random val NLL 最小, 三模型共用冻结',
                        'val_nll': [nll_base, nll_T]},
        'model': 'LGBM multiclass %s' % json.dumps(params, ensure_ascii=False),
        'protocol': 'random val 早停@%d; heldout/time 固定轮锚定' % best,
        'anchor': {'EMT 三元有效准确率': ANCHOR_EMT, 'v0 三元 time': V0['tri_time']},
        'quiz': {'incr_fau_emt': FAU_EMT_TABLES, 'incr_emt2026': EMT2026_TABLES,
                 '主切分': 'time_2018', 'v1 三元读数': acc_b}}
    with open(MF, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    P('manifest: %s' % MF)
    P('总用时 %.1f min' % ((time.time() - t_start) / 60))
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log) + '\n')


if __name__ == '__main__':
    main()
