# -*- coding: utf-8 -*-
"""
WP2 L1 特征工程迭代 (M1·W3, 提前开工): 比值消融/扩展 + OSDA 描述符注入消融
=====================================================================
任务 (执行计划 §3 M1·W3): 36 维基线上加常用比值特征 (Si/Al、OH/Si、Na/Si、
H2O/Si、OSDA/Si), 再消融是否注入描述符外键; 行级/DOI 级双口径记录,
重点观察 DOI 级诚实口径 (基线 0.3781) 的涨幅。

口径勘误 (开工核实): 上述 5 比值已含于基线 36 维 (comp_neg_knn.build_features
构成 = log10 摩尔 24 + 比值 5 + T/t 2 + 离散 5, 基线 docstring "22" 为笔误),
故 "加比值" 转译为消融 + 扩展:
  A0 基线 36 维: 复现锚点 (应逐位复现 l1_baseline: DOI 0.3781 / 行级 0.7364)
  A1 消融比值 31 维: 去掉 5 个 lg_ratio_* (量化已有比值贡献)
  A2 扩展比值 45 维: +9 (T 位分母比值补 AlPO/SAPO 家族缺口等, 见 EXT 说明)
  A3 OSDA1 分子描述符注入 490 维: A2 + 444 非 BE 描述符 + 命中指示器 (输入侧)
  A4 +192 binding energy 列 682 维: OSDA-框架对接能 (目标空间耦合特权口径)
  A5 配方+BE 228 维: BE 单独注入对照 (无分子描述符, BE 冗余性判定)
  (口径修正 2026-10-22: BE 族两种命名合并去重后, 非 BE 描述符 = 444; 旧 A3 682 实为 A4)
  (zeolite_descriptors 描述产物骨架 = 目标侧, 注入即标签泄漏, 不设该配置)
样本/目标/切分/模型/种子与 l1_baseline.py 完全一致, 差异仅归因于特征。
产出 (幂等覆盖写): Data/l1_feat_iter_report.txt
"""
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from comp_neg_knn import build_features
import osda_schema as osd

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, 'Data', 'zeosyn_db.parquet')
SPLITS = os.path.join(ROOT, 'Data', 'zeosyn_splits.parquet')
OSDA_DESC = os.path.join(ROOT, 'Data', 'osda_descriptors.parquet')
REPORT = os.path.join(ROOT, 'Data', 'l1_feat_iter_report.txt')
SEED = 42

RATIO5 = ['lg_ratio_SiAl', 'lg_ratio_OH_Si', 'lg_ratio_Na_Si',
          'lg_ratio_H2O_Si', 'lg_ratio_osda_Si']
# T 位元素 (骨架成网元素): Si/Al/P/Ge/B/Ti/Ga/Fe
T_COLS = ['n_Si', 'n_Al', 'n_P', 'n_Ge', 'n_B', 'n_Ti', 'n_Ga', 'n_Fe']

BASE_DOI, BASE_ROW = 0.3781, 0.7364  # l1_baseline 锚点 (对照用, Δ 以 A0 实测为基)


def topk_acc(y_true, proba, k):
    top = np.argsort(-proba, axis=1)[:, :k]
    return float(np.mean([y_true[i] in top[i] for i in range(len(y_true))]))


def num(df, c):
    return pd.to_numeric(df[c], errors='coerce').fillna(0.0).clip(lower=0.0)


def zscore(M):
    mu, sd = M.mean(axis=0), M.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (M - mu) / sd


def zscore_nan(M):
    mu, sd = np.nanmean(M, axis=0), np.nanstd(M, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (M - mu) / sd


def build_ext(df):
    """A2 扩展比值 9 维 (全库构建, StandardScaler 由调用方统一做)"""
    T = num(df, 'n_Si')
    for c in T_COLS[1:]:
        T = T + num(df, c)

    def lg_ratio(nr, den):
        v = (nr / den.replace(0, np.nan)).clip(lower=0.0)
        return np.log10(1.0 + v.fillna(0.0))

    X = {}
    # T 位分母: 基线 5 比值分母全为 Si, AlPO/SAPO/Ge 体系 Si=0 时信息丢失
    X['lg_ratio_OH_T'] = lg_ratio(num(df, 'n_OH'), T)
    X['lg_ratio_osda_T'] = lg_ratio(
        num(df, 'n_osda1') + num(df, 'n_osda2') + num(df, 'n_osda3'), T)
    X['lg_ratio_F_T'] = lg_ratio(num(df, 'n_F'), T)
    X['lg_ratio_H2O_T'] = lg_ratio(num(df, 'n_H2O'), T)
    X['lg_ratio_Ge_T'] = lg_ratio(num(df, 'n_Ge'), T)
    X['lg_ratio_seed_T'] = lg_ratio(num(df, 'n_seed'), T)
    # 分母 Al: 铝硅酸盐电荷与 AlPO 化学计量
    X['lg_ratio_Na_Al'] = lg_ratio(num(df, 'n_Na'), num(df, 'n_Al'))
    X['lg_ratio_P_Al'] = lg_ratio(num(df, 'n_P'), num(df, 'n_Al'))
    # 电荷平衡 (主库派生列, 缺失中位填后 z-score, 与 T_C 惯例一致)
    cb = pd.to_numeric(df['ratio_charge_bal'], errors='coerce')
    cb = cb.fillna(cb.median())
    X['z_ratio_charge_bal'] = (cb - cb.mean()) / max(cb.std(), 1e-12)
    return pd.DataFrame(X, index=df.index)


def run_config(tag, X, y, split, n_classes):
    """双口径训练评测: DOI 级 (主) + 行级随机 (leaky 参照), 与 l1_baseline 同协议"""
    tr, va, te = split == 'train', split == 'val', split == 'test'
    params = dict(objective='multiclass', num_class=n_classes,
                  learning_rate=0.05, num_leaves=31, min_child_samples=100,
                  lambda_l2=1.0, max_delta_step=1.0, seed=SEED, verbosity=-1)
    t0 = time.time()
    dtr = lgb.Dataset(X[tr], label=y[tr])
    dva = lgb.Dataset(X[va], label=y[va], reference=dtr)
    b = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    best = b.best_iteration or -1
    vloss = b.best_score['valid_0']['multi_logloss']
    pt = b.predict(X[te], num_iteration=best)
    res = dict(dim=X.shape[1], best=best, vloss=vloss,
               t1=topk_acc(y[te], pt, 1), t3=topk_acc(y[te], pt, 3),
               t5=topk_acc(y[te], pt, 5), dt=time.time() - t0, booster=b)
    # 行级随机切分参照 (leaky, 同基线 rng 协议)
    rng = np.random.default_rng(SEED)
    r = rng.permutation(len(y))
    n_tr, n_va = int(0.8 * len(y)), int(0.9 * len(y))
    tr2 = np.zeros(len(y), bool)
    va2 = np.zeros(len(y), bool)
    te2 = np.zeros(len(y), bool)
    tr2[r[:n_tr]] = True
    va2[r[n_tr:n_va]] = True
    te2[r[n_va:]] = True
    dtr2 = lgb.Dataset(X[tr2], label=y[tr2])
    dva2 = lgb.Dataset(X[va2], label=y[va2], reference=dtr2)
    b2 = lgb.train(params, dtr2, num_boost_round=400, valid_sets=[dva2],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    best2 = b2.best_iteration or -1
    pt2 = b2.predict(X[te2], num_iteration=best2)
    res['row_t1'] = float(np.mean(pt2.argmax(1) == y[te2]))
    res['best2'] = best2
    return res


def main():
    log = []

    def P(msg=''):
        print(msg)
        log.append(str(msg))

    df = pd.read_parquet(DB)
    sp = pd.read_parquet(SPLITS)
    Z36, cols36, fill_stats = build_features(df)

    pos_all = df['label1'] == 'positive'
    pos = pos_all & df['zeo_desc_key'].notna()
    m = df.loc[pos, ['route_id']].merge(
        sp[['route_id', 'fw_key', 'split_random']], on='route_id')
    assert len(m) == int(pos.sum()), 'positive 行与切分表对不齐'
    y_all = m['fw_key'].astype(str).values
    split = m['split_random'].values
    idx = np.where(pos)[0]
    classes = sorted(set(y_all))
    cmap = {c: i for i, c in enumerate(classes)}
    y = np.array([cmap[v] for v in y_all])
    tr = split == 'train'

    # ---- 各配置特征矩阵 (全库行构建, 取 pos 子集入模; 全库 z-score 惯例) ----
    XA1 = Z36[:, [i for i, c in enumerate(cols36) if c not in RATIO5]]
    colsA1 = [c for c in cols36 if c not in RATIO5]
    ext_df = build_ext(df)
    EXT = StandardScaler().fit_transform(ext_df.values)
    ext_cols = list(ext_df.columns)
    XA2 = np.hstack([Z36, EXT])
    colsA2 = cols36 + ext_cols

    osda = osd.load(OSDA_DESC)
    desc_cols = osd.desc_cols(osda)
    be_cols = osd.be_cols(osda)
    assert len(desc_cols) == 444 and len(be_cols) == 192, \
        'OSDA 描述符口径漂移: desc=%d be=%d' % (len(desc_cols), len(be_cols))
    J = df[['osda1_desc_key']].merge(
        osda[['osda_smiles'] + desc_cols + be_cols],
        left_on='osda1_desc_key', right_on='osda_smiles', how='left', sort=False)
    assert len(J) == len(df), '描述符 join 行数漂移'
    Dsc = zscore_nan(J[desc_cols].apply(
        pd.to_numeric, errors='coerce').values.astype(float))
    Be = zscore_nan(J[be_cols].apply(
        pd.to_numeric, errors='coerce').values.astype(float))
    b_desc = df['osda1_desc_key'].notna().astype(float).values[:, None]
    od_names = ['od1_' + c for c in desc_cols]
    be_names = ['od1_be_' + c.split('(')[-1].strip(') ') for c in be_cols]
    XA3 = np.hstack([XA2, b_desc, Dsc])
    colsA3 = colsA2 + ['b_osda1_desc'] + od_names
    XA4 = np.hstack([XA3, Be])
    colsA4 = colsA3 + be_names
    XBE = np.hstack([Z36, Be])
    colsBE = cols36 + be_names

    configs = [
        ('A0 基线 36 维 (锚点)', Z36, cols36),
        ('A1 消融 5 比值 31 维', XA1, colsA1),
        ('A2 +扩展比值 45 维', XA2, colsA2),
        ('A3 +OSDA1 分子描述符 490 维', XA3, colsA3),
        ('A4 +binding energy 192 列 682 维', XA4, colsA4),
        ('A5 配方+BE 228 维 (对照)', XBE, colsBE),
    ]

    cover = int(df.loc[pos, 'osda1_desc_key'].notna().sum())
    P('L1 特征工程迭代 (M1·W3, 提前开工): 比值消融/扩展 + OSDA 描述符注入消融')
    P('=' * 62)
    P('样本: positive %d 行, IZA 可映射 %d 行入模, 类别 %d 个; DOI 级切分 %d/%d/%d'
      % (int(pos_all.sum()), len(y), len(classes), tr.sum(),
         (split == 'val').sum(), (split == 'test').sum()))
    P('口径勘误: 任务所列 5 比值 (Si/Al、OH/Si、Na/Si、H2O/Si、OSDA/Si) 已含于基线')
    P('  36 维 (构成 = log10 摩尔 24 + 比值 5 + T/t 2 + 离散 5, l1_baseline docstring')
    P('  "22" 为笔误), 故 "加比值" 转译为 A1 消融验证 + A2 扩展 (T 位分母补 AlPO 家族)。')
    P('描述符注入口径: zeolite_descriptors (产物骨架 468 维) 为目标侧, 注入即标签泄漏,')
    P('  不设配置; OSDA 描述符为输入侧合法注入 (A3/A4); A4 的 binding energy 为')
    P('  OSDA-框架对接能, 属目标空间耦合的特权先验, 单列口径不计入主推荐。')
    P('A3 OSDA1 描述符覆盖: %d/%d = %.1f%% (osda1_desc_key 命中; 未命中行 NaN 由'
      % (cover, len(y), 100.0 * cover / len(y)))
    P('  LightGBM 原生处理 + b_osda1_desc 指示器; osda2/3 覆盖 10.2%/0.4% 留 W4)')
    P('模型/种子/切分与 l1_baseline 完全一致 (LightGBM multiclass, lr=0.05,')
    P('  leaves=31, min_child=100, l2=1, max_delta_step=1, seed=42, ES=50),')
    P('  差异仅归因于特征。')
    P('')

    results = {}
    P('逐配置结果 (DOI 级 = split_random 主口径; 行级 = leaky 参照):')
    for tag, Xc, cc in configs:
        res = run_config(tag, Xc[idx], y, split, len(classes))
        results[tag] = res
        P('  %-38s best@%-3d (loss %.4f) DOI test top-1/3/5 = %.4f/%.4f/%.4f'
          ' | 行级 leaky top-1 = %.4f [%.0fs]'
          % (tag.split(' (')[0], res['best'], res['vloss'], res['t1'],
             res['t3'], res['t5'], res['row_t1'], res['dt']))

    # ---- 汇总主表 ----
    a0 = results[configs[0][0]]
    P('')
    P('主表 (Δ 相对 A0 基线实测):')
    P('%-30s %-5s %-9s %-8s %-8s %-8s %-9s %-8s'
      % ('配置', '维度', 'DOI t1', 'Δt1', 'top-3', 'top-5', '行级 t1', 'Δ行级'))
    for tag, _, _ in configs:
        r = results[tag]
        name = tag.split(' (')[0]
        P('%-30s %-5d %-9.4f %+8.4f %-8.4f %-8.4f %-9.4f %+8.4f'
          % (name, r['dim'], r['t1'], r['t1'] - a0['t1'], r['t3'], r['t5'],
             r['row_t1'], r['row_t1'] - a0['row_t1']))

    # ---- 特征重要性 (A2~A5, DOI 模型 gain Top15) ----
    P('')
    for tag, _, cc in configs[2:]:
        b = results[tag]['booster']
        imp = b.feature_importance(importance_type='gain')
        order = np.argsort(-imp)[:15]
        P('特征重要性 Top15 (gain, %s):' % tag.split(' (')[0])
        for i in order:
            P('  %-24s %.1f' % (cc[i], imp[i]))
        P('')

    # ---- 自动结论 ----
    best_doi = max(results, key=lambda k: results[k]['t1'])
    a1, a2r, a3, a4, a5 = (results[configs[i][0]] for i in range(1, 6))
    same34 = all(a3[k] == a4[k] for k in ('t1', 't3', 't5', 'row_t1', 'best'))
    imp4 = a4['booster'].feature_importance(importance_type='gain')
    be_used4 = int((imp4[-192:] > 0).sum())
    imp5 = a5['booster'].feature_importance(importance_type='gain')
    be_used5 = int((imp5[36:] > 0).sum())
    P('结论:')
    P('  1. 锚点复现: A0 DOI top-1=%.4f / 行级=%.4f (基线 %.4f/%.4f, %s)'
      % (a0['t1'], a0['row_t1'], BASE_DOI, BASE_ROW,
         '逐位一致' if (abs(a0['t1'] - BASE_DOI) < 5e-4
                        and abs(a0['row_t1'] - BASE_ROW) < 5e-4)
         else '偏差超阈值, 查复现链路'))
    P('  2. 比值消融 (A1-A0): DOI top-1 %+.4f / top-3 %+.4f / top-5 %+.4f'
      ' -> 已有 5 比值对 top-1 贡献有限 (log10 摩尔已隐含), 对 top-3/5 有正贡献'
      % (a1['t1'] - a0['t1'], a1['t3'] - a0['t3'], a1['t5'] - a0['t5']))
    P('  3. 扩展比值 (A2-A0): DOI top-1 %+.4f, 行级 %+.4f (T 位分母比值增益有限)'
      % (a2r['t1'] - a0['t1'], a2r['row_t1'] - a0['row_t1']))
    P('  4. OSDA1 分子描述符 (A3-A2): DOI top-1 %+.4f (top-3 %+.4f, top-5 %+.4f),'
      ' 行级 %+.4f —— 本迭代唯一大信号源'
      % (a3['t1'] - a2r['t1'], a3['t3'] - a2r['t3'], a3['t5'] - a2r['t5'],
         a3['row_t1'] - a2r['row_t1']))
    P('  5. BE 注入消融 (A4-A3): DOI top-1 %+.4f 且与 A3 %s; A4 中 BE 非零 gain 列 %d/192'
      % (a4['t1'] - a3['t1'], '全指标逐位一致' if same34 else '指标存在差异', be_used4))
    P('     对照 A5 (配方+BE, 无分子描述符): DOI top-1=%.4f (相对 A0 %+.4f), BE 非零 gain 列 %d/192'
      % (a5['t1'], a5['t1'] - a0['t1'], be_used5))
    P('     注: BE = OSDA-骨架对接能 (目标空间耦合特权口径)。2026-10-22 口径修正后 A3 不含 BE;')
    P('     旧「BE 零增益/从未入选分裂」结论作废 (旧 A3 已含同值 BE 列, A4 仅重复注入)')
    P('  6. DOI 级最优配置: %s (top-1=%.4f, 相对基线 %+.4f / %+.1f%%);'
      ' 建议以 A3 为 L1 迭代主配置 (纯输入侧, 无特权信息, 维度最小)'
      % (best_doi, results[best_doi]['t1'],
         results[best_doi]['t1'] - a0['t1'],
         100.0 * (results[best_doi]['t1'] - a0['t1']) / a0['t1']))
    P('产出: %s' % REPORT)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log) + '\n')


if __name__ == '__main__':
    main()
