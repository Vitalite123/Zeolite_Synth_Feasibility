# -*- coding: utf-8 -*-
"""
WP2 L1 基线收口 v1 (M1·W4): 主配置 A3 490 维 × 三切分定版
==========================================================
配置依据 (l1_feat_iter.py, 2026-10-08; 2026-10-22 BE 列名缺陷修正后):
  A3 = 36 维配方/条件 + 9 扩展比值 + b_osda1_desc 命中指示器
  + 444 OSDA1 分子描述符 (输入侧; binding energy 为目标耦合特权口径, 剔除后不采用;
  zeolite_descriptors 目标侧不设配置)
三切分 (M3 切分协议, 切分单位 = (框架,文献) 组, 审计 PASS):
  split_random    主口径, DOI 级 80/10/10, val 早停
  split_heldout   留出骨架 (LTL/SAS/ITG/RTH/SZR/LTA) 整组进测试, 训练无
                  val, 轮数锚定主口径 best; 测试按骨架可见性分层
  split_time_2018 <=2018 训练 / >=2019 测试 (无年份 excluded 行剔除),
                  轮数锚定主口径 best
产出 (幂等覆盖写):
  Data/l1_baseline_v1_report.txt    收口报告
  Data/l1_baseline_v1_pred.parquet  逐行预测 (主口径 random 模型全行)
"""
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from comp_neg_knn import build_features
from l1_feat_iter import build_ext, zscore_nan
import osda_schema as osd

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, 'Data', 'zeosyn_db.parquet')
SPLITS = os.path.join(ROOT, 'Data', 'zeosyn_splits.parquet')
OSDA_DESC = os.path.join(ROOT, 'Data', 'osda_descriptors.parquet')
REPORT = os.path.join(ROOT, 'Data', 'l1_baseline_v1_report.txt')
PRED = os.path.join(ROOT, 'Data', 'l1_baseline_v1_pred.parquet')
SEED = 42
ANCHOR_DOI, ANCHOR_ROW = 0.5638, 0.8458  # A3 490 维 (非 BE) 重训锚点, 源 l1_feat_iter 2026-10-22
HELDOUT_FW = ['LTL', 'SAS', 'ITG', 'RTH', 'SZR', 'LTA']


def topk_acc(y_true, proba, k):
    top = np.argsort(-proba, axis=1)[:, :k]
    return float(np.mean([y_true[i] in top[i] for i in range(len(y_true))]))


def build_a3(df):
    """A3 490 维特征 (36 + 9 + 1 + 444 非 BE 描述符; BE 列在读取层归一化后剔除)"""
    Z36, cols36, _ = build_features(df)
    ext_df = build_ext(df)
    EXT = StandardScaler().fit_transform(ext_df.values)
    osda = osd.load(OSDA_DESC)
    desc_cols = osd.desc_cols(osda)
    assert len(desc_cols) == 444, 'OSDA 非 BE 描述符口径漂移: %d' % len(desc_cols)
    J = df[['osda1_desc_key']].merge(
        osda[['osda_smiles'] + desc_cols],
        left_on='osda1_desc_key', right_on='osda_smiles',
        how='left', sort=False)
    assert len(J) == len(df), '描述符 join 行数漂移'
    Dsc = zscore_nan(J[desc_cols].apply(
        pd.to_numeric, errors='coerce').values.astype(float))
    b_desc = df['osda1_desc_key'].notna().astype(float).values[:, None]
    X = np.hstack([Z36, EXT, b_desc, Dsc])
    cols = (cols36 + list(ext_df.columns) + ['b_osda1_desc']
            + ['od1_' + c for c in desc_cols])
    return X, cols


def main():
    log = []

    def P(msg=''):
        print(msg)
        log.append(str(msg))

    df = pd.read_parquet(DB)
    sp = pd.read_parquet(SPLITS)
    X, cols = build_a3(df)
    assert X.shape == (len(df), 490), 'A3 维度漂移'

    pos = (df['label1'] == 'positive') & df['zeo_desc_key'].notna()
    idx = np.where(pos)[0]
    m = df.loc[pos, ['route_id']].merge(
        sp[['route_id', 'fw_key', 'split_random', 'split_heldout',
            'split_time_2018']], on='route_id')
    assert len(m) == len(idx), 'positive 行与切分表对不齐'
    y_all = m['fw_key'].astype(str).values
    sr = m['split_random'].values
    sh = m['split_heldout'].values
    st = m['split_time_2018'].values
    classes = sorted(set(y_all))
    cmap = {c: i for i, c in enumerate(classes)}
    y = np.array([cmap[v] for v in y_all])
    Xm = X[idx]

    params = dict(objective='multiclass', num_class=len(classes),
                  learning_rate=0.05, num_leaves=31, min_child_samples=100,
                  lambda_l2=1.0, max_delta_step=1.0, seed=SEED, verbosity=-1)

    P('L1 基线收口 v1 (M1·W4): 主配置 A3 490 维 × 三切分定版')
    P('=' * 62)
    P('样本: positive IZA 可映射 %d 行, 类别 %d 个; A3 特征构成 = 36 配方/条件'
      '+ 9 扩展比值 + 1 命中指示器 + 444 OSDA1 分子描述符 (输入侧)' % (len(y), len(classes)))
    P('配置依据: l1_feat_iter.py (2026-10-08) A3; 2026-10-22 BE 列名缺陷修正后 A3 不含 BE'
      ' (zeolite_descriptors 目标侧不设配置)')
    P('')

    # ---- 切分一: split_random 主口径 (val 早停) ----
    tr, va, te = sr == 'train', sr == 'val', sr == 'test'
    t0 = time.time()
    dtr = lgb.Dataset(Xm[tr], label=y[tr])
    dva = lgb.Dataset(Xm[va], label=y[va], reference=dtr)
    b_r = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
    best = b_r.best_iteration or -1
    pt_r = b_r.predict(Xm[te], num_iteration=best)
    t1_r = topk_acc(y[te], pt_r, 1)
    P('split_random (主口径, DOI 级 80/10/10, 早停@%d, %.0fs):'
      % (best, time.time() - t0))
    P('  val  top-1/3/5 = %.4f / %.4f / %.4f'
      % (topk_acc(y[va], b_r.predict(Xm[va], num_iteration=best), 1),
         topk_acc(y[va], b_r.predict(Xm[va], num_iteration=best), 3),
         topk_acc(y[va], b_r.predict(Xm[va], num_iteration=best), 5)))
    P('  test top-1/3/5 = %.4f / %.4f / %.4f'
      % (t1_r, topk_acc(y[te], pt_r, 3), topk_acc(y[te], pt_r, 5)))
    ok_doi = abs(t1_r - ANCHOR_DOI) < 5e-4
    P('  锚点核对: DOI top-1=%.4f vs 特征工程迭代 %.4f -> %s'
      % (t1_r, ANCHOR_DOI, '逐位一致' if ok_doi else '偏差超阈值, 查构造链路'))

    # 类支撑分层 (主口径 test)
    sup = np.bincount(y[tr], minlength=len(classes))
    hi = sup[y[te]] >= 30
    P('  分层: 训练支撑>=30 的测试行 %d top-1=%.4f; 低支撑 %d 行 top-1=%.4f'
      % (int(hi.sum()), topk_acc(y[te][hi], pt_r[hi], 1),
         int((~hi).sum()), topk_acc(y[te][~hi], pt_r[~hi], 1)))

    # ---- 行级 leaky 参照 (同 l1_baseline/l1_feat_iter rng 协议) ----
    rng = np.random.default_rng(SEED)
    r = rng.permutation(len(y))
    n_tr, n_va = int(0.8 * len(y)), int(0.9 * len(y))
    tr2 = np.zeros(len(y), bool)
    va2 = np.zeros(len(y), bool)
    te2 = np.zeros(len(y), bool)
    tr2[r[:n_tr]] = True
    va2[r[n_tr:n_va]] = True
    te2[r[n_va:]] = True
    dtr2 = lgb.Dataset(Xm[tr2], label=y[tr2])
    dva2 = lgb.Dataset(Xm[va2], label=y[va2], reference=dtr2)
    b2 = lgb.train(params, dtr2, num_boost_round=400, valid_sets=[dva2],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    row_t1 = float(np.mean(
        b2.predict(Xm[te2], num_iteration=b2.best_iteration or -1).argmax(1)
        == y[te2]))
    P('  行级 leaky 参照: top-1=%.4f vs 特征工程迭代 %.4f -> %s'
      % (row_t1, ANCHOR_ROW,
         '逐位一致' if abs(row_t1 - ANCHOR_ROW) < 5e-4 else '偏差超阈值'))
    P('')

    # ---- 切分二: split_heldout (留出骨架整组进测试; 轮数锚定主口径 best) ----
    tr_h = sh == 'train'
    te_h = sh == 'test'
    t0 = time.time()
    dtr_h = lgb.Dataset(Xm[tr_h], label=y[tr_h])
    b_h = lgb.train(params, dtr_h, num_boost_round=best)
    pt_h = b_h.predict(Xm[te_h])
    ho_cls = np.array([cmap[f] for f in HELDOUT_FW if f in cmap])
    is_ho = np.isin(y[te_h], ho_cls)
    P('split_heldout (留出骨架 %s 整组进测试; 固定 %d 轮锚定主口径, %.0fs):'
      % ('/'.join(HELDOUT_FW), best, time.time() - t0))
    P('  test %d 行 top-1/3/5 = %.4f / %.4f / %.4f'
      % (int(te_h.sum()), topk_acc(y[te_h], pt_h, 1),
         topk_acc(y[te_h], pt_h, 3), topk_acc(y[te_h], pt_h, 5)))
    if is_ho.any():
        P('  ├ 留出骨架行 (训练不可见, 计错误) %d 行 top-1=%.4f'
          % (int(is_ho.sum()), topk_acc(y[te_h][is_ho], pt_h[is_ho], 1)))
    if (~is_ho).any():
        P('  └ 训练内骨架行 (新文献同骨架泛化) %d 行 top-1=%.4f'
          % (int((~is_ho).sum()), topk_acc(y[te_h][~is_ho], pt_h[~is_ho], 1)))
    P('')

    # ---- 切分三: split_time_2018 (<=2018 训练 / >=2019 测试) ----
    tr_t = st == 'train'
    te_t = st == 'test'
    t0 = time.time()
    dtr_t = lgb.Dataset(Xm[tr_t], label=y[tr_t])
    b_t = lgb.train(params, dtr_t, num_boost_round=best)
    pt_t = b_t.predict(Xm[te_t])
    P('split_time_2018 (<=2018 训练 %d / >=2019 测试 %d; excluded 无年份 %d 行'
      '剔除; 固定 %d 轮, %.0fs):'
      % (int(tr_t.sum()), int(te_t.sum()), int((st == 'excluded').sum()),
         best, time.time() - t0))
    P('  test top-1/3/5 = %.4f / %.4f / %.4f'
      % (topk_acc(y[te_t], pt_t, 1), topk_acc(y[te_t], pt_t, 3),
         topk_acc(y[te_t], pt_t, 5)))
    # 时间测试集内训练外骨架行
    unseen_t = np.bincount(y[tr_t], minlength=len(classes))
    unseen = unseen_t[y[te_t]] == 0
    if unseen.any():
        P('  其中训练外骨架行 %d (top-1=%.4f, 计错误), 训练内骨架行 %d top-1=%.4f'
          % (int(unseen.sum()), topk_acc(y[te_t][unseen], pt_t[unseen], 1),
             int((~unseen).sum()),
             topk_acc(y[te_t][~unseen], pt_t[~unseen], 1)))
    P('')

    # ---- 三切分主表 ----
    P('三切分主表 (A3 490 维, DOI 级防泄漏口径):')
    P('  切分            test 行   top-1    top-3    top-5')
    P('  random (主)     %-7d %-9.4f %-9.4f %.4f'
      % (int(te.sum()), t1_r, topk_acc(y[te], pt_r, 3),
         topk_acc(y[te], pt_r, 5)))
    P('  heldout         %-7d %-9.4f %-9.4f %.4f'
      % (int(te_h.sum()), topk_acc(y[te_h], pt_h, 1),
         topk_acc(y[te_h], pt_h, 3), topk_acc(y[te_h], pt_h, 5)))
    P('  time_2018       %-7d %-9.4f %-9.4f %.4f'
      % (int(te_t.sum()), topk_acc(y[te_t], pt_t, 1),
         topk_acc(y[te_t], pt_t, 3), topk_acc(y[te_t], pt_t, 5)))
    P('  行级 leaky 参照 (random, 对齐论文报告口径): top-1=%.4f' % row_t1)
    P('')

    # ---- 混淆 Top 与特征重要性 (主口径) ----
    pred_te = pt_r.argmax(axis=1)
    P('错判最多 Top10 (真类 -> 预测类, 主口径 test):')
    err = pd.DataFrame({'t': y[te], 'p': pred_te})
    err = err[err.t != err.p].value_counts(['t', 'p']).head(10)
    for (t, p), n in err.items():
        P('  %-8s -> %-8s %d' % (classes[t], classes[p], n))
    P('')
    imp = b_r.feature_importance(importance_type='gain')
    order = np.argsort(-imp)[:15]
    P('特征重要性 Top15 (gain, 主口径):')
    for i in order:
        P('  %-24s %.1f' % (cols[i], imp[i]))
    P('')

    # ---- 逐行预测落盘 (主口径 random 模型全行) ----
    pa = b_r.predict(Xm, num_iteration=best)
    out = pd.DataFrame({
        'route_id': m['route_id'].values,
        'fw_true': [classes[v] for v in y],
        'fw_pred': [classes[v] for v in pa.argmax(axis=1)],
        'p_max': pa.max(axis=1),
        'split_random': sr,
        'split_heldout': sh,
        'split_time_2018': st,
    })
    out.to_parquet(PRED, engine='pyarrow', compression='zstd', index=False)

    # ---- 收口声明 ----
    P('收口声明:')
    P('  1. L1 迭代历程: 36 维首信号 (DOI 0.378 / 行级 0.736) -> A3 490 维'
      ' (DOI %.3f / 行级 %.3f), 双口径登记进 M4 检查表 #6'
      % (t1_r, row_t1))
    P('  2. G2 对照: 行级 %.4f (36 维时 0.7364 已过 72%% 线, 主配置再 +%0.1f pp)'
      % (row_t1, 100 * (row_t1 - 0.7364)))
    P('  3. 三切分定位: random=in-distribution 新文献; heldout=跨拓扑外推'
      ' (留出骨架行计错误, top-1=%.4f); time_2018=时间外推 (top-1=%.4f)'
      % (topk_acc(y[te_h], pt_h, 1), topk_acc(y[te_t], pt_t, 1)))
    P('  4. 后续: 三模型 (LightGBM/XGBoost/CatBoost) × 三切分对比表 + TabPFN'
      ' 小数据对照留 M3·W4 (启动会决策 3 建议提前至 10 月下旬), 特征与协议'
      ' 以本脚本 A3 配置为准')
    P('产出: %s (%d 行), %s' % (PRED, len(out), REPORT))
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log) + '\n')


if __name__ == '__main__':
    main()
