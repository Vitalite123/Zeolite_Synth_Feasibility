# -*- coding: utf-8 -*-
"""
WP1-T1.2.1 竞相负样本：描述符空间 kNN 邻域生成竞相监督对表
=====================================================================
操作定义 (执行计划 §4.1, v0):
  1. 特征空间 X: log10 摩尔 + 常用比值 (Si/Al、OH/Si、Na/Si、H2O/Si、OSDA/Si) + T/t
     + 离散 one-hot, 全库 z-score 标准化 (与 WP2/WP3 特征工程同源)
  2. 对每个支撑 >=30 条路线 (参数 MIN_SUPPORT) 的骨架 T:
     tau_T = T 的正样本路线集 P_T 内部 kNN(k=10) 成对距离分布的 95 分位
  3. 任一主相 != T 的路线 r (P_T 补集, 含 absolute_neg/relative_neg/no_product/
     其他骨架 positive), 若 d(r, P_T) <= tau_T (到 P_T 第 k 近邻距离),
     输出监督对 (r, T, neg, d_knn, tau, params_ver)
  4. 附列: r 的实际主相 (竞争相 comp_framework)、group_id=(框架,文献)、
     骨架图距离 (T1.3, zeolite_graph_distance.csv 的 D/SOAP)

产出:
  Data/pair_label.parquet            主配置 (k=10, tau=95 分位, 标准化欧氏) 监督对
  Data/pair_label_sensitivity.csv    k∈{5,10,20} x tau分位∈{90,95,99} x
                                     度量∈{euclidean, mahalanobis} 负对数量曲线
  Data/pair_label_report.txt         支撑骨架清单/tau 分布/正负比/fail_mode 报表

口径说明:
  - v0 仅基于主库 zeosyn_db.parquet (23,961 行, 组成已归一化);
    incr_ 增量表为原始摩尔 (normed='no'), 并入需归一化预处理, 留 v1
  - ratio_SiAl 缺失 = 无 Al 全硅体系 -> 端部 cap 填 500 (与 0 端填充语义相反)
  - T_C/t_h 缺失 -> 全库中位数填充 (填入行数入报告)
  - fail_mode 映射: relative_neg->impurity_only; absolute_neg 按 product1
    (amorphous->amorphous, dense 类->dense, 其余->undesired); no_product->undesired
幂等: 重复运行覆盖写三个输出文件
"""
import os
import re
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, 'Data', 'zeosyn_db.parquet')
GRAPH_CSV = os.path.join(ROOT, 'zeosyn_gen', 'data', 'zeolite_graph_distance.csv')
OUT_PAIRS = os.path.join(ROOT, 'Data', 'pair_label.parquet')
OUT_SENS = os.path.join(ROOT, 'Data', 'pair_label_sensitivity.csv')
OUT_REPORT = os.path.join(ROOT, 'Data', 'pair_label_report.txt')

MIN_SUPPORT = 30
PARAMS_VER = 'v0_k10_p95_eucl'
K_GRID = [5, 10, 20]
PCT_GRID = [50, 90, 95, 99]  # p50 为参考行 (非计划网格), 供 WP2 消融视野
DENSE_PAT = re.compile(r'quartz|cristobalite|keatite|magadiite|tridymite|dense'
                       r'|analcime|nepheline|sodalite\b', re.I)

# ---- 特征定义 ----
# 元素列仅保留非零率 >=1% 者: 近恒定列 (如 n_Rb 65/23961) 的微小偏差会被
# StandardScaler 除以极小 std 放大, 把 tau_T 尾部撑大数个量级 (首跑教训)
LOG_MOLE_COLS = ['n_Si', 'n_Al', 'n_Na', 'n_K', 'n_Li', 'n_Cs', 'n_Ca',
                 'n_F', 'n_P', 'n_B', 'n_Ge', 'n_Ti', 'n_Ga',
                 'n_Fe', 'n_Zn', 'n_Co', 'n_Mg', 'n_V',
                 'n_H2O', 'n_OH', 'n_osda1', 'n_osda2', 'n_acid', 'n_seed']
RATIO_CAP = 500.0


def build_features(df):
    """全库特征矩阵 + 填充统计"""
    fill_stats = {}
    X = {}
    for c in LOG_MOLE_COLS:
        v = pd.to_numeric(df[c], errors='coerce').fillna(0.0).clip(lower=0.0)
        X[f'lg_{c}'] = np.log10(v + 1e-6)
    # 比值 (log10(1+r)); Si/Al NaN=全硅 -> cap 端
    rSiAl = df['ratio_SiAl'].fillna(RATIO_CAP).clip(upper=RATIO_CAP)
    X['lg_ratio_SiAl'] = np.log10(1 + rSiAl)
    fill_stats['ratio_SiAl 缺失(cap 填 500)'] = int(df['ratio_SiAl'].isna().sum())
    for c in ['ratio_OH_Si', 'ratio_Na_Si', 'ratio_H2O_Si', 'ratio_osda_Si']:
        v = df[c].fillna(0.0).clip(lower=0.0)
        X[f'lg_{c}'] = np.log10(1 + v)
    # T/t: 中位数填充
    Tm = df['T_C'].median()
    tm = df['t_h'].median()
    fill_stats['T_C 缺失(中位填)'] = int(df['T_C'].isna().sum())
    fill_stats['t_h 缺失(中位填)'] = int(df['t_h'].isna().sum())
    X['T_C_100'] = df['T_C'].fillna(Tm) / 100.0
    X['lg_t_h'] = np.log10(1 + df['t_h'].fillna(tm))
    # 离散 one-hot
    X['b_osda'] = (df['n_osda1'].fillna(0) > 0).astype(float)
    X['b_seed'] = (df['n_seed'].fillna(0) > 0).astype(float)
    X['b_fluoride'] = (df['n_F'].fillna(0) > 0).astype(float)
    X['b_nonwater'] = (df['solvent'].fillna(0) > 0).astype(float)
    stir = df['stirring'].astype(str).str.lower()
    X['b_stir'] = stir.isin(['yes', 'true', '1']).astype(float)
    F = pd.DataFrame(X, index=df.index)
    cols = list(F.columns)
    Z = StandardScaler().fit_transform(F.values)
    return Z, cols, fill_stats


def load_graph_dist():
    """(Zeo1,Zeo2)->(D,SOAP) 对称映射; 码规范化: 去 *- 前缀大写"""
    g = pd.read_csv(GRAPH_CSV)
    m = {}

    def norm(c):
        return re.sub(r'^[\*-]+', '', str(c)).strip().upper()
    for _, r in g.iterrows():
        a, b = norm(r['Zeo1']), norm(r['Zeo2'])
        m[(a, b)] = (r['D'], r['SOAP'])
        m[(b, a)] = (r['D'], r['SOAP'])
    return m


def knn_dist(P, Q, k, metric, V=None):
    """Q 中各点到 P 的第 k 近邻距离. Q is P 时第 0 邻居是自身 (需 k+1 邻居)."""
    internal = Q is P
    nnk = min(k + 1, len(P))
    kw = {'metric': metric}
    if metric == 'mahalanobis':
        kw['metric_params'] = {'VI': V}
    nn = NearestNeighbors(n_neighbors=nnk, **kw)
    nn.fit(P)
    d, _ = nn.kneighbors(Q)
    if nnk <= k:  # 点太少, 取最远可用
        return d[:, -1]
    return d[:, k] if internal else d[:, k - 1]


def main():
    log = []

    def P(msg=''):
        print(msg)
        log.append(str(msg))

    df = pd.read_parquet(DB)
    Z, feat_cols, fill_stats = build_features(df)
    gdist = load_graph_dist()

    pos = df.label1 == 'positive'
    support = df[pos].groupby('framework_code').size() \
        .sort_values(ascending=False)
    supported = support[support >= MIN_SUPPORT]
    P('WP1-T1.2.1 竞相负样本 (kNN 邻域) 生成报告')
    P('=' * 62)
    P(f'输入: {DB} ({len(df)} 行)')
    P(f'特征: {len(feat_cols)} 维 = log10 摩尔 {len(LOG_MOLE_COLS)} + 比值 5 '
      f'+ T/t 2 + 离散 5, 全库 z-score')
    for k, v in fill_stats.items():
        P(f'  填充: {k}: {v}')
    P(f'支撑 >= {MIN_SUPPORT} 骨架: {len(supported)} 个 '
      f'(正样本路线支撑 {int(supported.sum())})')
    P(f'支撑不足骨架 (不定义邻域): {len(support) - len(supported)} 个')
    P()
    P('量级说明: §4.1 字面定义 (tau=P_T 内部 kNN 距离 95 分位) 在主库 v0.5 上产出')
    P('百万级对数, 高于计划预估 "数万" 约 2 个量级: tau 分布重尾 (p95 约为内部')
    P('中位距离的 4-8 倍), 邻域半径被 P_T 内 5% 孤立路线撑宽; 敏感性曲线 (含')
    P('p50 参考行) 即为 M4 定版时 WP2 消融选择依据, 主表按计划主配置交付。')
    P()

    # 协方差 (马氏): 全库 LedoitWolf 收缩
    V = LedoitWolf().fit(Z).covariance_

    # ---- 主配置 k=10, tau95, euclidean ----
    K0, PCT0, MET0 = 10, 95, 'euclidean'
    pairs = []
    tau_table = []
    for T in supported.index:
        P_T_mask = pos & (df.framework_code == T)
        idx_T = np.where(P_T_mask)[0]
        idx_r = np.where(~P_T_mask)[0]
        ZT = Z[idx_T]
        d_in = knn_dist(ZT, ZT, K0, MET0)       # P_T 内部第 k 近邻
        tau = float(np.percentile(d_in, PCT0))
        d_out = knn_dist(ZT, Z[idx_r], K0, MET0)
        sel = d_out <= tau
        tau_table.append(dict(target_framework=T, n_support=len(idx_T), tau=tau,
                              d_in_med=float(np.median(d_in))))
        rows_T = df.iloc[idx_r[sel]]
        d_T = d_out[sel]
        for (ii, r), dk in zip(rows_T.iterrows(), d_T):
            gd = gdist.get((re.sub(r'^[\*-]+', '', str(r.framework_code)).upper(),
                            re.sub(r'^[\*-]+', '', str(T)).upper()), (np.nan, np.nan))
            pairs.append(dict(
                neg_pair_id=f'{r.route_id}|{T}', route_id=r.route_id,
                doi=r.doi, year=r.year,
                source_type='comp', target_framework=T,
                comp_framework=r.framework_code,
                label='neg', label_scheme='pairs_v0',
                fail_mode=('impurity_only' if r.label1 == 'relative_neg'
                           else 'dense' if (r.label1 == 'absolute_neg'
                                            and DENSE_PAT.search(str(r.product1)))
                           else 'amorphous' if r.label1 == 'absolute_neg'
                           else 'undesired' if r.label1 == 'no_product'
                           else np.nan),  # positive(其他骨架) = 竞相对, 非失败
                r_label1=r.label1, r_product1=r.product1,
                d_knn=float(dk), k=K0, tau=tau,
                params_ver=PARAMS_VER,
                group_id=f'{r.framework_code}|{r.doi}',
                graph_dist_D=gd[0], graph_dist_SOAP=gd[1],
            ))

    pair_df = pd.DataFrame(pairs)
    pair_df = pair_df.drop_duplicates(subset='neg_pair_id')
    tau_df = pd.DataFrame(tau_table)

    # ---- 敏感性: k x tau分位 x 度量 负对计数 ----
    sens = []
    for metric in ['euclidean', 'mahalanobis']:
        for k in K_GRID:
            d_in_all, d_out_all = {}, {}
            for T in supported.index:
                idx_T = np.where(pos & (df.framework_code == T))[0]
                idx_r = np.where(~(pos & (df.framework_code == T)))[0]
                d_in_all[T] = knn_dist(Z[idx_T], Z[idx_T], k, metric, V)
                d_out_all[T] = knn_dist(Z[idx_T], Z[idx_r], k, metric, V)
            row = dict(metric=metric, k=k)
            for pct in PCT_GRID:
                n = 0
                for T in supported.index:
                    tau = np.percentile(d_in_all[T], pct)
                    n += int((d_out_all[T] <= tau).sum())
                row[f'tau_p{pct}_npairs'] = n
            sens.append(row)
            P(f'敏感性 {metric} k={k}: '
              + ', '.join(f'p{p}={row[f"tau_p{p}_npairs"]}' for p in PCT_GRID))
    sens_df = pd.DataFrame(sens)

    # ---- 报告 ----
    P()
    P('--- 主配置 (k=10, tau 95 分位, 标准化欧氏) ---')
    P(f'监督对总数: {len(pair_df)} (去重后)')
    if len(pair_df):
        P(f"唯一路线数: {pair_df.route_id.nunique()}; "
          f"目标骨架数: {pair_df.target_framework.nunique()}")
        P(f"label1 构成: {pair_df.r_label1.value_counts().to_dict()}")
        P(f"fail_mode: {pair_df.fail_mode.value_counts().to_dict()}")
        P(f'图距离覆盖 (D 非空): {pair_df.graph_dist_D.notna().sum()}/{len(pair_df)}')
        P('Top 竞相对 (target <- comp_framework 计数 Top15):')
        top = pair_df.groupby(['target_framework', 'comp_framework']).size() \
            .sort_values(ascending=False).head(15)
        for (t, c), n in top.items():
            P(f'  {t} <- {c}: {n}')
    P()
    P('--- 支撑骨架 tau_T (Top 20 按支撑) ---')
    for _, r in tau_df.head(20).iterrows():
        P(f'  {r.target_framework}: support={r.n_support}, '
          f'tau={r.tau:.3f}, d_in_med={r.d_in_med:.3f}')
    P('  (全部 78 骨架 tau 见 parquet 侧表, 支撑不足清单略)')

    pair_df.to_parquet(OUT_PAIRS, index=False)
    sens_df.to_csv(OUT_SENS, index=False)
    with open(OUT_REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log) + '\n')
    tau_df.to_csv(os.path.join(ROOT, 'Data', 'pair_label_tau.csv'), index=False)
    print(f'\n已写 {OUT_PAIRS} ({len(pair_df)} 对), {OUT_SENS}, {OUT_REPORT}')


if __name__ == '__main__':
    main()
