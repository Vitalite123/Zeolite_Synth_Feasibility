# -*- coding: utf-8 -*-
"""
WP1 扩 OSDA 描述符表 v1.1 —— 域内缺 key 行补录 (L3 恢复通道)
=====================================================================
背景 (L3 v1 收口登记的恢复通道 + 2026-10-18 探查定案):
  - L3 域 18,031 行 (去重前) 中 1,485 行有 osda1_name 但 osda1_desc_key 缺失
    (120 个唯一名; '0' 等垃圾名剔除后 119 名), 主库 name<->smiles 配对可解析
    117 名, 其中仅 2 名 (18-crown-6 / tetrabutylammonium, 18 行) 的 SMILES 已在
    843 行官方表 -> 纯映射修复; 其余 115 名需新增描述符行。
  - EMT 考题侧结构性无 OSDA (Li 2023 SI 明文 organic template-free +
    Olanrewaju 2026 EMT_data 无模板设计): 扩表不改变考题行特征 (仅 SM_Table_S7
    2 行 18-crown-6 可触达), 收益面 = 域内描述符覆盖率 79.9% -> ~87%。

复算闸门 (probe 2026-10-18, 18-crown-6 全列对照 + 首跑 FAIL 修正):
  - 官方管线 = RDKit 确证: bertz_ct (H-added) 逐位 12 位小数一致, MolWt/
    num_bonds/num_rot_bonds EXACT (全 843 行断言)。
  - formal_charge carve-out: 官方 6/843 行与 SMILES 语义矛盾 (咪唑鎓阳离子记
    -1/-4、中性分子记 -1 等, 疑官方源缺陷), 不阻断闸门, 差异清单登记报告;
    新行取 RDKit 正确值 (其余 837 行官方与 RDKit 一致)。
  - 3D 列定义级复算 (15-conf ETKDGv3 + MMFF94s): rog rel 0.8% / volume 0.1% /
    GETAWAY 中位 1.1% (190/273 <2%) / WHIM 中位 5.3%; std 同量级。
  - geometry 组 (pmi/npr/asp/ecc/sph/inertial) 系综噪声底 0.05: 四协议扫描
    (mmff94s/mmff94/uff x15 confs, mmff94s x50 confs, 10 mol seed=42) 中位 rel
    0.0417-0.0580, 协议不敏感 (UFF 更差, 50 confs 仅边际改善) -> 判定为构象
    系综噪声而非协议错配, 阈值设 0.05 并登记扫描证据。
  - 规模守卫 (2026-10-19 实战教训): 新分子重原子数 > MAX_HEAVY=60 判 too_large 跳过
    (t20 = 肽类 319 原子 5 行 + t1 = 大环内酯 62 原子 8 行, 名字均为可疑短码);
    前次运行即在首个分子 (t20, 字母序最前) 上 GETAWAY/ComputeMolVolume 组合爆炸
    挂死 14.5 h 零输出, 故补守卫 + 逐分子进度打印。
  - 多组分守卫 (2026-10-19 实战教训②, 规模守卫生效后暴露的真根因): 新分子含点号
    断开碎片 (GetMolFrags > 1) 判 multi_component 跳过; 依据 = 官方 843 表多组分行数
    0 (官方管线从未计算多组分), 故跳过与官方口径一致而非放弃可比信息; 待新增 76
    分子中 18 个 (24%, 盐/反离子/多核配合物) 属此类。实证: 加规模守卫后运行仍卡在
    第 2 个分子 C1CC[N+]2(CC1)CCCCC2.C1CC[N+]2(CC1)CCCCC2 (22 重原子 2 片段)
    >10 min 无输出 (ETKDG/MMFF/体积网格对断开碎片病态), 这才是真正的耗时根因。
  - box(24)/axes(4)/free_sasa(2) 官方定义无法反推 (box_mean_2 为负值等) -> 新行 NaN。
  - binding_energy(192) 为 DFT 特权列, 新行 NaN (L1/L3 desc_cols 本就不含)。

口径分层 (每列登记复算等级, 不虚构造象协议外精度):
  tier2d   = 逐位 EXACT (严格 4 列全 843 行断言; formal_charge 官方 6 行语义
             矛盾 carve-out, 新行取 RDKit 正确值)
  tier3d   = 定义级等价 (同 RDKit 定义, 异构象系综; gate 阈值见 GATE)
  tier_nan = 定义不明 (box/axes/free_sasa), 新行留 NaN (LightGBM 原生处理)

产出 (幂等覆盖写):
  Data/osda_descriptors_v1_1.parquet   843 + N_new 行 x 830 列 (列序同原表)
  Data/osda_desc_key_sidecar.csv       名字归一化 -> desc_key 映射 (L3 侧回填用)
  Data/osda_desc_expand_report.txt     闸门报告 + 名字清单 + 口径分层声明
不动: zeobench_db_v1_8 (冻结) / osda_descriptors.parquet (v1.0 原表) / L1/L2 锚点链。
"""
import os
import time
import hashlib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors as rdmd
from rdkit.Chem import Descriptors as dsc2d

RDLogger.DisableLog('rdApp.*')

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, 'Data', 'zeobench_db_v1_8.parquet')
OSDA_V0 = os.path.join(ROOT, 'Data', 'osda_descriptors.parquet')
OUT_V11 = os.path.join(ROOT, 'Data', 'osda_descriptors_v1_1.parquet')
OUT_SIDE = os.path.join(ROOT, 'Data', 'osda_desc_key_sidecar.csv')
REPORT = os.path.join(ROOT, 'Data', 'osda_desc_expand_report.txt')

N_CONF = 15
EMB_SEED = 0xF00D
VAL_SAMPLE = 30
VAL_SEED = 42
# 新分子重原子数上限 (2026-10-19 实战教训): 官方表 p99=49 / 中位 13, 沸石 OSDA 几乎
# 全 <60 重原子; 超限的新分子判为非 OSDA 且触发 GETAWAY/ComputeMolVolume 组合爆炸
# (t20 = 319 重原子肽类 5 行 + t1 = 62 重原子大环内酯 8 行, 名字均为可疑短码),
# 按 too_large 跳过不阻断, 记入报告与 sidecar。
MAX_HEAVY = 60
# gate 阈值 (定义级复算验收): 2D 全 843 行逐位 (formal_charge carve-out); 3D 中位 rel 上限
# geometry=0.05 依据: 四协议扫描中位 rel 0.0417-0.0580, 系综噪声底 (见 docstring)
GATE = {'getaway': 0.05, 'whim': 0.08, 'geometry': 0.05, 'rog': 0.03, 'volume': 0.03}

TIER2D_STRICT = ['bertz_ct_mean_0', 'mol_weight',
                 'num_bonds_mean_0', 'num_rot_bonds_mean_0']
TIER2D_ALL = TIER2D_STRICT + ['formal_charge']
P3D = ('getaway', 'whim', 'gyration_radius', 'volume', 'pmi1', 'pmi2', 'pmi3',
       'npr1', 'npr2', 'asphericity', 'eccentricity', 'spherocity_index',
       'inertial_shape_factor')
TIER_NAN_PREFIX = ('box', 'axes', 'free_sasa')

HYP = dict(zip(map(ord, '‐‑‒–—−'), '-' * 6))


def norm_name(s):
    """名字归一化: 小写/strip/unicode 连字符统一/引号统一/空白压缩"""
    t = str(s).translate(HYP).strip().lower()
    t = t.replace('\u2019', "'").replace('\u2032', "'").replace('\u00b4', "'")
    return ' '.join(t.split())


def ok_name(s):
    t = norm_name(s)
    return bool(t) and any(c.isalpha() for c in t)


def canon(smiles):
    m = Chem.MolFromSmiles(smiles)
    return None if m is None else Chem.MolToSmiles(m)


def embed(mol_h, n_conf=N_CONF):
    ps = AllChem.ETKDGv3()
    ps.randomSeed = EMB_SEED
    cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=n_conf, params=ps))
    return cids


def calc_conf(mol_h, cid):
    out = {}
    out['getaway'] = np.asarray(rdmd.CalcGETAWAY(mol_h, confId=cid), float)
    out['whim'] = np.asarray(rdmd.CalcWHIM(mol_h, confId=cid), float)
    out['gyration_radius'] = rdmd.CalcRadiusOfGyration(mol_h, confId=cid)
    out['volume'] = AllChem.ComputeMolVolume(mol_h, confId=cid)
    out['pmi1'] = rdmd.CalcPMI1(mol_h, confId=cid)
    out['pmi2'] = rdmd.CalcPMI2(mol_h, confId=cid)
    out['pmi3'] = rdmd.CalcPMI3(mol_h, confId=cid)
    out['npr1'] = rdmd.CalcNPR1(mol_h, confId=cid)
    out['npr2'] = rdmd.CalcNPR2(mol_h, confId=cid)
    out['asphericity'] = rdmd.CalcAsphericity(mol_h, confId=cid)
    out['eccentricity'] = rdmd.CalcEccentricity(mol_h, confId=cid)
    out['spherocity_index'] = rdmd.CalcSpherocityIndex(mol_h, confId=cid)
    out['inertial_shape_factor'] = rdmd.CalcInertialShapeFactor(mol_h, confId=cid)
    return out


GEO_KEYS = ['gyration_radius', 'volume', 'pmi1', 'pmi2', 'pmi3', 'npr1', 'npr2',
            'asphericity', 'eccentricity', 'spherocity_index',
            'inertial_shape_factor']


def calc_mol(smiles):
    """完整描述符 dict (830 列口径)。返回 (row_dict, info) 或 (None, err)。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, 'parse_fail'
    mh = Chem.AddHs(mol)
    cids = embed(mh)
    if len(cids) < 3:
        return None, 'embed_fail(%d)' % len(cids)
    AllChem.MMFFOptimizeMoleculeConfs(mh, mmffVariant='MMFF94s', maxIters=2000)
    per = [calc_conf(mh, cid) for cid in cids]
    row = {}
    # tier2d: 逐位口径 (bertz 于 H-added; 键数/旋转键于重原子; MolWt 两态同值)
    row['bertz_ct_mean_0'] = dsc2d.BertzCT(mh)
    row['mol_weight'] = dsc2d.MolWt(mh)
    row['formal_charge'] = float(Chem.GetFormalCharge(mol))
    row['num_bonds_mean_0'] = float(mol.GetNumBonds())
    row['num_rot_bonds_mean_0'] = float(rdmd.CalcNumRotatableBonds(mol))
    # tier3d mean/std (ddof=0)
    g = np.stack([p['getaway'] for p in per])
    w = np.stack([p['whim'] for p in per])
    for i in range(g.shape[1]):
        row['getaway_mean_%d' % i] = g[:, i].mean()
        row['getaway_std_%d' % i] = g[:, i].std(ddof=0)
    for i in range(w.shape[1]):
        row['whim_mean_%d' % i] = w[:, i].mean()
        row['whim_std_%d' % i] = w[:, i].std(ddof=0)
    for k in GEO_KEYS:
        v = np.array([p[k] for p in per], float)
        row['%s_mean_0' % k] = v.mean()
        row['%s_std_0' % k] = v.std(ddof=0)
    return row, 'ok(n_conf=%d)' % len(cids)


def rel_stats(off, mine, cols):
    o = np.array([off[c] for c in cols], float)
    m = np.array([mine[c] for c in cols], float)
    ok = np.isfinite(o) & np.isfinite(m)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.abs(o[ok] - m[ok]) / np.maximum(np.abs(o[ok]), np.abs(m[ok]))
    rel = np.where(np.isfinite(rel), rel, np.nan)
    return rel


def main():
    log = []

    def P(msg=''):
        print(msg, flush=True)
        log.append(str(msg))

    t0 = time.time()
    osda_v0 = pd.read_parquet(OSDA_V0)
    cols_v0 = list(osda_v0.columns)
    db = pd.read_parquet(DB)

    P('WP1 扩 OSDA 描述符表 v1.1 —— 域内缺 key 行补录')
    P('=' * 68)
    P('输入: %s (%d 行) + %s (%d 行)' % (os.path.basename(DB), len(db),
                                         os.path.basename(OSDA_V0), len(osda_v0)))

    # ---- 1. 缺 key 名字清单 + 主库 name->smiles 配对 ----
    dom = db[(db['label1'].isin(['positive', 'relative_neg']))
             & db['zeo_desc_key'].notna()]
    miss = dom[dom['osda1_desc_key'].isna() & dom['osda1_name'].notna()]
    P('域 (去重前) %d 行 | 有名缺 key %d 行 / %d 唯一名'
      % (len(dom), len(miss), miss['osda1_name'].nunique()))

    pairs = {}
    for i in (1, 2, 3):
        for name, sm in zip(db['osda%d_name' % i], db['osda%d_smiles' % i]):
            if pd.notna(name) and pd.notna(sm) and ok_name(name):
                pairs.setdefault(norm_name(name), set()).add(str(sm).strip())

    names = miss[['osda1_name']].dropna().copy()
    names['name_norm'] = names['osda1_name'].map(norm_name)
    names = names[names['osda1_name'].map(ok_name)]
    rowcnt = names.groupby('name_norm').size()
    uniq = sorted(rowcnt.index)
    P('垃圾名剔除后 %d 个唯一名 (覆盖 %d 行)' % (len(uniq), int(rowcnt.sum())))

    # 一致性: 归一化合并后一名多 SMILES?
    conflict = {k: v for k, v in pairs.items() if k in set(uniq) and len(v) > 1}
    P('归一化后一名多 SMILES 冲突: %d 个 %s' % (len(conflict), list(conflict)[:3]))

    # ---- 2. SMILES 解析 + 官方 key 映射 ----
    off_canon = {}
    for sm in osda_v0['osda_smiles']:
        c = canon(str(sm))
        if c is not None:
            off_canon[c] = sm
    side_rows, new_mols, unresolved = [], [], []
    for nm in uniq:
        sms = pairs.get(nm)
        if not sms:
            unresolved.append(nm)
            side_rows.append(dict(osda_name_norm=nm, n_rows=int(rowcnt[nm]),
                                  desc_key='', target='unresolved_no_smiles'))
            continue
        sm = sorted(sms)[0]
        c = canon(sm)
        if c is None:
            unresolved.append(nm)
            side_rows.append(dict(osda_name_norm=nm, n_rows=int(rowcnt[nm]),
                                  desc_key='', target='parse_fail'))
            continue
        if c in off_canon:
            side_rows.append(dict(osda_name_norm=nm, n_rows=int(rowcnt[nm]),
                                  desc_key=off_canon[c], target='existing'))
        else:
            new_mols.append((nm, c, sm, int(rowcnt[nm])))
            side_rows.append(dict(osda_name_norm=nm, n_rows=int(rowcnt[nm]),
                                  desc_key=c, target='new_row'))
    n_exist = sum(1 for r in side_rows if r['target'] == 'existing')
    n_exist_rows = sum(r['n_rows'] for r in side_rows if r['target'] == 'existing')
    n_new_rows = sum(t[3] for t in new_mols)
    P('名字解析: 已在 843 表 %d 名 (%d 行, 纯映射修复) | 需新行 %d 唯一 SMILES'
      ' (%d 行) | 不可解 %d 名 (%d 行)'
      % (n_exist, n_exist_rows, len(new_mols), n_new_rows,
         len(unresolved), sum(r['n_rows'] for r in side_rows
                              if r['target'].startswith(('unresolved', 'parse')))))
    new_smiles = sorted(set(t[1] for t in new_mols))
    P('新行去重: %d 个唯一 canonical SMILES (多名同分子合并)' % len(new_smiles))

    # ---- 3. gate: 2D 全 843 行逐位断言 + 3D 抽样对照 ----
    P('')
    P('gate ①: 2D 逐位复算全 843 行断言 (严格 4 列 + formal_charge carve-out)...')
    bad2d, bad_fc = [], []
    for _, r in osda_v0.iterrows():
        mol = Chem.MolFromSmiles(str(r['osda_smiles']))
        if mol is None:
            continue
        mh = Chem.AddHs(mol)
        chk = {'bertz_ct_mean_0': dsc2d.BertzCT(mh),
               'mol_weight': dsc2d.MolWt(mh),
               'formal_charge': float(Chem.GetFormalCharge(mol)),
               'num_bonds_mean_0': float(mol.GetNumBonds()),
               'num_rot_bonds_mean_0': float(rdmd.CalcNumRotatableBonds(mol))}
        for k, v in chk.items():
            off = r[k]
            if k == 'formal_charge':
                if pd.isna(off) or abs(float(off) - float(v)) > 0.5:
                    bad_fc.append((r['osda_smiles'], k, float(off), float(v)))
            elif pd.isna(off) or abs(float(off) - float(v)) > 1e-6 * max(1.0, abs(float(v))):
                bad2d.append((r['osda_smiles'], k, float(off), float(v)))
    P('  2D 严格 4 列: %d/%d 行 EXACT (偏差 %d 个)'
      % (len(osda_v0) - len(set(s for s, *_ in bad2d)), len(osda_v0), len(bad2d)))
    for b in bad2d[:10]:
        P('    DIFF %s %s: 官方=%s RDKit=%s' % b)
    P('  formal_charge carve-out: 官方语义矛盾 %d/843 行 (不阻断, 差异如下,'
      ' 新行取 RDKit 正确值)' % len(bad_fc))
    for b in bad_fc:
        P('    FC-DIFF %s %s: 官方=%s RDKit=%s' % b)
    gate2d_ok = len(bad2d) == 0

    P('gate ②: 3D 定义级复算抽样 %d 行 (seed=%d)...' % (VAL_SAMPLE, VAL_SEED))
    rng = np.random.default_rng(VAL_SEED)
    val_idx = rng.choice(len(osda_v0), VAL_SAMPLE, replace=False)
    groups = {'getaway': [], 'whim': [], 'geometry': [], 'rog': [], 'volume': []}
    n_3d_fail = 0
    for i in val_idx:
        r = osda_v0.iloc[i]
        row, info = calc_mol(str(r['osda_smiles']))
        if row is None:
            n_3d_fail += 1
            continue
        gcols = [c for c in cols_v0 if c.startswith('getaway_')]
        wcols = [c for c in cols_v0 if c.startswith('whim_')]
        geocols = [c for c in cols_v0 if c.endswith('_mean_0')
                   and c.split('_mean_0')[0] in ('pmi1', 'pmi2', 'pmi3', 'npr1',
                                                 'npr2', 'asphericity',
                                                 'eccentricity', 'spherocity_index',
                                                 'inertial_shape_factor')]
        rel = rel_stats(r, row, gcols)
        groups['getaway'].append(np.nanmedian(rel))
        rel = rel_stats(r, row, wcols)
        groups['whim'].append(np.nanmedian(rel))
        rel = rel_stats(r, row, geocols)
        groups['geometry'].append(np.nanmedian(rel))
        rel = abs(float(r['gyration_radius_mean_0']) - row['gyration_radius_mean_0']) \
            / max(abs(float(r['gyration_radius_mean_0'])), 1e-9)
        groups['rog'].append(rel)
        rel = abs(float(r['volume_mean_0']) - row['volume_mean_0']) \
            / max(abs(float(r['volume_mean_0'])), 1e-9)
        groups['volume'].append(rel)
    gate3d = {}
    for gname, vals in groups.items():
        vals = np.array(vals)
        gate3d[gname] = float(np.median(vals))
        P('  %-8s 中位 rel = %.4f (p90 %.4f, n=%d) 阈值 %.2f %s'
          % (gname, np.median(vals), np.percentile(vals, 90), len(vals),
             GATE[gname], 'PASS' if np.median(vals) < GATE[gname] else 'FAIL'))
    P('  3D 失败分子 (parse/embed): %d/%d' % (n_3d_fail, VAL_SAMPLE))
    gate3d_ok = all(gate3d[g] < GATE[g] for g in GATE)
    P('GATE: 2D %s | 3D %s -> %s'
      % ('PASS' if gate2d_ok else 'FAIL', 'PASS' if gate3d_ok else 'FAIL',
         '通过, 写 v1_1 表' if (gate2d_ok and gate3d_ok) else '未过, 不写表!'))
    if not (gate2d_ok and gate3d_ok):
        P('闸门未过: 产出仅 sidecar + 报告, 不写描述符表 (纪律: 校验不过不入表)。')
        pd.DataFrame(side_rows).to_csv(OUT_SIDE, index=False, encoding='utf-8-sig')
        with open(REPORT, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log) + '\n')
        return

    # ---- 4. 新分子描述符计算 + v1_1 表 ----
    P('')
    P('新分子描述符计算 (tier2d 逐位 + tier3d 定义级, N_CONF=%d, MAX_HEAVY=%d)...'
      % (N_CONF, MAX_HEAVY))
    new_rows, fails = [], []
    for k, sm in enumerate(new_smiles, 1):
        m0 = Chem.MolFromSmiles(sm)
        nh = 0 if m0 is None else m0.GetNumHeavyAtoms()
        nfrag = 0 if m0 is None else len(Chem.GetMolFrags(m0))
        if m0 is None:
            info = 'parse_fail'
        elif nh > MAX_HEAVY:
            info = 'too_large(%d)' % nh
        elif nfrag > 1:
            info = 'multi_component(%d)' % nfrag
        else:
            info = None
        if info is not None:
            fails.append((sm, info))
            P('  [%d/%d] SKIP %s (%s)' % (k, len(new_smiles), info, sm[:48]))
            continue
        ts = time.time()
        row, info = calc_mol(sm)
        if row is None:
            fails.append((sm, info))
            P('  [%d/%d] SKIP %s (%s)' % (k, len(new_smiles), info, sm[:48]))
            continue
        full = {c: np.nan for c in cols_v0}
        full.update(row)
        full['osda_row_id'] = np.nan            # 落盘时统一编号
        full['osda_smiles'] = sm
        new_rows.append(full)
        P('  [%d/%d] ok %s in %.1fs' % (k, len(new_smiles), info, time.time() - ts))
    P('  成功 %d / 跳过 %d %s' % (len(new_rows), len(fails),
                                  [f[1] for f in fails] if fails else ''))
    new_df = pd.DataFrame(new_rows, columns=cols_v0)
    new_df['osda_row_id'] = np.arange(len(osda_v0) + 1, len(osda_v0) + 1 + len(new_df))
    v11 = pd.concat([osda_v0, new_df], ignore_index=True)[cols_v0]
    # 口径分层披露: 新行 NaN 列族
    nan_fam = {}
    for c in cols_v0:
        if c in TIER2D_ALL or c == 'osda_row_id' or c == 'osda_smiles':
            continue
        if v11.iloc[len(osda_v0):][c].isna().all():
            fam = c.split('_mean')[0].split('_std')[0].split('_')[0]
            nan_fam[fam] = nan_fam.get(fam, 0) + 1
    P('  新行 NaN 列族: %s (定义不明/DFT 特权, 共 %d 列)'
      % (dict(sorted(nan_fam.items())), sum(nan_fam.values())))
    v11.to_parquet(OUT_V11, engine='pyarrow', compression='zstd', index=False)
    P('  写出 %s (%d 行 x %d 列)' % (os.path.basename(OUT_V11), len(v11), v11.shape[1]))

    # sidecar 补跳过 SMILES 的名字为不可解 (区分 too_large / multi_component / parse_fail)
    def _fail_tag(info):
        for t in ('too_large', 'multi_component'):
            if info.startswith(t):
                return t
        return 'parse_fail'

    fail_map = {sm: _fail_tag(info) for sm, info in fails}
    for r in side_rows:
        if r['target'] == 'new_row' and r['desc_key'] in fail_map:
            r['target'] = fail_map[r['desc_key']]
            r['desc_key'] = ''
    side = pd.DataFrame(side_rows)
    side.to_csv(OUT_SIDE, index=False, encoding='utf-8-sig')
    P('  写出 %s (%d 行: existing %d / new_row %d / too_large %d /'
      ' multi_component %d / 不可解 %d)'
      % (os.path.basename(OUT_SIDE), len(side), n_exist,
         int((side['target'] == 'new_row').sum()),
         int((side['target'] == 'too_large').sum()),
         int((side['target'] == 'multi_component').sum()),
         int(side['target'].isin(['unresolved_no_smiles', 'parse_fail']).sum())))

    # ---- 5. 回填效果预估 (域内, 去重前口径) ----
    P('')
    smap = dict(zip(side.loc[side['desc_key'] != '', 'osda_name_norm'],
                    side.loc[side['desc_key'] != '', 'desc_key']))
    filled = miss['osda1_name'].map(norm_name).map(smap).notna().sum()
    P('回填效果预估 (去重前): 缺 key %d 行 -> 可回填 %d 行 (%.1f%%);'
      ' 域内描述符覆盖 %.1f%% -> %.1f%%'
      % (len(miss), int(filled), 100.0 * filled / max(len(miss), 1),
         100.0 * dom['osda1_desc_key'].notna().mean(),
         100.0 * (dom['osda1_desc_key'].notna().sum() + filled) / len(dom)))

    P('')
    P('收口声明:')
    P('  1. 官方管线 = RDKit 确证 (2D 严格 4 列全 843 行逐位 EXACT;'
      ' formal_charge 官方 %d 行语义矛盾 carve-out 登记)。' % len(bad_fc))
    P('  2. 口径分层: tier2d 逐位 | tier3d 定义级 (构象系综差异, gate 阈值内;'
      ' geometry 组系综噪声底 0.05 = 四协议扫描实证 0.0417-0.0580 协议不敏感)'
      ' | box/axes/free_sasa/binding_energy 新行 NaN。')
    P('  3. EMT 考题侧结构性无 OSDA (Li 2023 organic template-free 实锤 +'
      ' Olanrewaju 2026 无模板设计): 扩表收益 = 域内覆盖率, 非考题行。')
    P('  4. zeobench_db_v1_8 冻结不动; 回填仅 L3 侧经 sidecar 生效 (v1_1 表不动 L1/L2)。')
    P('  5. 规模守卫: 新分子重原子数 > %d 判 too_large 跳过; 官方 843 表不受该守卫'
      '约束 (3D 重算闸门口径不变)。' % MAX_HEAVY)
    P('  6. 多组分守卫: 新分子含点号断开碎片 (>1 片段) 判 multi_component 跳过;'
      ' 依据 = 官方 843 表多组分行数 0 (官方管线从未计算多组分), 故跳过与官方口径'
      '一致而非放弃可比信息。')
    P('产出: %s / %s / %s' % (os.path.basename(OUT_V11), os.path.basename(OUT_SIDE),
                              os.path.basename(REPORT)))
    P('总用时 %.1f min' % ((time.time() - t0) / 60.0))

    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log) + '\n')


if __name__ == '__main__':
    main()
