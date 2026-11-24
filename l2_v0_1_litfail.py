# -*- coding: utf-8 -*-
"""
M6 行动项 A5: lit_fail 308 条 (A+B) ETL 摩尔配比映射并入 L2 训练侧
→ 训练集 v0.1 + xtal 主口径三切分对照报告 (Q20 批准; Q28 G2 提升路径首步)
================================================================================
【映射方案 (数据博士预审口径, 全文载报告 §一)】
1. 标签协议 —— 按实际产物 label1 v2 (主相优先), 与主库/增量分区冻结协议一致:
   - L2 标签语义 = 结果事件 (该配方是否结晶出沸石主相), 意图 (intended_framework)
     不进特征空间, 按"失败意图"折叠标签会复制 Q19 已否定的系统性标签噪声;
   - actual_product 文本解析: ';' 为时间序列取末段 (无相词则回退全文), '+' 为
     相混合 (首段=主相), '/' 为混合列表 (首段=主相), ' or ' 为歧义 (保守负);
   - 主相为 amorphous/layered/致密矿物 (DENSE_PAT 同 comp_neg_knn 操作定义:
     quartz/cristobalite/tridymite/keatite/magadiite/analcime/nepheline/sodalite
     + Li2SiO3/chkalovite/pollucite 等)/unknown → absolute_neg (0/0);
   - 'No product / No X / no MOR / not crystallized' → no_product (0/0);
   - 沸石主相: 纯单相 → positive (1/1); 带次相 (任何 +/trace/impurity) →
     relative_neg (0/1); 低结晶度限定词 (poorly crystallized 等) 只记旗标不改标签
     (主库 label1 无结晶度阈值惯例, 同构);
   - clathrasil 家族 (Nonasil/Octadecasil/ZSM-39/RUB-10: IZA 合法码 NON/AST/MTN/
     RUT) 按 IZA 惯例判沸石主相, 与策展 fail_mode=dense 的分歧记 lf_diverge 旗标
     供周例会复核 (analcime/sodalite 依 DENSE_PAT 判致密, 与策展一致);
   - C 级 1 行 (litfail_v0_010) 不入库不计验收 (WP1 §2.3), 实并入 308 = A 184 + B 124。
   - lf_diverge 56 行复核 (2026-10-08, 周例会项提前): 54 行维持 (协议性分歧:
     low_cryst 31 非晶为主+沸石次相首段主相 / undesired 12 layered·analcime·
     unknown / dense 6 clathrasil IZA 惯例 / impurity_only 2 sodalite·cancrinite
     致密) + 2 行人工改判 (MANUAL_OVERRIDE, 采纳策展语义), 双口径计数不变。
2. 组成映射 —— 基准 SiO2=1 (元素摩尔, 与增量分区口径同构), 解析后全 n_ 列
   (除 n_seed) 除以行和归一化 sum=1 (v1.1 并入口径 1); B 级缺项按 WP1 §2.3
   "置 0 + 指示位" (指示位入训练集旗标列, 不扩 A3 特征空间保 490 维对照):
   - si_al: Si/Al=x / 裸数 / SAR / (gel) → n_Al=1/x; SiO2/Al2O3=R → n_Al=2/R;
     Si/B / SiO2/B2O3 → n_B; Si/Ge / SiO2/Ga2O3 → n_Ge/n_Ga; Zn/Si → n_Zn;
     mole frac 行直取; 纯硅 (pure silica/∞/no Al) → ratio_SiAl=NaN (cap 500 端,
     主库惯例); 区间取中点; M2O3 (M=B or Al) 元素歧义/未解析 → 置 0 + 旗标;
   - alkali: XOH/Si=r → 阳离子 r + n_OH=r; Na/Si, Na+/SiO2, OH-/SiO2 直取;
     Na2O/Al2O3=R → n_Na=R·n_Al (氧化物比=原子比); Na2O=a, Cs2O=b (按
     Al2O3=1 基准, 旗标假设); (X+Y)OH=V (X2O:Y2O=R1:R2) → n_OH=V,
     阳离子按 R1:R2 分摊; NH4F (F/SiO2=r) → n_F=r; HF (HF/Ch+=x) 依赖 Ch 量
     (未报 → 置 0 + lf_f_unresolved 旗标, 周例会可回填); 区间取中点;
   - osda: TEAOH/Si=r → n_osda1=r + n_OH+=r (OSDA 氢氧化物计 OH, 同
     incr_zeo1_2021 口径); 名称已知量未报 (choline/TEA+/hexamethonium/TBAOH/
     论文代号 T06/C11/...) → 置 0 + 旗标; none (含 ISDA/solvent-free/seeded) → 0;
   - osda1_desc_key: 别名归一 (TEA/TEOH→tetraethylammonium, choline, TBAOH,
     HMI, hexamethonium, EDA, T06→adamantyl 亚串, MSPT→spartein 亚串) 匹配
     主库 osda 名/SMILES → 描述符 444 维; 未匹配行描述符 NaN + b_osda1_desc=0;
   - T_C/t_h: 数值直取, 区间取中点, >x/<x 取边界; 未报 → NaN (特征侧主库中位填,
     与主库缺省惯例同构); n_H2O/solvent/stirring 未数字化按主库惯例记 0。
3. 切分纪律 (测试集与 v0 逐位一致, 只加训练行):
   - split_random: 308 行全 'train' (DOI 泄漏审计: lit_fail 24 个非空 DOI 与主库
     24,389 行 DOI 重叠 = 0);
   - split_heldout: intended_framework/actual_product 含留出骨架 token
     (LTL/SAS/ITG/RTH/SZR/LTA, 含 zeolite A / 单独 'L' 相词) → 'excluded',
     其余 'train';
   - split_time_2018: year≤2018 'train' / year≥2019 'excluded' (时间边界纪律,
     不以后见文献污染 ≤2018 训练侧; 亦不改测试集保对照)。
4. 特征协议: A3 490 维空间冻结; 主库行特征与 v0 逐位一致 (build_a3 复刻断言),
   lit_fail 行用主库拟合参数 (StandardScaler/中位数/z-score 统计) transform,
   保证 v0.1 与 v0 差异 = 且仅 = 训练侧加行。
5. 模型协议: LGBM binary (lr=0.05, leaves=31, min_child=100, L2=1.0, seed=42),
   random val 早停, heldout/time 轮数锚定 pure 口径 best (v0 协议镜像);
   v0 锚点核对: 仅主库行复现 v0 六读数 (assert; 2026-10-22 BE 修复后在 A3 490
   空间重锚, 旧 682 口径读数失效)。
产出 (幂等覆盖写):
  Data/l2_trainset_v0_1.parquet      训练集 v0.1 (23,488 主库 + 308 lit_fail)
  Data/l2_v0_1_litfail_report.txt   对照报告 (含映射方案全文/解析覆盖/审计)
"""
import os
import re
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from comp_neg_knn import LOG_MOLE_COLS, RATIO_CAP
from l1_baseline_v1 import build_a3, SEED
from l1_feat_iter import T_COLS
import osda_schema as osd

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, 'Data', 'zeosyn_db_v1_1.parquet')
SPLITS = os.path.join(ROOT, 'Data', 'zeosyn_splits_v1_1.parquet')
MANIFEST = os.path.join(ROOT, 'Data', 'zeosyn_db_dedup_manifest.csv')
LITFAIL = os.path.join(ROOT, 'Data', 'lit_fail_v1.csv')
OSDA_DESC = os.path.join(ROOT, 'Data', 'osda_descriptors.parquet')
TRAINSET = os.path.join(ROOT, 'Data', 'l2_trainset_v0_1.parquet')
REPORT = os.path.join(ROOT, 'Data', 'l2_v0_1_litfail_report.txt')
V0_TRAINSET = os.path.join(ROOT, 'Data', 'l2_trainset_v0.parquet')

# v0 报告读数 (锚点, 4dp); 2026-10-22 BE 列名修复后 A3 490 维重锚,
# 旧 682 口径读数 (pure 0.6078 / xtal 0.6530 等) 失效
V0 = {'pure_random': 0.6127, 'pure_random_ap': 0.7433,
      'xtal_random': 0.6247, 'xtal_random_ap': 0.8748,
      'pure_heldout': 0.5265, 'xtal_heldout': 0.5102,
      'pure_time': 0.6417, 'xtal_time': 0.6648}

# ============ 1. lit_fail 标签协议解析 (实际产物, label1 v2 主相优先) ============

DENSE_RE = re.compile(
    r'quartz|cristobalite|keatite|magadiite|tridymite|dense|analcime|nepheline'
    r'|sodalite|li2sio3|lial2\(oh\)7|lialo2|alli o2|chkalovite|khsi2o5'
    r'|pollucite|lithium disilicate|aluminum oxide|sr\(oh\)2|nab\(oh\)4'
    r'|unreacted lioh', re.I)
AMORPH_RE = re.compile(r'amorphous|hydrogel|no crystalline|no molecular sieve',
                       re.I)
LAYER_RE = re.compile(r'layered|lamellar', re.I)
NOPROD_RE = re.compile(r'^no product|^no [a-z]|^nothing|could not be synthesized'
                      r'|not crystallized', re.I)
ZEO_RE = re.compile(
    r'\((?:[A-Z*]{3})\)|\b(?:SSZ|ZSM|ITQ|VPI|CIT|ECS|ERS|EU|RUB|UZM|ECR|ZOL|RUT)'
    r'-?\s?\d|\bZOL-[A-Z]|\b(?:Li-ABW|MTW|MTW\(B\)|MFI|MOR|MER|RHO|LTL|LTA|CHA'
    r'|FAU|GIS|ERI|EUO|OFF|NON|FER|STT|MWW|PHI|BEA|Beta|\*BEA|AST|SOD|CAN'
    r'|P\))\b'
    r'|\boffretite\b|\berionite\b|\bchabazite\b|\bmordenite\b|\bferrierite\b'
    r'|\bphilipsite\b|\bgismondine\b|\bfaujasite\b|\bZeolite A\b|\bzeolite L\b'
    r'|\boctadecasil\b|\bNonasil\b|\bcancrinite\b|\(P\)'
    r'|(?<![\w(])P(?![\w)])|(?<![\w(])L(?![\w)+])',
    re.I)
LOWCRY_RE = re.compile(r'poorly crystallized|low crystallinity|crystallinity'
                       r'|weakened|incomplete crystallization|decreased', re.I)
IMPURE_RE = re.compile(r'impur', re.I)


def classify_segment(seg):
    """单段相分类: 位置优先 (最先出现的相词 = 主相)
    修复: 'impure UZM-15 (unknown dense phase cocrystallizing)' 类括注
    误判 DENSE; 'non-ERS-7 phases' 否定式整段 UNKN"""
    if re.match(r'non-', seg.strip(), re.I):
        return 'UNKN', 'negated'
    best = ('UNKN', 'unknown', 10 ** 9)
    for cls, pat in (('AMORPH', AMORPH_RE), ('LAYER', LAYER_RE),
                     ('DENSE', DENSE_RE), ('ZEO', ZEO_RE)):
        mm = pat.search(seg)
        if mm and mm.start() < best[2]:
            best = (cls, mm.group(0), mm.start())
    if NOPROD_RE.match(seg.strip()):
        return 'NOPROD', 'no_product'
    return best[0], best[1]


def classify_product(text):
    """actual_product → (label1_lf, 主相类, 次相沸石旗标, 低结晶旗标, 歧义旗标)
    协议: ';' 时间序列取末段 (末段无相词回退全文); '+' 相混合首段主相;
    '/' 混合列表首段主相; ' or ' 歧义保守负。"""
    text = str(text).strip()
    flags = {'zeo_sec': 0, 'low_cryst': 0, 'ambig': 0, 'unkn_main': 0}
    if re.search(r'\bor\b', text) and 'and/or' not in text:
        # "pure OFF or amorphous" 型歧义 → 保守绝对负
        flags['ambig'] = 1
        return 'absolute_neg', 'AMORPH', flags
    work = text
    if ';' in work:
        segs = [s for s in work.split(';') if s.strip()]
        last = segs[-1]
        if any(re.search(p, last) for p in
               (AMORPH_RE, LAYER_RE, DENSE_RE, ZEO_RE, NOPROD_RE)):
            work = last  # 末时间态可判 → 取末段
    segs = [s.strip() for s in re.split(r'\+|/| and ', work) if s.strip()]
    segs = [s for s in segs if not re.fullmatch(r'\(?.?[0-9.~%±约]+.*', s)
            or re.search(r'[A-Za-z]{3}', s)]
    if not segs:
        segs = [work]
    cls = [classify_segment(s) for s in segs]
    main_cls = cls[0][0]
    if LOWCRY_RE.search(text):
        flags['low_cryst'] = 1
    if main_cls == 'ZEO':
        # 'Impure ZSM-43 (26 samples)': 单段但文本声明杂质 → relative_neg
        if len(segs) == 1 and IMPURE_RE.search(work):
            return 'relative_neg', main_cls, flags
        label = 'positive' if len(segs) == 1 else 'relative_neg'
        return label, main_cls, flags
    if main_cls == 'NOPROD':
        return 'no_product', main_cls, flags
    if any(c == 'ZEO' for c, _ in cls[1:]):
        flags['zeo_sec'] = 1
    if main_cls == 'UNKN':
        flags['unkn_main'] = 1
    return 'absolute_neg', main_cls, flags


# ============ 2. 组成文本解析 (基准 SiO2=1 元素摩尔) ============

def _num_or_mid(s):
    """'0.14' → 0.14; '0.25-0.5' → 中点; '5-∞' → 取下界 5; '>2.5' → 2.5; '<30' → 30"""
    s = str(s).strip()
    m = re.search(r'(\d+\.?\d*)\s*[-~]\s*(?:(\d+\.?\d*)|∞)', s)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2)) if m.group(2) else lo  # 开上界 ∞ → 下界
        return (lo + hi) / 2, True
    m = re.search(r'>=?\s*(\d+\.?\d*)', s)
    if m:
        return float(m.group(1)), True
    m = re.search(r'<=?\s*(\d+\.?\d*)', s)
    if m:
        return float(m.group(1)), True
    m = re.search(r'(\d+\.?\d*)', s)
    return (float(m.group(1)), False) if m else (np.nan, False)


def parse_si_al(t):
    """si_al 文本 → (out, basis, 旗标列表)
    out: n_Si/n_Al/n_B/n_Ge/n_Ga/n_Zn/n_Co/n_Na/n_K/n_Li/n_Cs/n_Rb (基准
    SiO2=1, 配方式为绝对摩尔); basis: 'recipe' 配方式绝对摩尔 (碱金属氧化物
    n_M=2v 直取, 探针定案: zeolite Y Na/Si=0.86 / MOR Na/Si=0.4 交叉验证) /
    'ratio' 比值式 (alkali 氧化物按 per-Al2O3=1 假设 n_M=v·n_Al) / 'frac' mole frac"""
    out = dict(n_Si=1.0, n_Al=0.0, n_B=0.0, n_Ge=0.0, n_Ga=0.0, n_Zn=0.0,
               n_Co=0.0, n_Na=0.0, n_K=0.0, n_Li=0.0, n_Cs=0.0, n_Rb=0.0)
    fl = []
    if not isinstance(t, str) or not t.strip():
        return out, None, ['si_al_missing']
    s = t.strip()
    # ---- 配方式: x SiO2 : [b] Al2O3 [: c M2O][: d GeO2][: e B2O3] (绝对摩尔) ----
    m = re.search(r'([\d.]+)\s*SiO2\s*:', s, re.I)
    if m is None:  # 紧凑式 'SiO2/0.04Al2O3' (Al2O3 系数并入)
        m2 = re.search(r'(?:([\d.]+)\s*)?SiO2\s*/\s*([\d.]+)\s*Al2O3', s, re.I)
        if m2:
            out['n_Si'] = float(m2.group(1)) if m2.group(1) else 1.0
            out['n_Al'] = 2.0 * float(m2.group(2))
            return out, 'recipe', fl
    if m:
        out['n_Si'] = float(m.group(1))
        m2 = re.search(r':\s*(?:([\d.]+)\s*)?Al2O3', s, re.I)
        if m2:  # '10 SiO2 : Al2O3' 缺省系数 1
            out['n_Al'] = 2.0 * (float(m2.group(1)) if m2.group(1) else 1.0)
        for el in ('Na', 'K', 'Li', 'Cs', 'Rb'):  # M2O 因子 2 (元素摩尔)
            m3 = re.search(r':\s*([\d.]+)\s*%s2O' % el, s, re.I)
            if m3:
                out['n_' + el] = 2.0 * float(m3.group(1))
        if re.search(r':\s*[\d.]+\s*W2O', s, re.I):
            fl.append('si_al_w2o_skipped')  # 钨氧化物忽略
        m4 = re.search(r':\s*([\d.]+)\s*GeO2', s, re.I)
        if m4:
            out['n_Ge'] = float(m4.group(1))
        m5 = re.search(r':\s*([\d.]+)\s*B2O3', s, re.I)
        if m5:
            out['n_B'] = 2.0 * float(m5.group(1))
        return out, 'recipe', fl
    # ---- '5-∞': Si/Al 范围开上界 → 取下界 ----
    m = re.fullmatch(r'([\d.]+)\s*[-~]\s*∞', s)
    if m:
        out['n_Al'] = 1.0 / float(m.group(1))
        return out, 'ratio', ['si_al_range_open']
    # ---- mole frac 行: 'Si=0.50, Al=0.10 (mole frac)' ----
    m = re.search(r'Si=([\d.]+),\s*Al=([\d.]+)', s)
    if m:
        out['n_Si'] = float(m.group(1))
        out['n_Al'] = float(m.group(2))
        return out, 'frac', fl
    # ---- 纯硅系 ('20-∞' 范围中 ∞ 前为 '-' 不判纯硅) ----
    if re.search(r'pure silica|silica-only|no Al|(?<![-\d])∞|metakaolin',
                 s, re.I):
        if 'metakaolin' in s.lower():
            fl.append('si_al_metakaolin')
        return out, 'ratio', fl or ['si_al_silica']
    # ---- M2O3 (M=B or Al) 元素歧义 → 置 0 + 旗标 (不做数值回退) ----
    if re.search(r'M2O3', s, re.I) and out['n_Al'] == 0 and out['n_B'] == 0:
        return out, 'ratio', ['si_al_hetero_ambiguous']
    # ---- Al/(Al+Si)=v → n_Al = v/(1-v)·n_Si ----
    m = re.search(r'Al/\(Al\+Si\)=([\d.]+)', s)
    if m:
        v = float(m.group(1))
        out['n_Al'] = v / (1.0 - v) * out['n_Si']
        return out, 'ratio', fl
    # ---- 直接量纲 (per Si): Zn/Si / Al/Si ----
    m = re.search(r'Zn/Si\s*=\s*([\d.]+)', s, re.I)
    if m:
        out['n_Zn'] = _num_or_mid(m.group(1))[0]
    m = re.search(r'Al/Si\s*=\s*([\d.]+)', s, re.I)
    if m:
        out['n_Al'] = _num_or_mid(m.group(1))[0] * out['n_Si']
    # ---- 复合行: 'Si/Ge=15-50, Al/(Si+Ge)=0.02-0.067' ----
    m = re.search(r'Al/\(Si\+Ge\)=([\d.]+)', s)
    al_sige = None
    if m:
        al_sige = _num_or_mid(m.group(1))[0]
        fl.append('si_al_range_mid')
    # ---- Si/X 比值族 (可带边界/范围); 命中即 break 免回退分支误读 ----
    for pat, key in ((r'Si/Al\s*[=<>]+\s*([\d.]+(?:\s*[-~]\s*[\d.]+)?)', 'n_Al'),
                     (r'Si/B\s*[=<>]+\s*([\d.]+(?:\s*[-~]\s*[\d.]+)?)', 'n_B'),
                     (r'Si/Ge\s*[=<>]+\s*([\d.]+(?:\s*[-~]\s*[\d.]+)?)', 'n_Ge'),
                     (r'Si/Ga\s*[=<>]+\s*([\d.]+(?:\s*[-~]\s*[\d.]+)?)', 'n_Ga'),
                     (r'Si/Co\s*[=<>]+\s*([\d.]+(?:\s*[-~]\s*[\d.]+)?)', 'n_Co')):
        mm = re.search(pat, s, re.I)
        if mm:
            v, rng = _num_or_mid(mm.group(1))
            out[key] = 1.0 / v
            if rng:
                fl.append('si_al_range_mid')
            break
    else:
        # 氧化物比 SiO2/X2O3=R (可带边界/范围)
        for pat, key in ((r'SiO2/Al2O3\s*[=<>]+\s*([\d.]+(?:\s*[-~]\s*[\d.]+)?)',
                          'n_Al'),
                         (r'SiO2/B2O3\s*[=<>]+\s*([\d.]+(?:\s*[-~]\s*[\d.]+)?)',
                          'n_B'),
                         (r'SiO2/Ga2O3\s*[=<>]+\s*([\d.]+(?:\s*[-~]\s*[\d.]+)?)',
                          'n_Ga')):
            mm = re.search(pat, s, re.I)
            if mm:
                out[key] = 2.0 / _num_or_mid(mm.group(1))[0]
        # '15 (SiO2/Al2O3=30)': 括注给氧化物比, 值 = Si/Al
        m = re.search(r'(\d+\.?\d*)\s*\(SiO2/Al2O3', s)
        if m and out['n_Al'] == 0:
            out['n_Al'] = 1.0 / float(m.group(1))
        # SAR x / 裸数 / (gel)/(product)/(target)/边界 → 一律 Si/Al
        # (无任何异质元素解析时才进回退, 防 'SiO2/B2O3=30-60' 的 '=30' 误读)
        if not any(out[k] > 0 for k in ('n_Al', 'n_B', 'n_Ge', 'n_Ga',
                                        'n_Zn', 'n_Co')):
            m = re.fullmatch(
                r'\s*(?:SAR\s*)?([<>]?)\s*([\d.]+)\s*(?:\([^)]*\))?\s*', s)
            if m:
                out['n_Al'] = 1.0 / float(m.group(2))
                if m.group(1):
                    fl.append('si_al_bound_value')
            else:
                m = re.search(r'(?:SAR|=)\s*([\d.]+)', s)
                if m:
                    out['n_Al'] = 1.0 / float(m.group(1))
    if al_sige is not None and out['n_Si'] > 0:
        out['n_Al'] = al_sige * (out['n_Si'] + out['n_Ge'])
    if re.search(r'>\d', s) and not fl:
        fl.append('si_al_bound_value')
    if (out['n_Al'] == 0 and out['n_B'] == 0 and out['n_Ge'] == 0
            and out['n_Ga'] == 0 and out['n_Zn'] == 0 and out['n_Co'] == 0
            and not fl):
        fl.append('si_al_unparsed')
    return out, 'ratio', fl


def parse_alkali(t, n_Si, n_Al, n_Ge, basis):
    """alkali 文本 → dict(n_Na/n_K/n_Li/n_Cs/n_Rb/n_F/n_OH) + 旗标
    basis 分流 (探针定案): 'recipe' → 氧化物量按绝对摩尔 n_M=2v;
    其他 (ratio/frac) → per-Al2O3=1 假设 n_M=v·n_Al (氧化物比=元素比)"""
    out = dict(n_Na=0.0, n_K=0.0, n_Li=0.0, n_Cs=0.0, n_Rb=0.0, n_F=0.0,
               n_OH=0.0)
    fl = []
    if not isinstance(t, str) or not t.strip():
        return out, ['alkali_missing']
    s = t.strip()
    if re.match(r'none', s, re.I):  # 'none (no Li/Na)' → resolved-zero
        return out, ['alk_none']
    got = False
    # ---- 混合碱 (X+Y)OH=V (X2O:Y2O=R1:R2) —— 'Li' 两字符, 修复单字符类 ----
    m = re.search(r'\((Na|K|Li)\+(Na|K|Li)\)OH=([\d.]+)\s*\('
                  r'(Na|K|Li|Cs|Rb)2O:(Na|K|Li|Cs|Rb)2O=(\d+):(\d+)\)', s)
    if m:
        V = float(m.group(3))
        r1, r2 = float(m.group(6)), float(m.group(7))
        out['n_OH'] = V
        out['n_' + m.group(1)] = V * r1 / (r1 + r2)
        out['n_' + m.group(2)] = V * r2 / (r1 + r2)
        got = True
    # ---- 氧化物量族: 前缀式 'x Na2O' / 等值式 'Na2O=x' (basis 分流) ----
    OX = r'(Na2O|K2O|Li2O|Cs2O|Rb2O)'

    def _ox_apply(el, v):
        nonlocal got
        if basis == 'recipe':
            out['n_' + el] += 2.0 * v
        else:
            out['n_' + el] += v * n_Al
            if 'alk_oxide_per_al_assumed' not in fl:
                fl.append('alk_oxide_per_al_assumed')
        got = True

    for mm in re.finditer(r'([\d.]+)\s*' + OX, s):
        _ox_apply(mm.group(2)[:2].replace('2', ''), float(mm.group(1)))
    # (?<!:) 排除混合碱括注 'Na2O:Li2O=4:1' 的比例段误读
    for mm in re.finditer(r'(?<!:)' + OX + r'\s*=\s*([\d.]+)', s):
        _ox_apply(mm.group(1)[:2].replace('2', ''), float(mm.group(2)))
    # ---- 氢氧化物前缀式 'x NaOH' (绝对摩尔) ----
    for mm in re.finditer(r'([\d.]+)\s*(NaOH|KOH|LiOH|CsOH|RbOH)', s):
        v = float(mm.group(1))
        el = mm.group(2).replace('OH', '')
        if out['n_' + el] == 0:
            out['n_' + el] += v
        if out['n_OH'] == 0:
            out['n_OH'] += v
        got = True
    # ---- XOH/Si=r 族 (阳离子 + OH 双计) ----
    for el in ('Li', 'Na', 'K', 'Cs', 'Rb'):
        m = re.search(r'%sOH/Si=([\d.]+)' % el, s)
        if m:
            v = _num_or_mid(m.group(1))[0]
            if out['n_' + el] == 0:
                out['n_' + el] = v
            if out['n_OH'] == 0:
                out['n_OH'] = v
            got = True
    # NaOH (NaOH/SiO2~0.64) / NaOH (0.50) 括注式
    m = re.search(r'NaOH\s*\(OH-/SiO2=([\d.]+)', s)
    if m and out['n_OH'] == 0:
        out['n_OH'] = float(m.group(1))
        m2 = re.search(r'NaOH\s*\(([\d.]+)\)', s)
        if m2 and out['n_Na'] == 0:
            out['n_Na'] = float(m2.group(1))
        got = True
    m = re.search(r'NaOH/SiO2~?=?\s*([\d.]+)', s)
    if m:
        v = _num_or_mid(m.group(1))[0]
        if out['n_Na'] == 0:
            out['n_Na'] = v
        if out['n_OH'] == 0:
            out['n_OH'] = v
        got = True
    # ---- 元素比直取族 ----
    m = re.search(r'Na/Si=([\d.]+)', s)
    if m and out['n_Na'] == 0:
        out['n_Na'] = _num_or_mid(m.group(1))[0]
        got = True
    m = re.search(r'Na/SiO2=([\d.]+)', s)
    if m and out['n_Na'] == 0:
        out['n_Na'] = _num_or_mid(m.group(1))[0]
        got = True
    m = re.search(r'NaCl/Si=([\d.]+)', s)
    if m:
        out['n_Na'] += _num_or_mid(m.group(1))[0]  # Cl 不计
        got = True
    m = re.search(r'(?<![a-zA-Z])K/Si\s*[<>=~]+\s*([\d.]+)', s)
    if m and out['n_K'] == 0:
        out['n_K'] = _num_or_mid(m.group(1))[0]
        got = True
        fl.append('alk_bound_value')
    m = re.search(r'(?<![a-zA-Z])OH/Si\s*[<>=~]+\s*([\d.]+)', s)
    if m and out['n_OH'] == 0:
        out['n_OH'] = _num_or_mid(m.group(1))[0]
        got = True
        fl.append('alk_bound_value')
    # ---- per-Al / per-(Si+X) 族 ----
    m = re.search(r'Na/Al\s*[<>=~]+\s*([\d.]+)', s)
    if m and out['n_Na'] == 0:
        out['n_Na'] = _num_or_mid(m.group(1))[0] * n_Al
        got = True
    m = re.search(r'K/Al\s*[<>=~]+\s*([\d.]+)', s)
    if m and out['n_K'] == 0:
        out['n_K'] = _num_or_mid(m.group(1))[0] * n_Al
        got = True
    m = re.search(r'LiOH·?H2O,?\s*Li/Al=([\d.]+)(?:/[\d.]+)*', s)
    if m:
        v = float(m.group(1))
        out['n_Li'] = v * n_Al
        out['n_OH'] = v * n_Al  # LiOH 1:1
        got = True
        if re.search(r'Li/Al=[\d.]+/', s):
            fl.append('alk_multi_value_first')  # '0.6/1.1/1.7' 取首值
    m = re.search(r'Na/\(Si\+Al\)=([\d.]+(?:\s*[-~]\s*[\d.]+)?)', s)
    if m:
        v, rng = _num_or_mid(m.group(1))
        if out['n_Na'] == 0:
            out['n_Na'] = v * (n_Si + n_Al)
        got = True
        if rng:
            fl.append('alk_range_mid')
    m = re.search(r'OH/\(Si\+Ge\)=([\d.]+(?:\s*[-~]\s*[\d.]+)?)', s)
    if m:
        v, rng = _num_or_mid(m.group(1))
        if out['n_OH'] == 0:
            out['n_OH'] = v * (n_Si + n_Ge)
        got = True
        if rng:
            fl.append('alk_range_mid')
    m = re.search(r'Si/Na\s*[<>=~]+\s*([\d.]+)', s)
    if m and out['n_Na'] == 0:
        out['n_Na'] = 1.0 / _num_or_mid(m.group(1))[0]
        got = True
        fl.append('alk_bound_value')
    # ---- 氧化物比族 ----
    m = re.search(r'Na2O/Al2O3=([\d.]+)', s)
    if m and out['n_Na'] == 0:
        out['n_Na'] = _num_or_mid(m.group(1))[0] * n_Al  # 氧化物比=元素比
        got = True
    for el in ('Na', 'Cs', 'Rb', 'K'):
        m = re.search(r'%s2O/SiO2=([\d.]+)' % el, s)
        if m and out['n_' + el] == 0:
            out['n_' + el] = 2.0 * float(m.group(1))
            got = True
    # ---- OH-/SiO2, Na+/SiO2 ----
    m = re.search(r'OH-/SiO2=([\d.]+)', s)
    if m and out['n_OH'] == 0:
        out['n_OH'] = _num_or_mid(m.group(1))[0]
        got = True
    m = re.search(r'Na\+/SiO2=([\d.]+)', s)
    if m and out['n_Na'] == 0:
        out['n_Na'] = _num_or_mid(m.group(1))[0]
        got = True
    # ---- LiOH=V (Li2O=V) 型 ----
    m = re.search(r'LiOH=([\d.]+)\s*\(Li2O', s)
    if m:
        v = float(m.group(1))
        out['n_Li'] = v
        out['n_OH'] = v
        fl.append('alk_lioh_li2o_note')
        got = True
    # ---- F 源 ----
    m = re.search(r'F/SiO2=([\d.]+)', s)
    if m and out['n_F'] == 0:
        out['n_F'] = _num_or_mid(m.group(1))[0]
        got = True
    m = re.search(r'fluoride \(F/\(Si\+Ge\)=([\d.]+)', s)
    if m:
        v, rng = _num_or_mid(m.group(1))
        out['n_F'] = v * (n_Si + n_Ge)
        got = True
        fl.append('alk_f_range_mid')
    if re.search(r'NH4F|HF|fluoride|F-', s, re.I) and out['n_F'] == 0:
        fl.append('alk_f_present_unresolved')  # 氟路线但 F 量不可折算
    # ---- 元素歧义旗标 ----
    if re.search(r'\(Cs\|Rb\)OH|AM/Si', s, re.I):
        fl.append('alk_m_ambiguous')
    if not got:
        fl.append('alk_unparsed')
    return out, fl


def parse_T(t):
    if pd.isna(t):
        return np.nan, []
    v, rng = _num_or_mid(t)
    return v, (['t_range_mid'] if rng else [])


OSDA_ALIAS = {
    'tea': 'tetraethylammonium', 'tea+': 'tetraethylammonium',
    'teoh': 'tetraethylammonium', 'teaoh': 'tetraethylammonium',
    'choline': 'choline', 'ch+': 'choline',
    'tbaoh': 'tetrabutylammonium', 'hmi': 'hexamethyleneimine',
    'hexamethonium': 'hexamethonium', 'hmbr': 'hexamethonium',
    'eda': 'ethylenediamine', 'tmaa+': 'adamantyl', 't06': 'adamantyl',
    'mspt': 'spartein', 'tpa': 'tetrapropylammonium',
    'tpabr': 'tetrapropylammonium',}


def parse_osda(t, n_Si, n_Al):
    """osda 文本 → (n_osda1, 匹配用名, OH 伴随, 旗标)"""
    if not isinstance(t, str) or not t.strip():
        return 0.0, None, 0.0, ['osda_missing']
    s = t.strip()
    if re.match(r'none', s, re.I):
        return 0.0, None, 0.0, []
    fl = []
    m = re.search(r'TEAOH/Si=([\d.]+)', s)
    if m:
        v = _num_or_mid(m.group(1))[0]
        return v, 'tetraethylammonium', v, fl
    m = re.match(r'([\d.]+)\s*TEAOH\b', s)  # '0.20 TEAOH' 前缀量
    if m:
        v = float(m.group(1))
        return v, 'tetraethylammonium', v, fl
    m = re.match(r'([\d.]+)\s*TPABr\b', s)  # '0.5 TPABr + HF' 前缀量
    if m:
        return float(m.group(1)), 'tetrapropylammonium', 0.0, fl
    m = re.match(r'([\d.]+)\s*EDA\b', s)  # '70 EDA' 前缀量
    if m:
        return float(m.group(1)), 'ethylenediamine', 0.0, fl
    m = re.search(r'TEA/\(Si\+Al\)=([\d.]+)', s)
    if m:
        v = _num_or_mid(m.group(1))[0]
        return v * (n_Si + n_Al), 'tetraethylammonium', 0.0, fl + ['osda_range_mid']
    m = re.search(r'HMI/SiO2=([\d.]+)', s)
    if m:
        return float(m.group(1)), 'hexamethyleneimine', 0.0, fl
    m = re.search(r'TMAA2O/SiO2=([\d.]+)', s)
    if m:
        v = 2.0 * float(m.group(1))
        return v, 'adamantyl', 0.0, fl + ['osda_tmaa2o_x2']
    m = re.search(r'MSPT.*?([\d.]+)-([\d.]+)', s)
    if m:
        v = (float(m.group(1)) + float(m.group(2))) / 2
        return v, 'spartein', 0.0, fl + ['osda_range_mid']
    # 名称已知量未报 → 0 + 旗标; 别名映射供 desc 匹配
    head = re.split(r'[\s(+]', s)[0].lower()
    name = OSDA_ALIAS.get(head, OSDA_ALIAS.get(s.lower()))
    if name is None:
        low = s.lower()
        for k, v2 in OSDA_ALIAS.items():
            if low.startswith(k):
                name = v2
                break
    return 0.0, name, 0.0, ['osda_present_unquantified']


# ============ 3. 组装 lit_fail 合成行 ============

# lf_diverge 复核人工改判表 (2026-10-08, 56 行复核中 2 行采纳策展语义)
MANUAL_OVERRIDE = {
    'litfail_v1_130': ('absolute_neg',
                      'ECS-9 (amorphous) 括注表明产物非晶, 采纳策展 amorphous'),
    'litfail_v1_079': ('relative_neg',
                      'CIT-6 主相带杂质, 采纳策展 impurity_only (文本顺序≠主相关系)'),
}


def build_lf_rows():
    lf = pd.read_csv(LITFAIL)
    n_all = len(lf)
    lf = lf[lf['completeness'].isin(['A', 'B'])].reset_index(drop=True)
    n_c = n_all - len(lf)
    osda_db = pd.read_parquet(
        OSDA_DESC, columns=['osda_smiles', 'formal_charge'])
    chg_map = dict(zip(osda_db['osda_smiles'].astype(str),
                       osda_db['formal_charge']))
    # 主库 osda 名→SMILES 映射 (别名匹配用)
    db_cols = ['osda%d_%s' % (i, c) for i in (1, 2, 3)
               for c in ('name', 'smiles')]
    db = pd.read_parquet(DB, columns=db_cols + ['osda1_desc_key', 'doi'])
    name2smi = {}
    for i in (1, 2, 3):
        sub = db[['osda%d_name' % i, 'osda%d_smiles' % i]].dropna()
        for n, sm in zip(sub.iloc[:, 0], sub.iloc[:, 1]):
            name2smi[str(n).lower().strip()] = str(sm)
    smi_set = set(osda_db['osda_smiles'].astype(str))
    # osda1_desc_key 候选: 名精确→SMILES∈描述符表; 否则子串唯一命中
    desc_keys = {}
    for nm, sm in name2smi.items():
        if sm in smi_set:
            desc_keys.setdefault(nm, sm)

    def match_desc(name):
        if not name:
            return None
        sm = desc_keys.get(name)
        if sm:
            return sm
        subs = [k for k in desc_keys if name in k or k in name]
        if len(set(desc_keys[k] for k in subs)) == 1:
            return desc_keys[subs[0]]
        return None

    rows, stat = [], dict(n_all=n_all, n_c=n_c, flags={}, desc_hit=0,
                          T_missing=0, t_missing=0)
    for r in lf.itertuples():
        fl = []
        comp, basis, f1 = parse_si_al(r.si_al)
        fl += f1
        alk, f2 = parse_alkali(r.alkali, comp['n_Si'], comp['n_Al'],
                               comp['n_Ge'], basis)
        fl += f2
        os_q, os_name, os_oh, f3 = parse_osda(r.osda, comp['n_Si'],
                                              comp['n_Al'])
        fl += f3
        T_C, f4 = parse_T(r.T_C)
        fl += f4
        t_h, f5 = parse_T(r.t_h)
        fl += f5
        if np.isnan(T_C):
            stat['T_missing'] += 1
        if np.isnan(t_h):
            stat['t_missing'] += 1
        # HF/Ch+=x 型: 依赖 OSDA 量折算
        if isinstance(r.alkali, str):
            m = re.search(r'HF/Ch\+=([\d.]+)', r.alkali)
            if m and os_q > 0:
                alk['n_F'] = float(m.group(1)) * os_q
                fl = [x for x in fl if x not in
                      ('alk_f_present_unresolved', 'alk_unparsed')]
        # 归一化 (v1.1 口径 sum=1); 配方侧 M2O 与 alkali 列相加合并 —— 探针
        # 确认配方行 alkali 列均为定性 ('K-major + Cs-minor (9:1)' 型), 无双计
        moles = dict(comp)
        for k, v in alk.items():
            moles[k] = moles.get(k, 0.0) + v
        moles['n_osda1'] = os_q
        moles['n_OH'] = moles.get('n_OH', 0.0) + os_oh
        tot = sum(v for k, v in moles.items() if k != 'n_seed')
        norm = {k: (v / tot if k != 'n_seed' else 0.0)
                for k, v in moles.items()}
        # 派生比值 (scale 不变量, 用归一前摩尔)
        rSiAl = (comp['n_Si'] / comp['n_Al']) if comp['n_Al'] > 0 else np.nan
        rOH = moles.get('n_OH', 0.0)
        rNa = moles.get('n_Na', 0.0)
        rOS = os_q
        key = match_desc(os_name)
        if key is not None:
            stat['desc_hit'] += 1
        # charge_bal (qc_config 公式镜像; 修复 dict 键重复只留 n_Cs 的求和 bug)
        pos = sum(norm.get('n_' + e, 0.0) for e in ('Li', 'Na', 'K', 'Rb', 'Cs'))
        pos += 2 * (norm.get('n_Sr', 0.0) + norm.get('n_Ca', 0.0)
                    + norm.get('n_Ba', 0.0) + norm.get('n_Mg', 0.0))
        q = chg_map.get(str(key), np.nan) if key else np.nan
        pos += (0.0 if pd.isna(q) else q) * norm['n_osda1']
        neg = norm['n_Al'] + norm['n_OH'] + norm['n_F']
        cb = pos / neg if (neg > 0 and comp['n_Al'] > 0) else np.nan
        row = {
            'route_id': r.lit_id, 'label1': None, 'provenance': 'lit_failure',
            'batch': r.batch, 'doi': r.doi, 'year': r.year,
            'intended_framework': r.intended_framework,
            'actual_product': r.actual_product, 'fail_mode': r.fail_mode,
            'completeness': r.completeness,
            'n_Si': norm['n_Si'], 'n_Al': norm.get('n_Al', 0.0),
            'n_B': norm.get('n_B', 0.0), 'n_Ge': norm.get('n_Ge', 0.0),
            'n_Ga': norm.get('n_Ga', 0.0), 'n_Zn': norm.get('n_Zn', 0.0),
            'n_Co': norm.get('n_Co', 0.0), 'n_Rb': norm.get('n_Rb', 0.0),
            'n_Na': norm.get('n_Na', 0.0), 'n_K': norm.get('n_K', 0.0),
            'n_Li': norm.get('n_Li', 0.0), 'n_Cs': norm.get('n_Cs', 0.0),
            'n_Ca': 0.0, 'n_P': 0.0, 'n_Ti': 0.0, 'n_Fe': 0.0,
            'n_Mg': 0.0, 'n_V': 0.0, 'n_acid': 0.0,
            'n_F': norm.get('n_F', 0.0), 'n_OH': norm.get('n_OH', 0.0),
            'n_H2O': 0.0, 'n_seed': 0.0,
            'n_osda1': norm['n_osda1'], 'n_osda2': 0.0, 'n_osda3': 0.0,
            'ratio_SiAl': rSiAl,
            'ratio_OH_Si': rOH / comp['n_Si'] if comp['n_Si'] > 0 else np.nan,
            'ratio_Na_Si': (rNa / comp['n_Si']
                            if comp['n_Si'] > 0 else np.nan),
            'ratio_H2O_Si': 0.0,
            'ratio_osda_Si': (rOS / comp['n_Si']
                              if (rOS > 0 and comp['n_Si'] > 0) else np.nan),
            'ratio_charge_bal': cb, 'T_C': T_C, 't_h': t_h,
            'solvent': 0.0, 'stirring': 'no', 'osda1_desc_key': key,
            'osda1_name_matched': os_name,
        }
        lbl, main_cls, lflags = classify_product(r.actual_product)
        if r.lit_id in MANUAL_OVERRIDE:
            lbl, note = MANUAL_OVERRIDE[r.lit_id]
            fl.append('lf_manual_override')
        row['label1'] = lbl
        row['lf_main_cls'] = main_cls
        row['lf_zeo_secondary'] = lflags['zeo_sec']
        row['lf_low_cryst'] = lflags['low_cryst']
        row['lf_ambiguous'] = lflags['ambig']
        row['lf_unknown_main'] = lflags['unkn_main']
        pos_l = lbl == 'positive'
        rel_l = lbl == 'relative_neg'
        row['l2_pure'] = int(pos_l)
        row['l2_xtal'] = int(pos_l or rel_l)
        div = ((r.fail_mode in ('amorphous', 'dense', 'no_product')
                and (pos_l or rel_l))
               or (r.fail_mode in ('undesired', 'impurity_only', 'low_cryst')
                   and lbl in ('absolute_neg', 'no_product')))
        row['lf_diverge'] = int(div)
        row['lf_flags'] = ';'.join(sorted(set(fl)))
        rows.append(row)
        for f in set(fl):
            stat['flags'][f] = stat['flags'].get(f, 0) + 1
    return pd.DataFrame(rows), stat, db


# ============ 4. 特征: 主库拟合参数 transform lit_fail (A3 490 维) ============

def num(df, c):
    """build_ext 镜像: 与 l1_feat_iter.num 一致 (fillna(0) + clip(lower=0))"""
    return pd.to_numeric(df[c], errors='coerce').fillna(0.0).clip(lower=0.0)


def feat_raw(df, med_T, med_t):
    """build_features 镜像 (未 scaler), 列序一致"""
    X = {}
    for c in LOG_MOLE_COLS:
        v = pd.to_numeric(df[c], errors='coerce').fillna(0.0).clip(lower=0.0)
        X[f'lg_{c}'] = np.log10(v + 1e-6)
    rSiAl = df['ratio_SiAl'].fillna(RATIO_CAP).clip(upper=RATIO_CAP)
    X['lg_ratio_SiAl'] = np.log10(1 + rSiAl)
    for c in ['ratio_OH_Si', 'ratio_Na_Si', 'ratio_H2O_Si', 'ratio_osda_Si']:
        v = df[c].fillna(0.0).clip(lower=0.0)
        X[f'lg_{c}'] = np.log10(1 + v)
    X['T_C_100'] = df['T_C'].fillna(med_T) / 100.0
    X['lg_t_h'] = np.log10(1 + df['t_h'].fillna(med_t))
    X['b_osda'] = (df['n_osda1'].fillna(0) > 0).astype(float)
    X['b_seed'] = (df['n_seed'].fillna(0) > 0).astype(float)
    X['b_fluoride'] = (df['n_F'].fillna(0) > 0).astype(float)
    X['b_nonwater'] = (df['solvent'].fillna(0) > 0).astype(float)
    stir = df['stirring'].astype(str).str.lower()
    X['b_stir'] = stir.isin(['yes', 'true', '1']).astype(float)
    return pd.DataFrame(X, index=df.index)


def ext_raw(df, cb_med, cb_mean, cb_std):
    """build_ext 镜像 (未 scaler), 列序一致"""
    T = num(df, 'n_Si')
    for c in T_COLS[1:]:
        T = T + num(df, c)

    def lg_ratio(nr, den):
        v = (nr / den.replace(0, np.nan)).clip(lower=0.0)
        return np.log10(1.0 + v.fillna(0.0))

    X = {'lg_ratio_OH_T': lg_ratio(num(df, 'n_OH'), T),
         'lg_ratio_osda_T': lg_ratio(num(df, 'n_osda1') + num(df, 'n_osda2')
                                     + num(df, 'n_osda3'), T),
         'lg_ratio_F_T': lg_ratio(num(df, 'n_F'), T),
         'lg_ratio_H2O_T': lg_ratio(num(df, 'n_H2O'), T),
         'lg_ratio_Ge_T': lg_ratio(num(df, 'n_Ge'), T),
         'lg_ratio_seed_T': lg_ratio(num(df, 'n_seed'), T),
         'lg_ratio_Na_Al': lg_ratio(num(df, 'n_Na'), num(df, 'n_Al')),
         'lg_ratio_P_Al': lg_ratio(num(df, 'n_P'), num(df, 'n_Al'))}
    cb = pd.to_numeric(df['ratio_charge_bal'], errors='coerce').fillna(cb_med)
    X['z_ratio_charge_bal'] = (cb - cb_mean) / max(cb_std, 1e-12)
    return pd.DataFrame(X, index=df.index)


def build_features_matrix(df_main, df_lf):
    """主库拟合 → 两侧行特征 + 列名; 主库侧与 build_a3 逐位一致断言"""
    med_T, med_t = df_main['T_C'].median(), df_main['t_h'].median()
    cb_main = pd.to_numeric(df_main['ratio_charge_bal'], errors='coerce')
    cb_med = cb_main.median()
    cb_filled = cb_main.fillna(cb_med)
    cb_mean, cb_std = cb_filled.mean(), cb_filled.std()
    F_m, F_l = feat_raw(df_main, med_T, med_t), feat_raw(df_lf, med_T, med_t)
    sc = StandardScaler().fit(F_m.values)
    Z36 = np.vstack([sc.transform(F_m.values), sc.transform(F_l.values)])
    E_m = ext_raw(df_main, cb_med, cb_mean, cb_std)
    E_l = ext_raw(df_lf, cb_med, cb_mean, cb_std)
    sc2 = StandardScaler().fit(E_m.values)
    EXT = np.vstack([sc2.transform(E_m.values), sc2.transform(E_l.values)])
    # OSDA 描述符
    osda = osd.load(OSDA_DESC)
    desc_cols = osd.desc_cols(osda)
    assert len(desc_cols) == 444, 'OSDA 非 BE 描述符口径漂移: %d' % len(desc_cols)

    def desc_mat(df):
        J = df[['osda1_desc_key']].merge(
            osda[['osda_smiles'] + desc_cols],
            left_on='osda1_desc_key', right_on='osda_smiles',
            how='left', sort=False)
        assert len(J) == len(df), '描述符 join 行数漂移'
        return J[desc_cols].apply(pd.to_numeric,
                                  errors='coerce').values.astype(float)

    D_m = desc_mat(df_main)
    mu = np.nanmean(D_m, axis=0)
    sd = np.nanstd(D_m, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    D_l = desc_mat(df_lf)
    Dsc = np.vstack([(D_m - mu) / sd, (D_l - mu) / sd])
    b = np.vstack([df_main['osda1_desc_key'].notna().astype(float).values[:, None],
                   df_lf['osda1_desc_key'].notna().astype(float).values[:, None]])
    X = np.hstack([Z36, EXT, b, Dsc])
    cols = (list(F_m.columns) + list(E_m.columns) + ['b_osda1_desc']
            + ['od1_' + c for c in desc_cols])
    # 断言: 主库侧与 build_a3 逐位一致 (列序 + 数值)
    X_ref, cols_ref = build_a3(df_main)
    assert X_ref.shape == (len(df_main), 490), '参照维度漂移'
    assert cols == cols_ref, '列序与 build_a3 不一致'
    assert np.allclose(X[:len(df_main)], X_ref, atol=1e-9, equal_nan=True), \
        '主库特征复刻与 build_a3 不一致'
    return X, cols


# ============ 5. 主流程 ============

def main():
    log = []

    def P(msg=''):
        print(msg)
        log.append(str(msg))

    t_start = time.time()
    df = pd.read_parquet(DB)
    sp = pd.read_parquet(SPLITS)
    man = pd.read_csv(MANIFEST)
    dup_ids = set(man.loc[man['tier'] == 'A', 'route_id'].dropna())
    df = df[~df['route_id'].isin(dup_ids)].reset_index(drop=True)
    m = df[['route_id', 'label1', 'osda1_desc_key']].merge(
        sp, on='route_id', how='left')
    assert m['split_random'].notna().all(), '主库切分对不齐'

    lf_df, stat, db = build_lf_rows()
    n_lf = len(lf_df)
    P('M6 行动项 A5: lit_fail 并入 L2 训练侧 v0.1 —— xtal 主口径三切分对照')
    P('=' * 68)
    P('【一、映射方案 (数据博士预审口径, 详见脚本 docstring)】')
    P('  标签: 实际产物协议 (label1 v2 主相优先) —— 意图不进特征空间, 按结果事件')
    P('  标注; 主相 amorphous/layered/致密/unknown→absolute_neg, No product→')
    P('  no_product, 纯沸石主相→positive, 带次相→relative_neg; 低结晶度只记')
    P('  旗标; clathrasil (NON/AST/MTN/RUT) 按 IZA 惯例判沸石, 与策展 fail_mode')
    P('  分歧记 lf_diverge 供周例会复核; C 级 1 行不入库 (WP1 §2.3)。')
    P('  组成: SiO2=1 基准元素摩尔 → 归一化 sum=1 (v1.1 口径 1); B 级缺项置 0')
    P('  + 指示位旗标; T_C/t_h 区间中点/边界值; 氢氧化物 OSDA 计 n_OH。')
    P('  切分: random 全 train; heldout 留出骨架 token 排除; time year≤2018')
    P('  train / ≥2019 excluded; 测试集与 v0 逐位一致, 只加训练行。')
    P('')
    P('入库: lit_fail v1 全量 %d 条 = A %d + B %d + C %d → 并入 %d (A+B),'
      ' C 级剔除 %d' % (stat['n_all'], int((lf_df.completeness == 'A').sum()),
                       int((lf_df.completeness == 'B').sum()), stat['n_c'],
                       n_lf, stat['n_c']))
    # DOI 泄漏审计
    doi_main = set(db['doi'].dropna().astype(str))
    doi_lf = set(lf_df['doi'].dropna().astype(str))
    n_doi_ov = len(doi_main & doi_lf)
    P('泄漏审计 ①: lit_fail DOI 与主库重叠 = %d (24 个非空 DOI 全量比对)'
      ' → %s' % (n_doi_ov, 'PASS' if n_doi_ov == 0 else 'FAIL 需排除'))
    # heldout token 排除
    held_pat = re.compile(
        r'\b(LTL|SAS|ITG|RTH|SZR|LTA)\b|zeolite A|Zeolite A|(?<![\w(])L(?![\w)])',
        re.I)
    held_hit = lf_df['intended_framework'].fillna('').str.contains(
        held_pat, regex=True) | lf_df['actual_product'].fillna(
        '').str.contains(held_pat, regex=True)
    lf_df['split_random'] = 'train'
    lf_df['split_heldout'] = np.where(held_hit, 'excluded', 'train')
    lf_df['split_time_2018'] = np.where(lf_df['year'] >= 2019,
                                        'excluded', 'train')
    P('泄漏审计 ②: heldout 留出骨架 (LTL/SAS/ITG/RTH/SZR/LTA, 含 zeolite A/'
      '单相 L 词) token 命中 %d 行 → split_heldout=excluded; 其余 %d train'
      % (int(held_hit.sum()), n_lf - int(held_hit.sum())))
    P('泄漏审计 ③: time 纪律 —— year≥2019 %d 行 excluded (防后见污染 ≤2018'
      ' 训练侧), year≤2018 %d 行 train'
      % (int((lf_df['split_time_2018'] == 'excluded').sum()),
         int((lf_df['split_time_2018'] == 'train').sum())))
    P('')
    P('【二、解析覆盖 (B 级置 0 + 指示位; 旗标计数)】')
    cov = {'si_al': 0, 'alkali': 0, 'osda': 0}
    for _, r in lf_df.iterrows():
        f = set(str(r['lf_flags']).split(';')) if r['lf_flags'] else set()
        if not (f & {'si_al_missing', 'si_al_unparsed',
                     'si_al_hetero_ambiguous', 'si_al_metakaolin'}):
            cov['si_al'] += 1
        if not (f & {'alkali_missing', 'alk_unparsed'}):
            cov['alkali'] += 1
        if not (f & {'osda_missing', 'osda_present_unquantified'}):
            cov['osda'] += 1
    P('  si_al 解析定量 %d/%d (%.0f%%) | alkali %d/%d (%.0f%%) | osda 定量'
      ' %d/%d (%.0f%%) | OSDA 描述符命中 %d (%.0f%%) | T_C 缺 %d / t_h 缺 %d'
      % (cov['si_al'], n_lf, 100 * cov['si_al'] / n_lf,
         cov['alkali'], n_lf, 100 * cov['alkali'] / n_lf,
         cov['osda'], n_lf, 100 * cov['osda'] / n_lf,
         stat['desc_hit'], 100 * stat['desc_hit'] / n_lf,
         stat['T_missing'], stat['t_missing']))
    P('  旗标分布: %s' % dict(sorted(stat['flags'].items(),
                                     key=lambda x: -x[1])))
    lf_src = pd.read_csv(LITFAIL)
    unp_ids = lf_df.loc[lf_df['lf_flags'].str.contains(
        'si_al_unparsed|si_al_hetero', regex=True, na=False), 'route_id']
    samp = lf_src.loc[lf_src['lit_id'].isin(unp_ids), ['lit_id', 'si_al']].head(8)
    P('  置 0 未解析 si_al 样例 (前 8): %s' % samp.values.tolist())
    P('')
    P('【三、标签协议解析交叉表 (fail_mode × label1_lf)】')
    ct = pd.crosstab(lf_df['fail_mode'], lf_df['label1'])
    P(ct.to_string())
    P('  L2 双口径计数: l2_pure 1/0 = %d/%d, l2_xtal 1/0 = %d/%d'
      % (int(lf_df['l2_pure'].sum()), int((1 - lf_df['l2_pure']).sum()),
         int(lf_df['l2_xtal'].sum()), int((1 - lf_df['l2_xtal']).sum())))
    P('  策展 fail_mode ↔ 派生标签分歧 (lf_diverge=1): %d 行; 复核结论 (2026-'
      '10-08, 周例会项提前): 54 行维持 (协议性分歧: low_cryst 31 非晶为主+沸石'
      '次相首段主相 / undesired 12 layered·analcime·unknown / dense 6 clathrasil'
      ' IZA 惯例 / impurity_only 2 sodalite·cancrinite 致密) + 2 行人工改判'
      ' lf_manual_override (litfail_v1_130 relative_neg→absolute_neg: ECS-9'
      ' (amorphous) 括注非晶; litfail_v1_079 absolute_neg→relative_neg: CIT-6'
      ' 主相带杂质, 采纳策展 impurity_only); 双口径计数不变。'
      % int(lf_df['lf_diverge'].sum()))
    P('  低结晶旗标 lf_low_cryst=1: %d 行 | 次相含沸石 lf_zeo_secondary=1: %d 行'
      % (int(lf_df['lf_low_cryst'].sum()),
         int(lf_df['lf_zeo_secondary'].sum())))
    P('  合成行校验: 归一化 n_ 行和 = %.6f±%.1e (v1.1 口径 sum=1)'
      % (lf_df[['n_Si', 'n_Al', 'n_B', 'n_Ge', 'n_Ga', 'n_Zn', 'n_Na', 'n_K',
                'n_Li', 'n_Cs', 'n_F', 'n_OH', 'n_osda1']].sum(axis=1).mean(),
         lf_df[['n_Si', 'n_Al', 'n_B', 'n_Ge', 'n_Ga', 'n_Zn', 'n_Na', 'n_K',
                'n_Li', 'n_Cs', 'n_F', 'n_OH', 'n_osda1']].sum(axis=1).std()))
    P('')
    # ---- 特征 ----
    df_lf_cols = lf_df.copy()
    X, cols = build_features_matrix(df, df_lf_cols)
    X_main, X_lf = X[:len(df)], X[len(df):]
    cols_ok = X.shape[1] == 490
    P('【四、特征协议】A3 490 维空间冻结; 主库 %d 行特征与 build_a3 逐位一致'
      ' (assert PASS); lit_fail %d 行以主库拟合参数 (Scaler/中位数/z 统计)'
      ' transform → v0.1 与 v0 差异 = 且仅 = 训练侧加行。%s'
      % (len(df), n_lf, '' if cols_ok else ' 维度漂移!'))
    P('')
    # ---- 标签/切分矩阵 ----
    y_pure_m = np.where(df['label1'] == 'positive', 1.0, 0.0)
    y_xtal_m = np.where(df['label1'].isin(['positive', 'relative_neg']), 1.0, 0.0)
    y_pure = np.concatenate([y_pure_m, lf_df['l2_pure'].values.astype(float)])
    y_xtal = np.concatenate([y_xtal_m, lf_df['l2_xtal'].values.astype(float)])
    sr = np.concatenate([m['split_random'].values,
                         lf_df['split_random'].values])
    sh = np.concatenate([m['split_heldout'].values,
                         lf_df['split_heldout'].values])
    st = np.concatenate([m['split_time_2018'].values,
                          lf_df['split_time_2018'].values])
    params = dict(objective='binary', learning_rate=0.05, num_leaves=31,
                  min_child_samples=100, lambda_l2=1.0, max_delta_step=1.0,
                  seed=SEED, verbosity=-1)

    def run(tr, te, y, rounds=None, va=None):
        dtr = lgb.Dataset(X[tr], label=y[tr])
        if va is not None:
            dva = lgb.Dataset(X[va], label=y[va], reference=dtr)
            b = lgb.train(params, dtr, num_boost_round=400,
                          valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            rounds = b.best_iteration or -1
        else:
            b = lgb.train(params, dtr, num_boost_round=rounds)
        p = b.predict(X[te], num_iteration=rounds)
        return b, rounds, p

    # ---- v0 锚点核对 (仅主库行) ----
    P('【五、v0 锚点核对 (仅主库行, 协议/seed 同 v0)】')
    va_m = np.zeros(len(X), bool)
    tr_m, va_r, te_r = (sr == 'train'), (sr == 'val'), (sr == 'test')
    va_m[:len(df)] = True
    tr0 = tr_m & va_m
    va0 = va_r & va_m
    te0 = te_r & va_m
    anchor = {}
    for tag, y in (('pure', y_pure), ('xtal', y_xtal)):
        _, best, p = run(tr0, te0, y, va=va0)
        a = roc_auc_score(y[te0], p)
        ap = average_precision_score(y[te0], p)
        anchor[tag] = (best, a, ap)
        P('  %s: random 早停@%d AUROC=%.4f AUCPR=%.4f (v0: %.4f/%.4f) -> %s'
          % (tag, best, a, ap, V0['%s_random' % tag],
             V0['%s_random_ap' % tag],
             '一致' if abs(a - V0['%s_random' % tag]) < 1e-3 else '漂移!'))
        assert abs(a - V0['%s_random' % tag]) < 1e-3, \
            'v0 锚点漂移: %s random AUROC' % tag
        assert abs(ap - V0['%s_random_ap' % tag]) < 1e-3, \
            'v0 锚点漂移: %s random AUCPR' % tag
    for tag, y in (('pure', y_pure), ('xtal', y_xtal)):
        for sname, sv, v0k in (('heldout', sh, 'heldout'),
                               ('time_2018', st, 'time')):
            tr2 = (sv == 'train') & va_m
            te2 = (sv == 'test') & va_m
            _, _, p2 = run(tr2, te2, y, rounds=anchor['pure'][0])
            a2 = roc_auc_score(y[te2], p2)
            P('  %s %s: AUROC=%.4f (v0: %.4f) -> %s'
              % (tag, sname, a2, V0['%s_%s' % (tag, v0k)],
                 '一致' if abs(a2 - V0['%s_%s' % (tag, v0k)]) < 1e-3 else '漂移!'))
            assert abs(a2 - V0['%s_%s' % (tag, v0k)]) < 1e-3, \
                'v0 锚点漂移: %s %s AUROC' % (tag, sname)
    P('  六读数锚点 assert 全部 PASS (AUROC×6 + AUCPR×2, |Δ|<1e-3)')
    P('')
    # ---- v0.1 (加 lit_fail 训练行) ----
    P('【六、v0.1 三切分对照 (xtal 主口径 Q19 / pure 参照; 测试集 = v0 原集)】')
    res = {}
    lf_masks = {'random': np.zeros(len(X), bool),
                'heldout': np.zeros(len(X), bool),
                'time_2018': np.zeros(len(X), bool)}
    lf_masks['random'][len(df):] = lf_df['split_random'].values == 'train'
    lf_masks['heldout'][len(df):] = lf_df['split_heldout'].values == 'train'
    lf_masks['time_2018'][len(df):] = lf_df['split_time_2018'].values == 'train'
    P('  lit_fail 训练行计入: random %d / heldout %d / time_2018 %d'
      % (int(lf_masks['random'].sum()), int(lf_masks['heldout'].sum()),
         int(lf_masks['time_2018'].sum())))
    for tag, y in (('pure', y_pure), ('xtal', y_xtal)):
        tr = (sr == 'train') | lf_masks['random']
        va, te = (sr == 'val'), (sr == 'test')
        b_r, best, p_r = run(tr, te, y, va=va)
        a = roc_auc_score(y[te], p_r)
        ap = average_precision_score(y[te], p_r)
        res[tag, 'random'] = (int(te.sum()), a, ap, best)
        d_a = 100 * (a - V0['%s_random' % tag])
        d_ap = 100 * (ap - V0['%s_random_ap' % tag])
        P('')
        P('  ---- 口径 %s (%s) ----' % (tag, '主口径 结晶可行性' if tag == 'xtal'
                                       else '参照口径 纯相可行性'))
        P('  random (val 早停@%d): test AUROC=%.4f (Δ%+.2fpp vs v0 %.4f)'
          ' AUCPR=%.4f (Δ%+.2fpp vs v0 %.4f)'
          % (best, a, d_a, V0['%s_random' % tag], ap, d_ap,
             V0['%s_random_ap' % tag]))
        for sname, sv in (('heldout', sh), ('time_2018', st)):
            tr2 = (sv == 'train') | lf_masks[sname]
            te2 = sv == 'test'
            _, _, p2 = run(tr2, te2, y, rounds=res['pure', 'random'][3])
            a2 = roc_auc_score(y[te2], p2)
            res[tag, sname] = (int(te2.sum()), a2)
            d2 = 100 * (a2 - V0['%s_%s' % (tag, sname.split('_')[0]
                                            if sname == 'heldout' else 'time')])
            P('  %s (锚定 pure best@%d): test %d 行 AUROC=%.4f'
              ' (Δ%+.2fpp vs v0 %.4f)'
              % (sname, res['pure', 'random'][3], int(te2.sum()), a2, d2,
                 V0['%s_%s' % (tag, sname.split('_')[0]
                               if sname == 'heldout' else 'time')]))
    # 特征重要性 (xtal 主口径 random v0.1, 复用 §六 循环末次 b_r, 免重训)
    P('')
    P('  特征重要性 Top10 (gain, l2_xtal 主口径 random v0.1):')
    imp = b_r.feature_importance(importance_type='gain')
    for i in np.argsort(-imp)[:10]:
        P('    %-24s %.1f' % (cols[i], imp[i]))
    # ---- 训练集落盘 ----
    main_ts = pd.read_parquet(V0_TRAINSET)
    main_ts['provenance'] = 'main'
    lf_out = lf_df[[
        'route_id', 'label1', 'l2_pure', 'l2_xtal', 'split_random',
        'split_heldout', 'split_time_2018', 'provenance', 'batch', 'doi',
        'year', 'intended_framework', 'actual_product', 'fail_mode',
        'completeness', 'lf_main_cls', 'lf_zeo_secondary', 'lf_low_cryst',
        'lf_ambiguous', 'lf_unknown_main', 'lf_diverge', 'lf_flags',
        'osda1_name_matched', 'ratio_SiAl', 'ratio_charge_bal']].copy()
    ts = pd.concat([main_ts, lf_out], ignore_index=True, sort=False)
    ts.to_parquet(TRAINSET, engine='pyarrow', compression='zstd', index=False)
    # ---- 收口 ----
    P('')
    P('【七、判读 (Q20 增益定位对照)】')
    P('  预期增益侧: heldout/外推 (文献失败多覆盖未见拓扑空间) 与 AUCPR'
      ' (负样本密度); random 侧不设预期。')
    P('  实测: xtal random Δ%+.2fpp / AUCPR Δ%+.2fpp; heldout Δ%+.2fpp;'
      ' time Δ%+.2fpp (v0.1 - v0)'
      % (100 * (res['xtal', 'random'][1] - V0['xtal_random']),
         100 * (res['xtal', 'random'][2] - V0['xtal_random_ap']),
         100 * (res['xtal', 'heldout'][1] - V0['xtal_heldout']),
         100 * (res['xtal', 'time_2018'][1] - V0['xtal_time'])))
    P('  主口径 xtal random AUROC=%.4f (G2 线 0.85 -> %s); 后续: Q28 执行序'
      ' ② 难例专项 T1.3 → ③ 神经基线接入 (M7) → ④ 增量数据滚动扩容'
      % (res['xtal', 'random'][1],
         '达标' if res['xtal', 'random'][1] >= 0.85 else '未达'))
    P('')
    P('收口声明:')
    P('  1. 数据版本: 训练集 v0.1 = zeosyn_db_v1_1 + splits_v1_1 (主库'
      ' 23,488 行, Tier A 剔 901) + lit_fail_v1 A+B 308 行'
      ' (provenance=lit_failure); 测试集与 v0 逐位一致。')
    P('  2. 标签协议: lit_fail 按实际产物 label1 v2 解析 (主相优先), 非按')
    P('  失败意图折叠 (Q19 系统性标签噪声教训); 策展分歧 %d 行已复核'
      ' (54 维持 + 2 人工改判 MANUAL_OVERRIDE), lf_diverge 旗标留档。'
      % int(lf_df['lf_diverge'].sum()))
    P('  3. 双口径并报 (Q19): xtal 主口径 %.4f / pure 参照 %.4f;'
      ' G2 0.85 验收线绑定 xtal。'
      % (res['xtal', 'random'][1], res['pure', 'random'][1]))
    P('  4. 泄漏审计: DOI 重叠 0; heldout token 排除 %d 行; time ≥2019'
      ' excluded %d 行; C 级不入库。'
      % (int(held_hit.sum()),
         int((lf_df['split_time_2018'] == 'excluded').sum())))
    P('产出: %s (%d 行 = %d 主库 + %d lit_fail), %s; 耗时 %.0fs'
      % (TRAINSET, len(ts), len(main_ts), n_lf, REPORT,
         time.time() - t_start))
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log) + '\n')


if __name__ == '__main__':
    main()
