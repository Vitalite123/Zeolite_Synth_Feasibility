# -*- coding: utf-8 -*-
"""
OSDA 描述符列口径唯一事实源
====================================================================================
修复 binding energy 列名缺陷 (2026-10-22):
  osda_descriptors 表 830 列中, BE 族有两种命名 (binding energy (FW) /
  binding_energy (FW)), 两组同 token、同值。旧规则 startswith('binding_energy')
  只排除其中一组, 另 192 列进入特征空间 (违背 2026-10-04「BE 不混入建模特征」决议)。

口径:
  830 列 = 2 元数据 + 444 非 BE 描述符 + 384 BE (192 token x 2 命名)
  canonical 后 = 2 + 444 + 192 = 638 列; BE 统一为 binding_energy (FW)
数据表不动; 本模块只在读取层归一化。

用法:
  osda = load(path)             # 读表 + canonicalize
  D = desc_cols(osda)           # 444 非 BE 描述符
  B = be_cols(osda)             # 192 canonical BE
"""
import re

import numpy as np
import pandas as pd

META_COLS = ('osda_row_id', 'osda_smiles')
_BE_RE = re.compile(r'^binding[\s_]+energy\s*\(([A-Za-z0-9_\-]+)\)\s*$', re.I)
TOKEN_FMT = 'binding_energy (%s)'


def be_token(col):
    """BE 列名 -> 框架 token; 非 BE 列返回 None"""
    m = _BE_RE.match(str(col))
    return m.group(1) if m else None


def is_be(col):
    return be_token(col) is not None


def canonicalize(osda):
    """把两种 BE 命名合并为 canonical 列族并去重 (192 组)。
    组内数值必须逐位一致 (equal_nan), 否则直接报错。"""
    groups = {}
    for c in osda.columns:
        t = be_token(c)
        if t is not None:
            groups.setdefault(t, []).append(c)
    for t, cols in groups.items():
        if len(cols) > 1:
            base = pd.to_numeric(osda[cols[0]], errors='coerce').values
            for c in cols[1:]:
                other = pd.to_numeric(osda[c], errors='coerce').values
                if not np.allclose(base, other, rtol=0.0, atol=0.0, equal_nan=True):
                    raise AssertionError('BE 重复列数值不一致: %s' % (cols,))
    out, seen = {}, set()
    for c in osda.columns:
        t = be_token(c)
        if t is None:
            out[c] = osda[c]
        elif t not in seen:
            seen.add(t)
            out[TOKEN_FMT % t] = osda[c]
    return pd.DataFrame(out, index=osda.index)


def load(path):
    """读 parquet 并归一化 BE 列名 (冻结表原样, 不改盘)"""
    return canonicalize(pd.read_parquet(path))


def desc_cols(osda):
    return [c for c in osda.columns if c not in META_COLS and not is_be(c)]


def be_cols(osda):
    return [c for c in osda.columns if is_be(c)]
