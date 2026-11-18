import json
from collections import Counter

import pytest

from src.config import ROOT

IDX = ROOT / "data/position_index.json"


@pytest.mark.skipif(not IDX.exists(), reason="нет data/position_index.json")
def test_v2_position_index_shape_and_single_fact_per_variant():
    idx = json.loads(IDX.read_text(encoding="utf-8"))
    if len(idx) != 57:
        pytest.skip("индекс не V2 (13 фактов на 30K и 6 на 100K, позиции 0/50/100)")
    c = Counter((i["size"], i["target_pct"]) for i in idx)
    assert c == {("30k", 0): 13, ("30k", 50): 13, ("30k", 100): 13, ("100k", 0): 6, ("100k", 50): 6, ("100k", 100): 6}
    gold = {f["fact_id"]: f for f in json.loads((ROOT / "data/gold.json").read_text(encoding="utf-8"))["facts"]}
    first = idx[0]
    f = ROOT / first["file"]
    if f.exists():
        assert f.read_text(encoding="utf-8").count(gold[first["fact_id"]]["statement"]) == 1
    assert all(abs(i["actual_pct"] - i["target_pct"]) <= 3 for i in idx)
