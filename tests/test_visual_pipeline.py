import numpy as np
import pandas as pd
import pytest

from banc_control import parse_visual_activity

MAP_FILE = "visual_pipeline/flyvis_banc_map.csv"


def test_map_file_schema():
    m = pd.read_csv(MAP_FILE)
    assert {"eye", "flyvis_index", "flyvis_type", "u", "v", "banc_888_id", "banc_cell_type"} <= set(m.columns)
    assert set(m.eye) == {"left", "right"}
    assert not m.duplicated(["eye", "flyvis_index"]).any()
    assert m.banc_888_id.notna().all()


def test_retina_mapper_is_bijection():
    pytest.importorskip("flygym")
    pytest.importorskip("flyvis")
    from visual_pipeline.retina_mapper import RetinaMapper

    mapper = RetinaMapper()
    for eye in ("left", "right"):
        assert sorted(mapper.flygym_to_flyvis_idx[eye]) == list(range(721))


def test_bridge_emits_contract_records():
    pytest.importorskip("flygym")
    flyvis = pytest.importorskip("flyvis")
    if not (flyvis.results_dir / "flow/0000/000").exists():
        pytest.skip("brak wag FlyVis: flyvis download-pretrained")
    from visual_pipeline import VisionBridge

    bridge = VisionBridge()
    frame = np.full((bridge.retina.nrows, bridge.retina.ncols, 3), 128, np.uint8)
    records = parse_visual_activity([r.__dict__ for r in bridge.step(frame, frame)])
    assert len(records) == len(bridge.root_ids) > 0
    assert len({r.banc_root_id for r in records}) == len(records)
    assert np.isfinite([r.activity for r in records]).all()
