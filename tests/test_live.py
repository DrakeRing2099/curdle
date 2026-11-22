"""The live mirror: following the feed must keep the twin closer to reality than ignoring it."""
import numpy as np
from dairytwin.live import LiveMirror


def run(sync):
    m = LiveMirror()
    m.sync = sync
    errs = []
    for d in range(45):
        if d == 15:
            m.inject("heat", days=14)
            m.inject("demand", 1.4, days=14)
        errs.append(m.step()["stock_err"])
        if d == 25:
            m.mid_scale = (m.scale_est, m.scale_true)
    return np.mean(errs[20:]), m


def test_following_the_feed_beats_ignoring_it():
    off, _ = run("off")
    full, m = run("full")
    assert full < 0.75 * off                               # only 30% of shops have sensors, so not zero
    est, true = m.mid_scale
    assert abs(est - true) < 0.15                          # demand level is recovered from footfall
    assert m.twin.regimes.weather == m.reality.regimes.weather


def test_forecast_shapes_and_common_futures():
    m = LiveMirror(futures=3, horizon=5)
    for _ in range(10):
        m.step()
    f = m.forecast(["Habit"])
    p = f["policies"]["Habit"]
    assert len(p["fill_mean"]) == 5 and all(0 <= v <= 1 for v in p["fill_mean"])
    assert m.twin.day == m.reality.day == 10               # forecasting leaves the live twin untouched
