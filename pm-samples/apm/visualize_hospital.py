"""Generate Petri net and DFG images for hospital_log_high dataset.

Strategy for square-ish + readable:
  - rankdir=TB (top-bottom) for natural flow
  - large nodesep to push parallel branches apart horizontally (widens layout)
  - small ranksep to bring sequential ranks closer vertically (shortens layout)
  - high dpi for crisp text
  - NO forced size/ratio (those scale nodes and break readability)
"""
from pathlib import Path
import pm4py
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.dfg import visualizer as dfg_visualizer

XES = Path("benchmark/hospital/hospital_log_high.xes")
OUT_DIR = Path("HARDER")
OUT_DIR.mkdir(exist_ok=True)


def tune(gviz, nodesep="1.2", ranksep="0.35", dpi="150", rankdir="TB"):
    gviz.graph_attr["rankdir"] = rankdir
    gviz.graph_attr["nodesep"] = nodesep   # horizontal gap between siblings
    gviz.graph_attr["ranksep"] = ranksep   # vertical gap between ranks
    gviz.graph_attr["dpi"] = dpi
    return gviz


print(f"Loading {XES} ...", flush=True)
log = pm4py.read_xes(str(XES))

print("Discovering Petri net (Inductive Miner) ...", flush=True)
net, im, fm = pm4py.discover_petri_net_inductive(log)
pn_gviz = pn_visualizer.apply(net, im, fm)
tune(pn_gviz)
pn_path = OUT_DIR / "hospital_petri_net.png"
pn_visualizer.save(pn_gviz, str(pn_path))
print(f"  -> {pn_path}", flush=True)

print("Discovering DFG ...", flush=True)
dfg, start_acts, end_acts = pm4py.discover_dfg(log)
dfg_gviz = dfg_visualizer.apply(
    dfg,
    parameters={
        dfg_visualizer.Variants.FREQUENCY.value.Parameters.START_ACTIVITIES: start_acts,
        dfg_visualizer.Variants.FREQUENCY.value.Parameters.END_ACTIVITIES: end_acts,
    },
)
tune(dfg_gviz)
dfg_path = OUT_DIR / "hospital_dfg.png"
dfg_visualizer.save(dfg_gviz, str(dfg_path))
print(f"  -> {dfg_path}", flush=True)

print("Done.")
