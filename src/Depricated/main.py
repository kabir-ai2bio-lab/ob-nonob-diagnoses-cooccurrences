# Figures produced:
#   [Global]
#     1) Bar chart: Top-20 diagnoses by betweenness
#     2) Community meta-graph (context)
#   [Pregnancy]
#     3) Ego network of the top bridge diagnosis
#     4) Bar chart: Top-20 by betweenness (pregnancy)
#     5) Δ-bridging bar: betweenness z (pregnancy − overall)
#     6) Δ-bridging bar: participation z (pregnancy − overall)
#     7) Cross-community backbone (OB ↔ non-OB)
# ─────────────────────────────────────────────────────────────────────────────

import os
import itertools as it
import heapq
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
})

# Import project config
from config import (
    verify_data_files,
    DIAG_PATH,
    CONCEPT_PATH,
    FIGURES_DIR,
    MIN_DIAG_COUNT,
    TOP_N_DIAG,
    EDGE_MIN_WEIGHT,
)

# Verify data exists
verify_data_files()

# Ensure figures dir exists
os.makedirs(FIGURES_DIR, exist_ok=True)

# Utility function. It handles printing tables, label wrapping, saving figures
def print_table(title, df, cols=None, head=None):
    print("\n" + "="*80)
    print(title)
    print("="*80)
    if cols is not None:
        df = df.loc[:, [c for c in cols if c in df.columns]]
    if head is not None:
        df = df.head(head)
    with pd.option_context("display.max_rows", 200,
                           "display.max_columns", 20,
                           "display.width", 140,
                           "display.colheader_justify", "left",
                           "display.max_colwidth", 60):
        print(df.to_string(index=False))

# Wrap a list-like of strings to a fixed width. This is used for tick labels.
def _wrap_text_list(labels, width=30):
    
    from textwrap import fill
    out = []
    for s in labels:
        s = "" if s is None else str(s)
        # shrink very long uninterrupted tokens to avoid overflow (rare)
        if len(s) > 120 and " " not in s:
            s = s[:120] + "…"
        out.append(fill(s, width=width, break_long_words=True, replace_whitespace=False))
    return out

# Wrap current y tick labels on an Axes in-place.
def wrap_y_ticklabels(ax, width=30):
    
    ticks = ax.get_yticks()
    labels = [l.get_text() for l in ax.get_yticklabels()]
    if not labels or all(lbl == "" for lbl in labels):
        return
    wrapped = _wrap_text_list(labels, width=width)
    ax.set_yticks(ticks)
    ax.set_yticklabels(wrapped)


def savefig_safe(filename):
    path = os.path.join(FIGURES_DIR, filename)
    # tight bbox to prevent titles/labels from clipping
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"[Saved] {path}")

# Weighted betweenness centrality with a tqdm progress bar.
def betweenness_centrality_tqdm(G, weight="distance", normalized=True, desc="Betweenness"):
    
    betw = dict.fromkeys(G, 0.0)
    nodes = list(G)
    n = len(nodes)

    with tqdm(nodes, desc=desc, ncols=80, unit="node") as pbar:
        for s in pbar:
            # Dijkstra from source s
            S = []
            P = {v: [] for v in G}
            sigma = dict.fromkeys(G, 0.0)
            dist = dict.fromkeys(G, float("inf"))
            sigma[s] = 1.0
            dist[s] = 0.0

            Q = [(0.0, s)]
            while Q:
                dv, v = heapq.heappop(Q)
                if dv > dist[v]:
                    continue
                S.append(v)
                for w, ed in G[v].items():
                    vw = float(ed.get(weight, 1.0))
                    alt = dv + vw
                    if dist[w] > alt:
                        dist[w] = alt
                        heapq.heappush(Q, (alt, w))
                        sigma[w] = sigma[v]
                        P[w] = [v]
                    elif dist[w] == alt:
                        sigma[w] += sigma[v]
                        P[w].append(v)

            # Accumulation
            delta = dict.fromkeys(G, 0.0)
            while S:
                w = S.pop()
                for v in P[w]:
                    if sigma[w] != 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    betw[w] += delta[w]

    if normalized and n > 2:
        scale = 1.0 / ((n - 1) * (n - 2) / 2.0)
        for v in betw:
            betw[v] *= scale
    else:
        if not G.is_directed():
            for v in betw:
                betw[v] /= 2.0
    return betw


# ============================ LOAD ===========================================
usecols = ["person_id", "visit_occurrence_id", "condition_concept_id"]
df = pd.read_csv(DIAG_PATH, usecols=usecols, low_memory=False)

df = df.dropna(subset=["person_id", "visit_occurrence_id", "condition_concept_id"])
df["patient_id"] = df["person_id"].astype(str)
df["visit_id"]   = df["visit_occurrence_id"].astype(str)
df["diag_code"]  = df["condition_concept_id"].astype(str)
df = df[["patient_id", "visit_id", "diag_code"]]

# ============================ FILTER =========================================
df = df[df["diag_code"].str.strip().ne("")]  # drop empty codes

vc = df["diag_code"].value_counts()
if MIN_DIAG_COUNT:
    df = df[df["diag_code"].isin(vc[vc >= MIN_DIAG_COUNT].index)]

# keep only the TOP_N_DIAG most common diagnosis codes
if TOP_N_DIAG is not None and TOP_N_DIAG > 0:
    vc2 = (
        df["diag_code"].value_counts()
          .rename("cnt")
          .reset_index()
          .rename(columns={"index": "diag_code"})
          .sort_values(["cnt", "diag_code"], ascending=[False, True])
    )
    keep = set(vc2.head(TOP_N_DIAG)["diag_code"])
    df = df[df["diag_code"].isin(keep)]

# ============================ EDGES ==========================================
pairs_counter = Counter()
for _, grp in df.groupby(["patient_id", "visit_id"], sort=False):
    codes = sorted(set(grp["diag_code"]))  # unique + sorted for stability
    for a, b in it.combinations(codes, 2):
        pairs_counter[(a, b)] += 1

edges = pd.DataFrame(
    [(a, b, w) for (a, b), w in pairs_counter.items()],
    columns=["diag_a", "diag_b", "weight"]
)
edges = edges[edges["weight"] >= EDGE_MIN_WEIGHT]
edges = edges.sort_values(["diag_a", "diag_b"], kind="mergesort").reset_index(drop=True)

print(f"Built {len(edges)} edges over {df['diag_code'].nunique()} diagnosis nodes (pre-isolate-prune).")

# ============================ GRAPH ==========================================
G = nx.Graph()
for a, b, w in edges.itertuples(index=False):
    w = int(w)
    G.add_edge(a, b, weight=w, distance=1.0 / w)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

# ============================ COMMUNITIES ====================================
comm_id_of = {}
try:
    import community as community_louvain 

    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    comm_id_of = partition

    from collections import defaultdict as _dd
    groups = _dd(set)
    for n, c in partition.items():
        groups[c].add(n)
    communities = list(groups.values())

    from networkx.algorithms.community.quality import modularity as nx_modularity
    mod_val = nx_modularity(G, communities, weight="weight")
    print(f"[Louvain] Detected {len(communities)} communities; modularity={mod_val:.3f}")

except Exception as e:
    print(f"WARNING: python-louvain not available ({e}); using deterministic greedy modularity.")
    mapping = {n: i for i, n in enumerate(sorted(G.nodes()))}
    G_relabeled = nx.relabel_nodes(G, mapping, copy=True)

    from networkx.algorithms.community import greedy_modularity_communities, modularity
    comm_sets = list(greedy_modularity_communities(G_relabeled, weight="weight"))

    inv_map = {i: n for n, i in mapping.items()}
    communities = [{inv_map[i] for i in s} for s in comm_sets]

    for cid, comm in enumerate(communities):
        for n in comm:
            comm_id_of[n] = cid

    mod_val = modularity(G_relabeled, comm_sets, weight="weight")
    print(f"[Greedy] Detected {len(communities)} communities; modularity={mod_val:.3f}")

# ============================ BRIDGING METRICS ================================
print("\n[Computing betweenness centrality for global graph...]")
betw = betweenness_centrality_tqdm(G, weight="distance", normalized=True, desc="Global betweenness")
articulation = set(nx.articulation_points(G))

def participation_coeff(G, comm_of):
    P = {}
    for u in G.nodes():
        k = G.degree(u)
        if k == 0:
            P[u] = 0.0
            continue
        counts = defaultdict(int)
        for v in G.neighbors(u):
            c = comm_of.get(v, -1)
            counts[c] += 1
        P[u] = 1.0 - sum((cnt / k) ** 2 for cnt in counts.values())
    return P

part = participation_coeff(G, comm_id_of)

# ============================ NODE TABLE ==========================
rows = []
for n in G.nodes():
    rows.append({
        "diagnosis": n,
        "community_id": comm_id_of.get(n, -1),
        "degree": G.degree(n),
        "weighted_degree": int(sum(G[n][nbr]["weight"] for nbr in G.neighbors(n))),
        "betweenness": betw.get(n, 0.0),
        "participation_coeff": part.get(n, 0.0),
        "is_articulation": n in articulation
    })
nodes_df = pd.DataFrame(rows)
nodes_df["diagnosis"] = nodes_df["diagnosis"].astype(str)

nodes_df["rank_score"] = (
    (nodes_df["betweenness"] / (nodes_df["betweenness"].max() or 1))
  + 0.6 * (nodes_df["participation_coeff"] / (nodes_df["participation_coeff"].max() or 1))
  + 0.3 * (nodes_df["degree"] / (nodes_df["degree"].max() or 1))
)

# ============================ NAMES ================================
HAVE_NAMES = False
label_col_global = "diagnosis"
try:
    concept = pd.read_csv(CONCEPT_PATH, usecols=["concept_id","concept_name","domain_id"])
    cond_dict = concept[concept["domain_id"]=="Condition"].rename(columns={"concept_id":"cid"})

    nodes_named = nodes_df.copy()
    nodes_named["cid"] = pd.to_numeric(nodes_named["diagnosis"], errors="coerce")
    nodes_named = nodes_named.merge(cond_dict, on="cid", how="left")
    nodes_named["diagnosis_label"] = nodes_named["concept_name"].fillna(nodes_named["diagnosis"].astype(str))
    nodes_named.drop(columns=["concept_name","cid"], inplace=True)
    HAVE_NAMES = True
    label_col_global = "diagnosis_label"
except Exception as e:
    print(f"[WARN] Could not attach names from concept.csv ({e}). Using IDs in plots.")
    nodes_named = nodes_df.copy()
    nodes_named["diagnosis_label"] = nodes_named["diagnosis"]

# ============================================================================ 
# ===================== PREGNANCY-COHORT BRIDGE ANALYSIS =====================
# ============================================================================
PREGNANCY_KEYWORDS = [
    "pregnan", "obstetric", "prenatal", "antepartum", "intrapartum", "postpartum",
    "puerper", "gestation", "gestational", "preeclampsia", "eclampsia", "hyperemesis",
    "placenta", "preterm", "premature rupture", "prom", "hellp", "chorioamnionitis",
    "obstetric hemorrhage", "labor", "delivery"
]

try:
    concept_small = concept[concept["domain_id"]=="Condition"].copy()
except NameError:
    # Failsafe - load concepts from path if 'concept' wasn't created above
    concept_full = pd.read_csv(CONCEPT_PATH, usecols=["concept_id","concept_name","domain_id"])
    concept_small = concept_full[concept_full["domain_id"]=="Condition"].copy()

concept_small["name_low"] = concept_small["concept_name"].str.lower()
preg_mask = False
for kw in PREGNANCY_KEYWORDS:
    preg_mask = preg_mask | concept_small["name_low"].str.contains(kw, na=False)

pregnancy_codes = set(concept_small.loc[preg_mask, "concept_id"].astype(str))
print(f"[Pregnancy] Found {len(pregnancy_codes)} obstetric/pregnancy concept_ids from concept.csv")

vis_has_preg = (
    df.assign(is_preg=df["diag_code"].isin(pregnancy_codes))
      .groupby(["patient_id","visit_id"], sort=False)["is_preg"].any()
      .reset_index()
)
preg_visits = set(map(tuple, vis_has_preg[vis_has_preg["is_preg"]][["patient_id","visit_id"]].values))
print(f"[Pregnancy] Identified {len(preg_visits)} visits with ≥1 pregnancy/obstetric diagnosis")

df_preg = df[df[["patient_id","visit_id"]].apply(tuple, axis=1).isin(preg_visits)].copy()

PREG_MIN_COUNT = max(10, MIN_DIAG_COUNT // 2)
EDGE_MIN_WEIGHT_PREG = max(2, EDGE_MIN_WEIGHT - 1)

vc_preg = df_preg["diag_code"].value_counts()
df_preg = df_preg[df_preg["diag_code"].isin(vc_preg[vc_preg >= PREG_MIN_COUNT].index)]

pairs_counter_p = Counter()
for _, grp in df_preg.groupby(["patient_id","visit_id"], sort=False):
    codes = sorted(set(grp["diag_code"]))
    for a, b in it.combinations(codes, 2):
        pairs_counter_p[(a,b)] += 1

edges_preg = pd.DataFrame([(a,b,w) for (a,b),w in pairs_counter_p.items()],
                          columns=["diag_a","diag_b","weight"])
edges_preg = edges_preg[edges_preg["weight"] >= EDGE_MIN_WEIGHT_PREG]
edges_preg = edges_preg.sort_values(["diag_a","diag_b"], kind="mergesort").reset_index(drop=True)
print(f"[Pregnancy] Built {len(edges_preg)} edges over {df_preg['diag_code'].nunique()} diagnosis nodes.")

G_preg = nx.Graph()
for a,b,w in edges_preg.itertuples(index=False):
    w = int(w)
    G_preg.add_edge(a,b, weight=w, distance=1.0/max(1,w))

print(f"[Pregnancy] Graph: {G_preg.number_of_nodes()} nodes, {G_preg.number_of_edges()} edges.")

# Communities for pregnancy
comm_id_of_preg = {}
try:
    import community as community_louvain
    part_p = community_louvain.best_partition(G_preg, weight="weight", random_state=42)
    comm_id_of_preg = part_p
    from collections import defaultdict as _dd2
    gs = _dd2(set)
    for n,c in part_p.items():
        gs[c].add(n)
    communities_preg = list(gs.values())
    from networkx.algorithms.community.quality import modularity as nx_mod
    mod_p = nx_mod(G_preg, communities_preg, weight="weight")
    print(f"[Pregnancy] Louvain: {len(communities_preg)} communities; modularity={mod_p:.3f}")
except Exception as e:
    print(f"[Pregnancy][WARN] Louvain failed ({e}); using greedy modularity.")
    mapping_p = {n:i for i,n in enumerate(sorted(G_preg.nodes()))}
    Gp_rel = nx.relabel_nodes(G_preg, mapping_p, copy=True)
    from networkx.algorithms.community import greedy_modularity_communities, modularity
    comm_sets_p = list(greedy_modularity_communities(Gp_rel, weight="weight"))
    inv_mp = {i:n for n,i in mapping_p.items()}
    communities_preg = [{inv_mp[i] for i in s} for s in comm_sets_p]
    for cid,comm in enumerate(communities_preg):
        for n in comm:
            comm_id_of_preg[n] = cid
    mod_p = modularity(Gp_rel, comm_sets_p, weight="weight")
    print(f"[Pregnancy] Greedy: {len(communities_preg)} communities; modularity={mod_p:.3f}")

# Metrics for pregnancy
print("\n[Pregnancy] Computing betweenness centrality...")
betw_p = betweenness_centrality_tqdm(G_preg, weight="distance", normalized=True, desc="Preg betweenness")
artic_p = set(nx.articulation_points(G_preg))

def participation_coeff_local(Gx, comm_of):
    P = {}
    for u in Gx.nodes():
        k = Gx.degree(u)
        if k == 0:
            P[u] = 0.0
            continue
        counts = defaultdict(int)
        for v in Gx.neighbors(u):
            counts[comm_of.get(v, -1)] += 1
        P[u] = 1.0 - sum((cnt/k) ** 2 for cnt in counts.values())
    return P

part_p = participation_coeff_local(G_preg, comm_id_of_preg)

rows_p = []
for n in G_preg.nodes():
    rows_p.append({
        "diagnosis": n,
        "community_id_preg": comm_id_of_preg.get(n, -1),
        "degree_preg": G_preg.degree(n),
        "weighted_degree_preg": int(sum(G_preg[n][nbr]["weight"] for nbr in G_preg.neighbors(n))),
        "betweenness_preg": betw_p.get(n, 0.0),
        "participation_coeff_preg": part_p.get(n, 0.0),
        "is_articulation_preg": n in artic_p
    })
nodes_preg = pd.DataFrame(rows_p)
nodes_preg["diagnosis"] = nodes_preg["diagnosis"].astype(str)

# OB vs non-OB labeling and cross ties
OB_COMM_SHARE_THRESH = 0.20
comm_to_nodes = defaultdict(list)
for n,c in comm_id_of_preg.items():
    comm_to_nodes[c].append(n)

comm_is_obst = {}
for cid, nodes_in_c in comm_to_nodes.items():
    share = sum(1 for n in nodes_in_c if n in pregnancy_codes) / max(1, len(nodes_in_c))
    comm_is_obst[cid] = (share >= OB_COMM_SHARE_THRESH)

def cross_share_to_non_ob(Gx, comm_of, comm_is_obst_flag, u):
    k = Gx.degree(u)
    if k == 0:
        return 0.0, False
    total = 0
    to_non = 0
    has_ob = False
    has_non = False
    for v in Gx.neighbors(u):
        total += 1
        c = comm_of.get(v, -1)
        if comm_is_obst_flag.get(c, False):
            has_ob = True
        else:
            has_non = True
            to_non += 1
    return (to_non / total), (has_ob and has_non)

nodes_preg["cross_share_to_non_ob"] = nodes_preg["diagnosis"].apply(
    lambda n: cross_share_to_non_ob(G_preg, comm_id_of_preg, comm_is_obst, n)[0]
)
nodes_preg["has_ob_and_non_ob_neighbors"] = nodes_preg["diagnosis"].apply(
    lambda n: cross_share_to_non_ob(G_preg, comm_id_of_preg, comm_is_obst, n)[1]
)

# z-score comparison
def zscore(s):
    s = s.astype(float)
    return (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1.0)

overall = nodes_df[["diagnosis","betweenness","participation_coeff"]].copy()
overall["diagnosis"] = overall["diagnosis"].astype(str)
overall["betweenness_z_overall"]   = zscore(overall["betweenness"])
overall["participation_z_overall"] = zscore(overall["participation_coeff"])

preg = nodes_preg[["diagnosis","betweenness_preg","participation_coeff_preg",
                   "cross_share_to_non_ob","has_ob_and_non_ob_neighbors"]].copy()
preg["diagnosis"] = preg["diagnosis"].astype(str)
preg["betweenness_z_preg"]   = zscore(preg["betweenness_preg"])
preg["participation_z_preg"] = zscore(preg["participation_coeff_preg"])

cmp = overall.merge(preg, on="diagnosis", how="inner")
cmp["delta_betweenness_z"]   = cmp["betweenness_z_preg"]   - cmp["betweenness_z_overall"]
cmp["delta_participation_z"] = cmp["participation_z_preg"] - cmp["participation_z_overall"]

# Attach readable labels
def add_label(df_in, src_nodes_df, label_col_name="diagnosis_label"):
    lab = src_nodes_df[["diagnosis", label_col_name]].drop_duplicates().copy()
    lab["diagnosis"] = lab["diagnosis"].astype(str)
    out = df_in.copy()
    out["diagnosis"] = out["diagnosis"].astype(str)
    out = out.merge(lab, on="diagnosis", how="left")
    if label_col_name not in out.columns:
        out[label_col_name] = out["diagnosis"]
    return out

# nodes_named already has diagnosis_label
nodes_preg_named = add_label(nodes_preg, nodes_named, "diagnosis_label")
cmp_named        = add_label(cmp,        nodes_named, "diagnosis_label")

# ======================= FIGURES & TABLES =============================

# ---------- [Global] Top-20 by betweenness
plot_nodes = nodes_named.copy()
label_col = "diagnosis_label"
plot_nodes["diagnosis"] = plot_nodes["diagnosis"].astype(str)
top_global = plot_nodes.sort_values("betweenness", ascending=False).head(20)

plt.figure(figsize=(12, 7))
plt.barh(range(len(top_global)), top_global["betweenness"].values)
plt.yticks(range(len(top_global)), top_global[label_col].values, fontsize=9)
plt.gca().invert_yaxis()
wrap_y_ticklabels(plt.gca(), width=40)
plt.xlabel("betweenness")
plt.title("Global: Top 20 diagnoses by betweenness")
savefig_safe("global_top20_betweenness.png")

print_table(
    "[Global] Top-20 by betweenness",
    top_global[[label_col,"betweenness","degree","weighted_degree"]],
)

# ---------- [Pregnancy] Prep named tables
nodes_preg_plot = nodes_preg_named.copy()
label_col_preg = "diagnosis_label"
nodes_preg_plot["diagnosis"] = nodes_preg_plot["diagnosis"].astype(str)

# (3) Ego network of the top pregnancy bridge
cand = nodes_preg_plot.copy()
if "has_ob_and_non_ob_neighbors" in cand.columns:
    cand = cand[cand["has_ob_and_non_ob_neighbors"] == True]
if cand.empty:
    cand = nodes_preg_plot.copy()

top_row_preg = cand.sort_values("betweenness_preg", ascending=False).iloc[0]
top_node_preg = str(top_row_preg["diagnosis"])
top_label_preg = str(top_row_preg.get(label_col_preg, top_node_preg))

N_NEIGHBORS_PREG = 30
ego_p = nx.ego_graph(G_preg, top_node_preg, radius=1)
top_neighbors_p = sorted(ego_p.neighbors(top_node_preg),
                         key=lambda v: ego_p[top_node_preg][v]["weight"],
                         reverse=True)[:N_NEIGHBORS_PREG]
Hp = ego_p.subgraph([top_node_preg] + top_neighbors_p).copy()

subp = nodes_preg_plot.set_index("diagnosis").loc[list(Hp.nodes())]
sizes_p = 220 * (subp["betweenness_preg"] / (subp["betweenness_preg"].max() or 1) + 0.2)
colors_p = [comm_id_of_preg.get(n, -1) for n in Hp.nodes()]

pos = nx.spring_layout(Hp, weight="weight", seed=42)
plt.figure(figsize=(12,10))
nx.draw_networkx_nodes(Hp, pos, node_size=sizes_p, node_color=colors_p, cmap="tab20")
nx.draw_networkx_edges(Hp, pos, alpha=0.5)

labels = {top_node_preg: top_label_preg}
for nbr in top_neighbors_p[:15]:
    lbl = nodes_preg_plot.set_index("diagnosis").loc[str(nbr), label_col_preg]
    labels[str(nbr)] = str(lbl)
# wrap long node labels for readability
labels_wrapped = {k: _wrap_text_list([v], width=30)[0] for k, v in labels.items()}
nx.draw_networkx_labels(Hp, pos, labels=labels_wrapped, font_size=9)

plt.title(f"[Pregnancy] Ego network of top bridge: {top_label_preg}")
plt.axis("off")
savefig_safe("pregnancy_ego_top_bridge.png")

ego_nodes_tbl = (
    subp.reset_index()[["diagnosis", label_col_preg, "betweenness_preg", "degree_preg", "community_id_preg"]]
    .rename(columns={label_col_preg: "label"})
)
print_table("[Pregnancy] Ego-network nodes (top bridge)", ego_nodes_tbl)

# (4) Top-20 by betweenness (pregnancy)
top_preg = nodes_preg_plot.sort_values("betweenness_preg", ascending=False).head(20)
plt.figure(figsize=(12,7))
plt.barh(range(len(top_preg)), top_preg["betweenness_preg"].values)
plt.yticks(range(len(top_preg)), top_preg[label_col_preg].values, fontsize=9)
plt.gca().invert_yaxis()
wrap_y_ticklabels(plt.gca(), width=40)
plt.xlabel("betweenness_preg")
plt.title("[Pregnancy] Top 20 diagnoses by betweenness")
savefig_safe("pregnancy_top20_betweenness.png")

print_table(
    "[Pregnancy] Top-20 by betweenness",
    top_preg[[label_col_preg,"betweenness_preg","degree_preg","weighted_degree_preg"]].rename(columns={label_col_preg:"label"})
)

# (5) Δ betweenness z  &  (6) Δ participation z
cmp_plot = cmp_named.copy()
cmp_plot["diagnosis"] = cmp_plot["diagnosis"].astype(str)
label_col_cmp = "diagnosis_label"

def bar_and_table_delta(df, delta_col, k=20, title="", outfile="delta.png"):
    top = df.sort_values(delta_col, ascending=False).head(k)
    plt.figure(figsize=(12,7))
    plt.barh(range(len(top)), top[delta_col].values)
    plt.yticks(range(len(top)), top[label_col_cmp].values, fontsize=9)
    plt.gca().invert_yaxis()
    wrap_y_ticklabels(plt.gca(), width=40)
    plt.xlabel(delta_col)
    plt.title(title or f"Top {k} by {delta_col}")
    savefig_safe(outfile)
    print_table(title + " (table)", top[[label_col_cmp, delta_col]].rename(columns={label_col_cmp:"label"}))

bar_and_table_delta(
    cmp_plot, "delta_betweenness_z", 20,
    "Top Δ-bridging (betweenness z: pregnancy − overall)",
    outfile="delta_betweenness_z_preg_minus_overall.png"
)
bar_and_table_delta(
    cmp_plot, "delta_participation_z", 20,
    "Top Δ-bridging (participation z: pregnancy − overall)",
    outfile="delta_participation_z_preg_minus_overall.png"
)

# (7) Cross-community backbone (OB ↔ non-OB)
edge_weights = []
for u, v, d in G_preg.edges(data=True):
    cu, cv = comm_id_of_preg[u], comm_id_of_preg[v]
    if cu != cv and (comm_is_obst.get(cu, False) != comm_is_obst.get(cv, False)):
        edge_weights.append(d.get("weight", 1))

if len(edge_weights) == 0:
    print("[Pregnancy] No cross obstetric/non-obstetric edges found.")
else:
    thr = np.percentile(edge_weights, 75)  # top quartile cross-community edges
    B = nx.Graph()
    for u, v, d in G_preg.edges(data=True):
        cu, cv = comm_id_of_preg[u], comm_id_of_preg[v]
        if cu != cv and (comm_is_obst.get(cu, False) != comm_is_obst.get(cv, False)):
            if d.get("weight", 1) >= thr:
                B.add_edge(u, v, **d)

    if B.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(B), key=len)
        B = B.subgraph(largest_cc).copy()

        bet_map = {r["diagnosis"]: r["betweenness_preg"] for _, r in nodes_preg.iterrows()}
        sizes = [200 * (bet_map.get(n, 0) / (max(bet_map.values() or [1])) + 0.2) for n in B.nodes()]
        colors = ["tab:orange" if comm_is_obst.get(comm_id_of_preg.get(n, -1), False) else "tab:blue" for n in B.nodes()]

        pos = nx.spring_layout(B, weight="weight", seed=42)
        plt.figure(figsize=(12,10))
        nx.draw_networkx_nodes(B, pos, node_size=sizes, node_color=colors, alpha=0.9)
        nx.draw_networkx_edges(B, pos, alpha=0.6)

        k = 15
        lab_df = nodes_preg_plot.set_index("diagnosis")
        top_labels = sorted(B.nodes(), key=lambda n: bet_map.get(n, 0), reverse=True)[:k]
        labels = {str(n): str(lab_df.loc[str(n), label_col_preg]) if str(n) in lab_df.index else str(n)
                  for n in top_labels}
        labels_wrapped_B = {k: _wrap_text_list([v], width=28)[0] for k, v in labels.items()}
        nx.draw_networkx_labels(B, pos, labels=labels_wrapped_B, font_size=9)

        plt.title("[Pregnancy] Cross-community backbone (orange=obstetric, blue=non-obstetric)")
        plt.axis("off")
        savefig_safe("pregnancy_cross_community_backbone.png")

        bb_nodes_tbl = (
            pd.DataFrame({"diagnosis":[str(n) for n in B.nodes()]})
              .merge(nodes_preg_plot[["diagnosis", label_col_preg]].astype({"diagnosis":str}),
                     on="diagnosis", how="left")
              .rename(columns={label_col_preg:"label"})
        )
        print_table("[Pregnancy] Backbone nodes", bb_nodes_tbl)
    else:
        print("[Pregnancy] Cross-community backbone is empty after thresholding.")

print("\nAnalysis completed successfully! Figures saved under ./figures\n")