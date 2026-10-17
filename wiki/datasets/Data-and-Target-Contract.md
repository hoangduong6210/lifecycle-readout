---
title: Data and Target Contract
status: canonical contract
last_updated: 2026-08-19
paper_source: false
---

# Data and Target Contract

CoEdit NPZ must contain sources, destinations, timestamps, labels, features,
num_nodes, num_edges, and feat_dim. Event arrays share length `num_edges`;
features have shape `(num_edges, feat_dim)`; node IDs are in range; timestamps
are nondecreasing. Lifecycle targets are derived/readout quantities, not
externally observed ground-truth state labels.
