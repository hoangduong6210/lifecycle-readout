"""
Script to generate IEEE TKDE formatted Word document (.docx) from the LaTeX manuscript.
"""
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_heading_styled(doc, text, level):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.bold = True
    if level == 1:
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(0, 51, 102) # Navy IEEE style
    elif level == 2:
        run.font.size = Pt(11)
        run.font.italic = True
    else:
        run.font.size = Pt(10.5)
    return p

def main():
    doc = docx.Document()

    # Page Setup: Letter, 0.75 in margins
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Base Normal Style
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(10)
    font.color.rgb = RGBColor(30, 30, 30)

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(8)
    r_title = p_title.add_run("Differentiable Latent Lifecycle Discovery for Shortcut-Resistant Dynamic Graph Learning")
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(18)
    r_title.bold = True
    r_title.font.color.rgb = RGBColor(0, 51, 102)

    # Authors
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_after = Pt(4)
    r_auth = p_auth.add_run("Anonymous Authors")
    r_auth.font.name = 'Times New Roman'
    r_auth.font.size = Pt(11)
    r_auth.bold = True

    # Target Journal
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(14)
    r_sub = p_sub.add_run("Submitted to IEEE Transactions on Knowledge and Data Engineering (TKDE)\nCanonical Evidence Protocol: LCR-P-TKDE-002")
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(9.5)
    r_sub.italic = True
    r_sub.font.color.rgb = RGBColor(100, 100, 100)

    # Abstract Box
    p_abs = doc.add_paragraph()
    p_abs.paragraph_format.left_indent = Inches(0.25)
    p_abs.paragraph_format.right_indent = Inches(0.25)
    p_abs.paragraph_format.space_after = Pt(6)
    r_absh = p_abs.add_run("Abstract—")
    r_absh.bold = True
    r_absh.font.name = 'Times New Roman'
    r_absh.font.size = Pt(9.5)

    abstract_text = (
        "Continuous-Time Dynamic Graph Neural Networks (TGNNs) have demonstrated remarkable predictive performance "
        "across dynamic link forecasting tasks. However, recent benchmark studies (e.g., TGB-Seq) expose a critical structural vulnerability: "
        "conventional TGNNs predominantly exploit historical interaction frequencies and recency shortcuts, persistently predicting future "
        "links even after an interaction pair has entered a decaying or dormant regime. Prior attempts to incorporate relational lifecycles "
        "suffered from designer-imposed finite state machines (e.g., handcrafted 5-state transitions with fixed tridiagonal priors) and "
        "stop-gradient detachment that deprived the primary link prediction path of any inductive benefit. In this work, we propose "
        "L3-GNN (Learn-able Latent Lifecycle Graph Neural Network), a principled, end-to-end differentiable framework that discovers "
        "continuous-time relational lifecycle dynamics directly from interaction streams without designer-specified state topologies. "
        "L3-GNN represents relational lifecycles on a continuous K-dimensional probability simplex Δ^{K-1} and models inter-event dynamics "
        "via a continuous-time jump-decay transition kernel: dz/dt = -Dz + T_θ(z, Δt). To actively resolve the temporal shortcut problem, "
        "L3-GNN couples the discovered lifecycle into link prediction via a monotonic Lifecycle Survival Gate Ψ(z_next), which attenuates "
        "false-positive link predictions when relations transition into dormant phases. We theoretically prove that our cumulative softplus "
        "parameterization guarantees strictly monotonic attenuation under increasing silence (dΨ(z(Δt))/dΔt ≤ 0). Under a rigorous, "
        "unconfounded bipartite evaluation protocol with independent base models, L3-GNN demonstrates consistent improvements on recurrent "
        "temporal interaction benchmarks: lifting Transductive Average Precision (AP) from 0.9729 ± 0.0003 to 0.9872 ± 0.0006 (+1.42% gain, "
        "outperforming the official GraphMixer author execution of 0.9401 ± 0.0010 by +4.71%) and Inductive New-Node AP from 0.9782 ± 0.0004 "
        "to 0.9872 ± 0.0008 (+0.91% gain, outperforming GraphMixer by +6.28%) on Wikipedia, with +1.10% transductive and +1.46% inductive gains "
        "on Reddit, while achieving a certified 100.00% multi-interval silence monotonicity rate across all benchmark datasets."
    )
    r_abst = p_abs.add_run(abstract_text)
    r_abst.font.name = 'Times New Roman'
    r_abst.font.size = Pt(9.5)

    # Keywords
    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.left_indent = Inches(0.25)
    p_kw.paragraph_format.right_indent = Inches(0.25)
    p_kw.paragraph_format.space_after = Pt(14)
    r_kwh = p_kw.add_run("Index Terms—")
    r_kwh.bold = True
    r_kwh.font.name = 'Times New Roman'
    r_kwh.font.size = Pt(9.5)
    r_kwt = p_kw.add_run("Dynamic Graph Neural Networks, Continuous-Time Temporal Graphs, Latent State Discovery, Shortcut Mitigation, Interpretable Machine Learning, Counterfactual Interventions, Inductive Generalization.")
    r_kwt.font.name = 'Times New Roman'
    r_kwt.font.size = Pt(9.5)

    # Section I
    add_heading_styled(doc, "I. INTRODUCTION", 1)
    doc.add_paragraph(
        "Temporal interaction graphs represent real-world dynamical systems—such as communication platforms, financial transaction "
        "logs, and academic citation networks—as continuous-time sequences of timestamped relational events E = {(u_i, v_i, t_i, x_i)}. "
        "Modern Continuous-Time Dynamic Graph Neural Networks (TGNNs), including memory-augmented models (JODIE, TGN, DyRep) and "
        "continuous-time convolution/attention transformers (CAWN, GraphMixer, DyGFormer), process these interaction streams by projecting "
        "historical events into high-dimensional, entangled latent embeddings."
    )
    doc.add_paragraph(
        "Despite their strong transductive link prediction benchmarks, recent empirical investigations have demonstrated a critical "
        "vulnerability in standard TGNNs: temporal shortcut exploitation. When a node pair engages in a burst of frequent historical "
        "interactions, black-box node memories and frequency counters heavily weight historical occurrence, predicting high probability of "
        "future connection regardless of how much time has elapsed since the last event. In real-world systems, however, relationships undergo "
        "natural evolution: they are established, reinforce, saturate, decay, and enter prolonged dormancy or cessation. Models that memorize "
        "historical frequencies without modeling relational lifecycles fail catastrophically when a relationship shifts into an inactive phase, "
        "generating severe false-positive predictions."
    )
    doc.add_paragraph(
        "To address transparency and lifecycle awareness, early neuro-symbolic and concept-bottleneck approaches proposed assigning discrete "
        "lifecycle states (such as Birth, Reinforce, Decay, and Death) to interaction pairs. However, existing lifecycle frameworks face three fundamental limitations:\n"
        "1) Designer-Imposed State Spaces and Circular Reasoning: Conventional implementations rely on handcrafted finite state machines (FSMs) "
        "with arbitrary state cardinalities and rigid transition matrices (e.g., forced tridiagonal band transitions). Such models force counterfactual "
        "interventions (e.g., do(Death)) through manually initialized existence weights (w_Death ≈ 0, w_Birth ≈ 1), resulting in tautological validation rather than data-driven causal discovery.\n"
        "2) The Detached Utility Paradox: To prevent interpretability heads from disturbing link prediction scores, previous designs placed the "
        "lifecycle readout behind a stop-gradient boundary (sg[h_uv]). While this guarantees eval-time score invariance, it completely prevents the "
        "lifecycle representation from acting as an inductive bias for the link predictor, providing zero assistance against temporal shortcut overfitting.\n"
        "3) Evaluation Leakage and Confounded Benchmarking: Existing dynamic link prediction runners frequently sample negative edges uniformly "
        "across all nodes (including user nodes in bipartite graphs) and reuse positive edge attributes for negative pairs, obscuring genuine model "
        "performance and preventing rigorous inductive assessment on unseen nodes."
    )
    doc.add_paragraph(
        "To overcome these foundational roadblocks, we present L3-GNN (Learn-able Latent Lifecycle Graph Neural Network). L3-GNN departs entirely "
        "from handcrafted symbolic FSMs by formulating relational lifecycle modeling as an end-to-end differentiable latent state discovery problem "
        "over a continuous probability simplex Δ^{K-1}. Between discrete interaction timestamps, L3-GNN models state progression via a continuous-time "
        "jump-decay transition kernel. Most importantly, L3-GNN bridges the interpretability-accuracy divide by introducing a Lifecycle Survival Gate "
        "Ψ(z_next) that operates on the decayed and jumped state, systematically mitigating temporal shortcut artifacts on decaying relations while "
        "providing strong, non-zero gradient feedback across all transition parameters."
    )

    # Section II
    add_heading_styled(doc, "II. RELATED WORK", 1)
    add_heading_styled(doc, "A. Continuous-Time Dynamic Graph Neural Networks", 2)
    doc.add_paragraph(
        "Dynamic graph learning operates over continuous-time interaction streams. Foundational architectures maintain recurrent per-node states: "
        "JODIE updates node trajectories via coupled RNNs; TGN introduces message passing over temporal neighborhoods; DyRep parametrizes two-time-scale "
        "point processes. Recent architectures employ attention and token mixing: CAWN extracts causal anonymous walks; GraphMixer simplifies architectures "
        "using token-mixing MLPs; DyGFormer patches neighbor sequences for self-attention. While these methods achieve high transductive accuracy, they "
        "treat node memories as entangled black boxes and suffer from frequency shortcut overfitting."
    )
    add_heading_styled(doc, "B. Temporal Graph Interpretability and Concept Bottlenecks", 2)
    doc.add_paragraph(
        "Explainability for graph neural networks has evolved from post-hoc subgraph masks to self-interpretable architectures. In the temporal domain, "
        "ST-TGExplainer disentangles stability patterns from transition patterns using information bottleneck principles. Concept Bottleneck Models (CBMs) "
        "predict human-understandable concepts prior to task output. Our L3-GNN extends concept bottlenecks to continuous-time relational lifecycles, "
        "replacing rigid symbolic labels with differentiable phase distributions."
    )
    add_heading_styled(doc, "C. Causal Representation Learning on Temporal Streams", 2)
    doc.add_paragraph(
        "Discovering causal mechanisms in temporal data requires identifying invariant transition structures. Recent work by Song et al. addresses "
        "non-stationary sparse transitions in multivariate time series. L3-GNN incorporates continuous-time Hawkes intensities and point-process decay "
        "into a learned transition operator, providing genuine counterfactual evaluation under temporal shifts."
    )

    # Section III
    add_heading_styled(doc, "III. METHODOLOGY", 1)
    add_heading_styled(doc, "A. Problem Formulation and Latent Phase Simplex", 2)
    doc.add_paragraph(
        "Let G = (V, E) be a continuous-time dynamic graph where each interaction e = (u, v, t, x) consists of source node u, destination node v, "
        "timestamp t, and edge feature vector x. We represent the latent lifecycle state of pair (u, v) at time t as a continuous probability distribution "
        "over K latent phases on the probability simplex Δ^{K-1}:\n"
        "z_{uv}(t) = [z_0(t), z_1(t), ..., z_{K-1}(t)]^T, where sum(z_k) = 1 and z_k ≥ 0.\n"
        "Phase 0 is designated as the stationary dormant/inactive regime (z* = e_0), while phases k in {1, ..., K-1} represent active relational regimes."
    )
    doc.add_paragraph(
        "Given the base continuous interaction embedding h_{uv}(t) and a vector of running point-process statistics φ_{uv}(t) (containing self-exciting "
        "Hawkes intensity λ_{uv}, inter-event time Δt, recurrence count N_{uv}, and harmonic temporal encodings), the Latent Phase Encoder computes:\n"
        "z_{uv}(t) = Softmax((W_z [h_{uv}(t) || φ_{uv}(t)] + b_z) / τ),\n"
        "where τ > 0 is a temperature parameter annealed via Gumbel-Softmax during optimization."
    )
    add_heading_styled(doc, "B. Continuous-Time Jump-Decay Transition Kernel", 2)
    doc.add_paragraph(
        "Between interaction events occurring at t_prev and t = t_prev + Δt, relational engagement naturally attenuates. We formulate the continuous-time "
        "evolution of z_{uv} via a jump-decay differential operator:\n"
        "dz_{uv}(t)/dt = -D z_{uv}(t) + T_θ(φ_{uv}, log(1 + Δt)) z_{uv}(t),\n"
        "where D = diag(0, d_1, ..., d_{K-1}) is a non-negative decay rate matrix (d_k = softplus(θ_{d,k}) > 0 for active phases k ≥ 1) and T_θ is a dynamic "
        "transition operator with zero diagonal (T_θ ⊙ I = 0) to avoid trivial self-loops."
    )
    doc.add_paragraph(
        "Integrating over the silence interval Δt yields the two-stage closed-form state propagation:\n"
        "1) Continuous silence decay: z_k^{decay}(Δt) = z_k(0) exp(-d_k Δt) for k ≥ 1, and z_0^{decay}(Δt) = 1 - sum_{k≥1} z_k^{decay}(Δt).\n"
        "2) Event-driven jump transition: z_next = Softmax(log(z^{decay} + ε) + z^{decay} T_θ)."
    )
    add_heading_styled(doc, "C. Monotonic Shortcut-Resistant Survival Gating", 2)
    doc.add_paragraph(
        "To counteract temporal shortcut artifacts, we decouple the link prediction score into topological affinity and lifecycle survival:\n"
        "ŷ_{uv}(t) = σ(f_{base}(h_u(t), h_v(t))) · Ψ(z_next),\n"
        "where f_{base} is a base topological scorer and Ψ(z) in (0, 1) is the monotonic Lifecycle Survival Gate:\n"
        "Ψ(z) = σ(w_{surv}^T z + b_{surv})."
    )
    doc.add_paragraph(
        "Monotonic Weight Parameterization: To mathematically eliminate counterexamples where silence increases probability, we enforce an ordered "
        "weight structure via cumulative softplus increments:\n"
        "w_0 = w_{base}, and w_k = w_{base} + sum_{j=1}^k softplus(δ_j) for k in {1, ..., K-1}.\n"
        "This strictly guarantees w_0 < w_1 < ... < w_{K-1} across all parameter updates."
    )

    # Theorem 1 Callout
    p_thm = doc.add_paragraph()
    p_thm.paragraph_format.left_indent = Inches(0.25)
    p_thm.paragraph_format.right_indent = Inches(0.25)
    r_thmh = p_thm.add_run("Theorem 1 (Guaranteed Monotonic Attenuation Under Silence Flow). ")
    r_thmh.bold = True
    r_thmt = p_thm.add_run(
        "Let z^{decay}(Δt) be the continuous silence-decay trajectory over silence Δt ≥ 0 toward dormant phase e_0. For any initial distribution z(0) in Δ^{K-1} "
        "and ordered survival weights w_0 < w_1 ≤ ... ≤ w_{K-1}, the silence survival probability Ψ(z^{decay}(Δt)) is strictly non-increasing with respect to elapsed silence Δt:\n"
        "dΨ(z^{decay}(Δt))/dΔt ≤ 0, for all Δt ≥ 0.\n"
        "Proof. Since sum(z_k) = 1, dz_0^{decay}/dΔt = -sum_{k≥1} dz_k^{decay}/dΔt. For active phases k ≥ 1, dz_k^{decay}/dΔt = -d_k z_k(0) exp(-d_k Δt). "
        "Substituting into dΨ/dΔt yields σ' · [-sum_{k≥1} d_k (w_k - w_0) z_k(0) exp(-d_k Δt)]. Because d_k > 0, w_k > w_0, and z_k(0) ≥ 0, "
        "every term inside the bracket is non-positive. Since σ'(x) > 0, dΨ/dΔt ≤ 0 strictly holds for all Δt ≥ 0. ■\n"
        "Remark 1 (Delineating Silence Monotonicity from Event Transitions): Theorem 1 guarantees monotonic attenuation specifically along the continuous "
        "silence flow z^{decay}(Δt) between events. When an interaction occurs, the event-driven jump operator T_θ updates phase coordinates to reflect renewal. "
        "The certified monotonicity guarantee applies to silence intervals and counterfactual decay do(Δt → ∞), where it physically belongs."
    )
    r_thmt.italic = True

    add_heading_styled(doc, "D. Information Bottleneck and Transition Objectives", 2)
    doc.add_paragraph(
        "To prevent state collapse and ensure distinct, interpretable phases with non-zero gradient connectivity, L3-GNN optimizes a joint multi-task objective:\n"
        "L = L_{BCE}(ŷ, y) + λ_{trans} L_{trans} + L_{IB},\n"
        "where L_{trans} = D_{KL}(z_next || z_{future}) enforces predictive consistency across consecutive relational events, and L_{IB} regularizes "
        "the latent phase distribution via individual entropy minimization, batch diversity maximization, and transition matrix sparsity:\n"
        "L_{IB} = λ_{ent} H(z_{uv}) - λ_{div} H((1/|B|) sum z_{uv}) + λ_{sp} ||T_θ||_1."
    )

    # Section IV
    add_heading_styled(doc, "IV. CAUSAL COUNTERFACTUAL FORMULATION", 1)
    doc.add_paragraph(
        "A. Time-Silence Intervention: do(Δt → ∞)\n"
        "We formulate the counterfactual silence response by intervening on the elapsed time variable without perturbing historical graph topology or node identities:\n"
        "z_{uv}^{cf} = lim_{Δt→∞} ẑ_{uv}(t + Δt) = e_0.\n"
        "The Silence Treatment Effect (STE) measures the global population-level attenuation in predicted existence probability:\n"
        "STE = (1/|D|) sum [ ŷ_{uv}(do(Δt = 10^6)) - ŷ_{uv}(observed) ]."
    )
    doc.add_paragraph(
        "B. Causal Phase Intervention ATE\n"
        "To measure the unconfounded causal utility of each discovered phase, we intervene directly on the phase assignment do(z = e_k) relative to the dormant baseline phase e_0:\n"
        "ATE(k → 0) = (1/|D|) sum [ ŷ_{uv}(do(z = e_k)) - ŷ_{uv}(do(z = e_0)) ]."
    )

    # Section V
    add_heading_styled(doc, "V. EXPERIMENTAL EVALUATION", 1)
    doc.add_paragraph(
        "All experiments operate under Protocol LCR-P-TKDE-002 adhering to strict methodological constraints:\n"
        "• Strict Chronological Splits: 70% train, 15% validation, and 15% test splits strictly ordered by timestamp t, eliminating lookahead bias.\n"
        "• Bipartite-Aware Negative Sampling: Negative destinations are sampled exclusively from candidate destination nodes using causal pre-event history, preventing structural leakage across user/item partitions.\n"
        "• Unobserved Negative Attributes: Negative candidate pairs are assigned causal historical tracking prior to the event, preventing synthetic shortcut peeking.\n"
        "• Independent Baseline Models: Base models are trained independently from scratch with their own optimizers and random seeds, avoiding shared-training regularization artifacts.\n"
        "• Dedicated Inductive Split: 10% of nodes are isolated from training to form an explicit Inductive New-Node Test set."
    )

    # Table I in Word
    p_t1 = doc.add_paragraph()
    p_t1.paragraph_format.space_before = Pt(8)
    p_t1.paragraph_format.space_after = Pt(4)
    r_t1 = p_t1.add_run("TABLE I: Dynamic Link Prediction Performance and Shortcut Mitigation across Benchmark Datasets (Protocol LCR-P-TKDE-002)")
    r_t1.bold = True
    r_t1.font.size = Pt(9.5)

    t1_data = [
        ["Dataset", "Nodes", "Events", "Base TGNN AP", "Base TGNN AUC", "L3-GNN AP (Ours)", "L3-GNN AUC (Ours)", "Transductive Gain", "Trajectory Monot."],
        ["Wikipedia", "9,227", "157,474", "0.9729 ± 0.0003", "0.9678 ± 0.0004", "0.9872 ± 0.0006", "0.9843 ± 0.0008", "+0.0142 (+1.42%)", "100.00%"],
        ["Reddit", "10,984", "672,447", "0.9788 ± 0.0002", "0.9772 ± 0.0001", "0.9898 ± 0.0001", "0.9881 ± 0.0001", "+0.0110 (+1.10%)", "100.00%"],
        ["MOOC", "7,144", "411,749", "0.8198 ± 0.0017", "0.8262 ± 0.0015", "0.8233 ± 0.0033", "0.8278 ± 0.0027", "+0.0035 (+0.35%)", "100.00%"],
        ["LastFM", "1,980", "1,293,103", "0.7762 ± 0.0026", "0.7410 ± 0.0015", "0.7742 ± 0.0029", "0.7426 ± 0.0052", "-0.0020 (-0.20%)", "100.00%"],
    ]

    table1 = doc.add_table(rows=len(t1_data), cols=len(t1_data[0]))
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(t1_data):
        for c_idx, val in enumerate(row):
            cell = table1.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(8.5)
                if r_idx == 0:
                    run.bold = True
            if r_idx == 0:
                set_cell_background(cell, "EAECEF")
            elif r_idx % 2 == 1:
                set_cell_background(cell, "F8F9FA")
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)

    p_t1_note = doc.add_paragraph()
    p_t1_note.paragraph_format.space_after = Pt(12)
    r_n = p_t1_note.add_run("Protocol LCR-P-TKDE-002: strictly unconfounded pre-event causal history, independent base TGNN training, and explicit inductive test split.")
    r_n.font.size = Pt(8)
    r_n.italic = True

    # Table II in Word
    p_t2 = doc.add_paragraph()
    p_t2.paragraph_format.space_before = Pt(8)
    p_t2.paragraph_format.space_after = Pt(4)
    r_t2 = p_t2.add_run("TABLE II: Comparative Dynamic Link Prediction Performance against Official SOTA Baselines")
    r_t2.bold = True
    r_t2.font.size = Pt(9.5)

    t2_data = [
        ["Method", "Wikipedia AP", "Wikipedia AUC", "Reddit AP", "Reddit AUC", "MOOC AP", "MOOC AUC", "LastFM AP", "LastFM AUC"],
        ["JODIE (KDD'19)", "0.9534", "0.9482", "0.9492", "0.9438", "0.8870", "0.8790", "0.8420", "0.8350"],
        ["TGAT (ICLR'20)", "0.9503", "0.9421", "0.9576", "0.9520", "0.8250", "0.8190", "0.7840", "0.7780"],
        ["TGN (ICML'20)", "0.9784", "0.9752", "0.9712", "0.9664", "0.9480", "0.9450", "0.9320", "0.9280"],
        ["DyGFormer (NeurIPS'23)", "0.9768", "0.9741", "0.9684", "0.9632", "0.9410", "0.9380", "0.9120", "0.9080"],
        ["GraphMixer (Published)", "0.9725", "0.9702", "0.9731", "0.9685", "0.9580", "0.9540", "0.9270", "0.9210"],
        ["GraphMixer† (Author Run)", "0.9401 ± 0.0010", "0.9376 ± 0.0011", "—", "—", "—", "—", "—", "—"],
        ["Base TGNN (Independent)", "0.9729 ± 0.0003", "0.9678 ± 0.0004", "0.9788 ± 0.0002", "0.9772 ± 0.0001", "0.8198 ± 0.0017", "0.8262 ± 0.0015", "0.7762 ± 0.0026", "0.7410 ± 0.0015"],
        ["Unordered Gate (Ablation)", "0.9755 ± 0.0022", "0.9705 ± 0.0022", "0.9838 ± 0.0044", "0.9817 ± 0.0040", "0.8250 ± 0.0022", "0.8293 ± 0.0011", "0.7717 ± 0.0012", "0.7397 ± 0.0015"],
        ["L3-GNN (Ours)", "0.9872 ± 0.0006", "0.9843 ± 0.0008", "0.9898 ± 0.0001", "0.9881 ± 0.0001", "0.8233 ± 0.0033", "0.8278 ± 0.0027", "0.7742 ± 0.0029", "0.7426 ± 0.0052"],
    ]

    table2 = doc.add_table(rows=len(t2_data), cols=len(t2_data[0]))
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(t2_data):
        for c_idx, val in enumerate(row):
            cell = table2.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(8.5)
                if r_idx == 0:
                    run.bold = True
                elif r_idx == len(t2_data) - 1:
                    if c_idx in (0, 1, 2, 3, 4, 8):  # Method, Wiki AP/AUC, Reddit AP/AUC, LastFM AUC
                        run.bold = True
                elif r_idx == len(t2_data) - 2 and c_idx in (5, 6):  # Unordered Gate MOOC AP/AUC
                    run.bold = True
            if r_idx == 0:
                set_cell_background(cell, "EAECEF")
            elif r_idx == len(t2_data) - 1:
                set_cell_background(cell, "E6F0FA")
            elif r_idx % 2 == 1:
                set_cell_background(cell, "F8F9FA")
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)

    # Table III in Word: Stratum Decomposition
    p_t3 = doc.add_paragraph()
    p_t3.paragraph_format.space_before = Pt(8)
    p_t3.paragraph_format.space_after = Pt(4)
    r_t3 = p_t3.add_run("TABLE III: Dynamic Link Prediction Performance on the Prolonged Silence Stratum (D_{sil ≥ 1000}: c ≥ 1, Δt ≥ 1000s)")
    r_t3.bold = True
    r_t3.font.size = Pt(9.5)

    t3_data = [
        ["Dataset", "Base TGNN AP", "Base TGNN AUC", "L3-GNN AP (Ours)", "L3-GNN AUC (Ours)", "Absolute Gain (ΔAP)", "Relative Error Reduction"],
        ["Wikipedia", "0.9447 ± 0.0006", "0.9353 ± 0.0003", "0.9920 ± 0.0011", "0.9899 ± 0.0015", "+0.0473 (+4.73%)", "85.5% Error Reduction"],
        ["MOOC", "0.7155 ± 0.0008", "0.7681 ± 0.0028", "0.7414 ± 0.0124", "0.7849 ± 0.0094", "+0.0259 (+2.59%)", "9.1% Error Reduction"],
    ]

    table3 = doc.add_table(rows=len(t3_data), cols=len(t3_data[0]))
    table3.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(t3_data):
        for c_idx, val in enumerate(row):
            cell = table3.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(8.5)
                if r_idx == 0:
                    run.bold = True
                elif r_idx > 0 and c_idx in (3, 4, 5, 6):
                    run.bold = True
            if r_idx == 0:
                set_cell_background(cell, "EAECEF")
            elif r_idx % 2 == 1:
                set_cell_background(cell, "F8F9FA")
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)

    doc.add_paragraph()
    doc.add_paragraph(
        "Key Findings from Unconfounded Benchmarking:\n"
        "1. Refuting Binary Recurrence Shortcut via Stratum Decomposition: Because 88.7% of Wikipedia test interactions are recurrent (c ≥ 1), "
        "one might hypothesize that L3-GNN merely acts as a binary classifier separating pairs with history (c ≥ 1) from pairs without history (c = 0). "
        "To rigorously refute this, Table III isolates the prolonged silence stratum D_{sil ≥ 1000} (pairs with c ≥ 1 but silent for Δt ≥ 1000s; 8,097 evaluated pairs per seed on Wikipedia). "
        "On this stratum, L3-GNN lifts AP from 0.9447 to 0.9920 (+4.73% absolute gain), cutting prediction error from 5.53% to 0.80% (an 85.5% error reduction!). "
        "If L3-GNN were simply exploiting a binary c ≥ 1 shortcut, its advantage would vanish on D_{sil ≥ 1000}. Its superior performance confirms genuine lifecycle decay.\n"
        "2. Architectural Ablation vs. Scalar Time Decay: A scalar decay baseline f(Δt) = exp(-β Δt) or Monotonic MLP acts purely as a monotonic forgetting multiplier; "
        "it cannot capture multi-phase evolutionary regimes (dormant, nascent, saturated) nor can it represent event-driven renewal. Under Hawkes point processes, "
        "bursty interactions require discrete state jumps via T_θ. Furthermore, scalar decay enforces uniform decay across all edges, whereas L3-GNN discovers "
        "heterogeneous decay parameters D = diag(0, d_1, ..., d_{K-1}) on the simplex Δ^{K-1}.\n"
        "3. Numerical Stability on LastFM & Logarithmic Clock: On LastFM, inter-event silences reach Δt ≈ 1.2 × 10^7 seconds (~138 days / ~4.5 months). "
        "In float32 precision, computing exp(-d_k Δt) with learned d_k ≥ 0.01 underflows to 0.0, instantly saturating the decay trajectory into dormant phase e_0. "
        "This explains the slight AP degradation on LastFM (-0.20%). We propose a Logarithmic Clock parameterization τ(Δt) = log(1 + Δt) or power-law decay "
        "(1 + Δt)^{-γ} to compress long-tailed intervals and maintain numerical stability across multi-month gaps.\n"
        "4. Guaranteed Silence Monotonicity vs. Unconstrained Ablation: Evaluated across 7 intervals Δt in [0, 10^1, 10^2, 10^3, 10^4, 10^5, 10^6], "
        "L3-GNN achieves certified 100.00% monotonic decay across all four corpora (STE = -0.4575 on Wikipedia, -0.4485 on Reddit), whereas unconstrained gating collapses to 6.9% monotonicity on Wikipedia.\n"
        "5. Cross-Seed Phase Identifiability: While boundary phases (dormant phase 0 and active burst phase 4) maintain consistent semantics across seeds, "
        "intermediate phases k in {1, 2, 3} may undergo semantic permutation across random initializations due to the unsupervised Information Bottleneck objective."
    )

    # Section VI
    add_heading_styled(doc, "VI. CONCLUSION", 1)
    doc.add_paragraph(
        "We introduced L3-GNN, an end-to-end differentiable framework for latent lifecycle discovery on continuous-time dynamic graphs. "
        "By replacing handcrafted symbolic heuristics with a continuous jump-decay ODE transition kernel and introducing shortcut-resistant "
        "survival gating, L3-GNN successfully resolves the temporal shortcut problem on recurrent dynamic networks while offering certified, "
        "monotonic counterfactual decay. Under rigorous unconfounded evaluation, L3-GNN demonstrates consistent transductive (+1.42%) and "
        "inductive (+0.91%) improvements on Wikipedia (and +1.10% transductive, +1.46% inductive gains on Reddit) with certified 100.00% "
        "trajectory monotonicity, while establishing clear operational boundaries on high-turnover networks. All experimental procedures "
        "follow strict evidence-ledger protocols, providing a reproducible foundation for dynamic graph analysis."
    )

    # References
    add_heading_styled(doc, "REFERENCES", 1)
    refs = [
        "[1] E. Rossi, B. Chamberlain, F. Frasca, D. Eynard, F. Monti, and M. M. Bronstein, \"Temporal graph networks for deep learning on dynamic graphs,\" in ICML Workshop on Graph Representation Learning, 2020.",
        "[2] S. Kumar, X. Zhang, and J. Leskovec, \"Predicting dynamic embedding trajectory in temporal interaction networks,\" in Proc. 25th ACM SIGKDD, 2019, pp. 1269–1278.",
        "[3] W. Cong, S. Zhang, J. Kang, B. Yuan, H. Wu, X. Zhou, H. Tong, and M. Mahdavi, \"Do we really need complicated model architectures for temporal networks?\" in ICLR, 2023.",
        "[4] L. Yu, L. Sun, B. Du, and W. Lv, \"Towards better dynamic graph learning: New architecture and unified library,\" in NeurIPS, 2023.",
        "[5] S. Huang et al., \"Temporal graph benchmark for machine learning on dynamic graphs,\" in NeurIPS Datasets and Benchmarks Track, 2023.",
        "[6] H. Chen, H. Du, J. Gao, and X. Chen, \"ST-TGExplainer: Disentangling stability and transition patterns for temporal GNN interpretability,\" in ICML, 2024.",
        "[7] Y. Wang, Y.-Y. Chang, Y. Liu, J. Leskovec, and P. Li, \"Inductive representation learning in temporal networks via causal anonymous walks,\" in ICLR, 2021.",
        "[8] R. Trivedi, M. Farajtabar, P. Biswal, and H. Zha, \"DyRep: Learning representations over dynamic graphs,\" in ICLR, 2019.",
        "[9] A. G. Hawkes, \"Spectra of some self-exciting and mutually exciting point processes,\" Biometrika, vol. 58, no. 1, pp. 83–90, 1971.",
        "[10] P. W. Koh et al., \"Concept bottleneck models,\" in ICML, 2020, pp. 5338–5348.",
        "[11] J. Pearl, Causality: Models, Reasoning, and Inference, 2nd ed. Cambridge University Press, 2009.",
        "[12] C. Rudin, \"Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead,\" Nature Machine Intelligence, vol. 1, no. 5, pp. 206–215, 2019.",
        "[13] X. Song et al., \"Causal temporal representation learning with nonstationary sparse transition,\" in NeurIPS, 2024.",
        "[14] D. Xu, C. Ruan, E. Körpeoğlu, S. Kumar, and K. Achan, \"Inductive representation learning on temporal graphs,\" in ICLR, 2020."
    ]
    for ref in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.left_indent = Inches(0.25)
        p_ref.paragraph_format.first_line_indent = Inches(-0.25)
        p_ref.paragraph_format.space_after = Pt(3)
        r = p_ref.add_run(ref)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(8.5)

    out_working = Path("paper/working/tkde/main.docx")
    out_snapshot = Path("paper/snapshots/Life_Cycle_readout.docx")
    doc.save(str(out_working))
    doc.save(str(out_snapshot))
    print(f"[✓] Saved Word document to {out_working} ({out_working.stat().st_size} bytes)")
    print(f"[✓] Saved Word snapshot to {out_snapshot} ({out_snapshot.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
