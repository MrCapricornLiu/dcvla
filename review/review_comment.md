翻译成中文：

Desk Rejection Assessment:
Paper Length
Pass ✅.

Topic Compatibility
Pass ✅. The paper targets efficient inference for multimodal vision–language–action models, which fits ACL’s Multimodality and Language Grounding to Vision and Machine Learning for NLP thrusts.

Minimum Quality
Pass ✅. The submission includes Abstract, Introduction, Related Work, Methodology, Theoretical Analysis, Experiments, Conclusion, and Limitations. The work is technical and empirical, not a resource paper.

Prompt Injection and Hidden Manipulation Detection
Pass ✅. I did not detect any attempt to manipulate the review or hidden instructions.

Expected Review Outcome:
Paper Summary
The paper proposes a training-free method, Coordinated Cross-Modal Token Reuse (CTR), to accelerate and stabilize vision–language–action (VLA) inference. The key idea is a unified patch mask built via a two-stage criterion: temporal consistency between consecutive frames and text-to-vision attention from the previous step to identify task-relevant regions. This single mask is used to synchronize reuse decisions on both sides: caching and reusing per-patch sublayer outputs in the vision encoder, and reusing key–value caches for corresponding visual tokens in the LLM prefill. The method keeps architectures and token order untouched and introduces periodic keyframes to limit drift. Experiments on LIBERO and SimplerEnv report 12–20% TFLOPs savings with equal or higher success rates.

Summary Of Strengths
Cross-modal coordination idea: Using a single unified mask to drive both vision-encoder and LLM reuse is a clear, simple principle that addresses semantic mismatches seen when each side reuses tokens independently. Figure 2 communicates this well by aligning the two-stage mask construction with the two reuse pathways and explicitly showing that the same positions are reused across modules.
Training-free and architecture-invariant: No weight changes or sequence reordering; the approach should be broadly applicable to existing VLA stacks.
Quantitative results on multiple suites: The main LIBERO table (Table 1; shown on Page 6) reports consistent TFLOPs reductions of roughly 12–20% with stable or improved success rates, notably +9.7% on Goal for OpenVLA with simultaneous latency drops. The per-side ablation (Table 3 on Page 7) further shows complementary savings when enabling LLM-side and vision-side reuse together.
Complexity analysis connects design to expected savings: Equations (8)–(11) acknowledge that VE attention remains full-sequence, hence savings mainly arise from skipped projections and MLP on the vision side and from shorter effective sequences on the LLM side. This is honest and matches the implementation claims.
Figures support the narrative: Figure 1 shows the standard VLA pipeline and indicates where CTR integrates without altering token order. Figure 2 details the unified mask construction and the consistency of reuse across the vision encoder and LLM, making the cross-modal alignment tangible.
Algorithmic clarity: Algorithm 1 concisely summarizes the runtime procedure, including keyframe refresh and cache updates.
Summary Of Weaknesses
Vision-encoder reuse approximation and correctness are under-specified.
Section 3.4 states reused patches bypass attention and MLP computations by reusing post-block outputs from t−1, while attention is still evaluated over the full sequence. It is unclear how attention for non-reused patches is computed if K/V for reused patches are not reprojected from current inputs. If K/V for reused patches come from the previous frame, then non-reused patches at time t attend to stale K/V, which can produce cross-frame inconsistencies. Equation (4) and the paragraph below it acknowledge this is an approximation but do not quantify its effect or detail the exact attention data flow. This is central to soundness because ViT self-attention is globally coupled. Please clarify precisely which tensors are recomputed and which are taken from t−1, and whether any K/V are recomputed for reused patches.
The paper mentions keyframe refresh every K steps but provides no sensitivity study to K beyond a fixed K=5. Abrupt scene changes would be problematic if attention is stale; there is no stress test with camera motion or occlusions.
Missing comparisons to closely-related baselines.
The paper positions itself as coordinating vision- and LLM-side reuse, yet it does not include direct baselines that already perform strong token caching or pruning in VLMs or VLAs. In particular, there is no quantitative comparison to recent training-free methods specific to VLA acceleration (e.g., VLA-Cache-style baselines as implemented in prior work), or to vision-token pruning tailored for VLA. Without these, it is hard to attribute gains to cross-modal coordination rather than to an LLM-side cache alone.
Table 3 provides an internal ablation of toggling caches, which is useful, but it is not a substitute for comparisons against public methods that claim strong speedups with minimal accuracy loss.
Quantitative reporting inconsistencies and clarity gaps.
Table numbering and references are inconsistent. On Page 6, a large main LIBERO table is shown, and the text then cites “Table 1” for main results. Immediately below, “Table 1: Main results on LIBERO suites” appears again as a different small table reporting SimplerEnv. On Page 7, the ablation is presented as another table and later referred to as “Table 3,” followed by “Table 4” for mask consistency. Please fix numbering and ensure all tables are uniquely labeled and consistently cited in the text.
Latency reporting shows discrepancies. The large LIBERO table (Page 6) claims up to −18.3% latency for Long with OpenVLA + CTR, yet the per-module latencies in Table 3 suggest smaller absolute savings when summing VE and LLM milliseconds (e.g., Spatial: roughly 12.48+32.62 vs 11.90+31.36, which is closer to ~4–5% overall). Please reconcile end-to-end latency definitions across tables: are there differences in the counting window, action de-tokenization time, or environment overhead included in one table but not the other?
Mask design depends on t−1 attention for task relevance, which can lag.
In Section 3.3, a top-K attention set from t−1 is forced to recompute at t. When target attention shifts abruptly between frames, this lag could cause important regions at t to be reused erroneously. Figure 2 shows this flow clearly, but the method’s reliance on t−1 attention raises a failure mode not discussed in the main paper. Please add experiments or diagnostics for rapid motion, object occlusions, or fast distractors where attention shifts quickly, and consider variants that incorporate lightweight forward saliency at t.
Limited ablation depth.
Beyond the on/off switch (Table 3) and a coarse “mask variant” ablation (Table 4), there are missing studies: sensitivity to TopK and thresholds, keyframe interval K, proportion of layers pruned on the LLM side, and reuse ratios ρv and ρl. The current “LLM mask high/low” and “VE mask high/low” descriptors are not defined quantitatively, and the main text does not explain how these thresholds map to reuse ratios or attention entropy modulation. This limits interpretability and tuning guidance.
No analysis of selection-overhead vs. savings. Equations (6)–(7) bound overheads, but there are no empirical timings showing the wall-clock share of mask construction.
Reproducibility gaps.
Missing details include image resolution, patch size and count P, specific ViT and LLM layer indices where pruning or reuse is applied, attention aggregation formula across heads and layers for a(p), how TopK is chosen per sequence length, and exact entropy-based modulation settings. Seeds, number of rollouts per task, and standard deviations are also missing. Without these, it is hard to reproduce reported numbers.
There is no statement about releasing code or scripts.
Evaluation scope could be stronger.
Only two backbones are considered, both OpenVLA variants. Including an additional VLA family or a video-based backbone would help demonstrate generality. Also, SimplerEnv results (Table 2) are small and do not include efficiency numbers, only success rates. Reporting latency and TFLOPs there would strengthen the generalization claim.
Theoretical section is primarily a cost model; no analysis of approximation error.
Section 4 carefully accounts for computational cost but does not analyze the approximation error introduced by reusing t−1 sublayer outputs in the ViT. Even a simple bound that assumes limited scene motion or bounded patch-change would help legitimize Equation (4) and the claimed stability.
Figures and algorithm are helpful, but some critical design choices remain implicit.
Figure 2 and Algorithm 1 are clear about the pipeline, but neither states how many LLM layers are pruned or how reuse ratios vary by layer. Similarly, Figure 2 suggests a simple AND of static∧not-topK, but the paper later mentions attention-entropy-based modulation; this tension should be resolved with a definitive, implementable rule.
Minor but notable: the paper relies on very recent or future-dated citations; while permissible, the Related Work should position this method precisely relative to contemporaneous caching works on VLMs and VLAs with a more explicit contrast.
Specific references to elements in the paper:

Figure 2: Useful to understand the two-stage mask and how it routes reuse to both the ViT and LLM. It also reveals the potential t−1 attention lag.
Equation (4): Central to the VE-side approximation; merits deeper clarification of attention computation with partly stale tokens.
Equations (8)–(11): Coherent lower-bound cost reasoning, consistent with keeping full attention in the ViT and shortening the LLM sequence during prefill.
Table 1 (main LIBERO results on Page 6): Key evidence for the claimed 12–20% TFLOPs reduction with stable or improved success; however, latency claims need alignment with Table 3.
Table 3 (switch ablation): Demonstrates complementary effects from each side and the benefit of coordination. VE and LLM latencies and TFLOPs are helpful, but the total end-to-end numbers should match the main table’s accounting.
Table 4 (mask-consistency ablation): Supports the central thesis that a unified mask improves success, but definitions of “high/low” should be quantified.
Potentially Missing Related Work
Qin, S., Yu, H., Wu, C., “VLCache: Computing 2% Vision Tokens and Reusing 98% for Vision-Language Inference,” 2025 — General multimodal caching across images and text with strong reuse rates. Highly relevant to Section 2 and Section 3.5; should be cited and briefly contrasted, and if possible included as a baseline or at least discussed after Equation (5) concerning LLM-side reuse.
Liu, Z., Chen, Y., Cai, H., “VLA-Pruner: Temporal-Aware Dual-Level Visual Token Pruning for Efficient Vision-Language-Action Inference,” 2025 — Targets dual-level token importance with temporal awareness for VLAs. Directly relevant to the two-stage selection in Section 3.3; should be discussed in Related Work and compared in Experiments.
Yang, J., Xie, S., Li, S., “CoCM: Conditional Cross-Modal Learning for Vision-Language Models,” 2025 — Builds separate caches conditioned across modalities and dynamically adjusts fusion. Relevant to the unified mask idea in Section 3.2; should be discussed in Related Work as an alternative coordination strategy.
Chen, J., Song, W., Ding, P., “Unified Diffusion VLA: Vision-Language-Action Model via Joint Discrete Denoising Diffusion Process,” 2025 — Presents unified cross-modal modeling for VLA, conceptually related to coordination across modalities. Could be cited in Related Work to broaden positioning of unified cross-modal decision rules.
Note: Xu et al., 2025 (VLA-Cache) is already cited and discussed; no action needed there.

Comments Suggestions And Typos
Actionable questions and suggestions:

Clarify the exact attention computation for VE-side reuse. When m_t(p)=1, what K/V are used for that patch during the attention of other tokens at time t? If they come from t−1, please state this explicitly and add a small study measuring performance degradation on controlled synthetic shifts (camera panning, dynamic distractors) to quantify the approximation.
Reconcile latency numbers across tables. Please define precisely what “Latency (ms)” includes in the main table versus Table 3 and ensure consistency. If environment or action de-tokenization is included in one but not the other, state so, and consider reporting both model-only and end-to-end latency.
Expand ablations: report sensitivity to K (keyframe interval), TopK proportion, similarity thresholds, and reuse-layer schedules on the LLM side. Replace qualitative “mask high/low” with quantitative settings and show how reuse ratios and success vary.
Add or discuss strong baselines: at minimum, compare with a representative LLM-only cache reuse baseline equivalent to VLA-Cache, and cite or discuss pruner-style approaches for the vision side. If reimplementation is heavy, at least provide a careful head-to-head analysis on a subset of tasks.
Provide reproducibility details: patch size and P, image resolution, per-layer reuse schedule, attention aggregation for a(p), thresholds used, seeds, episodes per task, and standard deviations. If possible, release code or a minimal inference patch to reproduce key results. My recommendation could increase if these details and comparisons are added and the VE-side approximation is clarified and stress-tested.
Typos and presentation:

Table numbering is inconsistent throughout Pages 6–7. Ensure unique numbering and consistent references in text.
In Section 3.4: “This is an approximation because self-attention couples tokens; we do not shorten the sequence or modify the attention kernel.” This sentence is correct but warrants a short clarifying note regarding which tensors are kept from t−1.
Minor formatting nits: ensure consistent notation for sets S_t, A_{t−1}, R_t across equations; in Eq. (3), the indicator is written “ℝ^p [p∈R_t]” style; consider a standard indicator notation.
Evaluation-change criteria:

If the authors clarify VE-side attention computations and provide a stress test showing the approximation is robust, add consistent end-to-end latency accounting, and include one strong baseline comparison (e.g., a VLA-Cache-like LLM-only reuse), my overall score would likely increase to 3.5 (borderline conference). Code release and detailed ablations would further strengthen that case.
Confidence
4 Quite sure after careful checking: I tried to check the important points carefully. It’s unlikely, though conceivable, that I missed something that should affect my ratings.

Soundness
2.5 Between Poor and Acceptable: The cost model is sound, and experiments trend positively, but the VE-side approximation and latency inconsistencies need clarification and stronger validation.

Excitement
2.5 Between potentially interesting and interesting: The coordinated mask is a clean idea with practical appeal, but the current empirical and comparative depth feels limited.

Overall Assessment
2.5 Borderline Findings: The paper may meet minimal standards but requires improvements before being accepted to Findings. Rationale: The coordination idea is sensible and well presented, and results are promising. However, methodological clarity about VE-side reuse is insufficient, strong baselines are missing, table inconsistencies confuse the efficiency claims, and ablations are not yet deep enough for the main conference. With added clarity, stronger comparisons, and reproducibility details, this could reach borderline-conference quality.

Best Paper Justification
N/A.

Limitations And Societal Impact
The Limitations section acknowledges smaller wall-clock gains vs TFLOPs and the lack of sequence-length reduction in the ViT. Additional points to discuss:

Potential failure modes when attention shifts suddenly, occlusions occur, or with fast camera motion; using t−1 attention for task relevance may cause reuse of now-relevant regions.
Masking based on temporal stability could bias the policy toward background persistence and miss small but important object motions; consider safety implications in physical robotics when errors accumulate.
While there is no human data, deploying more aggressive caching might lead to unexpected control failures; mitigations such as runtime monitors or conservative fallbacks could be discussed.
Ethical concerns
None.

Needs Ethics Review
No.

Reproducibility
3 Reproducible with difficulty: Key implementation and evaluation details are missing, and code is not promised. Providing precise hyperparameters, layer schedules, and seeds would raise this.

Datasets
1 No usable datasets submitted.

Software
1 No usable software released.