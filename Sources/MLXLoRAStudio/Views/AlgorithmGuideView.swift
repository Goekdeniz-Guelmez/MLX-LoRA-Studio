import SwiftUI

// MARK: - Algorithm Guide
//
// A long-form, wiki-style reference for every training loop that
// `mlx-lm-lora` (the package this app is built on) ships with.
// Tapping a card in the picker at the top selects the algorithm —
// the rest of the page is one continuous article covering it: what
// the loss is, what the knobs do, what dataset shape it expects,
// when to reach for it, and which settings are typically tuned.

struct AlgorithmGuideView: View {
    @Binding var config: TrainingConfig
    @State private var selection: GuideSelection = .mode(.sft)

    enum GuideSelection: Hashable {
        case mode(TrainMode)
        case adaptation
        case optimizers
        case quantization
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                HeaderView(
                    title: "Algorithm Guide",
                    subtitle: "A long-form reference for every training loop in mlx-lm-lora",
                    symbol: "book"
                )

                introCard
                trainingLoopPicker
                foundationsPicker
                article
            }
            .padding(24)
        }
        .navigationTitle("Algorithm Guide")
        .onAppear { selection = .mode(config.trainMode) }
    }

    // MARK: Intro

    private var introCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("How to read this page")
                .font(.headline)
            Text("The mlx-lm-lora package ships nine training algorithms, grouped into three families: **supervised** (SFT), **preference** (DPO, CPO, ORPO), and **reinforcement / online** (GRPO, Online DPO, XPO, RLHF-REINFORCE, PPO). Tapping a card in **Training loops** below opens its full wiki article.")
                .fixedSize(horizontal: false, vertical: true)
            Text("Under **Foundations** you will find the four pieces that are orthogonal to the choice of loss: the **adaptation method** that decides which tensors are trainable (LoRA / DoRA / full), the **optimizer** that turns gradients into updates (Adam, AdamW, Muon), the **load-time quantisation** of the base model (4 / 6 / 8 / MXFP4 bit), and **Quantization-Aware Training** that simulates a quantised forward pass so the trained adapter survives deployment quantisation.")
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .liquidGlass(cornerRadius: 12)
    }

    // MARK: Pickers

    private var trainingLoopPicker: some View {
        VStack(alignment: .leading, spacing: 10) {
            SectionTitle("Training loops")
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 240), spacing: 12)], spacing: 12) {
                ForEach(TrainMode.allCases) { mode in
                    Button {
                        selection = .mode(mode)
                        config.trainMode = mode
                        config.applyDefaultDataset(for: mode)
                    } label: {
                        algorithmCard(mode: mode, isSelected: selection == .mode(mode))
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private var foundationsPicker: some View {
        VStack(alignment: .leading, spacing: 10) {
            SectionTitle("Foundations")
            HStack(spacing: 12) {
                foundationCard(
                    title: "Adaptation methods",
                    subtitle: "LoRA · DoRA · Full",
                    symbol: "square.stack.3d.up",
                    isSelected: selection == .adaptation
                ) { selection = .adaptation }

                foundationCard(
                    title: "Optimizers",
                    subtitle: "Adam · AdamW · Muon",
                    symbol: "function",
                    isSelected: selection == .optimizers
                ) { selection = .optimizers }

                foundationCard(
                    title: "Quantization & QAT",
                    subtitle: "4 / 6 / 8 / MXFP4 + QAT hook",
                    symbol: "cpu",
                    isSelected: selection == .quantization
                ) { selection = .quantization }
            }
        }
    }

    private func algorithmCard(mode: TrainMode, isSelected: Bool) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text(mode.title)
                    .font(.headline)
                Spacer()
                Text(mode.family)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Text(mode.summary)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 6) {
                if mode.needsReference {
                    InfoPill(text: "Reference", symbol: "person.2")
                }
                if mode.needsJudge {
                    InfoPill(text: "Judge", symbol: "scales")
                }
                if mode.supportsQAT {
                    InfoPill(text: "QAT", symbol: "cpu")
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, minHeight: 150, alignment: .topLeading)
        .contentShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .liquidGlass(cornerRadius: 12)
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(isSelected ? Color.accentColor : .clear, lineWidth: 2)
        )
    }

    private func foundationCard(
        title: String,
        subtitle: String,
        symbol: String,
        isSelected: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Image(systemName: symbol)
                        .font(.title3)
                        .frame(width: 32, height: 32)
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 6))
                    Text(title)
                        .font(.headline)
                    Spacer()
                }
                Text(subtitle)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            .padding(14)
            .frame(maxWidth: .infinity, minHeight: 110, alignment: .topLeading)
            .contentShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .liquidGlass(cornerRadius: 12)
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(isSelected ? Color.accentColor : .clear, lineWidth: 2)
            )
        }
        .buttonStyle(.plain)
    }

    // MARK: Article router

    @ViewBuilder
    private var article: some View {
        switch selection {
        case .mode(let mode):
            AlgorithmArticle(mode: mode)
        case .adaptation:
            AdaptationArticle(config: $config)
        case .optimizers:
            OptimizerArticle(config: $config)
        case .quantization:
            QuantizationArticle(config: $config)
        }
    }
}

// MARK: - Article

/// One algorithm rendered as a long, scrollable article. The structure
/// is identical for every mode so the page reads as a wiki.
private struct AlgorithmArticle: View {
    let mode: TrainMode

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            header
            section("Overview", text: overview)
            section("Intuition", text: intuition)
            section("Objective (math)", content: { mathView })
            section("What the settings change", content: { settingsTable })
            section("Dataset format", text: datasetFormat)
            section("When to use it", text: whenToUse)
            section("Tips & gotchas", content: { tipsList })
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .liquidGlass(cornerRadius: 14)
    }

    // MARK: Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text(mode.title)
                    .font(.system(.largeTitle, design: .rounded).bold())
                Spacer()
                Text(mode.family)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .liquidGlass(cornerRadius: 999, interactive: true)
            }
            Text(mode.summary)
                .font(.title3)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: Generic section helpers

    @ViewBuilder
    private func section(_ title: String, text: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.title3.bold())
            Text(.init(MathTypesetter.richify(text)))
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.title3.bold())
            content()
        }
    }

    // MARK: Per-mode content

    private var overview: String {
        switch mode {
        case .sft:
            """
            **Supervised fine-tuning (SFT)** is the canonical “teach the model to imitate a dataset” loop. Every example is a `(prompt, completion)` pair (or a chat-style `messages` list, or a raw `text` blob). The model is trained to maximise the log-probability it assigns to the completion tokens; prompt tokens are masked out so the loss only counts the answer.

            In `mlx-lm-lora` the SFT trainer is the substrate that every other algorithm reuses. The actual cross-entropy / NLL loss, gradient checkpointing, KV-cache handling, QAT hooks, and the sequence-chunked efficient-long-context forward pass all live in `sft_trainer.py`. Every preference and RL algorithm inherits those mechanics and only swaps the loss function and (sometimes) the dataset format.
            """
        case .dpo:
            """
            **Direct Preference Optimisation (Rafailov et al., 2023)** trains the model directly on human preference pairs `(chosen, rejected)` for the same prompt. There is no reward model and no sampling loop — just a closed-form loss that pulls the policy’s log-probability of the chosen completion up and the rejected completion down, regularised by a frozen **reference model** so the policy does not drift.

            The reference is normally the base model you started from. Loading a different one (e.g. an instruction-tuned SFT checkpoint) shifts the implicit reward baseline and is a common knob for steering the resulting behaviour.
            """
        case .cpo:
            """
            **CPO (Contrastive Preference Optimisation)** is DPO with the reference term dropped. The chosen-rejected log-prob gap is compared against an absolute target instead of a relative one — which means the policy can move further from the base model without a reference forward pass. It is faster to train and uses less memory, at the cost of being more sensitive to the `beta`/`delta` knobs.
            """
        case .orpo:
            """
            **ORPO (Hong et al., 2024)** folds the SFT cross-entropy and the odds-ratio preference term into a single loss. There is no reference model and no SFT warm-up step — the chosen response is pushed up and the rejected one pulled down by the same gradient that improves the model’s next-token log-likelihood. In `mlx_lm_lora` it accepts an optional `preference_score` per example so heterogeneous preference strengths (e.g. UltraFeedback-style 0..10 scores) can be used as a soft target.
            """
        case .grpo:
            """
            **GRPO (DeepSeekMath, 2024)** is an RL loop that does not need a learned reward model. For every prompt it samples `group_size` completions from the current policy, scores each completion with one or more user-supplied **reward functions** (format checks, accuracy checks, int-format, etc.), and computes a group-relative advantage: `A_i = (r_i − mean(r)) / (std(r) + ε)`. The PPO-style clipped objective is then applied at the token level.

            The KL term against the reference model is optional but recommended when the policy starts to drift in a way the rewards do not penalise.
            """
        case .onlineDPO:
            """
            **Online DPO** is DPO where the `chosen` and `rejected` completions are sampled at training time instead of read from a static dataset. For every prompt the trainer draws two completions, asks a **judge** (an LLM or a human) which one is better, treats that as the preference pair, and runs the DPO loss on the fly.

            The judge can be `human` (you label pairs interactively), or a Hugging Face model identifier / local path that the runner loads and prompts with a pairwise system prompt.
            """
        case .xpo:
            """
            **XPO (Exploratory Preference Optimisation)** is the online-DPO family plus an explicit **exploration bonus** proportional to the KL divergence between the current policy and the reference. Setting `alpha > 0` rewards the model for trying completions that are different from the reference, which mitigates the “stuck on the reference completion” pathology of plain online DPO. `alpha` can be a single float or a list (one per epoch) so the bonus can decay over training.
            """
        case .rlhfReinforce:
            """
            **RLHF-REINFORCE** is the classic policy-gradient RLHF loop. The trainer samples completions from the policy, scores them with a scalar reward (an LLM judge, normally configured via the `judge` and `judge_system` settings), and applies a per-token REINFORCE objective regularised by an optional KL penalty against a reference model.

            It is conceptually the simplest of the online algorithms (no clipping, no value head) but it has the highest variance, so it benefits from smaller learning rates and longer KL warm-up than DPO/PPO.
            """
        case .ppo:
            """
            **PPO (Schulman et al., 2017) as applied to LMs** is the textbook clipped policy optimisation. For each prompt the trainer samples two completions, asks the judge which is better, and treats them as `(chosen, rejected)`. It then computes log-ratios against the reference model and minimises the clipped surrogate objective on both sequences, plus a KL penalty.

            It is the most powerful and the most finicky of the loops: the `epsilon` clip range, the `beta` KL weight, and the judge quality all matter a lot.
            """
        }
    }

    private var intuition: String {
        switch mode {
        case .sft:
            """
            • The model gets a question, you give it the right answer, and you ask it to be more confident in the right answer next time.
            • Masking the prompt means the loss only punishes (or rewards) the tokens the model *generated* in its answer, not the question.
            • Iterations are stochastic gradient steps over mini-batches; the validation set is a held-out slice that you should never train on.
            • Because everything else in this guide is built on top of SFT, the SFT trainer is also where `grad_checkpoint`, `efficient_long_context`, `seq_step_size`, and `qat_*` live.
            """
        case .dpo:
            """
            • DPO is derived from the closed-form solution of the KL-constrained RL objective, so the loss is mathematically equivalent to RLHF with a learned reward model — but you skip the reward model entirely.
            • The implicit reward is `β · log(π_θ(chosen) / π_ref(chosen)) − β · log(π_θ(rejected) / π_ref(rejected))`. A larger `β` makes the loss more aggressive, a smaller one softer.
            • `loss_type = "sigmoid"` is the original DPO; `hinge` is a margin-style loss; `ipo` regularises toward a constant target (more robust to noise); `dpop` adds an explicit reference-drift penalty scaled by `delta`.
            """
        case .cpo:
            """
            • Without a reference, the policy can drift — the model is being told “chosen is better than rejected” but there is no anchor.
            • Setting `loss_type = "dpop"` adds a hinge penalty `max(0, ref − π)` (CPO substitutes the policy log-prob for the reference in the drift penalty) to keep that drift bounded by `delta`.
            • In practice CPO converges faster than DPO at the cost of needing slightly more careful `beta` and `delta` tuning.
            """
        case .orpo:
            """
            • ORPO’s signature trick is the log-odds term: `log σ( log π(chosen) − log π(rejected) )`. The same gradient that lowers NLL on the chosen completion also increases that gap.
            • `preference_score` (default 1.0) lets the trainer scale the chosen log-prob per example, so a soft preference of 0.3 still gets a smaller push than a hard preference of 1.0.
            • `reward_scaling` is accepted but the upstream implementation does not actually use it as a separate multiplier — it is reserved for future variants.
            """
        case .grpo:
            """
            • The reward is whatever functions you ship — string-matching accuracy, integer/format checks, XML-tag counting, etc. The default set is the `r1_*` family (DeepSeek-R1 style format + accuracy rewards).
            • Advantages are computed per prompt group, so the absolute scale of the reward functions does not matter — only their *relative* ordering within a group. This is what makes GRPO robust to reward function magnitude.
            • Importance sampling lets you decide whether the PPO ratio is a per-token quantity (default, `token`) or averaged across the sequence (`sequence`, more stable per the GSPO paper).
            • KL is optional. Set `beta = 0.0` to disable it; the trainer falls back to a Schulman-style unbiased estimator for logging only.
            """
        case .onlineDPO:
            """
            • Every step is: prompt → two completions → judge picks a winner → DPO loss.
            • Because the completions come from the *current* policy, the data distribution shifts as training progresses, which avoids the “off-policy stale data” failure mode of vanilla DPO.
            • The judge system prompt is a high-leverage string: it controls the rubric. A vague “which is better?” prompt produces noisy labels; a specific “reward conciseness, factual accuracy, and refusal of unsafe content” prompt is much sharper.
            • `loss_type` and `delta` behave exactly as in DPO.
            """
        case .xpo:
            """
            • XPO’s bonus term `alpha · (KL(policy ‖ ref) on chosen + KL on rejected)` rewards moving away from the reference. It is the opposite sign of a KL penalty.
            • The result is a model that is incentivised to *try new things* early in training and to settle into a stable region as the bonus decays.
            • Use a single `alpha` for constant exploration, or pass a list to decay it epoch by epoch.
            """
        case .rlhfReinforce:
            """
            • The policy gradient is `−(reward − β · KL) · log π(action)`, summed over the sampled trajectory. There is no clipping and no value head, which is why the variance is high.
            • In `mlx_lm_lora` the “reward” is a scalar produced by an LLM judge (the `judge` setting); there is no separate value model.
            • KL is computed per token between the policy and the reference; `beta` is the coefficient in front of the KL term in the advantage (`reward − β · KL`).
            """
        case .ppo:
            """
            • The objective is the standard PPO surrogate: `−min(ρ · A, clip(ρ, 1−ε, 1+ε) · A)`, with `ρ = π / π_ref` and `A` derived from the chosen-rejected reward gap.
            • The clip range `epsilon = 0.2` is the classic Schulman default; tightening it (e.g. 0.1) makes updates more conservative, loosening it (e.g. 0.3) lets the policy move further per step.
            • KL is added on top of the surrogate as a regulariser, not as part of the advantage.
            """
        }
    }

    private var mathView: some View {
        let parts = math
        return VStack(alignment: .leading, spacing: 10) {
            if !parts.prose.isEmpty {
                Text(.init(MathTypesetter.richify(parts.prose)))
                    .fixedSize(horizontal: false, vertical: true)
            }
            ForEach(Array(parts.equations.enumerated()), id: \.offset) { _, eq in
                MathBlock(source: eq)
            }
        }
    }

    private var math: (prose: String, equations: [String]) {
        switch mode {
        case .sft:
            return (
                prose: "Per-example next-token cross-entropy, masked so only the completion contributes. Implemented in `sft_trainer.default_loss`: it shifts the input by one to form `(inputs, targets)`, optionally restricts the loss to a `[length_start, length_end]` range via the cache offset (used for `efficient_long_context`), and averages over the masked token count.",
                equations: [
                    "**ℒ**ₛfₜ(θ) = − ∑ₜ  log *p*₍θ₎(*y*ₜ | prompt, *y*<sub>t</sub>)        for t ∈ completion tokens",
                ]
            )
        case .dpo:
            return (
                prose: "For a single preference pair `(*y*₍c₎, *y*₍r₎)` and prompt `*x*`. `β` is the temperature: higher `β` means the policy is pushed harder toward the implicit reward. `delta` is only used in `dpop` loss and controls how much drift from the reference is penalised.",
                equations: [
                    "logits  =  ( log π_θ(*y*₍c₎|*x*)  −  log π_θ(*y*₍r₎|*x*) )  −  ( log π_ref(*y*₍c₎|*x*)  −  log π_ref(*y*₍r₎|*x*) )",
                    "**ℒ**₍DPO₎   =  − log σ( β · logits )                                                  (sigmoid)\n=  max( 0,  1 − β · logits )                                                (hinge)\n=  ( logits − 1 ⁄ (2β) )²                                                  (ipo)\n=  − log σ( β · logits )  +  δ · max( 0,  log π_ref(*y*₍c₎|*x*) − log π_θ(*y*₍c₎|*x*) )    (dpop)",
                ]
            )
        case .cpo:
            return (
                prose: "Same shape as DPO without the reference term in the main loss. The `dpop` CPO variant substitutes the policy log-prob for the reference in the drift penalty, because the reference is not in the forward pass.",
                equations: [
                    "logits  =  log π_θ(*y*₍c₎|*x*)  −  log π_θ(*y*₍r₎|*x*)",
                    "**ℒ**₍CPO₎   =  − log σ( β · logits )                                                  (sigmoid)\n=  max( 0,  1 − β · logits )                                                (hinge)\n=  ( logits − 1 ⁄ (2β) )²                                                  (ipo)\n=  − log σ( β · logits )  +  δ · max( 0,  log π_θ(*y*₍r₎|*x*) − log π_θ(*y*₍c₎|*x*) )    (dpop)",
                ]
            )
        case .orpo:
            return (
                prose: "The same loss reduces NLL on the chosen completion because the chosen log-prob term appears in both halves of the gradient. The optional `preference_score` rescales the chosen log-prob before the subtraction, so a row with `preference_score = 0.3` contributes roughly a third of the gradient of a row with score 1.0.",
                equations: [
                    "log_odds      =  log π_θ(*y*₍c₎|*x*)  −  log π_θ(*y*₍r₎|*x*)              (mean over tokens)",
                    "**ℒ**₍ORPO₎    =  − β · log σ( log_odds )",
                ]
            )
        case .grpo:
            return (
                prose: "For a prompt with `*G*` sampled completions and reward functions `{*r*₍k₎}` with weights `{*w*₍k₎}`. `ε_low` and `ε_high` (`epsilon` and `epsilon_high`) are the asymmetric clip bounds from DAPO. `importance_sampling_level` decides whether `ratio` is per-token or averaged across the sequence.",
                equations: [
                    "*R*<sub>i</sub>       =  ∑<sub>k</sub>  *w*₍k₎ · *r*₍k₎(prompt, *y*<sub>i</sub>)                              (total reward)",
                    "*A*<sub>i</sub>       =  ( *R*<sub>i</sub>  −  meanⱼ *R*<sub>j</sub> )  ⁄  ( stdⱼ *R*<sub>j</sub> + 1×10⁻⁴ )      (group-normalised advantage)",
                    "ratio<sub>i,t</sub>   =  π_θ(*y*<sub>i,t</sub>)  ⁄  π_ref(*y*<sub>i,t</sub>)                              (importance ratio)",
                    "**ℒ**₍clip₎    =  − min( ratio · *A*,  clip( ratio,  1−ε_low,  1+ε_high ) · *A* )",
                    "**ℒ**₍KL₎      =  β · ( ratio · (π_ref ⁄ π_θ)  −  log(π_ref ⁄ π_θ)  −  1 )                      (unbiased KL)",
                    "**ℒ**₍GRPO₎    =  ( **ℒ**₍clip₎  +  **ℒ**₍KL₎ )   averaged over valid tokens",
                ]
            )
        case .onlineDPO:
            return (
                prose: "For each prompt `*x*`, sample two completions `(*y*<sub>1</sub>, *y*<sub>2</sub>)`, judge picks the winner `*w* ∈ {0, 1}`. The temperature and judge system prompt together control how informative the labels are; `beta` and `loss_type` are identical to DPO.",
                equations: [
                    "*y*<sub>chosen</sub>   =  *y*<sub>w</sub>",
                    "*y*<sub>rejected</sub> =  *y*<sub>1−w</sub>",
                    "**ℒ**          =  DPO loss on ( *y*<sub>chosen</sub>, *y*<sub>rejected</sub> )    — see DPO math",
                ]
            )
        case .xpo:
            return (
                prose: "XPO loss is DPO with an additive exploration bonus proportional to the KL. A positive `alpha` means *reward me for being different from the reference*. When `alpha` is a list, `get_current_alpha(step, total, schedule)` returns the schedule element for the current step (one entry per epoch, with the last entry held to the end).",
                equations: [
                    "bonus  =  α · ( KL(π_θ ∥ π_ref) on chosen  +  KL(π_θ ∥ π_ref) on rejected )",
                    "**ℒ**₍XPO₎  =  **ℒ**₍DPO₎  −  bonus",
                ]
            )
        case .rlhfReinforce:
            return (
                prose: "For a prompt `*x*` and sampled completion `*y*` with judge reward `*R*`. No clipping, no value head. `β` is the KL weight in the advantage (not in a separate regulariser).",
                equations: [
                    "*A*<sub>t</sub>         =  *R*  −  β · KL<sub>t</sub>                              (per-token advantage)",
                    "**ℒ**₍REINFORCE₎ =  − ∑<sub>t</sub>  *A*<sub>t</sub> · log π_θ(*y*<sub>t</sub> | *x*, *y*<sub><t</sub>)",
                ]
            )
        case .ppo:
            return (
                prose: "For a prompt, two completions, and a winner `*w*`. The advantage is *not* a reward-model output here — it is derived from the log-prob gap between chosen and rejected, after the judge has decided which is which.",
                equations: [
                    "*A*           =  log π_θ(*y*<sub>c</sub>)  −  log π_θ(*y*<sub>r</sub>)                              (per-sequence advantage)",
                    "*A*<sub>norm</sub>      =  ( *A*  −  mean )  ⁄  ( std + 1×10⁻⁸ )",
                    "ρ<sub>c</sub>         =  exp( log π_θ(*y*<sub>c</sub>)  −  log π_ref(*y*<sub>c</sub>) )",
                    "ρ<sub>r</sub>         =  exp( log π_θ(*y*<sub>r</sub>)  −  log π_ref(*y*<sub>r</sub>) )",
                    "**ℒ**₍surr₎      =  − min( ρ<sub>c</sub> · *A*<sub>norm</sub>,  clip( ρ<sub>c</sub>,  1−ε,  1+ε ) · *A*<sub>norm</sub> )\n=  − min( ρ<sub>r</sub> · (−*A*<sub>norm</sub>),  clip( ρ<sub>r</sub>,  1−ε,  1+ε ) · (−*A*<sub>norm</sub>) )",
                    "**ℒ**₍PPO₎       =  **ℒ**₍surr₎  +  β · ( mean( log π_θ  −  log π_ref ) )",
                ]
            )
        }
    }

    // MARK: Settings table

    private var settingsTable: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Setting")
                    .frame(width: 180, alignment: .leading)
                Text("Default")
                    .frame(width: 110, alignment: .leading)
                Text("What it actually changes")
            }
            .font(.caption.bold())
            .foregroundStyle(.secondary)
            .padding(.bottom, 6)

            Divider()
            ForEach(Array(settings.enumerated()), id: \.offset) { _, row in
                HStack(alignment: .top) {
                    Text(row.setting)
                        .font(.system(.body, design: .monospaced))
                        .frame(width: 180, alignment: .leading)
                    Text(row.defaultValue)
                        .font(.system(.body, design: .monospaced))
                        .frame(width: 110, alignment: .leading)
                    Text(row.explanation)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.vertical, 6)
                Divider()
            }
        }
        .padding(.top, 4)
    }

    private struct SettingRow {
        let setting: String
        let defaultValue: String
        let explanation: String
    }

    private var settings: [SettingRow] {
        // The first block is always the shared SFT substrate.
        let base: [SettingRow] = [
            .init(setting: "model", defaultValue: "Qwen/Qwen3-0.6B",
                  explanation: "Hugging Face id or local path of the base model. Loaded with the chosen quantisation and wrapped with LoRA/DoRA/full-finetune adapters."),
            .init(setting: "data", defaultValue: "auto",
                  explanation: "HF dataset id, local folder, or JSONL. The trainer's `create_dataset` picks the column names per `train_mode`; for the bundled defaults this is automatic."),
            .init(setting: "train_type", defaultValue: "lora",
                  explanation: "`lora`, `dora`, or `full`. LoRA adds low-rank adapters to attention/MLP projections; DoRA decomposes the weight into magnitude and direction updates; `full` fine-tunes every parameter."),
            .init(setting: "lora_parameters.rank", defaultValue: "8",
                  explanation: "Inner rank of the LoRA adapters. Higher rank = more capacity, more parameters. 4–16 is typical on a laptop; 32+ needs careful memory budgeting."),
            .init(setting: "lora_parameters.scale", defaultValue: "20.0",
                  explanation: "LoRA scaling factor (α in the original paper). Effective update is `(α/r) · B·A`, so the ratio of scale to rank is what matters."),
            .init(setting: "lora_parameters.dropout", defaultValue: "0.0",
                  explanation: "Dropout inside the LoRA branch. Only worth setting above 0 when the dataset is tiny or you see overfitting on the validation curve."),
            .init(setting: "num_layers", defaultValue: "16",
                  explanation: "How many transformer layers get LoRA adapters (counted from the top). Smaller models ⇒ all layers; larger models ⇒ last N only."),
            .init(setting: "batch_size", defaultValue: "1",
                  explanation: "Per-device minibatch size. Real batch size on disk is `batch_size · gradient_accumulation_steps`."),
            .init(setting: "gradient_accumulation_steps", defaultValue: "1",
                  explanation: "Number of micro-batches to accumulate before stepping the optimiser. Increase to simulate a larger effective batch when memory is tight."),
            .init(setting: "iters / epochs", defaultValue: "iters=1000",
                  explanation: "Either fixed iterations or full passes over the dataset. Setting `epochs > 0` switches the runner to epoch-counted training."),
            .init(setting: "learning_rate", defaultValue: "1e-5",
                  explanation: "Peak learning rate. 1e-5 to 5e-5 is a sensible LoRA range; full fine-tuning usually wants 1e-6 to 5e-6."),
            .init(setting: "steps_per_report", defaultValue: "10",
                  explanation: "Print the loss/reward line every N optimiser steps. Also drives the live-metrics tick."),
            .init(setting: "steps_per_eval", defaultValue: "200",
                  explanation: "Run the validation pass every N optimiser steps. Set to -1 to disable."),
            .init(setting: "val_batches", defaultValue: "25",
                  explanation: "Number of validation minibatches per eval pass. Set to -1 to use the entire validation set."),
            .init(setting: "save_every", defaultValue: "100",
                  explanation: "How often (in optimiser steps) to checkpoint the adapter weights under `<adapter_path>/<iter>_adapters.safetensors`."),
            .init(setting: "max_seq_length", defaultValue: "2048",
                  explanation: "Hard cap on token length per example. Longer examples are truncated. Drives the KV-cache memory budget."),
            .init(setting: "grad_checkpoint", defaultValue: "true",
                  explanation: "Rewrites every transformer layer's forward to use `mx.checkpoint`. Roughly halves activation memory at the cost of one extra recompute per layer."),
            .init(setting: "efficient_long_context", defaultValue: "false",
                  explanation: "Splits long sequences into `seq_step_size` chunks and reuses the KV-cache between them. Enable when `max_seq_length` is large and you are OOM-ing."),
            .init(setting: "seq_step_size", defaultValue: "512",
                  explanation: "Chunk size used by `efficient_long_context`. 256–1024 is a reasonable range; smaller = less memory, more recompute."),
            .init(setting: "mask_prompt", defaultValue: "false",
                  explanation: "If true, the loss is only computed over the completion portion of the example (the prompt is masked out). SFT and DPO enforce this internally; the flag is for custom datasets."),
            .init(setting: "fuse", defaultValue: "true",
                  explanation: "After training, merge the LoRA weights back into the base model and save a standalone checkpoint. Disable to keep the adapter files separate."),
            .init(setting: "lm_studio_name", defaultValue: "—",
                  explanation: "If set, the merged model is written into the LM Studio models directory under this name in addition to (or instead of) `adapter_path`."),
            .init(setting: "resume_adapter_file", defaultValue: "—",
                  explanation: "Path to a previously-saved `adapters.safetensors` to warm-start from. Useful for continuing a long run or for SFT-then-DPO pipelines."),
            .init(setting: "seed", defaultValue: "0",
                  explanation: "Seed for `numpy` and `mlx.core`. Set deterministically to reproduce a run; bear in mind that MLX GPU kernels are not bit-exact across driver versions."),
            .init(setting: "load_in_4bits / 6 / 8 / mxfp4", defaultValue: "4-bit",
                  explanation: "Quantisation used when loading the base model. `4-bit` and `MXFP4` are the most aggressive; 8-bit is the safe default for preference training."),
            .init(setting: "qat_enable", defaultValue: "false",
                  explanation: "Installs a symmetric fake-quantise hook on every `nn.Linear` after the first optimiser step (configurable). Gradients flow through via the straight-through estimator, so the model trains as if it will be quantised at inference."),
            .init(setting: "qat_bits / group_size / start_step / interval", defaultValue: "8 / 64 / 1 / 1",
                  explanation: "Bit-width, group size (0 = per-tensor), first optimiser step to enable, and how often to re-project. Defaults match MLX's affine quantiser scheme."),
            .init(setting: "test / test_batches", defaultValue: "false / 100",
                  explanation: "If `test` is true, the runner evaluates a held-out test split after training and writes the result into the run record."),
        ]
        return base + modeSpecific
    }

    private var modeSpecific: [SettingRow] {
        switch mode {
        case .sft:
            return [
                .init(setting: "prompt_feature / completion_feature", defaultValue: "auto",
                      explanation: "Column-name overrides for the `(prompt, completion)` schema. Empty = use the trainer's defaults."),
                .init(setting: "messages_feature / system_feature", defaultValue: "auto",
                      explanation: "Column-name overrides for the chat-template `messages` schema. Used when the dataset ships multi-turn conversations."),
                .init(setting: "text_feature", defaultValue: "auto",
                      explanation: "Column-name override for the raw-pretraining `text` schema."),
            ]

        case .dpo:
            return [
                .init(setting: "beta", defaultValue: "0.1",
                      explanation: "Temperature inside the DPO sigmoid. Lower = softer updates, higher = more aggressive divergence from the reference."),
                .init(setting: "dpo_cpo_loss_type", defaultValue: "sigmoid",
                      explanation: "Loss variant: `sigmoid` (vanilla DPO), `hinge` (margin loss), `ipo` (squared error around `1/(2β)`), `dpop` (DPO with reference-drift penalty)."),
                .init(setting: "delta", defaultValue: "50.0",
                      explanation: "Coefficient in front of the `dpop` reference-drift penalty. Ignored unless `dpo_cpo_loss_type = dpop`."),
                .init(setting: "reference_model_path", defaultValue: "—",
                      explanation: "Path or HF id of the frozen reference model. If empty, the trainer instantiates a second copy of the base model (doubles the GPU memory)."),
                .init(setting: "prompt_feature / chosen_feature / rejected_feature", defaultValue: "auto",
                      explanation: "Column-name overrides for the preference schema."),
            ]

        case .cpo:
            return [
                .init(setting: "beta", defaultValue: "0.1",
                      explanation: "Temperature inside the CPO sigmoid."),
                .init(setting: "dpo_cpo_loss_type", defaultValue: "sigmoid",
                      explanation: "Loss variant. CPO accepts the same four options as DPO; `dpop` here is the policy-side drift penalty."),
                .init(setting: "delta", defaultValue: "50.0",
                      explanation: "Coefficient for the CPO drift penalty. Only used with `loss_type = dpop`."),
                .init(setting: "chosen_feature / rejected_feature", defaultValue: "auto",
                      explanation: "Column-name overrides. CPO does not need a separate `prompt` column because the dataset format is the same as DPO."),
            ]

        case .orpo:
            return [
                .init(setting: "beta", defaultValue: "0.1",
                      explanation: "Multiplier on the ORPO log-odds term."),
                .init(setting: "reward_scaling", defaultValue: "1.0",
                      explanation: "Reserved for future variants — the current ORPO loss does not use it as a separate multiplier."),
                .init(setting: "chosen_feature / rejected_feature / preference_score_feature", defaultValue: "auto",
                      explanation: "Column-name overrides. The ORPO trainer expects `chosen`, `rejected`, and optionally `preference_score`."),
            ]

        case .grpo:
            return [
                .init(setting: "group_size", defaultValue: "4",
                      explanation: "Number of completions sampled per prompt. Higher = lower-variance advantage estimates, more compute per step."),
                .init(setting: "beta", defaultValue: "0.1",
                      explanation: "KL penalty coefficient against the reference model. Set to 0 to disable KL entirely."),
                .init(setting: "epsilon / epsilon_high", defaultValue: "1e-4 / —",
                      explanation: "Asymmetric PPO clip range (`epsilon_low`, `epsilon_high`). If `epsilon_high` is empty, both bounds default to `epsilon` (classic PPO)."),
                .init(setting: "max_completion_length", defaultValue: "512",
                      explanation: "Maximum number of tokens to sample per completion. Drives the time-per-step budget."),
                .init(setting: "temperature / top_p / top_k / min_p", defaultValue: "0.8 / 0.95 / 20 / 0.0",
                      explanation: "Sampler settings for the in-loop generation. `temperature=0` is invalid; raise it if you want more diversity, lower it for near-deterministic completions."),
                .init(setting: "reward_functions", defaultValue: "—",
                      explanation: "Comma-separated list of reward function names to use. If empty, the default DeepSeek-R1 family (`r1_accuracy`, `r1_int`, `r1_strict_format`, `r1_soft_format`, `r1_count_xml`) is loaded."),
                .init(setting: "reward_functions_file", defaultValue: "—",
                      explanation: "Path to a Python file that defines a `REWARD_FUNCTIONS` list. Loaded via `load_reward_functions_from_file`."),
                .init(setting: "reward_weights", defaultValue: "—",
                      explanation: "Comma-separated list of weights matching the reward function list. Empty = all weights = 1.0."),
                .init(setting: "importance_sampling_level", defaultValue: "—",
                      explanation: "`token` (default), `sequence`, or empty. `sequence` averages the log-ratio per sequence — recommended for stability per the GSPO paper."),
                .init(setting: "grpo_loss_type", defaultValue: "grpo",
                      explanation: "`grpo` (mean over all tokens), `bnpo` (normalised by the actual token count), or `dr_grpo` (divided by `batch_size · max_tokens`)."),
                .init(setting: "reference_model_path", defaultValue: "—",
                      explanation: "Path or HF id of the reference model used both for KL and (when `importance_sampling_level != none`) for the importance ratio."),
            ]

        case .onlineDPO:
            return [
                .init(setting: "beta", defaultValue: "0.1",
                      explanation: "DPO temperature."),
                .init(setting: "dpo_cpo_loss_type", defaultValue: "sigmoid",
                      explanation: "Loss variant. Same four options as DPO."),
                .init(setting: "delta", defaultValue: "50.0",
                      explanation: "Drift-penalty coefficient for `dpop` loss."),
                .init(setting: "judge", defaultValue: "Qwen/Qwen3-0.6B",
                      explanation: "Hugging Face id / local path of the judge LLM, or the literal string `human`. With `human` the runner pauses and asks you to label each pair."),
                .init(setting: "judge_system", defaultValue: "—",
                      explanation: "System prompt sent to the LLM judge. Treat this as the rubric — short, specific, and concrete works best."),
                .init(setting: "max_completion_length", defaultValue: "512",
                      explanation: "Maximum number of tokens to sample per completion in the in-loop generation."),
                .init(setting: "temperature", defaultValue: "0.8",
                      explanation: "Sampling temperature for the policy. Lower ⇒ both completions look more similar, which means the judge sees harder comparisons."),
                .init(setting: "reference_model_path", defaultValue: "—",
                      explanation: "Path or HF id of the frozen reference model. Empty ⇒ second copy of the base model."),
            ]

        case .xpo:
            return [
                .init(setting: "beta", defaultValue: "0.1",
                      explanation: "DPO temperature."),
                .init(setting: "alpha", defaultValue: "1e-5",
                      explanation: "Exploration-bonus coefficient. Single float = constant. A list (e.g. `[1e-4, 1e-5, 1e-6]`) gives a per-epoch schedule that decays over training."),
                .init(setting: "dpo_cpo_loss_type", defaultValue: "sigmoid",
                      explanation: "Base DPO loss variant."),
                .init(setting: "delta", defaultValue: "50.0",
                      explanation: "Drift-penalty coefficient for `dpop`."),
                .init(setting: "judge", defaultValue: "Qwen/Qwen3-0.6B",
                      explanation: "LLM or `human` for the pairwise judge."),
                .init(setting: "judge_system", defaultValue: "—",
                      explanation: "Rubric system prompt for the judge."),
                .init(setting: "max_completion_length", defaultValue: "512",
                      explanation: "Maximum sampled completion length."),
                .init(setting: "temperature", defaultValue: "0.8",
                      explanation: "Sampling temperature for the policy."),
                .init(setting: "reference_model_path", defaultValue: "—",
                      explanation: "Path or HF id of the frozen reference model."),
            ]

        case .rlhfReinforce:
            return [
                .init(setting: "beta", defaultValue: "0.1",
                      explanation: "Coefficient on the KL term in the per-token advantage (`A = R − β · KL`). Set to 0 to disable KL regularisation."),
                .init(setting: "judge", defaultValue: "Qwen/Qwen3-0.6B",
                      explanation: "LLM judge that produces the scalar reward. The judge is loaded once and called once per (prompt, completion pair)."),
                .init(setting: "max_completion_length", defaultValue: "128",
                      explanation: "Maximum number of tokens to sample per completion. RLHF-REINFORCE is variance-sensitive, so shorter completions usually help."),
                .init(setting: "reference_model_path", defaultValue: "—",
                      explanation: "Frozen reference model used to compute the per-token KL. Empty ⇒ second copy of the base model."),
            ]

        case .ppo:
            return [
                .init(setting: "beta", defaultValue: "0.1",
                      explanation: "KL regulariser weight, added to the clipped surrogate."),
                .init(setting: "epsilon", defaultValue: "0.2",
                      explanation: "PPO clip range for the importance ratio. The classic value; lower it for more conservative updates."),
                .init(setting: "dpo_cpo_loss_type", defaultValue: "sigmoid",
                      explanation: "Loss variant (the chosen/rejected split is the same, only the inner objective changes — the runner passes it through)."),
                .init(setting: "delta", defaultValue: "50.0",
                      explanation: "Drift-penalty coefficient for `dpop`."),
                .init(setting: "judge", defaultValue: "Qwen/Qwen3-0.6B",
                      explanation: "Pairwise judge (LLM or `human`)."),
                .init(setting: "judge_system", defaultValue: "—",
                      explanation: "Rubric system prompt for the judge."),
                .init(setting: "max_completion_length", defaultValue: "512",
                      explanation: "Maximum sampled completion length."),
                .init(setting: "temperature", defaultValue: "0.8",
                      explanation: "Sampling temperature for the policy completions."),
                .init(setting: "reference_model_path", defaultValue: "—",
                      explanation: "Frozen reference model used in the importance ratio."),
            ]
        }
    }

    // MARK: Dataset format

    private var datasetFormat: String {
        switch mode {
        case .sft:
            """
            SFT accepts three shapes, in priority order:

            1. **chat messages** — a `messages` list of `{"role", "content"}` dicts (with an optional `system` field). The trainer applies the tokenizer's chat template.
            2. **prompt / completion** — explicit `(prompt, completion)` strings; the prompt is masked out of the loss when `mask_prompt = true`.
            3. **text** — a single `text` string for raw next-token pretraining-style fine-tuning.

            The bundled default for SFT is `mlx-community/JOSIE-v2-Instruct-5K`, a 5K-row instruction-tuning set in the messages format.
            """
        case .dpo:
            """
            DPO expects a preference dataset with one row per preference, containing at minimum `chosen` and `rejected`. The `prompt` column is optional but recommended; if present it is prepended to both completions before tokenisation.

            The bundled default is `mlx-community/Human-Like-DPO`, which has the prompt/chosen/rejected shape.
            """
        case .cpo:
            """
            Same as DPO: one row per preference with `chosen` and `rejected`. CPO does not use the `prompt` column.
            """
        case .orpo:
            """
            ORPO requires `chosen` and `rejected` and **no** `prompt` column (the chosen and rejected strings are used verbatim, and the model is expected to learn the prompt–completion split on its own). Optionally a `preference_score` (float) column scales the per-example gradient.

            The bundled default is `mlx-community/Josiefied-Qwen3-dpo-v1-flat`, a flattened DPO dataset.
            """
        case .grpo:
            """
            GRPO needs at minimum a `prompt` field and, for the default `r1_*` reward functions, an `answer` field. The 4-tuple the trainer produces per row is `(prompt_tokens, answer_tokens, prompt_text, answer_text)`; an optional 5th element is the `type` used to switch reward functions per category.

            The bundled default is `mlx-community/Dolci-Think-RL-7B-2k`, a reasoning dataset.
            """
        case .onlineDPO, .xpo, .rlhfReinforce, .ppo:
            """
            These online loops only need a `prompt` field at training time. The completions are sampled from the policy itself, and the (chosen, rejected) pair or scalar reward comes from the judge, not from the dataset.

            The bundled default is `mlx-community/Human-Like-DPO` for its small size and standard prompt column.
            """
        }
    }

    // MARK: When to use

    private var whenToUse: String {
        switch mode {
        case .sft:
            """
            **Almost always first.** SFT establishes the model's behaviour — tone, format, refusal style, domain knowledge. Every other algorithm in this list assumes you have an SFT checkpoint that is already in the right ballpark for the task, and only fine-tunes alignment on top.

            Reasonable defaults: `lr=1e-5`, `batch_size=1`, `grad_accum=8`, `iters=1000`, `rank=8`, `scale=20.0`.
            """
        case .dpo:
            """
            After SFT, when you have a static preference dataset (UltraFeedback, HelpSteer, Anthropic HH). DPO is the cheapest preference algorithm — one forward pass per completion on the policy, one forward pass on the reference.

            Use `loss_type=ipo` if you have noisy or contradictory labels, `loss_type=dpop` if the policy starts drifting from the reference in ways the SFT loss did not catch.
            """
        case .cpo:
            """
            Same use case as DPO when you cannot afford a second reference model on the GPU, or when you want a more aggressive update. Pair with a slightly smaller `beta` than DPO would use.
            """
        case .orpo:
            """
            When you want a single-stage alternative to “SFT then DPO”. ORPO has been shown to work well on small chat models and is the natural choice for datasets that ship a per-example preference score (UltraFeedback-style `score_chosen - score_rejected`).
            """
        case .grpo:
            """
            When you have a *verifiable* reward (math correctness, code execution, format compliance) rather than a labelled preference dataset. GRPO is the workhorse behind recent reasoning models (DeepSeek-R1, Qwen3-Instruct reasoning mode). Expect completions to look very different from SFT outputs — that is the point.
            """
        case .onlineDPO:
            """
            When you have a strong LLM judge (or a human in the loop) and a base prompt distribution you can keep sampling from. Online DPO avoids the off-policy gap of static DPO and works well with iterative refinement of the same model.
            """
        case .xpo:
            """
            When online DPO collapses toward the reference completion and you want a controlled amount of exploration. Try `alpha=1e-4` first, then either hold it constant or schedule it down across epochs.
            """
        case .rlhfReinforce:
            """
            Educational / minimal RLHF. With the modern LLM judge, REINFORCE is competitive with PPO for short completions and is much simpler to debug. It is also the algorithm that is most sensitive to judge quality and learning rate — start with `lr=5e-6` and a `beta=0.05`.
            """
        case .ppo:
            """
            The classic, the most expressive, and the most finicky. Reach for PPO when the other loops are under-performing on a metric the judge captures well, and you have time to tune `epsilon` and `beta` together. Always log `clip_fraction` — if it sits at 0% the policy is not moving, if it sits at >30% the policy is moving too far per step.
            """
        }
    }

    // MARK: Tips

    private var tipsList: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(tips.enumerated()), id: \.offset) { _, tip in
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: "lightbulb")
                        .foregroundStyle(.tint)
                        .padding(.top, 2)
                    Text(.init(MathTypesetter.richify(tip)))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private var tips: [String] {
        switch mode {
        case .sft:
            [
                "Start with the bundled default dataset and run for a small number of iters — you should see `loss` drop within 100 steps. If it does not, the dataset probably has the wrong column names for SFT (the trainer raises `Unsupported data format`).",
                "If you are OOM-ing, enable `grad_checkpoint` first, then drop `max_seq_length`, then try `efficient_long_context` with a `seq_step_size` of 256.",
                "QAT is most useful when you plan to ship a 4-bit or 8-bit version of the adapter for LM Studio. Leave it off during development — it adds noticeable per-step overhead.",
                "`mask_prompt=true` is the default behaviour for SFT in the trainer; the flag is exposed for custom datasets that mix masked and unmasked rows.",
            ]
        case .dpo:
            [
                "Set `reference_model_path` to the SFT checkpoint, not the original base model — the implicit reward is then `Δ log-prob vs the SFT model`, which is what your labels reflect.",
                "Watch `accuracies` and `margins` in the live metrics. `accuracies > 0.7` with `margins > 0` means the loss is doing real work; if `margins` plateaus, raise `beta` slightly.",
                "If you see NaNs, the most common cause is `loss_type=dpop` with `delta` too large for the current `lr`. Drop `delta` to 10 first.",
                "`efficient_long_context` applies here too — preference datasets with long answers (e.g. full document diffs) benefit from chunked forward passes.",
            ]
        case .cpo:
            [
                "Use a smaller `beta` than DPO would use (start at 0.05) — without the reference term the gradient is unanchored and large `beta` overshoots.",
                "CPO is the only preference algorithm that does not need a second copy of the base model, so it is the right call when you are running on integrated Apple Silicon GPUs.",
                "If validation reward plateaus early, switch the loss to `dpop` and add a `delta` to keep drift bounded.",
            ]
        case .orpo:
            [
                "Verify your dataset does not have a `prompt` column — ORPO silently concatenates the prompt into the chosen/rejected if you let it, which is rarely what you want.",
                "`preference_score` should be normalised to roughly 0..1 before training; very large scores push the loss into saturated regions of `log_sigmoid`.",
                "ORPO does not need a separate SFT warm-up — the same loss improves the NLL. Skip a stage in your pipeline if you were planning ‘SFT then DPO’.",
            ]
        case .grpo:
            [
                "Start with the default DeepSeek-R1 reward family. They are designed for `<reasoning>...</reasoning><answer>...</answer>` completions; if your data does not have that structure, write a custom reward function.",
                "The most common failure mode is `hit_max_tokens_ratio = 1.0` — the model is generating until the limit and never reaching the answer tag. Lower `max_completion_length` or strengthen the format reward.",
                "`clip_ratio_total` should be in the 0.05–0.2 range. Below that the advantage signal is too weak, above that the policy is moving too aggressively per step.",
                "`importance_sampling_level = sequence` is a free stability win for reasoning tasks with long completions.",
                "GRPO is the slowest of the loops because every step does a generation pass; expect 4–8x the wall-clock time of an SFT step at the same `batch_size`.",
            ]
        case .onlineDPO:
            [
                "The judge system prompt is the single highest-leverage string in the whole config. Write it as a concrete rubric with bullet points, not as a vague ‘which is better?’.",
                "Use `temperature ≈ 0.8` and `top_p=0.95` for the policy — too low and both completions are identical, too high and the judge labels look random.",
                "If you set `judge = \"human\"`, the runner will pause and ask you to label every pair; budget your time accordingly (or drop `batch_size` to 1).",
            ]
        case .xpo:
            [
                "Default `alpha = 1e-5` is very small. If completions look indistinguishable from the reference, raise `alpha` to `1e-4` and watch `exploration_bonus` in the live metrics.",
                "For a 3-epoch run, try `alpha = [1e-4, 1e-5, 1e-6]` — strong exploration early, decaying to nothing by the final epoch.",
            ]
        case .rlhfReinforce:
            [
                "Use a small `max_completion_length` (128 is a good default). REINFORCE variance scales with the trajectory length.",
                "Prefer `lr=5e-6` to `1e-5`; REINFORCE has no clipping to catch a too-aggressive step.",
                "Always log `rewards` and `kl_penalty` — if `kl_penalty` is rising while `rewards` plateaus, the policy is drifting in a way the judge does not see.",
            ]
        case .ppo:
            [
                "Log `clip_fraction`. If it is consistently > 0.3, the policy is moving too far per step — either lower `lr` or tighten `epsilon` to 0.1.",
                "`epsilon = 0.2` is the original PPO default and is a fine starting point; do not lower it until you have seen the policy train for at least one full pass.",
                "A KL weight (`beta`) that is too small lets the policy drift far from the reference; a `beta` that is too large suppresses the policy before it learns anything. Start at 0.1 and adjust based on the `kl_penalty` log.",
                "The judge matters more than the algorithm. PPO amplifies whatever the judge rewards, for better or worse.",
            ]
        }
    }
}

// MARK: - Foundation Articles
//
// The training-loop articles above describe *what loss to minimise*.
// These three foundation articles describe the orthogonal pieces that
// are configured separately on the Train page: which tensors are
// trainable, how the optimiser turns gradients into updates, and how
// (or whether) the forward pass is quantised.

/// Shared scaffolding for a long-form article. Mirrors the
/// AlgorithmArticle layout (header → overview → intuition → math →
/// settings table → tips) so the page reads as a single wiki.
private struct FoundationArticle<Body: View>: View {
    let title: String
    let subtitle: String
    let symbol: String
    @ViewBuilder let content: Body

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Image(systemName: symbol)
                    .font(.title2)
                    .frame(width: 42, height: 42)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.system(.largeTitle, design: .rounded).bold())
                    Text(subtitle)
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }
            }
            content
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .liquidGlass(cornerRadius: 14)
    }
}

/// Reusable section header / text block for a wiki-style article.
private struct ArticleSection: View {
    let title: String
    let text: String

    init(_ title: String, _ text: String) {
        self.title = title
        self.text = text
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.title3.bold())
            Text(.init(MathTypesetter.richify(text)))
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

private struct ArticleTips: View {
    let tips: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Tips & gotchas")
                .font(.title3.bold())
            ForEach(Array(tips.enumerated()), id: \.offset) { _, tip in
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: "lightbulb")
                        .foregroundStyle(.tint)
                        .padding(.top, 2)
                    Text(.init(MathTypesetter.richify(tip)))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }
}

private struct SettingsTable: View {
    struct Row {
        let setting: String
        let defaultValue: String
        let explanation: String
    }

    let rows: [Row]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Setting")
                    .frame(width: 200, alignment: .leading)
                Text("Default")
                    .frame(width: 110, alignment: .leading)
                Text("What it actually changes")
            }
            .font(.caption.bold())
            .foregroundStyle(.secondary)
            .padding(.bottom, 6)

            Divider()
            ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                HStack(alignment: .top) {
                    Text(row.setting)
                        .font(.system(.body, design: .monospaced))
                        .frame(width: 200, alignment: .leading)
                    Text(row.defaultValue)
                        .font(.system(.body, design: .monospaced))
                        .frame(width: 110, alignment: .leading)
                    Text(row.explanation)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.vertical, 6)
                Divider()
            }
        }
        .padding(.top, 4)
    }
}

// MARK: - Math rendering
//
// Math equations are written in the article source as plain strings
// of Unicode math (∑, β, ², ₁, →, ‖·‖, ≈, ε, λ, …) with optional
// Markdown emphasis for variables (`*x*`) and vectors (`**v**`).
// Authors may also wrap sub/superscript runs in `<sub>…</sub>` and
// `<sup>…</sup>` tags; `MathTypesetter.richify` translates those
// into Unicode glyphs before the string is handed to SwiftUI's
// Markdown renderer (which does not understand HTML-style tags).
// `MathBlock` typesets the result with a serif font inside a tinted
// panel; `MathInline` does the same in-line for prose paragraphs.

/// Translates author-friendly `<sub>…</sub>` / `<sup>…</sup>` markup
/// into Unicode subscript/superscript glyphs, so a Markdown `Text`
/// renders it correctly. Multi-character content inside the tags is
/// passed through to the inner run (no Unicode glyph exists for "ₖₜ"),
/// which preserves the readable shape in serif italics and avoids the
/// confusing raw-tag leak users were seeing on screen.
enum MathTypesetter {
    /// Maps ASCII/digit runs to Unicode subscript glyphs where a
    /// single character exists; otherwise returns the input unchanged.
    private static let subscripts: [Character: Character] = [
        "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
        "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
        "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
        "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
        "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
        "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
        "v": "ᵥ", "x": "ₓ",
    ]

    private static let superscripts: [Character: Character] = [
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
        "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
        "n": "ⁿ", "i": "ⁱ",
    ]

    /// Translates `<sub>…</sub>` / `<sup>…</sup>` / `<super>…</super>`
    /// runs into Unicode glyphs. The parser accepts both `<sup>` and
    /// `<super>` as opening forms and uses the matching closer
    /// (`</sup>` or `</super>`) — they all map to the superscript
    /// glyph table. Nested tags are not supported (they are uncommon
    /// in this codebase and the parser would just get fragile).
    static func richify(_ source: String) -> String {
        // Quick exit: nothing to translate.
        guard source.contains("<sub>") || source.contains("<sup>") || source.contains("<super>") else { return source }
        var output = ""
        output.reserveCapacity(source.count)
        var i = source.startIndex
        while i < source.endIndex {
            if let sub = parseTag(source, at: i, open: "sub>", close: "</sub>") {
                output.append(contentsOf: translate(sub, table: subscripts))
                i = source.index(i, offsetBy: "<sub>".count + sub.count + "</sub>".count)
                continue
            }
            if let sup = parseSuper(source, at: i) {
                output.append(contentsOf: translate(sup.inner, table: superscripts))
                i = source.index(i, offsetBy: sup.openCount + sup.inner.count + sup.closeCount)
                continue
            }
            output.append(source[i])
            i = source.index(after: i)
        }
        return output
    }

    /// Result of `parseSuper`: the inner content plus how many
    /// characters the matching open and close tags each consumed,
    /// so the caller's index arithmetic stays accurate for both
    /// `<sup>` (4 chars) and `<super>` (7 chars).
    private struct SupHit {
        let inner: String
        let openCount: Int
        let closeCount: Int
    }

    /// Matches `<sup>…</sup>` and `<super>…</super>` at the given
    /// index and returns the inner content. Returns nil if neither
    /// opener appears at `i`.
    private static func parseSuper(_ source: String, at i: String.Index) -> SupHit? {
        let openers: [(String, String)] = [("<super>", "</super>"), ("<sup>", "</sup>")]
        for (open, close) in openers {
            guard source[i...].hasPrefix(open) else { continue }
            let afterOpen = source.index(i, offsetBy: open.count)
            guard let closeRange = source.range(of: close, range: afterOpen..<source.endIndex) else { continue }
            return SupHit(
                inner: String(source[afterOpen..<closeRange.lowerBound]),
                openCount: open.count,
                closeCount: close.count
            )
        }
        return nil
    }

    /// If `source` starting at `i` opens a `<sub>…</sub>` tag,
    /// returns the inner content. Returns nil if no tag starts here.
    private static func parseTag(_ source: String, at i: String.Index, open: String, close: String) -> String? {
        let openFull = "<" + open
        guard source[i...].hasPrefix(openFull) else { return nil }
        let afterOpen = source.index(i, offsetBy: openFull.count)
        guard let closeRange = source.range(of: close, range: afterOpen..<source.endIndex) else { return nil }
        return String(source[afterOpen..<closeRange.lowerBound])
    }

    private static func translate(_ inner: String, table: [Character: Character]) -> String {
        var out = ""
        out.reserveCapacity(inner.count)
        for ch in inner {
            if let mapped = table[ch] {
                out.append(mapped)
            } else {
                out.append(ch)
            }
        }
        return out
    }
}



/// Block-level equation: a tinted panel with a left accent rule and
/// the equation body rendered as rich text. Selectable so the user
/// can copy a formula.
struct MathBlock: View {
    let source: String
    var caption: String? = nil

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            Rectangle()
                .fill(LinearGradient(
                    colors: [.accentColor.opacity(0.85), .accentColor.opacity(0.20)],
                    startPoint: .top, endPoint: .bottom
                ))
                .frame(width: 3)

            VStack(alignment: .leading, spacing: 4) {
                Text(.init(MathTypesetter.richify(source)))
                    .font(.system(.body, design: .serif))
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.vertical, 10)
                    .padding(.horizontal, 14)
                if let caption {
                    Text(caption)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 14)
                        .padding(.bottom, 8)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(.tint.opacity(0.07))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(.tint.opacity(0.20), lineWidth: 0.5)
        )
    }
}

/// Inline math: same serif font, no panel chrome. Used inside the
/// prose paragraphs of "Intuition" / "Overview" sections.
struct MathInline: View {
    let source: String

    var body: some View {
        Text(.init(MathTypesetter.richify(source)))
            .font(.system(.body, design: .serif).italic())
            .textSelection(.enabled)
    }
}

// MARK: Adaptation Article

/// Covers the three `train_type` values: `lora` (low-rank adapters on
/// the attention/MLP projections), `dora` (the same with an extra
/// magnitude-direction decomposition), and `full` (every weight is
/// trainable). Backed by `mlx_lm_lora.utils.from_pretrained`, which
/// wraps the base model with `linear_to_lora_layers` and applies the
/// chosen quantisation.
private struct AdaptationArticle: View {
    @Binding var config: TrainingConfig

    var body: some View {
        FoundationArticle(
            title: "Adaptation methods",
            subtitle: "Which tensors receive gradients",
            symbol: "square.stack.3d.up"
        ) {
            ArticleSection("Overview", """
                Every trainer in mlx-lm-lora operates on the same base model loaded by `mlx_lm_lora.utils.from_pretrained`. What changes between `lora`, `dora` and `full` is which tensors are **trainable** and how the gradients are computed. LoRA keeps the base model frozen and learns a low-rank update to each targeted linear layer; DoRA keeps the same low-rank update but factorises the combined weight into a magnitude vector and a direction matrix so the two can be tuned independently; `full` unfreezes the entire model.
                """)

            ArticleSection("Intuition", """
                • **LoRA** pretends the optimal weight change `ΔW` is rank-deficient: `ΔW = (α/r) · B · A` where `A ∈ R^{r×in}`, `B ∈ R^{out×r}`, `r ≪ min(in, out)`. For a 7B model with `r=8` this drops trainable parameters from ~7B to ~4M.
                • **DoRA** decomposes the (frozen + LoRA) weight into a unit-norm direction `V` and a learnable magnitude `m`, then writes `W' = m · V / ‖V‖`. The motivation is empirical: full fine-tuning tends to update direction and magnitude by very different amounts, and DoRA reproduces that behaviour while keeping the parameter count near LoRA's.
                • **`full` fine-tuning** is the textbook case. Every weight is trainable, the optimiser state is proportional to the parameter count, and a 7B model needs ~28 GB of activation + optimiser memory at fp16. On a laptop this is rarely viable past the 1B scale.
                • All three share the same forward pass; only the parameter list passed to `nn.value_and_grad(model, …)` changes. That is why every algorithm above works with every adaptation method.
                """)

            VStack(alignment: .leading, spacing: 10) {
                Text("Objective (math)")
                    .font(.title3.bold())
                Text(.init(MathTypesetter.richify("Let *W*<sub>0</sub> ∈ ℝ<sup>out × in</sup> be the frozen base weight, *x* the input and *y* the output of the targeted `nn.Linear` layer.")))
                    .fixedSize(horizontal: false, vertical: true)

                Text("**LoRA** (Hu et al., 2021)")
                    .font(.body.bold())
                MathBlock(source: """
                    *A*  ~  𝒩( 0,  σ² )           (initialised)
                    *B*   =  **0**                 (initialised — first step is a no-op)
                    Δ*W*  =  ( α ⁄ *r* )  ·  *B* · *A*
                    *y*   =  ( *W*<sub>0</sub> + Δ*W* ) · *x*  +  dropout( Δ*W* · *x* )       (if dropout > 0)
                """)
                Text(.init(MathTypesetter.richify("The *α ⁄ r* ratio is what the original paper called the *scaling factor*; in mlx-lm-lora the same number is split into the `scale` (α) and `rank` (r) settings.")))
                    .fixedSize(horizontal: false, vertical: true)

                Text("**DoRA** (Liu et al., 2024)")
                    .font(.body.bold())
                MathBlock(source: """
                    *W*′        =  *W*<sub>0</sub>  +  ( α ⁄ *r* )  ·  *B* · *A*
                    **V**        =  *W*′                                       (frozen after each step)
                    *m*         =  ‖*W*<sub>0</sub>‖<sub>c</sub>               (per-column magnitude, learnable)
                    *W*<sub>dora</sub>  =  *m*  ·  **V**  ⁄  ‖**V**‖
                    *y*         =  *W*<sub>dora</sub> · *x*
                """)
                Text(.init(MathTypesetter.richify("The unit-norm rescaling on **V** is what makes DoRA different from *LoRA plus a magnitude multiplier*.")))
                    .fixedSize(horizontal: false, vertical: true)

                Text("**Full fine-tuning**")
                    .font(.body.bold())
                MathBlock(source: "*y*  =  *W*  ·  *x*")
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("What the settings change")
                    .font(.title3.bold())
                SettingsTable(rows: [
                    .init(setting: "train_type", defaultValue: "lora",
                          explanation: "Pick `lora` (default), `dora`, or `full`. Switching to `full` removes the adapter wrapper entirely and is the only configuration that can be used with `num_layers` ignored."),
                    .init(setting: "lora_parameters.rank", defaultValue: "8",
                          explanation: "Inner rank of the low-rank update. 4 = tiny, 8 = typical, 16 = high capacity, 32+ = usually overkill on a laptop. Higher rank multiplies memory and adapter-file size by the same factor."),
                    .init(setting: "lora_parameters.scale", defaultValue: "20.0",
                          explanation: "Magnitude scaling (α). The effective update is `(α/r) · B·A`, so the ratio to `rank` is what matters. mlx-lm-lora's reference defaults are `(rank=8, scale=20.0)` for a 2.5x ratio; many papers use 1x."),
                    .init(setting: "lora_parameters.dropout", defaultValue: "0.0",
                          explanation: "Dropout applied to `ΔW · x` before it is added to `W_0 · x`. Only worth setting above 0 when training on a small dataset or when the loss curve shows overfitting."),
                    .init(setting: "num_layers", defaultValue: "16",
                          explanation: "How many of the top transformer layers receive LoRA/DoRA adapters. The trainer counts from the top, so `num_layers = 8` on a 32-layer model targets layers 24–31."),
                    .init(setting: "resume_adapter_file", defaultValue: "—",
                          explanation: "Path to a previously-saved `adapters.safetensors`. Works for any `train_type` (the file just needs to match the layer names of the new run)."),
                    .init(setting: "fuse", defaultValue: "true",
                          explanation: "If true, the LoRA/DoRA updates are merged back into `W_0` after training and the merged model is what is saved to `adapter_path`. Disable to keep adapter files separate."),
                ])
            }

            ArticleSection("When to use which", """
                • **LoRA** is the right default for everything in this app. ~4M trainable parameters, fits in CPU RAM, fast iteration.
                • **DoRA** is worth trying when LoRA plateaus on a metric that tracks style or format compliance (DoRA is reported to match full fine-tuning more closely on instruction-following). It costs roughly the same memory and time as LoRA; the only downside is a slightly larger adapter file.
                • **Full** is reserved for cases where LoRA / DoRA are not expressive enough — e.g. training a small model from scratch on a domain corpus. On a 16 GB Apple Silicon machine the practical limit is ~1B parameters at fp16.
                """)

            ArticleTips(tips: [
                "Start with `rank=8, scale=20.0, dropout=0.0`. If you are seeing the loss plateau, raise `rank` to 16 *before* changing the learning rate — capacity is usually the bottleneck, not step size.",
                "DoRA's magnitude vector is per-output-column, so a layer with a large output dim stores more DoRA parameters than LoRA at the same rank. The difference is small (a few hundred floats per layer) but it does show up in the adapter file size.",
                "If you change `train_type` between runs, delete the old `adapters.safetensors` first — the layer name conventions differ and a stale file will silently fail to load.",
                "For `full` fine-tuning, set `grad_checkpoint = true` from the start; the activation memory alone is enough to OOM a 7B model on a laptop.",
            ])
        }
    }
}

// MARK: Optimizer Article

/// Covers the three optimizers the runner exposes through
/// `optim.Adam`, `optim.AdamW` and `optim.Muon`. The runner picks the
/// class from `args.optimizer` and forwards the matching
/// `optimizer_config[args.optimizer]` dict to its constructor.
private struct OptimizerArticle: View {
    @Binding var config: TrainingConfig

    var body: some View {
        FoundationArticle(
            title: "Optimizers",
            subtitle: "How gradients become weight updates",
            symbol: "function"
        ) {
            ArticleSection("Overview", """
                mlx-lm-lora ships three optimizers: **Adam**, **AdamW** and **Muon**. The first two are the workhorses; Muon is a relatively recent addition (Jordan et al., 2024) that has been shown to converge faster on transformer hidden weights. The selection is exposed through `config.optimizer` and the trainer picks the matching class in `mlx.optimizers` at construction time.
                """)

            ArticleSection("Intuition", """
                • **Adam** keeps a per-parameter exponential moving average of the first moment `m` and the second moment `v` of the gradient, and applies a bias-corrected update. The default `betas = (0.9, 0.999)` work well across a wide range of LRs.
                • **AdamW** is Adam with **decoupled weight decay**: the regularisation is `λ · W` added directly to the update rather than appearing inside the gradient. This is the correct way to do L2 regularisation on adaptive optimizers and is the standard choice for transformer fine-tuning.
                • **Muon** is "Adam with the second moment replaced by a Newton-Schulz orthogonalisation of the momentum". Concretely, the per-tensor update is projected to (approximately) an orthogonal matrix before being scaled. Empirically this gives faster convergence on hidden 2D weights (the QKV and MLP projections inside the transformer blocks) and is much worse on 1D weights (biases, norms) — which is why Muon implementations typically pair it with AdamW for the non-hidden params.
                • In mlx-lm-lora all three are invoked with the same call signature `opt = OptClass(learning_rate=lr, **optimizer_config[opt_name])`, so any per-optimizer hyperparameters (e.g. `betas`, `eps`, `weight_decay`, `momentum`) are forwarded verbatim from the `optimizer_config` dict in the spec.
                """)

            VStack(alignment: .leading, spacing: 10) {
                Text("Objective (math)")
                    .font(.title3.bold())
                Text(.init(MathTypesetter.richify("Let *g*<sub>t</sub> = ∇**ℒ**(θ<sub>t−1</sub>) be the gradient at step *t*, and let *lr* be the learning rate. All three optimizers store their own state in `optimizer.state`; the trainer seeds that state from `mx.random.state` for determinism.")))
                    .fixedSize(horizontal: false, vertical: true)

                Text("**Adam** (Kingma & Ba, 2014)")
                    .font(.body.bold())
                MathBlock(source: """
                    *m*<sub>t</sub>   =  β₁ · *m*<sub>t−1</sub>  +  ( 1 − β₁ ) · *g*<sub>t</sub>               (first moment)
                    *v*<sub>t</sub>   =  β₂ · *v*<sub>t−1</sub>  +  ( 1 − β₂ ) · *g*<sub>t</sub>²              (second moment)
                    *m̂*<sub>t</sub>  =  *m*<sub>t</sub> ⁄ ( 1 − β₁<super>t</super> )                          (bias correction)
                    *v̂*<sub>t</sub>  =  *v*<sub>t</sub> ⁄ ( 1 − β₂<super>t</super> )                          (bias correction)
                    θ<sub>t</sub>    =  θ<sub>t−1</sub>  −  *lr*  ·  *m̂*<sub>t</sub> ⁄ ( √*v̂*<sub>t</sub> + ε )
                """)
                Text(.init(MathTypesetter.richify("Default in `mlx.optimizers`: β₁ = 0.9, β₂ = 0.999, ε = 1×10⁻⁸. L2 regularisation is *not* applied — use AdamW if you want weight decay.")))
                    .fixedSize(horizontal: false, vertical: true)

                Text("**AdamW** (Loshchilov & Hutter, 2019)")
                    .font(.body.bold())
                MathBlock(source: """
                    ( *m*<sub>t</sub>,  *v*<sub>t</sub>,  *m̂*<sub>t</sub>,  *v̂*<sub>t</sub> )  ←  Adam update as above
                    θ<sub>t</sub>    =  θ<sub>t−1</sub>  −  *lr*  ·  (  *m̂*<sub>t</sub> ⁄ ( √*v̂*<sub>t</sub> + ε )  +  λ · θ<sub>t−1</sub>  )
                """)
                Text(.init(MathTypesetter.richify("The *λ · θ* term is the decoupled weight decay. Default `weight_decay = 0.01`; tune this to control how aggressively the model is pulled toward zero (and therefore how much the LoRA / DoRA adapters are encouraged to stay small).")))
                    .fixedSize(horizontal: false, vertical: true)

                Text("**Muon** (Jordan et al., 2024)")
                    .font(.body.bold())
                MathBlock(source: """
                    *m*<sub>t</sub>   =  μ · *m*<sub>t−1</sub>  +  *g*<sub>t</sub>                       (momentum buffer, μ ≈ 0.95)
                    **O**<sub>t</sub>   =  NewtonSchulz5( *m*<sub>t</sub> )                                (≈ orthogonalise the momentum)
                    scale    =  √( out · in )                                                              (spectral-norm-preserving scale)
                    θ<sub>t</sub>    =  θ<sub>t−1</sub>  −  *lr*  ·  scale  ·  **O**<sub>t</sub>
                """)
                Text(.init(MathTypesetter.richify("The Newton–Schulz iteration is a small fixed-point loop (5 steps in the paper) that maps a matrix to its nearest semi-orthogonal one. The update is a single matrix multiply per parameter, so the wall-clock cost is comparable to AdamW despite the extra iteration.")))
                    .fixedSize(horizontal: false, vertical: true)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("What the settings change")
                    .font(.title3.bold())
                SettingsTable(rows: [
                    .init(setting: "optimizer", defaultValue: "adamw",
                          explanation: "Pick `adam`, `adamw`, or `muon`. The class is loaded from `mlx.optimizers` and constructed with `learning_rate=lr` plus the matching `optimizer_config` dict."),
                    .init(setting: "learning_rate", defaultValue: "1e-5",
                          explanation: "Peak learning rate. Sensible LoRA/DoRA range: 1e-5 to 5e-5 for AdamW, 5e-4 to 5e-3 for Muon (Muon is parameterised differently and likes a much larger LR). Full fine-tuning usually wants 1e-6 to 5e-6 with AdamW."),
                    .init(setting: "lr_schedule", defaultValue: "—",
                          explanation: "Optional schedule from `mlx_lm.tuner.utils.build_schedule`. If non-empty, the runner wraps `learning_rate` with the schedule; otherwise `learning_rate` is used as a constant."),
                    .init(setting: "optimizer_config.adam", defaultValue: "{}",
                          explanation: "Extra kwargs forwarded to `optim.Adam(...)`. Recognised keys: `betas`, `eps`. Defaults to `mlx.optimizers.Adam`'s defaults."),
                    .init(setting: "optimizer_config.adamw", defaultValue: "{}",
                          explanation: "Extra kwargs forwarded to `optim.AdamW(...)`. Recognised keys: `betas`, `eps`, `weight_decay`. Default `weight_decay=0.01`; raise to 0.05 if you are seeing the LoRA magnitudes drift up over training."),
                    .init(setting: "optimizer_config.muon", defaultValue: "{}",
                          explanation: "Extra kwargs forwarded to `optim.Muon(...)`. Recognised keys: `momentum`, `nesterov`, `weight_decay`. Defaults to Muon's defaults (momentum=0.95, nesterov=True)."),
                ])
            }

            ArticleSection("When to use which", """
                • **AdamW** is the safe default for everything in this app. Pick it when you do not have a strong reason to use something else.
                • **Adam** (without weight decay) is a reasonable choice when you are explicitly trying to disable regularisation — e.g. when the base model is already regularised and you do not want the LoRA adapters to be penalised.
                • **Muon** is a free lunch on the LoRA/DoRA hidden weights of any 1B+ model; it converges in roughly half the iterations to the same loss. Pair it with a higher learning rate (5e-3 to 1e-2) and keep `weight_decay=0.0` unless you specifically want the LoRA magnitudes regularised.
                """)

            ArticleTips(tips: [
                "If you switch from AdamW to Muon, raise the learning rate by ~100x. The Muon update is parameterised differently and the AdamW default of 1e-5 will under-train.",
                "Watch the live-metrics loss curve for the first 50 steps after switching optimizers — a sudden divergence almost always means the LR is wrong, not the optimizer choice.",
                "If you set `weight_decay` for AdamW on a `full` fine-tuning run, expect a small loss bump in the first 100 steps as the regulariser pulls the weights toward zero. This is normal and usually recovers within an epoch.",
                "`lr_schedule` is a small DSL from upstream mlx-examples. Common choices: `cosine:<iters>:<min_lr>` for a cosine decay, `warmup_cosine:<warmup>:<iters>:<min_lr>` for a warm-up followed by decay.",
            ])
        }
    }
}

// MARK: Quantization & QAT Article

/// Covers the four load-time quantisation modes (`load_in_4bits`,
/// `load_in_6bits`, `load_in_8bits`, `load_in_mxfp4`) and the
/// in-training Quantization-Aware Training (QAT) hook installed by
/// `_install_qat_hooks` in the SFT trainer. Backed by
/// `mlx.nn.quantize` (load time) and the straight-through-estimator
/// fake-quantise hook (train time).
private struct QuantizationArticle: View {
    @Binding var config: TrainingConfig

    var body: some View {
        FoundationArticle(
            title: "Quantization & QAT",
            subtitle: "Loading the base model in low precision, and training against quantised weights",
            symbol: "cpu"
        ) {
            ArticleSection("Overview", """
                mlx-lm-lora supports **two distinct kinds of quantisation** that are easy to confuse:

                1. **Load-time quantisation** — when the base model is read from disk, `mlx.nn.quantize` replaces each `nn.Linear` with a `QuantizedLinear` that stores the weights in N-bit signed integers and a per-group scale. The base model stays in memory in that form for the rest of the run, and the LoRA / DoRA adapters are kept in full precision on top of it.
                2. **Quantization-Aware Training (QAT)** — a small hook installed on every `nn.Linear` after the first optimiser step. The hook fake-quantises the weight on the way *into* the forward pass (straight-through estimator), so the model trains as if it would be quantised at inference time. The optimiser still sees and updates the full-precision weights, so the gradient is unaffected.

                Both are exposed through the `quantization` setting and the `qat_*` settings respectively. They compose: a typical "QLoRA + QAT" run loads the base model in 4-bit and then trains with the QAT hook enabled so the LoRA updates are robust to that 4-bit precision.
                """)

            ArticleSection("Intuition", """
                • **Load-time quantisation** is the same technique used by every "QLoRA" pipeline. The matrix `W ∈ R^{out×in}` is split along the last axis into groups of `group_size` consecutive columns, each group is divided by a scale `s_g = max(|W_g|) / qmax`, and the values are rounded to signed N-bit integers. The quantised weight is `Q_g = round(W_g / s_g)` and the dequantised weight is `Q_g · s_g ≈ W_g`.
                • **QAT** is the difference between a model that *was* trained in fp16 and *deployed* in 4-bit (which loses accuracy on outlier channels) and a model that was trained *as if* it would be deployed in 4-bit (which learns to keep the outliers tame). The hook is installed by patching the `__call__` of every `nn.Linear` subclass; weights are restored to full precision after each forward so the optimiser still has a clean signal.
                • The straight-through estimator is the only "trick" in QAT: the forward uses the quantised weight and the backward is the identity, so gradients flow through unchanged. This is the reason QAT works at all — without the STE, the quantise-then-round step would be zero almost everywhere and gradients would vanish.
                • In mlx-lm-lora the QAT hook is enabled after the first optimiser step (`qat_start_step`) and re-applied every `qat_interval` steps. That deferred start is important: the very first optimiser step on a freshly initialised LoRA would quantise noise.
                """)

            VStack(alignment: .leading, spacing: 10) {
                Text("Objective (math)")
                    .font(.title3.bold())

                Text("**Affine quantisation** (load-time, all four modes)")
                    .font(.body.bold())
                MathBlock(source: """
                    # Per group of *group_size* consecutive columns of *W*:
                    *s*<sub>g</sub>     =  max(| *W*<sub>g</sub> |) ⁄ *q*<sub>max</sub>                 (positive scale)
                    *Q*<sub>g</sub>     =  clip( round( *W*<sub>g</sub> ⁄ *s*<sub>g</sub> ),  *q*<sub>min</sub>,  *q*<sub>max</sub> )    (N-bit signed int)
                    *Ŵ*<sub>g</sub>     =  *Q*<sub>g</sub>  ·  *s*<sub>g</sub>                          (dequantised)
                """)
                Text(.init(MathTypesetter.richify("with *q*<sub>max</sub> = 2<sup>N − 1</sup> − 1 and *q*<sub>min</sub> = −*q*<sub>max</sub> − 1. For N = 4, *q*<sub>max</sub> = 7 and *q*<sub>min</sub> = −8. For MXFP4 the format is identical but the group size is fixed at 32 and the scale is stored as an 8-bit E8M0 value (a *microscaling* format).")))
                    .fixedSize(horizontal: false, vertical: true)

                Text("**Symmetric fake quantise** (QAT, applied inside the forward)")
                    .font(.body.bold())
                MathBlock(source: """
                    # Same arithmetic, but the quantise+dequantise pair happens
                    # at runtime on the current weight tensor:
                    *Ŵ*   =  *s*  ·  clip( round( *W* ⁄ *s* ),  *q*<sub>min</sub>,  *q*<sub>max</sub> )

                    # The forward uses *Ŵ*; the backward uses an STE:
                    ∂**ℒ** ⁄ ∂*W*   =  ∂**ℒ** ⁄ ∂*Ŵ*                                  (identity — gradient flows through)
                """)
                Text("The hook is implemented as")
                    .font(.body.bold())
                MathBlock(source: """
                    self.weight  =  *w*  +  stop_gradient( quantize(*w*)  −  *w* )        # forward sees *Ŵ*
                    out          =  original_forward(self, *x*)
                    self.weight  =  *w*                                                  # restore for optimiser
                """)
                Text(.init(MathTypesetter.richify("Note the `stop_gradient` around `( quantize(*w*) − *w* )` — that is the STE. The `+ *w*` outside it means the forward value is exactly *Ŵ*, the backward value is 1, and the optimiser only ever touches the full-precision *w*.")))
                    .fixedSize(horizontal: false, vertical: true)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("What the settings change")
                    .font(.title3.bold())
                SettingsTable(rows: [
                    .init(setting: "load_in_4bits", defaultValue: "false",
                          explanation: "Quantise the base model to 4-bit on load (group size 128). The most aggressive option; pairs naturally with LoRA/DoRA. Doubles `rank`-for-`rank` capacity versus 8-bit on a 16 GB machine."),
                    .init(setting: "load_in_6bits", defaultValue: "false",
                          explanation: "6-bit quantisation (group size 128). Rarely the right choice — usually either 4-bit (smallest) or 8-bit (highest quality) wins."),
                    .init(setting: "load_in_8bits", defaultValue: "false",
                          explanation: "8-bit quantisation (group size 128). The safe default for preference training and for any run where the model has to read long contexts with precise numerics."),
                    .init(setting: "load_in_mxfp4", defaultValue: "false",
                          explanation: "MXFP4 (microscaling 4-bit) with group size 32. Used by recent NVIDIA/Microscaling hardware; on Apple Silicon it is functionally similar to 4-bit but with smaller groups and an 8-bit E8M0 scale."),
                    .init(setting: "qat_enable", defaultValue: "false",
                          explanation: "Install the STE fake-quantise hook on every `nn.Linear` after the first optimiser step. Only effective for the SFT/DPO/ORPO trainers (the others do not call `_install_qat_hooks`)."),
                    .init(setting: "qat_bits", defaultValue: "8",
                          explanation: "Bit-width used by the QAT hook. Match this to the inference quantisation: if you will deploy at 4-bit, train with `qat_bits=4`; if you will deploy at 8-bit, train with `qat_bits=8`."),
                    .init(setting: "qat_group_size", defaultValue: "64",
                          explanation: "Group size used by the QAT hook. 0 or negative means per-tensor. Match the deployment group size if you can; 64 or 128 are the most common values."),
                    .init(setting: "qat_start_step", defaultValue: "1",
                          explanation: "First optimiser step on which to install the hook. Set higher if your first few steps see NaN gradients — the hook is intolerant of weights that are still close to their initialisation."),
                    .init(setting: "qat_interval", defaultValue: "1",
                          explanation: "Re-apply the QAT projection every N optimiser steps. The default (`1`) projects on every step; raise to e.g. `4` if the projection is showing up in your training-time profile."),
                ])
            }

            ArticleSection("Which to pick", """
                • **Default (no quantisation, no QAT)**: best for small models (≤1B) on machines with ≥32 GB of unified memory. Easiest to debug.
                • **4-bit load + LoRA + QAT (8-bit)**: the "QLoRA" recipe. Smallest memory footprint, suitable for 7B–13B models on a 16 GB laptop. The QAT-on-8-bit hook makes the adapter robust to being merged back into a 4-bit base.
                • **8-bit load + LoRA + QAT (8-bit)**: slightly higher memory than the 4-bit recipe, but the inference quantisation is gentler and the loss curve is more stable.
                • **MXFP4**: only choose this if your deployment target uses MXFP4 (or you specifically want microscaling-style 4-bit during training). On Apple Silicon the throughput difference vs. regular 4-bit is small.
                • **No load quantisation + QAT (4-bit)**: useful when you want the *base* model to stay in fp16 but the *adapter* to be 4-bit-deployable. QAT is the only way to get a 4-bit adapter that does not lose accuracy on outlier channels.
                """)

            ArticleTips(tips: [
                "QAT does not help the *base* model — it only changes how the LoRA/DoRA adapters are trained. If you are doing `train_type=full`, QAT has no effect on the merged output.",
                "If you see the loss start to oscillate or NaN after a few hundred steps, the most common cause is QAT with a `qat_bits` that is too aggressive for the current learning rate. Drop `qat_bits` to 8 and try again.",
                "Group size 32 (the MXFP4 default) is the most sensitive to outliers; 128 is the most forgiving. When in doubt, raise the group size before lowering the bit-width.",
                "Quantisation happens at load time and is irreversible for the loaded model object — the only way to change the base precision is to re-load the model. Restart the runner after changing the `load_in_*` flags.",
            ])
        }
    }
}
