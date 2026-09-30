//! Model-family guidance layer for the generated system prompt (ADR 122).
//!
//! The psyche-derived identity, the operator section and the machine-owned
//! tool/conventions sections (see
//! [`assemble_system_prompt`](super::model_manager::assemble_system_prompt))
//! describe *who* the akh is and *what* it may call. They say nothing about
//! *how* to drive a tool loop: which arguments to send, when a failed call is
//! evidence, when to stop, or when a write must be announced first. The
//! GLM-5.3 and Kimi K3 runs examined in
//! `docs/ai/research/2026-09-30-glm-kimi-run-evaluation.md` fail in exactly
//! those places (next to harness faults that no prompt fixes).
//!
//! This module adds one optional, machine-owned section — **Execution
//! Discipline** — composed after the operating conventions, plus a short
//! family addendum for the model families with observed failure signatures.
//! With the layer on, `assemble_system_prompt` also reduces the operating
//! conventions to the lines whose tools are live in the turn.
//!
//! The layer is **opt-in per akh** (`prompt_guidance` in `akh.toml`,
//! default [`PromptGuidance::Off`]), so an akh that does not set it keeps its
//! system prompt (KV-cache prefix stability, WI-357; and no un-measured
//! behavior change on the live Romans or the dojo baseline).
//!
//! Family detection reads the *model identifier* (e.g. `glm-5.3:cloud`), the
//! same way [`detect_prompt_mode`](super::model_manager::detect_prompt_mode)
//! recognizes Gemma. It never inspects conversation text, so the project rule
//! against hard-coded natural-language matching does not apply; the guidance
//! text itself is structural (tool names, schema, receipts, rounds), not a
//! vocabulary list.

use serde::{Deserialize, Serialize};

/// Model family a guidance addendum is tailored to, detected from the model id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    /// Zhipu GLM (e.g. `glm-5.3`, `glm-5.3:cloud`).
    Glm,
    /// Moonshot Kimi (e.g. `kimi-k3`, `kimi-k2.7-code`).
    Kimi,
    /// Any other model: gets the common discipline block only.
    Other,
}

impl ModelFamily {
    /// Detect the family from a model identifier (case-insensitive).
    ///
    /// Matching is on the vendor prefix of the id, not on any text a user or
    /// model produced. An empty or unknown id is [`ModelFamily::Other`].
    pub fn detect(model: &str) -> Self {
        let id = model.trim().to_ascii_lowercase();
        // Strip a registry/namespace prefix such as `library/` or `zai-org/`.
        let base = id.rsplit('/').next().unwrap_or(&id);
        if base.starts_with("glm") {
            Self::Glm
        } else if base.starts_with("kimi") {
            Self::Kimi
        } else {
            Self::Other
        }
    }

    /// Stable lowercase label (logs, telemetry, docs).
    pub fn label(self) -> &'static str {
        match self {
            Self::Glm => "glm",
            Self::Kimi => "kimi",
            Self::Other => "other",
        }
    }
}

/// Per-akh opt-in for the guidance layer (`prompt_guidance` in `akh.toml`).
///
/// ```toml
/// prompt_guidance = "auto"   # off | auto | generic | glm | kimi
/// ```
///
/// Must sit above the first `[table]` header in `akh.toml` (top-level scalars
/// placed below a table are silently nested into it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptGuidance {
    /// No guidance section; the prompt is byte-identical to the pre-ADR-122
    /// assembly. The default.
    #[default]
    Off,
    /// Common discipline block plus the addendum for the family detected from
    /// the akh's pinned model id.
    Auto,
    /// Common discipline block only, whatever the model.
    Generic,
    /// Common block plus the GLM addendum, regardless of the model id.
    Glm,
    /// Common block plus the Kimi addendum, regardless of the model id.
    Kimi,
}

impl PromptGuidance {
    /// Whether the guidance layer is enabled at all.
    pub fn is_enabled(self) -> bool {
        self != Self::Off
    }

    /// `serde(skip_serializing_if)` helper: an off setting is not written back
    /// to `akh.toml`, so saving an untouched config stays byte-stable.
    pub fn is_off(&self) -> bool {
        *self == Self::Off
    }

    /// Resolve the setting against the akh's model id.
    ///
    /// Returns `None` when guidance is off, otherwise the family whose
    /// addendum applies ([`ModelFamily::Other`] = common block only).
    pub fn resolve(self, model: &str) -> Option<ModelFamily> {
        match self {
            Self::Off => None,
            Self::Auto => Some(ModelFamily::detect(model)),
            Self::Generic => Some(ModelFamily::Other),
            Self::Glm => Some(ModelFamily::Glm),
            Self::Kimi => Some(ModelFamily::Kimi),
        }
    }
}

/// Heading of the guidance section; tests and the dojo A/B locate it by this.
pub const GUIDANCE_HEADING: &str = "## Execution Discipline";

/// Common block: applies to every family once guidance is enabled.
///
/// Every rule answers a failure mode with transcript evidence in
/// `docs/ai/research/2026-09-30-glm-kimi-run-evaluation.md` (mode ids in
/// brackets here, kept out of the prompt text itself):
/// arguments [R-B, R-C, D-A, D-B], failed calls [R-C, D-A, D-E], evidence
/// [R-D, D-C], state changes [R-E], finishing [A-F6, A-F8, R-F, D-H].
const COMMON_DISCIPLINE: &str = "\
## Execution Discipline\n\
\n\
These rules govern how you drive tools. They take precedence over habits from other harnesses.\n\
\n\
### Tool arguments\n\
- Send only the parameters the tool's schema declares, with the declared types. Omit optional parameters (page size, limit, offset, cursor, sort, filters, format, verbosity) unless the task needs a value other than the default.\n\
- Send objects as JSON objects, arrays as JSON arrays and numbers as numbers, never as a string that contains JSON. Where the schema lists allowed values, use one of them exactly.\n\
- Copy identifiers (ids, keys, paths, names, commit hashes) exactly from the task, from an earlier successful result or from your notes. Never construct or guess one. Keep using the same key for the same thing across turns.\n\
- Before calling one tool many times in the same round, make one call and check that it succeeds.\n\
- If a call fails because of its arguments, read the error, change the arguments once, and retry. Never resend a call or a value that already failed, in this turn or in an earlier one.\n\
\n\
### Failed calls\n\
- A failed or errored tool result is not evidence. It shows only that the call failed.\n\
- After two failed calls to the same tool, stop using that tool for this item. Use another tool that can answer, or report the blocker.\n\
- If three rounds pass without a successful result that moves the task forward, stop and answer with what you have.\n\
\n\
### Evidence and claims\n\
- State a fact as observed only if a successful tool result in this conversation shows it. Name the tool that produced it.\n\
- Cite only successful results. Mention a failed read in prose; never cite it as a source or receipt.\n\
- Never claim that you ran, tested, wrote, sent, posted, merged, escalated or verified something unless a successful tool result shows it. An action that failed is reported as failed, in the first line of the answer, not only further down.\n\
- Do not report how many tool calls or rounds you used; the runtime records that.\n\
\n\
### Actions that change state\n\
- Before a call that creates, edits, deletes, sends, posts, merges, unparks, moves or updates anything, write one line saying what you will change and why. Then make the call.\n\
- Take only state-changing actions the task asks for or clearly requires. When unsure, do the read-only part and propose the change instead.\n\
- Read-only calls need no announcement.\n\
\n\
### Finishing\n\
- When the task names a deliverable (a verdict, a submission, a document), produce it before any optional step such as notes or memory writes.\n\
- Stop calling tools as soon as you can deliver. When the round or time budget is running out, deliver what you have and name the gap instead of exploring further.\n\
- Every turn ends with a written answer: the result, or what blocked you, what you tried and the next step. A turn never ends on a tool call, a plan or empty text.\n\
- If the task prescribes an output format (a verdict line, a JSON object, headings), the final answer is exactly that format, with no code fence and no text around it.\n\
- Keep your reasoning and self-narration out of the final answer.\n\
- Reply in the language the request was written in.";

/// GLM addendum (GLM-5.3 on the Romans, the dojo teacher and the Anima
/// `maat` / `imhotep` runtimes). Evidence: stringified nested arguments and
/// dangling-schema guesses [R-B], outcome headlines over failed actions and
/// wrong self-counts [R-D], leaked monologue [R-F], 2k–15k words of thinking
/// per dojo episode [D-F], parallel fan-out before a first success [D-A].
const GLM_ADDENDUM: &str = "\
### Notes for this model\n\
- Nested arguments are the most common failure: if a parameter is an object, the value you send starts with an opening brace, not a quote.\n\
- Decide after the first successful read that answers the question. Do not re-plan the whole task after each result.\n\
- The first line of the answer states the outcome as it is: done, failed, blocked or proposed.";

/// Kimi addendum (Kimi K3 on the Anima `sekhmet` / `tecton` runtimes).
/// Evidence: memory write before the verdict [A-F6], reconnaissance that used
/// the whole budget [A-F8], a whole merge diff read into context [A-F5],
/// multi-minute reasoning per iteration against a turn deadline [A-F2].
const KIMI_ADDENDUM: &str = "\
### Notes for this model\n\
- Make the deliverable call as soon as the evidence supports it. Notes, memory and follow-ups come after it.\n\
- Read summaries before content: file lists, counts and the first lines of output before whole files or whole diffs. For a merge commit, compare against its first parent only.\n\
- Keep each reasoning step short. The turn has a deadline, and long deliberation spends it.";

/// Render the guidance section for a resolved family.
pub fn guidance_section(family: ModelFamily) -> String {
    match family {
        ModelFamily::Glm => format!("{COMMON_DISCIPLINE}\n\n{GLM_ADDENDUM}"),
        ModelFamily::Kimi => format!("{COMMON_DISCIPLINE}\n\n{KIMI_ADDENDUM}"),
        ModelFamily::Other => COMMON_DISCIPLINE.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_families_from_model_ids() {
        assert_eq!(ModelFamily::detect("glm-5.3"), ModelFamily::Glm);
        assert_eq!(ModelFamily::detect("GLM-5.3:cloud"), ModelFamily::Glm);
        assert_eq!(ModelFamily::detect("zai-org/glm-5.3"), ModelFamily::Glm);
        assert_eq!(ModelFamily::detect("kimi-k3"), ModelFamily::Kimi);
        assert_eq!(
            ModelFamily::detect("kimi-k2.7-code:cloud"),
            ModelFamily::Kimi
        );
        assert_eq!(ModelFamily::detect("moonshotai/Kimi-K3"), ModelFamily::Kimi);
        assert_eq!(ModelFamily::detect("gemma4:e4b"), ModelFamily::Other);
        assert_eq!(ModelFamily::detect(""), ModelFamily::Other);
    }

    #[test]
    fn detection_is_prefix_anchored() {
        // A model whose id merely *contains* a family token elsewhere is not
        // that family (avoids e.g. a fine-tune named after its teacher).
        assert_eq!(ModelFamily::detect("lilith-from-glm"), ModelFamily::Other);
        assert_eq!(ModelFamily::detect("distilled-kimi"), ModelFamily::Other);
    }

    #[test]
    fn off_resolves_to_none_and_is_default() {
        assert_eq!(PromptGuidance::default(), PromptGuidance::Off);
        assert!(!PromptGuidance::Off.is_enabled());
        assert_eq!(PromptGuidance::Off.resolve("glm-5.3"), None);
    }

    #[test]
    fn auto_follows_model_id_and_forced_ignores_it() {
        assert_eq!(
            PromptGuidance::Auto.resolve("glm-5.3"),
            Some(ModelFamily::Glm)
        );
        assert_eq!(
            PromptGuidance::Auto.resolve("kimi-k3"),
            Some(ModelFamily::Kimi)
        );
        assert_eq!(
            PromptGuidance::Auto.resolve("qwen3.8"),
            Some(ModelFamily::Other)
        );
        assert_eq!(
            PromptGuidance::Generic.resolve("glm-5.3"),
            Some(ModelFamily::Other)
        );
        assert_eq!(
            PromptGuidance::Kimi.resolve("glm-5.3"),
            Some(ModelFamily::Kimi)
        );
        assert_eq!(PromptGuidance::Glm.resolve(""), Some(ModelFamily::Glm));
    }

    #[test]
    fn serde_round_trips_snake_case_labels() {
        #[derive(Deserialize, Serialize)]
        struct Wrap {
            prompt_guidance: PromptGuidance,
        }
        for (text, want) in [
            ("off", PromptGuidance::Off),
            ("auto", PromptGuidance::Auto),
            ("generic", PromptGuidance::Generic),
            ("glm", PromptGuidance::Glm),
            ("kimi", PromptGuidance::Kimi),
        ] {
            let parsed: Wrap = toml::from_str(&format!("prompt_guidance = \"{text}\"")).unwrap();
            assert_eq!(parsed.prompt_guidance, want);
            let back = toml::to_string(&parsed).unwrap();
            assert!(back.contains(text), "{back}");
        }
        assert!(toml::from_str::<Wrap>("prompt_guidance = \"loud\"").is_err());
    }

    #[test]
    fn every_family_section_carries_the_common_block() {
        for family in [ModelFamily::Glm, ModelFamily::Kimi, ModelFamily::Other] {
            let section = guidance_section(family);
            assert!(section.starts_with(GUIDANCE_HEADING), "{family:?}");
            for sub in [
                "### Tool arguments",
                "### Failed calls",
                "### Evidence and claims",
                "### Actions that change state",
                "### Finishing",
            ] {
                assert!(section.contains(sub), "{family:?} lacks {sub}");
            }
        }
        assert!(!guidance_section(ModelFamily::Other).contains("### Notes for this model"));
        assert!(guidance_section(ModelFamily::Glm).contains("### Notes for this model"));
        assert!(guidance_section(ModelFamily::Kimi).contains("### Notes for this model"));
    }

    #[test]
    fn guidance_names_no_tools() {
        // The section is tool-agnostic: it must not name registry tools, so it
        // cannot drift from the live catalog (cf. the operating-conventions
        // `operating_conventions_only_name_real_tools` invariant).
        for family in [ModelFamily::Glm, ModelFamily::Kimi, ModelFamily::Other] {
            assert!(!guidance_section(family).contains('`'), "{family:?}");
        }
    }
}
