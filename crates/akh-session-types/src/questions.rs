//! Provider-independent question forms carried inside PendingInput.prompt.
//! Answers are ordinary journaled user data, never tool permission or credentials.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct QuestionForm {
    pub format: String,
    pub questions: Vec<Question>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Question {
    pub id: String,
    pub header: String,
    pub question: String,
    pub options: Vec<QuestionOption>,
    pub multiple: bool,
    pub allow_text: bool,
    pub secret: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct QuestionOption {
    pub label: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preview: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct QuestionAnswers {
    pub cancelled: bool,
    pub answers: BTreeMap<String, Vec<String>>,
}
impl QuestionForm {
    pub const FORMAT: &str = "akh.questions.v1";
    pub fn validate(&self) -> Result<(), String> {
        if self.format != Self::FORMAT
            || self.questions.is_empty()
            || self.questions.len() > 16
            || serde_json::to_vec(self).map_err(|e| e.to_string())?.len() > 64 * 1024
        {
            return Err("unsupported or oversized question form".into());
        }
        let mut ids = HashSet::new();
        for q in &self.questions {
            if q.id.is_empty()
                || q.id.len() > 256
                || q.id.chars().any(char::is_control)
                || !ids.insert(&q.id)
                || q.question.trim().is_empty()
                || q.question.len() > 8192
                || q.header.len() > 256
                || q.options.len() > 32
                || (q.options.is_empty() && !q.allow_text)
            {
                return Err(
                    "question requires unique identity, prompt and usable answer choices".into(),
                );
            }
            let mut labels = HashSet::new();
            for option in &q.options {
                if option.label.trim().is_empty()
                    || option.label.len() > 2048
                    || option.description.len() > 8192
                    || option.preview.as_ref().is_some_and(|p| p.len() > 16384)
                    || !labels.insert(&option.label)
                {
                    return Err(
                        "question choices require distinct bounded labels and descriptions".into(),
                    );
                }
            }
        }
        Ok(())
    }
    pub fn validate_answers(&self, value: &QuestionAnswers) -> Result<(), String> {
        self.validate()?;
        if value.cancelled {
            return if value.answers.is_empty() {
                Ok(())
            } else {
                Err("skipped questions must not include answers".into())
            };
        }
        if value.answers.len() != self.questions.len() || self.questions.iter().any(|q| q.secret) {
            return Err(
                "every non-secret question needs an answer; secrets need a private input channel"
                    .into(),
            );
        }
        for q in &self.questions {
            let answers = value
                .answers
                .get(&q.id)
                .ok_or("answer identity differs from the offered question")?;
            let mut unique = HashSet::new();
            if answers.is_empty()
                || answers.len() > 32
                || (!q.multiple && answers.len() != 1)
                || answers.iter().any(|a| {
                    a.trim().is_empty()
                        || a.len() > 8192
                        || !unique.insert(a)
                        || (!q.allow_text && !q.options.iter().any(|o| o.label == *a))
                })
            {
                return Err(
                    "answer is outside the question's offered choices or cardinality".into(),
                );
            }
        }
        if serde_json::to_vec(value).map_err(|e| e.to_string())?.len() > 64 * 1024 {
            return Err("question answers exceed 64 KiB".into());
        }
        Ok(())
    }
}
