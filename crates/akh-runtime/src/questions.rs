//! Translate only at the provider boundary. Clients never need provider keys.
use akh_session_types::questions::{Question, QuestionAnswers, QuestionForm, QuestionOption};
use serde_json::{Value, json};
use std::collections::HashSet;

#[derive(Clone, Copy)]
pub enum Provider {
    Codex,
    Claude,
}
fn field(value: &Value, key: &str) -> Result<String, String> {
    value[key]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("question is missing {key}"))
}
pub fn ask(
    host: &mut dyn crate::Host,
    id: &str,
    raw: &Value,
    provider: Provider,
) -> Result<Value, String> {
    let form = normalize(raw, provider)?;
    let answer = host.question(id, &serde_json::to_value(&form).map_err(|e| e.to_string())?)?;
    let answer: QuestionAnswers =
        serde_json::from_value(answer).map_err(|e| format!("invalid question answer: {e}"))?;
    encode(&form, &answer, provider)
}
fn normalize(raw: &Value, provider: Provider) -> Result<QuestionForm, String> {
    let questions = raw["questions"]
        .as_array()
        .ok_or("provider question requires an array")?;
    let mut form = QuestionForm {
        format: QuestionForm::FORMAT.into(),
        questions: vec![],
    };
    let mut texts = HashSet::new();
    for (index, q) in questions.iter().enumerate() {
        let text = field(q, "question")?;
        if matches!(provider, Provider::Claude) && !texts.insert(text.clone()) {
            return Err("Claude question texts must be unique".into());
        }
        let options = match &q["options"] {
            Value::Null => vec![],
            Value::Array(options) => options
                .iter()
                .map(|o| {
                    Ok(QuestionOption {
                        label: field(o, "label")?,
                        description: field(o, "description")?,
                        preview: o.get("preview").map(|_| field(o, "preview")).transpose()?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
            _ => return Err("invalid provider question options".into()),
        };
        let (key, multiple, allow_text, secret) = match provider {
            Provider::Codex => (
                field(q, "id")?,
                false,
                q["isOther"] == true || options.is_empty(),
                q["isSecret"] == true,
            ),
            Provider::Claude => (
                format!("question-{index}"),
                q["multiSelect"] == true,
                true,
                false,
            ),
        };
        form.questions.push(Question {
            id: key,
            header: field(q, "header")?,
            question: text,
            options,
            multiple,
            allow_text,
            secret,
        });
    }
    form.validate()?;
    Ok(form)
}
fn encode(
    form: &QuestionForm,
    answer: &QuestionAnswers,
    provider: Provider,
) -> Result<Value, String> {
    form.validate_answers(answer)?;
    let mut mapped = serde_json::Map::new();
    if !answer.cancelled {
        for q in &form.questions {
            let values = &answer.answers[&q.id];
            match provider {
                Provider::Codex => {
                    mapped.insert(q.id.clone(), json!({"answers":values}));
                }
                Provider::Claude => {
                    mapped.insert(
                        q.question.clone(),
                        if q.multiple {
                            json!(values)
                        } else {
                            json!(values[0])
                        },
                    );
                }
            }
        }
    }
    Ok(match provider {
        Provider::Codex => json!({"answers":mapped}),
        Provider::Claude => json!({"answers":mapped,"cancelled":answer.cancelled}),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_ids_choices_and_multiselect_are_preserved_without_permission_semantics() {
        let codex=normalize(&json!({"questions":[{"id":"db","header":"DB","question":"Which database?","options":[{"label":"Postgres","description":"Shared"}],"isOther":true}]}),Provider::Codex).unwrap();
        let answer =
            serde_json::from_value(json!({"cancelled":false,"answers":{"db":["SQLite"]}})).unwrap();
        assert_eq!(
            encode(&codex, &answer, Provider::Codex).unwrap(),
            json!({"answers":{"db":{"answers":["SQLite"]}}})
        );
        let claude=normalize(&json!({"questions":[{"header":"Features","question":"Choose features","multiSelect":true,"options":[{"label":"A","description":"Alpha"},{"label":"B","description":"Beta"}]}]}),Provider::Claude).unwrap();
        let answer =
            serde_json::from_value(json!({"cancelled":false,"answers":{"question-0":["A","B"]}}))
                .unwrap();
        assert_eq!(
            encode(&claude, &answer, Provider::Claude).unwrap(),
            json!({"answers":{"Choose features":["A","B"]},"cancelled":false})
        );
        let cancelled = serde_json::from_value(json!({"cancelled":true,"answers":{}})).unwrap();
        assert_eq!(
            encode(&codex, &cancelled, Provider::Codex).unwrap(),
            json!({"answers":{}})
        );
        assert_eq!(
            encode(&claude, &cancelled, Provider::Claude).unwrap()["cancelled"],
            true
        );
        assert!(serde_json::from_value::<QuestionAnswers>(json!({"approved":true})).is_err());
    }
    #[test]
    fn malformed_extra_and_secret_answers_do_not_reach_the_native_provider() {
        let form=normalize(&json!({"questions":[{"id":"q","header":"Q","question":"Choose","options":[{"label":"A","description":"A"}]}]}),Provider::Codex).unwrap();
        for value in [
            json!({"cancelled":false,"answers":{"q":["B"]}}),
            json!({"cancelled":false,"answers":{"wrong":["A"]}}),
            json!({"cancelled":false,"answers":{"q":["A","A"]}}),
            json!({"cancelled":true,"answers":{"q":["A"]}}),
        ] {
            assert!(
                encode(
                    &form,
                    &serde_json::from_value(value).unwrap(),
                    Provider::Codex
                )
                .is_err()
            );
        }
        let mut secret = form.clone();
        secret.questions[0].secret = true;
        assert!(
            encode(
                &secret,
                &serde_json::from_value(json!({"cancelled":false,"answers":{"q":["A"]}})).unwrap(),
                Provider::Codex
            )
            .is_err()
        );
        assert!(normalize(&json!({"questions":[]}), Provider::Codex).is_err());
        let mut duplicate = form.clone();
        duplicate.questions.push(form.questions[0].clone());
        assert!(duplicate.validate().is_err());
    }
}
