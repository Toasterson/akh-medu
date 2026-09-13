use akh_session_types::questions::{QuestionAnswers, QuestionForm};
use serde_json::{Value, json};

#[test]
fn published_form_round_trips_and_validates_the_shared_client_answers() {
    let value: Value = serde_json::from_str(include_str!("fixtures/questions.json")).unwrap();
    let form: QuestionForm = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(serde_json::to_value(&form).unwrap(), value);
    form.validate().unwrap();
    let answer: QuestionAnswers = serde_json::from_value(
        json!({"cancelled":false,"answers":{"features":["Search","Export"],"note":["Keep local"]}}),
    )
    .unwrap();
    form.validate_answers(&answer).unwrap();
}
