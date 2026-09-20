mod common;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use bytes::Bytes;
use common::*;
use futures::stream;
use sokar::index::Index;
use sokar_client::*;
use tower::ServiceExt;
fn generated_body() -> reqwest::Body {
    // 100 MiB generated incrementally; no test-side whole-body allocation.
    reqwest::Body::wrap_stream(stream::iter((0..1600).map(|chunk| {
        let start = chunk * 65536;
        Ok::<_, std::io::Error>(Bytes::from(
            (start..start + 65536)
                .map(|n| (n % 251) as u8)
                .collect::<Vec<_>>(),
        ))
    })))
}
#[tokio::test]
async fn fs_100_mib_stream_dedup_range_etag_and_reindex() {
    let fixture = Fixture::new(true).await;
    let server = fixture.serve().await;
    let c = server
        .client
        .create_collection(&create("large"))
        .await
        .unwrap();
    let first = server
        .client
        .upload(c.id, &options("reports/first.bin"), generated_body())
        .await
        .unwrap();
    assert_eq!(first.outcome, UploadOutcome::Stored);
    assert_eq!(first.artifact.blob.size, 100 * 1024 * 1024);
    let second = server
        .client
        .upload(c.id, &options("reports/second.bin"), generated_body())
        .await
        .unwrap();
    assert_eq!(second.outcome, UploadOutcome::Deduplicated);
    assert_eq!(first.artifact.blob.digest, second.artifact.blob.digest);
    let store = fixture.state.stores.get(&c.store).unwrap();
    let files = store.op.list_with("blobs/").recursive(true).await.unwrap();
    assert_eq!(files.iter().filter(|e| e.metadata().is_file()).count(), 1);
    let disk = fixture
        .dir
        .path()
        .join("objects/test-prefix")
        .join(first.artifact.blob.key.clone());
    assert_eq!(std::fs::metadata(disk).unwrap().len(), 100 * 1024 * 1024);
    let response = server
        .client
        .content(first.artifact.id, Some("bytes=1000-1999"), None)
        .await
        .unwrap();
    assert_eq!(response.status(), 206);
    assert_eq!(
        response.headers()["content-range"],
        "bytes 1000-1999/104857600"
    );
    assert_eq!(response.headers()["accept-ranges"], "bytes");
    assert!(
        response.headers()["content-disposition"]
            .to_str()
            .unwrap()
            .starts_with("attachment; filename*=UTF-8''")
    );
    let etag = response.headers()["etag"].to_str().unwrap().to_owned();
    assert_eq!(
        response.bytes().await.unwrap().as_ref(),
        (1000..2000).map(|n| (n % 251) as u8).collect::<Vec<_>>()
    );
    let response = server
        .client
        .content(first.artifact.id, None, Some(&etag))
        .await
        .unwrap();
    assert_eq!(response.status(), 304);
    assert!(response.bytes().await.unwrap().is_empty());
    fixture.memory.clear().await;
    let restored = fixture.state.reindex(&c.store).await.unwrap();
    assert_eq!(
        (restored.collections, restored.artifacts, restored.blobs),
        (1, 2, 1)
    );
    assert_eq!(fixture.memory.collection(c.id).await.unwrap(), c);
    assert_eq!(
        fixture.memory.artifact(first.artifact.id).await.unwrap(),
        first.artifact
    );
    assert_eq!(
        fixture.memory.artifact(second.artifact.id).await.unwrap(),
        second.artifact
    );
    assert_eq!(
        fixture
            .memory
            .refcount(&c.store, first.artifact.blob.digest)
            .await,
        2
    );
    assert_eq!(fixture.state.reindex(&c.store).await.unwrap().artifacts, 2);
    server
        .client
        .delete_artifact(first.artifact.id)
        .await
        .unwrap();
    assert!(store.op.exists(&second.artifact.blob.key).await.unwrap());
    server
        .client
        .delete_artifact(second.artifact.id)
        .await
        .unwrap();
    assert!(store.op.exists(&second.artifact.blob.key).await.unwrap());
    assert!(store.op.exists(&second.artifact.key()).await.unwrap());
    let tombstone: Artifact = store.read_json(&second.artifact.key()).await.unwrap();
    assert!(tombstone.deleted_at.is_some());
}
#[tokio::test]
async fn api_multipart_idempotency_versions_paging_and_client_problems() {
    let fixture = Fixture::new(false).await;
    let server = fixture.serve().await;
    let c = server
        .client
        .create_collection(&create("api"))
        .await
        .unwrap();
    assert_eq!(server.client.collection(c.id).await.unwrap(), c);
    assert_eq!(
        server.client.collection_by_scope(&c.scope).await.unwrap(),
        c
    );
    assert_eq!(
        server
            .client
            .create_collection(&create("api"))
            .await
            .unwrap(),
        c
    );
    let mut opts = options("unicode/résumé a%.txt");
    opts.media_type = "text/plain".parse().unwrap();
    opts.idempotency_key = Some("retry-1".into());
    let a = server
        .client
        .upload_multipart(
            c.id,
            &opts,
            reqwest::multipart::Part::bytes("hello".as_bytes()),
        )
        .await
        .unwrap();
    let replay = server.client.upload(c.id, &opts, "hello").await.unwrap();
    assert_eq!(replay, a);
    let error = server
        .client
        .upload(c.id, &opts, "different")
        .await
        .unwrap_err();
    assert_eq!(error.problem().unwrap().status, 409);
    opts.idempotency_key = None;
    let b = server.client.upload(c.id, &opts, "world").await.unwrap();
    assert_eq!(b.artifact.previous, Some(a.artifact.id));
    assert_eq!(
        server
            .client
            .artifact_by_path(c.id, &opts.path)
            .await
            .unwrap(),
        b.artifact
    );
    assert_eq!(
        server.client.artifact(a.artifact.id).await.unwrap(),
        a.artifact
    );
    let page = server
        .client
        .list_artifacts(
            c.id,
            &ListArtifacts {
                prefix: Some("unicode/".into()),
                kind: Some(ArtifactKind::Document),
                limit: Some(1),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(page.artifacts, vec![a.artifact.clone()]);
    let page = server
        .client
        .list_artifacts(
            c.id,
            &ListArtifacts {
                cursor: page.next_cursor,
                limit: Some(1),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(page.artifacts, vec![b.artifact]);
    assert!(page.next_cursor.is_none());
    let response = server
        .client
        .content(a.artifact.id, Some("bytes=-2"), None)
        .await
        .unwrap();
    assert_eq!(response.bytes().await.unwrap(), "lo");
    let error = server
        .client
        .content(a.artifact.id, Some("bytes=99-100"), None)
        .await
        .unwrap_err();
    assert_eq!(error.problem().unwrap().code, "sokar::delivery");
    let bad = Client::new(&server.base, "wrong")
        .unwrap()
        .artifact(a.artifact.id)
        .await
        .unwrap_err();
    assert_eq!(bad.problem().unwrap().status, 401);
    fixture.memory.clear().await;
    fixture.state.reindex(&c.store).await.unwrap();
    assert_eq!(
        fixture.memory.artifact(a.artifact.id).await.unwrap(),
        a.artifact
    );
}
#[tokio::test]
async fn auth_fail_closed_rotation_sha256_and_admin_separation() {
    use sha2::{Digest, Sha256};
    let fixture = Fixture::new(false).await;
    let api = sokar::http::api(fixture.state.clone());
    let response = api
        .clone()
        .oneshot(Request::get("/v1/collections").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response.headers()["content-type"],
        "application/problem+json"
    );
    assert!(response.headers().contains_key("x-request-id"));
    let mut headers = axum::http::HeaderMap::new();
    headers.insert("authorization", format!("Bearer {TOKEN}").parse().unwrap());
    let mut principals = fixture.config.principals.clone();
    assert!(sokar::auth::authenticate(&principals, &headers).is_ok());
    principals.push(principals[0].clone());
    assert!(sokar::auth::authenticate(&principals, &headers).is_err());
    principals.pop();
    let file = fixture.dir.path().join("sha256");
    std::fs::write(&file, hex::encode(Sha256::digest(TOKEN))).unwrap();
    principals[0].token_file = None;
    principals[0].token_sha256_file = Some(file);
    assert!(sokar::auth::authenticate(&principals, &headers).is_ok());
    headers.insert(
        "authorization",
        format!("Bearer {}", hex::encode(Sha256::digest(TOKEN)))
            .parse()
            .unwrap(),
    );
    assert!(sokar::auth::authenticate(&principals, &headers).is_err());
    std::fs::write(
        fixture.config.principals[0].token_file.as_ref().unwrap(),
        "rotated-token-at-least-24-characters",
    )
    .unwrap();
    headers.insert(
        "authorization",
        "Bearer rotated-token-at-least-24-characters"
            .parse()
            .unwrap(),
    );
    assert!(sokar::auth::authenticate(&fixture.config.principals, &headers).is_ok());
    let admin = sokar::http::admin(fixture.state.clone());
    for route in ["/healthz", "/readyz", "/metrics"] {
        let response = admin
            .clone()
            .oneshot(Request::get(route).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
    }
    let response = api
        .oneshot(
            Request::get("/metrics")
                .header(
                    "authorization",
                    "Bearer rotated-token-at-least-24-characters",
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 404);
}
#[tokio::test]
async fn streaming_limits_problem_ids_and_inline_allowlist() {
    use std::sync::Arc;
    let fixture = Fixture::new(false).await;
    let mut config = (*fixture.config).clone();
    config.limits.max_body_bytes = 8;
    config.delivery.inline_media_types = vec!["text/plain".parse().unwrap()];
    let state = Arc::new(sokar::Sokar::new(
        Arc::new(config),
        fixture.state.stores.clone(),
        fixture.memory.clone(),
    ));
    let c = fixture.collection().await;
    let app = sokar::http::api(state);
    let uri = format!("/v1/collections/{}/artifacts", c.id);
    for body in [
        Body::from("123456789"),
        Body::from_stream(stream::iter([
            Ok::<_, std::io::Error>(Bytes::from_static(b"1234")),
            Ok(Bytes::from_static(b"56789")),
        ])),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::post(&uri)
                    .header("authorization", format!("Bearer {TOKEN}"))
                    .header("content-type", "text/plain")
                    .header("sokar-kind", "document")
                    .header("sokar-path", "limit.txt")
                    .body(body)
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 413);
        let id = response.headers()["x-request-id"]
            .to_str()
            .unwrap()
            .to_owned();
        let problem: Problem = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), 65536)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(problem.code, "sokar::body_limit");
        assert_eq!(problem.request_id, id);
    }
    let response = app
        .clone()
        .oneshot(
            Request::post(&uri)
                .header("authorization", format!("Bearer {TOKEN}"))
                .header("content-type", "text/plain")
                .header("sokar-kind", "document")
                .header("sokar-path", "inline.txt")
                .body(Body::from("safe"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 201);
    let uploaded: UploadResponse = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap(),
    )
    .unwrap();
    let response = app
        .clone()
        .oneshot(
            Request::get(format!("/v1/artifacts/{}/content", uploaded.artifact.id))
                .header("authorization", format!("Bearer {TOKEN}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.headers()["content-disposition"], "inline");
    for route in ["/v1/artifacts/not-a-uuid", "/v1/not-found"] {
        let response = app
            .clone()
            .oneshot(
                Request::get(route)
                    .header("authorization", format!("Bearer {TOKEN}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.headers()["content-type"],
            "application/problem+json"
        );
    }
}
#[tokio::test]
async fn multipart_file_before_path_and_duplicate_fields() {
    let fixture = Fixture::new(false).await;
    let server = fixture.serve().await;
    let c = fixture.collection().await;
    let route = format!("{}/v1/collections/{}/artifacts", server.base, c.id);
    let client = reqwest::Client::new();
    let form = reqwest::multipart::Form::new()
        .part("file", reqwest::multipart::Part::bytes("first".as_bytes()))
        .text("path", "nested/file-first.bin");
    let response = client
        .post(&route)
        .bearer_auth(TOKEN)
        .header("sokar-kind", "document")
        .multipart(form)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 201);
    let a: UploadResponse = response.json().await.unwrap();
    assert_eq!(
        server
            .client
            .content(a.artifact.id, None, None)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap(),
        "first"
    );
    let form = reqwest::multipart::Form::new()
        .text("path", "a")
        .text("path", "b")
        .part("file", reqwest::multipart::Part::bytes("x".as_bytes()));
    let response = client
        .post(&route)
        .bearer_auth(TOKEN)
        .header("sokar-kind", "document")
        .multipart(form)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn sokard_process_100_mib_roundtrip_and_restart_reindex() {
    let fixture = Fixture::new(true).await;
    let daemon = fixture.serve_binary().await;
    let c = daemon
        .client
        .create_collection(&create("binary"))
        .await
        .unwrap();
    let a = daemon
        .client
        .upload(c.id, &options("first"), generated_body())
        .await
        .unwrap();
    let b = daemon
        .client
        .upload(c.id, &options("second"), generated_body())
        .await
        .unwrap();
    assert_eq!(a.outcome, UploadOutcome::Stored);
    assert_eq!(b.outcome, UploadOutcome::Deduplicated);
    let response = daemon
        .client
        .content(a.artifact.id, Some("bytes=1000-1999"), None)
        .await
        .unwrap();
    assert_eq!(response.status(), 206);
    assert_eq!(
        response.bytes().await.unwrap().as_ref(),
        (1000..2000).map(|n| (n % 251) as u8).collect::<Vec<_>>()
    );
    drop(daemon);
    // Omitted DB uses a fresh ephemeral index on every process start.
    let daemon = fixture.serve_binary().await;
    assert_eq!(daemon.client.collection(c.id).await.unwrap(), c);
    assert_eq!(
        daemon.client.artifact(a.artifact.id).await.unwrap(),
        a.artifact
    );
    assert_eq!(
        daemon.client.artifact(b.artifact.id).await.unwrap(),
        b.artifact
    );
}

#[tokio::test]
async fn empty_artifact_and_live_reindex_preserves_idempotency() {
    let fixture = Fixture::new(false).await;
    let server = fixture.serve().await;
    let c = fixture.collection().await;
    let mut opts = options("empty");
    opts.idempotency_key = Some("empty-retry".into());
    let a = server.client.upload(c.id, &opts, "").await.unwrap();
    assert_eq!(a.artifact.blob.size, 0);
    let response = server
        .client
        .content(a.artifact.id, None, None)
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert!(response.bytes().await.unwrap().is_empty());
    assert_eq!(
        server
            .client
            .content(a.artifact.id, Some("bytes=0-"), None)
            .await
            .unwrap_err()
            .problem()
            .unwrap()
            .status,
        416
    );
    fixture.state.reindex(&c.store).await.unwrap();
    assert_eq!(server.client.upload(c.id, &opts, "").await.unwrap(), a);
}

#[tokio::test]
async fn collection_create_or_get_and_incompatible_store() {
    let fixture = Fixture::new(false).await;
    let server = fixture.serve().await;
    let http = reqwest::Client::new();
    let url = format!("{}/v1/collections", server.base);
    let mut input = create("create-or-get");
    let first = http
        .post(&url)
        .bearer_auth(TOKEN)
        .json(&input)
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), 201);
    let collection: Collection = first.json().await.unwrap();
    for store in [None, Some(collection.store.clone())] {
        input.store = store;
        let response = http
            .post(&url)
            .bearer_auth(TOKEN)
            .json(&input)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(response.json::<Collection>().await.unwrap(), collection);
    }
    input.store = Some("different".parse().unwrap());
    let response = http
        .post(&url)
        .bearer_auth(TOKEN)
        .json(&input)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 409);
    assert_eq!(
        response.json::<Problem>().await.unwrap().code,
        "sokar::scope_exists"
    );
}

#[tokio::test]
async fn credential_format_matches_session_services() {
    let fixture = Fixture::new(false).await;
    let config = &fixture.config.principals[0];
    let file = config.token_file.as_ref().unwrap();
    for token in [
        "short",
        "a-long-token-with whitespace",
        "a-long-token-with\twhitespace",
    ] {
        std::fs::write(file, token).unwrap();
        assert!(config.credential().is_err());
    }
    std::fs::write(file, format!("  {TOKEN}\n")).unwrap();
    assert_eq!(config.credential().unwrap().0, TOKEN);
}
