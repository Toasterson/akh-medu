//! Portable bucket manifests and API records. Unknown fields are deliberately tolerated.
use crate::Error;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fmt, str::FromStr};
use time::OffsetDateTime;
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

macro_rules! uuid_id {
    ($name:ident) => {
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(try_from = "Uuid", into = "Uuid")]
        pub struct $name(Uuid);
        impl $name {
            pub fn new() -> Self {
                Self(Uuid::now_v7())
            }
            pub fn as_uuid(self) -> Uuid {
                self.0
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
        impl TryFrom<Uuid> for $name {
            type Error = Error;
            fn try_from(value: Uuid) -> Result<Self, Error> {
                if value.get_version_num() != 7 {
                    return Err(Error::Invalid("expected a UUIDv7".into()));
                }
                Ok(Self(value))
            }
        }
        impl From<$name> for Uuid {
            fn from(value: $name) -> Uuid {
                value.0
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
        impl FromStr for $name {
            type Err = Error;
            fn from_str(value: &str) -> Result<Self, Error> {
                Self::try_from(
                    Uuid::parse_str(value).map_err(|_| Error::Invalid("invalid UUID".into()))?,
                )
            }
        }
    };
}
uuid_id!(CollectionId);
uuid_id!(ArtifactId);

macro_rules! string_type {
    ($name:ident, $validate:expr) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(try_from = "String", into = "String")]
        pub struct $name(Box<str>);
        impl $name {
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
        impl TryFrom<String> for $name {
            type Error = Error;
            fn try_from(value: String) -> Result<Self, Error> {
                ($validate)(&value)?;
                Ok(Self(value.into_boxed_str()))
            }
        }
        impl From<$name> for String {
            fn from(value: $name) -> String {
                value.0.into_string()
            }
        }
        impl FromStr for $name {
            type Err = Error;
            fn from_str(s: &str) -> Result<Self, Error> {
                Self::try_from(s.to_owned())
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}
string_type!(StoreName, |v: &str| {
    if v.is_empty()
        || v.len() > 63
        || !v
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
    {
        Err(Error::Invalid(
            "store name must contain 1..63 ASCII letters, digits, underscores or hyphens".into(),
        ))
    } else {
        Ok(())
    }
});
string_type!(ArtifactPath, |v: &str| {
    if v.is_empty()
        || v.len() > 4096
        || v.contains('\\')
        || v.chars().any(char::is_control)
        || v.split('/').any(|c| c.is_empty() || c == "." || c == "..")
        || !v.nfc().eq(v.chars())
    {
        Err(Error::Invalid(
            "artifact path must be relative NFC with no empty, dot or parent components".into(),
        ))
    } else {
        Ok(())
    }
});
string_type!(MediaType, |v: &str| {
    let parsed = v
        .parse::<mime::Mime>()
        .map_err(|_| Error::Invalid("invalid media type".into()))?;
    if parsed.type_() == mime::STAR
        || parsed.subtype() == mime::STAR
        || v.chars().any(char::is_control)
    {
        Err(Error::Invalid("media type must be concrete".into()))
    } else {
        Ok(())
    }
});
impl MediaType {
    pub fn essence(&self) -> String {
        self.0
            .parse::<mime::Mime>()
            .expect("validated MIME")
            .essence_str()
            .to_owned()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct BlobDigest(pub [u8; 32]);
impl BlobDigest {
    pub fn key(&self) -> String {
        let h = self.to_string();
        format!("blobs/sha256/{}/{}/{h}", &h[..2], &h[2..4])
    }
}
impl fmt::Display for BlobDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex::encode(self.0))
    }
}
impl From<BlobDigest> for String {
    fn from(value: BlobDigest) -> String {
        value.to_string()
    }
}
impl TryFrom<String> for BlobDigest {
    type Error = Error;
    fn try_from(value: String) -> Result<Self, Error> {
        let mut bytes = [0; 32];
        hex::decode_to_slice(value, &mut bytes)
            .map_err(|_| Error::Invalid("expected a 32-byte hex digest".into()))?;
        Ok(Self(bytes))
    }
}
impl FromStr for BlobDigest {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        Self::try_from(s.to_owned())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum CollectionScope {
    AnimaWorkItem {
        project: Uuid,
        work_item: Uuid,
    },
    AnimaSession {
        work_item: Option<Uuid>,
        session: Uuid,
    },
    AkhSession {
        akh: String,
        session: Uuid,
    },
    AkhWorkspace {
        akh: String,
        workspace: String,
    },
    Named {
        name: String,
    },
}
impl CollectionScope {
    pub fn key(&self) -> String {
        serde_json::to_string(self).expect("scope serialization is infallible")
    }
}
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RetentionPolicy {
    #[default]
    Forever,
    Ttl {
        seconds: u64,
    },
    UntilScopeClosed {
        grace_seconds: u64,
    },
    KeepLast {
        count: u32,
    },
}
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Quota {
    pub max_bytes: Option<u64>,
    pub max_artifacts: Option<u64>,
}
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum ArtifactKind {
    Document,
    Image,
    Audio,
    Video,
    Archive,
    Transcript,
    DispatchPrompt,
    Site,
    SurfaceLog,
    #[default]
    Other,
}
impl fmt::Display for ArtifactKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            serde_json::to_value(self)
                .expect("kind")
                .as_str()
                .expect("string"),
        )
    }
}
impl FromStr for ArtifactKind {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        serde_json::from_value(serde_json::Value::String(s.into()))
            .map_err(|_| Error::Invalid("invalid artifact kind".into()))
    }
}
pub type Labels = BTreeMap<String, String>;
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobRef {
    pub digest: BlobDigest,
    pub size: u64,
    pub media_type: MediaType,
    pub key: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Collection {
    pub schema: String,
    pub id: CollectionId,
    pub scope: CollectionScope,
    pub owner: String,
    pub store: StoreName,
    pub retention: RetentionPolicy,
    pub quota: Quota,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
}
impl Collection {
    pub const SCHEMA: &'static str = "sokar.collection.v1";
    pub fn validate(&self) -> Result<(), Error> {
        check_schema(&self.schema, Self::SCHEMA)
    }
    pub fn key(&self) -> String {
        format!("collections/{}/manifest.json", self.id)
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Artifact {
    pub schema: String,
    pub id: ArtifactId,
    pub collection: CollectionId,
    pub path: ArtifactPath,
    pub kind: ArtifactKind,
    pub blob: BlobRef,
    pub labels: Labels,
    pub created_by: String,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339::option")]
    pub expires_at: Option<OffsetDateTime>,
    pub previous: Option<ArtifactId>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "time::serde::rfc3339::option"
    )]
    pub deleted_at: Option<OffsetDateTime>,
}
impl Artifact {
    pub const SCHEMA: &'static str = "sokar.artifact.v1";
    pub fn validate(&self) -> Result<(), Error> {
        check_schema(&self.schema, Self::SCHEMA)?;
        if self.blob.key != self.blob.digest.key() {
            return Err(Error::Invalid("blob key does not match digest".into()));
        }
        Ok(())
    }
    pub fn key(&self) -> String {
        format!("artifacts/{}/manifest.json", self.id)
    }
}
pub type CollectionManifest = Collection;
pub type ArtifactManifest = Artifact;
/// Parsed version tag. The owned manifest field preserves unknown tags for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestSchema {
    pub kind: String,
    pub major: u32,
}
impl std::str::FromStr for ManifestSchema {
    type Err = Error;
    fn from_str(value: &str) -> Result<Self, Error> {
        let invalid = || Error::Invalid(format!("invalid manifest schema: {value}"));
        let (kind, major) = value
            .strip_prefix("sokar.")
            .and_then(|v| v.split_once(".v"))
            .ok_or_else(invalid)?;
        if kind.is_empty()
            || !kind.bytes().all(|b| b.is_ascii_lowercase())
            || major.is_empty()
            || !major.bytes().all(|b| b.is_ascii_digit())
        {
            return Err(invalid());
        }
        Ok(Self {
            kind: kind.into(),
            major: major.parse().map_err(|_| invalid())?,
        })
    }
}
pub fn check_schema(actual: &str, expected: &'static str) -> Result<(), Error> {
    match (
        actual.parse::<ManifestSchema>(),
        expected.parse::<ManifestSchema>(),
    ) {
        (Ok(actual), Ok(expected))
            if actual.kind == expected.kind && actual.major == expected.major =>
        {
            Ok(())
        }
        _ => Err(Error::ManifestSchema {
            actual: actual.into(),
            expected,
        }),
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateCollection {
    pub scope: CollectionScope,
    pub store: Option<StoreName>,
    #[serde(default)]
    pub retention: RetentionPolicy,
    #[serde(default)]
    pub quota: Quota,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UploadOutcome {
    Stored,
    Deduplicated,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UploadResponse {
    pub artifact: Artifact,
    pub outcome: UploadOutcome,
}
#[derive(Debug, Clone)]
pub struct UploadOptions {
    pub path: ArtifactPath,
    pub media_type: MediaType,
    pub kind: ArtifactKind,
    pub labels: Labels,
    pub idempotency_key: Option<String>,
}
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ListArtifacts {
    pub prefix: Option<String>,
    pub kind: Option<ArtifactKind>,
    pub cursor: Option<String>,
    pub limit: Option<u32>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactPage {
    pub artifacts: Vec<Artifact>,
    pub next_cursor: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Problem {
    #[serde(rename = "type")]
    pub type_url: String,
    pub title: String,
    pub status: u16,
    pub detail: String,
    pub code: String,
    pub request_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn typed_manifest_schema_versions_and_kinds() {
        for (tag, kind, major) in [
            ("sokar.artifact.v1", "artifact", 1),
            ("sokar.artifact.v2", "artifact", 2),
            ("sokar.collection.v1", "collection", 1),
        ] {
            assert_eq!(
                tag.parse::<ManifestSchema>().unwrap(),
                ManifestSchema {
                    kind: kind.into(),
                    major
                }
            );
        }
        check_schema("sokar.artifact.v1", Artifact::SCHEMA).unwrap();
        check_schema("sokar.collection.v1", Collection::SCHEMA).unwrap();
        for tag in [
            "sokar.artifact.v2",
            "sokar.collection.v1",
            "sokar.artifact.v0",
            "sokar.artifact.vx",
            "other.artifact.v1",
            "sokar..v1",
            "sokar.artifact.v1.extra",
        ] {
            assert!(matches!(
                check_schema(tag, Artifact::SCHEMA),
                Err(Error::ManifestSchema { .. })
            ));
        }
    }
    #[test]
    fn validated_types() {
        for path in [
            "/etc", "a/../b", "a//b", "a/./b", "a\\b", "", "e\u{301}", "a\n",
        ] {
            assert!(path.parse::<ArtifactPath>().is_err(), "{path:?}");
        }
        assert!("réports/a b.pdf".parse::<ArtifactPath>().is_ok());
        assert!("text/html".parse::<MediaType>().is_ok());
        assert!("garbage".parse::<MediaType>().is_err());
        assert!("../store".parse::<StoreName>().is_err());
        assert!(CollectionId::try_from(Uuid::nil()).is_err());
        let digest = BlobDigest([42; 32]);
        assert_eq!(
            serde_json::from_str::<BlobDigest>(&serde_json::to_string(&digest).unwrap()).unwrap(),
            digest
        );
    }
    #[test]
    fn manifests_roundtrip_and_forward_fields() {
        let c = Collection {
            schema: Collection::SCHEMA.into(),
            id: CollectionId::new(),
            scope: CollectionScope::Named {
                name: "test".into(),
            },
            owner: "test".into(),
            store: "memory".parse().unwrap(),
            retention: RetentionPolicy::Forever,
            quota: Quota::default(),
            created_at: OffsetDateTime::now_utc(),
        };
        let mut value = serde_json::to_value(&c).unwrap();
        value["future_field"] = true.into();
        let restored: Collection = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(restored, c);
        restored.validate().unwrap();
        value["schema"] = "sokar.collection.v2".into();
        assert!(
            serde_json::from_value::<Collection>(value)
                .unwrap()
                .validate()
                .is_err()
        );
        let digest = BlobDigest([1; 32]);
        let a = Artifact {
            schema: Artifact::SCHEMA.into(),
            id: ArtifactId::new(),
            collection: c.id,
            path: "file.txt".parse().unwrap(),
            kind: ArtifactKind::Document,
            blob: BlobRef {
                digest,
                size: 10,
                media_type: "text/plain".parse().unwrap(),
                key: digest.key(),
            },
            labels: Labels::new(),
            created_by: "test".into(),
            created_at: c.created_at,
            expires_at: None,
            previous: None,
            deleted_at: None,
        };
        let restored: Artifact = serde_json::from_slice(&serde_json::to_vec(&a).unwrap()).unwrap();
        assert_eq!(a, restored);
        restored.validate().unwrap();
        let mut tombstone = a.clone();
        tombstone.deleted_at = Some(a.created_at);
        let restored: Artifact =
            serde_json::from_slice(&serde_json::to_vec(&tombstone).unwrap()).unwrap();
        assert_eq!(restored, tombstone);
        assert!(check_schema("sokar.artifact.v2", Artifact::SCHEMA).is_err());
    }
}
