use crate::{SokarError, SokarResult, config::DeliveryConfig, store::Store};
use axum::{
    body::Body,
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use futures::TryStreamExt;
use sokar_client::Artifact;
use url::Url;
pub struct StreamBody {
    pub body: Body,
    pub headers: HeaderMap,
    pub status: StatusCode,
}
pub enum Delivery {
    Redirect(Url),
    Stream(StreamBody),
}
impl IntoResponse for Delivery {
    fn into_response(self) -> Response {
        match self {
            Self::Redirect(url) => {
                axum::response::Redirect::temporary(url.as_str()).into_response()
            }
            Self::Stream(stream) => (stream.status, stream.headers, stream.body).into_response(),
        }
    }
}
fn header_value(value: &str) -> SokarResult<HeaderValue> {
    HeaderValue::from_str(value)
        .map_err(|_| SokarError::Invalid("invalid response header value".into()))
}
pub async fn content(
    store: &Store,
    artifact: &Artifact,
    request: &HeaderMap,
    config: &DeliveryConfig,
) -> SokarResult<Delivery> {
    let mut headers = HeaderMap::new();
    let etag = format!("\"sha256-{}\"", artifact.blob.digest);
    headers.insert(header::ETAG, header_value(&etag)?);
    headers.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, immutable, max-age=31536000"),
    );
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::CONTENT_TYPE,
        header_value(artifact.blob.media_type.as_str())?,
    );
    let basename = artifact
        .path
        .as_str()
        .rsplit('/')
        .next()
        .expect("nonempty path");
    let encoded =
        percent_encoding::utf8_percent_encode(basename, percent_encoding::NON_ALPHANUMERIC);
    let disposition = if config
        .inline_media_types
        .iter()
        .any(|m| m.essence() == artifact.blob.media_type.essence())
    {
        "inline".to_owned()
    } else {
        format!("attachment; filename*=UTF-8''{encoded}")
    };
    headers.insert(header::CONTENT_DISPOSITION, header_value(&disposition)?);
    if request
        .get(header::IF_NONE_MATCH)
        .and_then(|h| h.to_str().ok())
        .is_some_and(|v| {
            v.split(',').any(|tag| {
                let tag = tag.trim();
                tag == "*" || tag.strip_prefix("W/").unwrap_or(tag) == etag
            })
        })
    {
        return Ok(Delivery::Stream(StreamBody {
            body: Body::empty(),
            headers,
            status: StatusCode::NOT_MODIFIED,
        }));
    }
    let size = artifact.blob.size;
    // If-Range only supports our strong validator. Dates or mismatches return the full body.
    let use_range = request
        .get(header::IF_RANGE)
        .is_none_or(|v| v.as_bytes() == etag.as_bytes());
    let range = if use_range {
        request
            .get(header::RANGE)
            .map(|h| {
                h.to_str()
                    .map_err(|_| SokarError::Range { size })
                    .and_then(|v| parse_range(v, size))
            })
            .transpose()?
    } else {
        None
    };
    let (start, end, status) = if let Some((start, end)) = range {
        headers.insert(
            header::CONTENT_RANGE,
            header_value(&format!("bytes {start}-{}/{size}", end - 1))?,
        );
        (start, end, StatusCode::PARTIAL_CONTENT)
    } else {
        (0, size, StatusCode::OK)
    };
    headers.insert(
        header::CONTENT_LENGTH,
        header_value(&(end - start).to_string())?,
    );
    let body = if end == start {
        Body::empty()
    } else {
        let reader = store
            .op
            .reader_with(&artifact.blob.key)
            .chunk(1024 * 1024)
            .await
            .map_err(|e| store.error("open content", e))?;
        let stream = reader
            .into_bytes_stream(start..end)
            .await
            .map_err(|e| store.error("read range", e))?;
        let name = store.name.clone();
        Body::from_stream(stream.map_err(move |e| {
            SokarError::storage(
                &name,
                "stream content",
                opendal::Error::new(opendal::ErrorKind::Unexpected, "content stream failed")
                    .set_source(e),
            )
        }))
    };
    Ok(Delivery::Stream(StreamBody {
        body,
        headers,
        status,
    }))
}
/// One RFC byte range, inclusive wire end converted to exclusive storage end.
pub fn parse_range(value: &str, size: u64) -> SokarResult<(u64, u64)> {
    let error = || SokarError::Range { size };
    let value = value.strip_prefix("bytes=").ok_or_else(error)?;
    let (start, end) = value.split_once('-').ok_or_else(error)?;
    if size == 0 || value.contains(',') {
        return Err(error());
    }
    if start.is_empty() {
        let suffix = end.parse::<u64>().map_err(|_| error())?;
        if suffix == 0 {
            return Err(error());
        }
        return Ok((size.saturating_sub(suffix), size));
    }
    let start = start.parse::<u64>().map_err(|_| error())?;
    let end = if end.is_empty() {
        size
    } else {
        end.parse::<u64>()
            .map_err(|_| error())?
            .saturating_add(1)
            .min(size)
    };
    if start >= size || end <= start {
        return Err(error());
    }
    Ok((start, end))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ranges() {
        assert_eq!(parse_range("bytes=1000-1999", 3000).unwrap(), (1000, 2000));
        assert_eq!(parse_range("bytes=-10", 5).unwrap(), (0, 5));
        assert_eq!(parse_range("bytes=3-", 5).unwrap(), (3, 5));
        assert_eq!(parse_range("bytes=0-999", 5).unwrap(), (0, 5));
        for r in [
            "bytes=5-",
            "bytes=-0",
            "bytes=2-1",
            "bytes=0-1,3-4",
            "items=0-1",
        ] {
            assert!(parse_range(r, 5).is_err());
        }
        assert!(parse_range("bytes=0-", 0).is_err());
    }
}
