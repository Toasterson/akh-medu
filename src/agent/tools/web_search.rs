//! Web search tool: query the web via the DuckDuckGo HTML endpoint.
//!
//! Uses `ureq` for the synchronous HTTP request (mirroring `http_fetch`) and
//! the `scraper` crate to extract result rows from the returned HTML. The
//! DuckDuckGo HTML endpoint is keyless and zero-config: it answers a plain
//! `GET` with a desktop `User-Agent`. Result anchors carry the `result__a`
//! class (title + redirect href) and snippets the `result__snippet` class.
//! DuckDuckGo wraps each href in a redirect (`//duckduckgo.com/l/?uddg=<url>`),
//! so the real destination is recovered by decoding the `uddg` query parameter.

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use std::collections::HashSet;

/// DuckDuckGo HTML search endpoint (keyless).
const DDG_ENDPOINT: &str = "https://html.duckduckgo.com/html/";

/// Desktop User-Agent so DuckDuckGo returns the full HTML result page.
const USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
     AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";

/// Request timeout in seconds.
const TIMEOUT_SECS: u64 = 10;

/// Default number of results to return when `max_results` is omitted.
const DEFAULT_MAX_RESULTS: u64 = 5;

/// Search the web via the DuckDuckGo HTML endpoint.
pub struct WebSearchTool;

impl Tool for WebSearchTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "web_search".into(),
            description:
                "Search the web via DuckDuckGo (keyless) and return titles, URLs, and snippets."
                    .into(),
            parameters: vec![
                ToolParam {
                    name: "query".into(),
                    description: "Search query.".into(),
                    required: true,
                },
                ToolParam {
                    name: "max_results".into(),
                    description: "Maximum number of results to return (default: 5).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let query = input.require("query", "web_search")?;
        let max_results = input
            .get_u64("max_results")
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_MAX_RESULTS) as usize;

        // Build the request URL with a percent-encoded query string.
        let url = format!("{DDG_ENDPOINT}?q={}", percent_encode(query));

        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(TIMEOUT_SECS))
            .build();

        let body = match agent.get(&url).set("User-Agent", USER_AGENT).call() {
            Ok(response) => match response.into_string() {
                Ok(body) => body,
                Err(e) => {
                    return Ok(ToolOutput::err(format!(
                        "web search failed: could not read response body: {e}"
                    )));
                }
            },
            Err(ureq::Error::Status(code, _response)) => {
                return Ok(ToolOutput::err(format!(
                    "web search failed: HTTP status {code}"
                )));
            }
            Err(ureq::Error::Transport(transport)) => {
                return Ok(ToolOutput::err(format!("web search failed: {transport}")));
            }
        };

        let results = parse_results(&body, max_results);

        if results.is_empty() {
            return Ok(ToolOutput::ok(format!("No results for \"{query}\".")));
        }

        let mut out = String::new();
        for (i, r) in results.iter().enumerate() {
            out.push_str(&format!(
                "{}. {}\n   {}\n   {}\n",
                i + 1,
                r.title,
                r.url,
                r.snippet,
            ));
        }

        Ok(ToolOutput::ok(out.trim_end().to_string()))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "web_search".into(),
            description: "Searches the web via DuckDuckGo — network access.".into(),
            parameters: vec![
                ToolParamSchema::required("query", "Search query."),
                ToolParamSchema::optional(
                    "max_results",
                    "Maximum number of results to return (default: 5).",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::Network]),
                description: "Searches the web via DuckDuckGo — network access.".into(),
                shadow_triggers: vec!["network".into()],
            },
            source: ToolSource::Native,
        }
    }
}

/// A single parsed search result.
struct SearchResult {
    title: String,
    url: String,
    snippet: String,
}

/// Parse DuckDuckGo HTML result rows: title + href from `result__a` anchors and
/// the matching `result__snippet`. Returns up to `max_results` entries.
fn parse_results(html: &str, max_results: usize) -> Vec<SearchResult> {
    // Scoped so the non-Send `scraper` types never cross an await/`?` boundary.
    let document = scraper::Html::parse_document(html);

    // Selectors are static and known-valid; unwrap is safe here.
    let result_sel = scraper::Selector::parse("div.result").unwrap();
    let anchor_sel = scraper::Selector::parse("a.result__a").unwrap();
    let snippet_sel = scraper::Selector::parse("a.result__snippet").unwrap();

    let mut results = Vec::new();

    for row in document.select(&result_sel) {
        if results.len() >= max_results {
            break;
        }

        let Some(anchor) = row.select(&anchor_sel).next() else {
            continue;
        };

        let title = collapse_ws(&anchor.text().collect::<String>());
        if title.is_empty() {
            continue;
        }

        let raw_href = anchor.value().attr("href").unwrap_or_default();
        let url = decode_ddg_href(raw_href);
        if url.is_empty() {
            continue;
        }

        let snippet = row
            .select(&snippet_sel)
            .next()
            .map(|s| collapse_ws(&s.text().collect::<String>()))
            .unwrap_or_default();

        results.push(SearchResult {
            title,
            url,
            snippet,
        });
    }

    results
}

/// Decode a DuckDuckGo redirect href into the real destination URL.
///
/// DuckDuckGo wraps results in `//duckduckgo.com/l/?uddg=<percent-encoded url>`
/// (optionally with extra params). When the `uddg` parameter is present its
/// percent-decoded value is the real URL; otherwise the href is returned as-is
/// (with a leading protocol-relative `//` promoted to `https://`).
fn decode_ddg_href(href: &str) -> String {
    if let Some(query_start) = href.find('?') {
        let query = &href[query_start + 1..];
        for pair in query.split('&') {
            if let Some(value) = pair.strip_prefix("uddg=") {
                return percent_decode(value);
            }
        }
    }

    if let Some(rest) = href.strip_prefix("//") {
        format!("https://{rest}")
    } else {
        href.to_string()
    }
}

/// Percent-encode a string for use in a URL query value (RFC 3986
/// unreserved characters pass through; everything else is `%XX`-escaped).
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Percent-decode a URL query value, turning `+` into a space and `%XX`
/// escapes into their byte values. Invalid escapes are passed through.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hi = hex_val(bytes[i + 1]);
                let lo = hex_val(bytes[i + 2]);
                match (hi, lo) {
                    (Some(h), Some(l)) => {
                        out.push((h << 4) | l);
                        i += 3;
                    }
                    _ => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            other => {
                out.push(other);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Hex digit to its numeric value, or `None` if not a hex digit.
fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Collapse runs of whitespace (including newlines from HTML text nodes) into
/// single spaces and trim the ends.
fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tool_manifest::{Capability, DangerLevel};
    use crate::engine::{Engine, EngineConfig};
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .expect("test engine")
    }

    #[test]
    fn web_search_signature_and_manifest() {
        // No network: only inspect static metadata.
        let tool = WebSearchTool;

        let sig = tool.signature();
        assert_eq!(sig.name, "web_search");
        assert_eq!(sig.parameters.len(), 2);
        assert_eq!(sig.parameters[0].name, "query");
        assert!(sig.parameters[0].required);
        assert_eq!(sig.parameters[1].name, "max_results");
        assert!(!sig.parameters[1].required);

        let manifest = tool.manifest();
        assert_eq!(manifest.name, "web_search");
        // Manifest name must match the signature name.
        assert_eq!(manifest.name, sig.name);
        assert_eq!(manifest.parameters.len(), 2);
        assert!(manifest.parameters[0].required);
        assert!(!manifest.parameters[1].required);
        assert_eq!(manifest.danger.level, DangerLevel::Cautious);
        assert!(manifest.danger.capabilities.contains(&Capability::Network));
        assert_eq!(manifest.danger.capabilities.len(), 1);
        assert_eq!(manifest.danger.shadow_triggers, vec!["network".to_string()]);
    }

    #[test]
    fn web_search_requires_query() {
        // Missing required `query` must hard-error via `require` before any
        // network call is attempted.
        let tool = WebSearchTool;
        let engine = test_engine();
        let input = ToolInput::new();
        let result = tool.execute(&engine, input);
        assert!(result.is_err());
    }

    #[test]
    fn percent_encode_escapes_query() {
        assert_eq!(percent_encode("rust lang"), "rust%20lang");
        assert_eq!(percent_encode("a+b&c=d"), "a%2Bb%26c%3Dd");
        assert_eq!(percent_encode("safe-_.~"), "safe-_.~");
    }

    #[test]
    fn percent_decode_recovers_url() {
        assert_eq!(
            percent_decode("https%3A%2F%2Fexample.com%2Fa%20b"),
            "https://example.com/a b"
        );
        assert_eq!(percent_decode("a+b"), "a b");
    }

    #[test]
    fn decode_ddg_href_unwraps_redirect() {
        let href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Frust-lang.org%2F&rut=abc";
        assert_eq!(decode_ddg_href(href), "https://rust-lang.org/");
    }

    #[test]
    fn decode_ddg_href_promotes_protocol_relative() {
        assert_eq!(
            decode_ddg_href("//example.com/page"),
            "https://example.com/page"
        );
    }

    #[test]
    fn parse_results_extracts_rows_and_respects_limit() {
        let html = r#"
            <html><body>
            <div class="result">
                <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone">First Title</a>
                <a class="result__snippet">Snippet one text.</a>
            </div>
            <div class="result">
                <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Ftwo">Second Title</a>
                <a class="result__snippet">Snippet two text.</a>
            </div>
            </body></html>
        "#;

        let results = parse_results(html, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "First Title");
        assert_eq!(results[0].url, "https://example.com/one");
        assert_eq!(results[0].snippet, "Snippet one text.");

        let all = parse_results(html, 5);
        assert_eq!(all.len(), 2);
        assert_eq!(all[1].url, "https://example.org/two");
    }

    #[test]
    fn parse_results_empty_on_no_rows() {
        let results = parse_results("<html><body>no results here</body></html>", 5);
        assert!(results.is_empty());
    }
}
