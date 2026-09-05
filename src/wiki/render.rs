//! Static HTML rendering pipeline — Phase 22c.
//!
//! Renders wiki articles, the index, and category pages to a static output
//! directory.  Requires the `wiki` feature flag (comrak + minijinja + syntect).
//!
//! # Pipeline
//!
//! ```text
//! ArticleStore → comrak (Markdown → HTML) → minijinja template → write file
//! ```

#[cfg(feature = "wiki")]
mod inner {
    use std::fs;
    use std::path::{Path, PathBuf};

    use comrak::{ComrakOptions, markdown_to_html};
    use miette::Diagnostic;
    use minijinja::Environment;
    use thiserror::Error;

    use crate::wiki::store::{Article, ArticleStore};

    // ─── Error ───────────────────────────────────────────────────────────

    #[derive(Debug, Error, Diagnostic)]
    pub enum RenderError {
        #[error("I/O error writing to {path}: {message}")]
        #[diagnostic(
            code(akh::wiki::render::io),
            help("Check output directory permissions.")
        )]
        Io { path: String, message: String },

        #[error("template error: {message}")]
        #[diagnostic(
            code(akh::wiki::render::template),
            help("Check the minijinja template syntax.")
        )]
        Template { message: String },
    }

    pub type RenderResult<T> = std::result::Result<T, RenderError>;

    // ─── Config ──────────────────────────────────────────────────────────

    /// Configuration for the rendering pipeline.
    #[derive(Debug, Clone)]
    pub struct RenderConfig {
        /// Output directory for the static site.
        pub output_dir: PathBuf,
        /// Site title shown in the HTML `<title>` tag.
        pub site_title: String,
        /// Base URL for inter-article links.
        pub base_url: String,
    }

    impl Default for RenderConfig {
        fn default() -> Self {
            Self {
                output_dir: PathBuf::from("site"),
                site_title: "Akhipedia".into(),
                base_url: "/".into(),
            }
        }
    }

    // ─── WikiRenderer ────────────────────────────────────────────────────

    /// Renders an `ArticleStore` to a static HTML site.
    pub struct WikiRenderer {
        config: RenderConfig,
    }

    impl WikiRenderer {
        pub fn new(config: RenderConfig) -> Self {
            Self { config }
        }

        /// Render the full static site from an `ArticleStore`.
        pub fn render_site(&self, store: &ArticleStore) -> RenderResult<usize> {
            fs::create_dir_all(&self.config.output_dir).map_err(|e| RenderError::Io {
                path: self.config.output_dir.display().to_string(),
                message: e.to_string(),
            })?;

            let mut count = 0;
            for article in store.articles.values() {
                self.render_article(article)?;
                count += 1;
            }
            self.render_index(store)?;
            Ok(count)
        }

        /// Render a single article to `<output_dir>/<slug>.html`.
        pub fn render_article(&self, article: &Article) -> RenderResult<PathBuf> {
            let html_body = markdown_to_html(&article.body, &comrak_options());
            let slug = slugify(&article.header.title);
            let out_path = self.config.output_dir.join(format!("{slug}.html"));

            let block_labels: Vec<&str> = article
                .header
                .blocks
                .iter()
                .map(|b| b.block_type.as_label())
                .collect();

            let responds_to_str = article
                .header
                .responds_to
                .map(|s| s.get().to_string())
                .unwrap_or_default();

            let html = render_article_page(
                &self.config.site_title,
                &self.config.base_url,
                &article.header.title,
                &html_body,
                article.header.author.get(),
                article.header.created,
                article.header.modified,
                &block_labels,
                article.header.cites.len(),
                &responds_to_str,
            )?;

            write_file(&out_path, &html)?;
            Ok(out_path)
        }

        /// Render the index page listing all articles.
        pub fn render_index(&self, store: &ArticleStore) -> RenderResult<PathBuf> {
            let out_path = self.config.output_dir.join("index.html");

            let mut articles: Vec<(&str, String, usize, u64)> = store
                .articles
                .values()
                .map(|a| {
                    (
                        a.header.title.as_str(),
                        slugify(&a.header.title),
                        a.header.blocks.len(),
                        a.header.created,
                    )
                })
                .collect();

            articles.sort_by_key(|(title, _, _, _)| *title);

            let html = render_index_page(
                &self.config.site_title,
                &self.config.base_url,
                &articles,
                store.count(),
            )?;

            write_file(&out_path, &html)?;
            Ok(out_path)
        }
    }

    // ─── Pure rendering functions ─────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn render_article_page(
        site_title: &str,
        base_url: &str,
        title: &str,
        body_html: &str,
        author_id: u64,
        created: u64,
        modified: u64,
        block_labels: &[&str],
        cites_count: usize,
        responds_to: &str,
    ) -> RenderResult<String> {
        let mut env = Environment::new();
        env.add_template("article.html", ARTICLE_TEMPLATE)
            .map_err(|e| RenderError::Template {
                message: e.to_string(),
            })?;

        let tmpl = env
            .get_template("article.html")
            .map_err(|e| RenderError::Template {
                message: e.to_string(),
            })?;

        let ctx = minijinja::context! {
            site_title,
            base_url,
            title,
            body_html,
            author_id,
            created,
            modified,
            block_labels,
            cites_count,
            responds_to,
        };

        tmpl.render(ctx).map_err(|e| RenderError::Template {
            message: e.to_string(),
        })
    }

    fn render_index_page(
        site_title: &str,
        base_url: &str,
        articles: &[(&str, String, usize, u64)],
        total: usize,
    ) -> RenderResult<String> {
        let mut env = Environment::new();
        env.add_template("index.html", INDEX_TEMPLATE)
            .map_err(|e| RenderError::Template {
                message: e.to_string(),
            })?;

        let tmpl = env
            .get_template("index.html")
            .map_err(|e| RenderError::Template {
                message: e.to_string(),
            })?;

        // Build a list of context objects for the template.
        let article_list: Vec<_> = articles
            .iter()
            .map(|(title, slug, block_count, _)| {
                minijinja::context! {
                    title => *title,
                    slug => slug.as_str(),
                    block_count => *block_count,
                }
            })
            .collect();

        let ctx = minijinja::context! {
            site_title,
            base_url,
            articles => article_list,
            total,
        };

        tmpl.render(ctx).map_err(|e| RenderError::Template {
            message: e.to_string(),
        })
    }

    // ─── Helpers ─────────────────────────────────────────────────────────

    fn comrak_options() -> ComrakOptions {
        let mut opts = ComrakOptions::default();
        opts.extension.strikethrough = true;
        opts.extension.table = true;
        opts.extension.autolink = true;
        opts.extension.tasklist = true;
        opts.render.unsafe_ = false;
        opts
    }

    pub fn slugify(title: &str) -> String {
        let slug: String = title
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect();
        // Collapse consecutive hyphens and trim leading/trailing.
        let slug = slug.trim_matches('-').to_string();
        // Simple multi-hyphen collapse.
        let mut prev = ' ';
        slug.chars()
            .filter(|&c| {
                if c == '-' && prev == '-' {
                    false
                } else {
                    prev = c;
                    true
                }
            })
            .collect()
    }

    fn write_file(path: &Path, content: &str) -> RenderResult<()> {
        fs::write(path, content).map_err(|e| RenderError::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        })
    }

    // ─── Templates ───────────────────────────────────────────────────────

    const ARTICLE_TEMPLATE: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{{ title }} — {{ site_title }}</title>
  <style>
    body { font-family: sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
    header { border-bottom: 1px solid #ccc; margin-bottom: 1.5rem; }
    .meta { color: #666; font-size: 0.85em; }
    nav a { margin-right: 1rem; }
    blockquote { border-left: 3px solid #aaa; padding-left: 1rem; color: #444; }
    pre { background: #f5f5f5; padding: 1rem; overflow-x: auto; }
  </style>
</head>
<body>
  <header>
    <nav><a href="{{ base_url }}index.html">← Index</a></nav>
    <h1>{{ title }}</h1>
    <p class="meta">
      Author: #{{ author_id }} ·
      Created: {{ created }} ·
      Modified: {{ modified }}
      {% if responds_to %} · Responds to: #{{ responds_to }}{% endif %}
      {% if cites_count > 0 %} · Cites: {{ cites_count }} article(s){% endif %}
    </p>
    {% if block_labels %}
    <p class="meta">Blocks: {{ block_labels | join(sep=", ") }}</p>
    {% endif %}
  </header>
  <main>{{ body_html | safe }}</main>
</body>
</html>
"#;

    const INDEX_TEMPLATE: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{{ site_title }}</title>
  <style>
    body { font-family: sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
    h1 { border-bottom: 1px solid #ccc; }
    li { margin: 0.4rem 0; }
    .meta { color: #666; font-size: 0.8em; }
  </style>
</head>
<body>
  <h1>{{ site_title }}</h1>
  <p>{{ total }} article(s)</p>
  <ul>
  {% for art in articles %}
    <li>
      <a href="{{ base_url }}{{ art.slug }}.html">{{ art.title }}</a>
      <span class="meta">({{ art.block_count }} blocks)</span>
    </li>
  {% endfor %}
  </ul>
</body>
</html>
"#;

    // ─── Tests ───────────────────────────────────────────────────────────

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::engine::{Engine, EngineConfig};
        use crate::symbol::SymbolKind;
        use crate::wiki::store::ArticleStore;
        use tempfile::TempDir;

        fn engine() -> Engine {
            Engine::new(EngineConfig::default()).unwrap()
        }

        #[test]
        fn slugify_works() {
            assert_eq!(slugify("Rust Basics"), "rust-basics");
            assert_eq!(slugify("Hello World"), "hello-world");
            assert_eq!(slugify("A  B"), "a-b");
        }

        #[test]
        fn render_empty_index() {
            let tmp = TempDir::new().unwrap();
            let config = RenderConfig {
                output_dir: tmp.path().to_path_buf(),
                ..Default::default()
            };
            let renderer = WikiRenderer::new(config);
            let store = ArticleStore::new();
            let n = renderer.render_site(&store).unwrap();
            assert_eq!(n, 0);
            assert!(tmp.path().join("index.html").exists());
        }

        #[test]
        fn render_single_article() {
            let tmp = TempDir::new().unwrap();
            let eng = engine();
            let author = eng
                .create_symbol(SymbolKind::Entity, "render-author")
                .unwrap()
                .id;
            let mut store = ArticleStore::new();
            store
                .create_article(
                    "Hello World",
                    "# Hello\n\nContent.",
                    author,
                    &[],
                    vec![],
                    &eng,
                )
                .unwrap();

            let config = RenderConfig {
                output_dir: tmp.path().to_path_buf(),
                ..Default::default()
            };
            let renderer = WikiRenderer::new(config);
            let n = renderer.render_site(&store).unwrap();
            assert_eq!(n, 1);
            let slug_path = tmp.path().join("hello-world.html");
            assert!(slug_path.exists());
            let html = fs::read_to_string(&slug_path).unwrap();
            assert!(html.contains("Hello World"));
            assert!(html.contains("<h1>"));
        }

        #[test]
        fn render_index_lists_articles() {
            let tmp = TempDir::new().unwrap();
            let eng = engine();
            let author = eng
                .create_symbol(SymbolKind::Entity, "idx-author")
                .unwrap()
                .id;
            let mut store = ArticleStore::new();
            store
                .create_article("Alpha", "body a", author, &[], vec![], &eng)
                .unwrap();
            store
                .create_article("Beta", "body b", author, &[], vec![], &eng)
                .unwrap();

            let config = RenderConfig {
                output_dir: tmp.path().to_path_buf(),
                ..Default::default()
            };
            let renderer = WikiRenderer::new(config);
            renderer.render_site(&store).unwrap();

            let index_html = fs::read_to_string(tmp.path().join("index.html")).unwrap();
            assert!(index_html.contains("Alpha"));
            assert!(index_html.contains("Beta"));
            assert!(index_html.contains("2 article(s)"));
        }

        #[test]
        fn render_article_contains_metadata() {
            let tmp = TempDir::new().unwrap();
            let eng = engine();
            let author = eng
                .create_symbol(SymbolKind::Entity, "meta-author")
                .unwrap()
                .id;
            let mut store = ArticleStore::new();
            store
                .create_article("Meta Test", "meta body", author, &[], vec![], &eng)
                .unwrap();

            let config = RenderConfig {
                output_dir: tmp.path().to_path_buf(),
                site_title: "TestWiki".into(),
                ..Default::default()
            };
            let renderer = WikiRenderer::new(config);
            renderer.render_site(&store).unwrap();

            let html = fs::read_to_string(tmp.path().join("meta-test.html")).unwrap();
            assert!(html.contains("TestWiki"));
            assert!(html.contains("Meta Test"));
        }

        #[test]
        fn render_markdown_formatting() {
            let tmp = TempDir::new().unwrap();
            let eng = engine();
            let author = eng
                .create_symbol(SymbolKind::Entity, "md-author")
                .unwrap()
                .id;
            let mut store = ArticleStore::new();
            store
                .create_article(
                    "MD Test",
                    "# Header\n\n**bold** and _italic_.\n\n- item 1\n- item 2",
                    author,
                    &[],
                    vec![],
                    &eng,
                )
                .unwrap();

            let config = RenderConfig {
                output_dir: tmp.path().to_path_buf(),
                ..Default::default()
            };
            let renderer = WikiRenderer::new(config);
            renderer.render_site(&store).unwrap();

            let html = fs::read_to_string(tmp.path().join("md-test.html")).unwrap();
            assert!(html.contains("<strong>bold</strong>"));
            assert!(html.contains("<em>italic</em>"));
            assert!(html.contains("<li>"));
        }

        #[test]
        fn render_multiple_articles_all_written() {
            let tmp = TempDir::new().unwrap();
            let eng = engine();
            let author = eng
                .create_symbol(SymbolKind::Entity, "multi-author")
                .unwrap()
                .id;
            let mut store = ArticleStore::new();
            for i in 0..5 {
                store
                    .create_article(
                        &format!("Article {i}"),
                        &format!("body {i}"),
                        author,
                        &[],
                        vec![],
                        &eng,
                    )
                    .unwrap();
            }
            let config = RenderConfig {
                output_dir: tmp.path().to_path_buf(),
                ..Default::default()
            };
            let renderer = WikiRenderer::new(config);
            let n = renderer.render_site(&store).unwrap();
            assert_eq!(n, 5);
        }
    }
}

#[cfg(feature = "wiki")]
pub use inner::{RenderConfig, RenderError, RenderResult, WikiRenderer, slugify};
