//! JWZ email threading algorithm (RFC 5256 / Jamie Zawinski).
//!
//! Groups parsed emails into conversation threads using Message-ID,
//! In-Reply-To, and References headers. Produces a `ThreadTree` with
//! roots and parent-child relationships.

use std::collections::HashMap;

use super::parser::ParsedEmail;

// ── ThreadNode ──────────────────────────────────────────────────────────

/// A node in the thread tree, representing one message (or a phantom
/// placeholder for a referenced but unseen message).
#[derive(Debug, Clone)]
pub struct ThreadNode {
    /// The Message-ID this node represents.
    pub message_id: String,
    /// Index of the parent node (None if this is a root).
    pub parent: Option<usize>,
    /// Indices of child nodes.
    pub children: Vec<usize>,
    /// The parsed email data. `None` for phantom nodes (referenced
    /// messages we haven't fetched).
    pub parsed: Option<ParsedEmail>,
}

impl ThreadNode {
    /// Whether this is a phantom (placeholder) node with no actual message.
    pub fn is_phantom(&self) -> bool {
        self.parsed.is_none()
    }

    /// The subject of this message (or "[missing]" for phantoms).
    pub fn subject(&self) -> &str {
        self.parsed
            .as_ref()
            .map(|p| p.subject.as_str())
            .unwrap_or("[missing]")
    }

    /// The date timestamp (for sorting). Phantoms sort to 0.
    pub fn date(&self) -> u64 {
        self.parsed.as_ref().and_then(|p| p.date).unwrap_or(0)
    }
}

// ── ThreadTree ──────────────────────────────────────────────────────────

/// A forest of threaded email conversations.
#[derive(Debug, Clone)]
pub struct ThreadTree {
    /// All nodes (both real and phantom).
    pub nodes: Vec<ThreadNode>,
    /// Indices of root nodes (conversations that have no known parent).
    pub roots: Vec<usize>,
    /// Index from Message-ID to node index.
    index: HashMap<String, usize>,
}

impl ThreadTree {
    /// Find the root thread index for a given message.
    pub fn thread_id_for(&self, message_id: &str) -> Option<usize> {
        let idx = *self.index.get(message_id)?;
        Some(self.find_root(idx))
    }

    /// Get a node by index.
    pub fn get(&self, idx: usize) -> Option<&ThreadNode> {
        self.nodes.get(idx)
    }

    /// Number of distinct thread roots (conversations).
    pub fn thread_count(&self) -> usize {
        self.roots.len()
    }

    /// Total number of nodes (including phantoms).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Walk up parent links to find the root of a thread.
    fn find_root(&self, mut idx: usize) -> usize {
        let mut visited = std::collections::HashSet::new();
        while let Some(parent) = self.nodes[idx].parent {
            if !visited.insert(idx) {
                break; // Cycle protection.
            }
            idx = parent;
        }
        idx
    }

    /// Collect all message IDs in a thread (given a root index).
    pub fn thread_messages(&self, root_idx: usize) -> Vec<&str> {
        let mut result = Vec::new();
        let mut stack = vec![root_idx];
        while let Some(idx) = stack.pop() {
            if let Some(node) = self.nodes.get(idx) {
                result.push(node.message_id.as_str());
                for &child in &node.children {
                    stack.push(child);
                }
            }
        }
        result
    }
}

// ── build_threads ───────────────────────────────────────────────────────

/// Build a `ThreadTree` from a set of parsed emails using the JWZ algorithm.
///
/// Five steps:
/// 1. Build an ID table: create a node for each Message-ID.
/// 2. Link parents: use In-Reply-To and References to establish parent-child.
/// 3. Find root set: nodes with no parent are roots.
/// 4. Prune empty containers: remove phantom nodes with no children.
/// 5. Sort siblings by date.
pub fn build_threads(emails: &[ParsedEmail]) -> ThreadTree {
    let mut nodes: Vec<ThreadNode> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();

    // Step 1: Build ID table — create or find a node for each message.
    for email in emails {
        let msg_id = email.message_id.clone();
        let node_idx = get_or_create_node(&mut nodes, &mut index, &msg_id);
        // Attach the parsed email data to this node.
        nodes[node_idx].parsed = Some(email.clone());
    }

    // Step 2: Link parents via In-Reply-To and References.
    for email in emails {
        let child_idx = index[&email.message_id];

        // References chain: each references[i] is the parent of references[i+1].
        let mut prev_idx: Option<usize> = None;
        for ref_id in &email.references {
            let ref_idx = get_or_create_node(&mut nodes, &mut index, ref_id);
            if let Some(prev) = prev_idx {
                // Only link if not already parented and no cycle.
                if nodes[ref_idx].parent.is_none() && ref_idx != prev {
                    link_parent_child(&mut nodes, prev, ref_idx);
                }
            }
            prev_idx = Some(ref_idx);
        }

        // In-Reply-To: the direct parent of this message.
        if let Some(ref reply_to) = email.in_reply_to {
            let parent_idx = get_or_create_node(&mut nodes, &mut index, reply_to);
            // Only set parent if not already parented and no self-reference.
            if nodes[child_idx].parent.is_none() && child_idx != parent_idx {
                link_parent_child(&mut nodes, parent_idx, child_idx);
            }
        } else if let Some(last_ref) = prev_idx {
            // If no In-Reply-To but References exist, use the last reference.
            if nodes[child_idx].parent.is_none() && child_idx != last_ref {
                link_parent_child(&mut nodes, last_ref, child_idx);
            }
        }
    }

    // Step 3: Find root set (nodes with no parent).
    let roots: Vec<usize> = (0..nodes.len())
        .filter(|&i| nodes[i].parent.is_none())
        .collect();

    // Step 4: Prune phantom nodes that have no children and no message data.
    // (We don't actually remove them to keep indices stable — just exclude
    //  from roots.)
    let roots: Vec<usize> = roots
        .into_iter()
        .filter(|&i| !nodes[i].is_phantom() || !nodes[i].children.is_empty())
        .collect();

    // Step 5: Sort siblings by date.
    // Collect date info first to avoid borrow conflict.
    let dates: Vec<u64> = nodes.iter().map(|n| n.date()).collect();
    for node in &mut nodes {
        node.children.sort_by_key(|&child_idx| {
            dates.get(child_idx).copied().unwrap_or(0)
        });
    }

    // Sort roots by date of earliest message.
    let mut roots = roots;
    roots.sort_by_key(|&root_idx| nodes[root_idx].date());

    ThreadTree {
        nodes,
        roots,
        index,
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Get an existing node index for a Message-ID, or create a phantom node.
fn get_or_create_node(
    nodes: &mut Vec<ThreadNode>,
    index: &mut HashMap<String, usize>,
    message_id: &str,
) -> usize {
    if let Some(&idx) = index.get(message_id) {
        idx
    } else {
        let idx = nodes.len();
        nodes.push(ThreadNode {
            message_id: message_id.to_string(),
            parent: None,
            children: Vec::new(),
            parsed: None,
        });
        index.insert(message_id.to_string(), idx);
        idx
    }
}

/// Establish a parent-child link between two nodes.
fn link_parent_child(nodes: &mut [ThreadNode], parent_idx: usize, child_idx: usize) {
    // Prevent cycles: walk up from parent to check we don't reach child.
    let mut current = parent_idx;
    let mut depth = 0;
    while let Some(p) = nodes[current].parent {
        if p == child_idx {
            return; // Would create a cycle.
        }
        current = p;
        depth += 1;
        if depth > 1000 {
            return; // Safety limit.
        }
    }

    nodes[child_idx].parent = Some(parent_idx);
    if !nodes[parent_idx].children.contains(&child_idx) {
        nodes[parent_idx].children.push(child_idx);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_email(
        message_id: &str,
        subject: &str,
        in_reply_to: Option<&str>,
        references: &[&str],
        date: Option<u64>,
    ) -> ParsedEmail {
        ParsedEmail {
            message_id: message_id.to_string(),
            from: "sender@example.com".to_string(),
            from_display: None,
            to: vec!["recipient@example.com".to_string()],
            cc: Vec::new(),
            subject: subject.to_string(),
            date,
            in_reply_to: in_reply_to.map(|s| s.to_string()),
            references: references.iter().map(|s| s.to_string()).collect(),
            body_text: Some("body".to_string()),
            body_html: None,
            has_attachments: false,
            list_id: None,
            content_type: "text/plain".to_string(),
        }
    }

    #[test]
    fn single_message_is_root() {
        let emails = vec![make_email("<a@example>", "Hello", None, &[], Some(1000))];
        let tree = build_threads(&emails);
        assert_eq!(tree.thread_count(), 1);
        assert_eq!(tree.node_count(), 1);
        assert_eq!(tree.roots.len(), 1);
    }

    #[test]
    fn linear_chain() {
        // A → B → C
        let emails = vec![
            make_email("<a@ex>", "Thread start", None, &[], Some(100)),
            make_email(
                "<b@ex>",
                "Re: Thread start",
                Some("<a@ex>"),
                &["<a@ex>"],
                Some(200),
            ),
            make_email(
                "<c@ex>",
                "Re: Re: Thread start",
                Some("<b@ex>"),
                &["<a@ex>", "<b@ex>"],
                Some(300),
            ),
        ];

        let tree = build_threads(&emails);

        // Only one root (the first message).
        assert_eq!(tree.thread_count(), 1);
        assert_eq!(tree.node_count(), 3);

        // A is root, B is child of A, C is child of B.
        let a = tree.index["<a@ex>"];
        let b = tree.index["<b@ex>"];
        let c = tree.index["<c@ex>"];

        assert!(tree.nodes[a].parent.is_none());
        assert_eq!(tree.nodes[b].parent, Some(a));
        assert_eq!(tree.nodes[c].parent, Some(b));
    }

    #[test]
    fn branching_thread() {
        // A → B, A → C (two replies to same parent)
        let emails = vec![
            make_email("<root@ex>", "Root", None, &[], Some(100)),
            make_email(
                "<reply1@ex>",
                "Re: Root",
                Some("<root@ex>"),
                &["<root@ex>"],
                Some(200),
            ),
            make_email(
                "<reply2@ex>",
                "Re: Root",
                Some("<root@ex>"),
                &["<root@ex>"],
                Some(300),
            ),
        ];

        let tree = build_threads(&emails);
        assert_eq!(tree.thread_count(), 1);

        let root = tree.index["<root@ex>"];
        assert_eq!(tree.nodes[root].children.len(), 2);
    }

    #[test]
    fn phantom_parent() {
        // B references A, but A is not in the set.
        let emails = vec![make_email(
            "<b@ex>",
            "Re: Missing parent",
            Some("<phantom@ex>"),
            &["<phantom@ex>"],
            Some(200),
        )];

        let tree = build_threads(&emails);

        // The phantom becomes a root with B as child.
        let phantom = tree.index["<phantom@ex>"];
        assert!(tree.nodes[phantom].is_phantom());
        assert_eq!(tree.nodes[phantom].children.len(), 1);
        assert!(tree.roots.contains(&phantom));
    }

    #[test]
    fn orphan_message() {
        // Two unrelated messages.
        let emails = vec![
            make_email("<x@ex>", "Msg X", None, &[], Some(100)),
            make_email("<y@ex>", "Msg Y", None, &[], Some(200)),
        ];

        let tree = build_threads(&emails);
        assert_eq!(tree.thread_count(), 2);
    }

    #[test]
    fn thread_id_for_lookup() {
        let emails = vec![
            make_email("<a@ex>", "Root", None, &[], Some(100)),
            make_email(
                "<b@ex>",
                "Reply",
                Some("<a@ex>"),
                &["<a@ex>"],
                Some(200),
            ),
        ];

        let tree = build_threads(&emails);

        let a_root = tree.thread_id_for("<a@ex>").unwrap();
        let b_root = tree.thread_id_for("<b@ex>").unwrap();
        assert_eq!(a_root, b_root); // Same thread.
    }

    #[test]
    fn thread_messages_collects_all() {
        let emails = vec![
            make_email("<a@ex>", "Root", None, &[], Some(100)),
            make_email(
                "<b@ex>",
                "Reply 1",
                Some("<a@ex>"),
                &["<a@ex>"],
                Some(200),
            ),
            make_email(
                "<c@ex>",
                "Reply 2",
                Some("<a@ex>"),
                &["<a@ex>"],
                Some(300),
            ),
        ];

        let tree = build_threads(&emails);
        let root = tree.thread_id_for("<a@ex>").unwrap();
        let msgs = tree.thread_messages(root);

        assert_eq!(msgs.len(), 3);
        assert!(msgs.contains(&"<a@ex>"));
        assert!(msgs.contains(&"<b@ex>"));
        assert!(msgs.contains(&"<c@ex>"));
    }

    #[test]
    fn siblings_sorted_by_date() {
        let emails = vec![
            make_email("<root@ex>", "Root", None, &[], Some(100)),
            make_email(
                "<late@ex>",
                "Late",
                Some("<root@ex>"),
                &["<root@ex>"],
                Some(500),
            ),
            make_email(
                "<early@ex>",
                "Early",
                Some("<root@ex>"),
                &["<root@ex>"],
                Some(200),
            ),
        ];

        let tree = build_threads(&emails);
        let root = tree.index["<root@ex>"];
        let children = &tree.nodes[root].children;

        // Early (200) should come before Late (500).
        let first_child = &tree.nodes[children[0]];
        let second_child = &tree.nodes[children[1]];
        assert!(first_child.date() <= second_child.date());
    }
}
