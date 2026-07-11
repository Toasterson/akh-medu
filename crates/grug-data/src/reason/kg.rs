//! The synthetic knowledge graph: typed entities, a fixed functional-relation
//! bank, and deterministic (seeded) fact assignment.
//!
//! Every relation is **functional**: each entity of the relation's source
//! type is assigned exactly one object (of the relation's target type),
//! chosen deterministically from `seed`. Every [`EntityType`] has at least one
//! outgoing relation, so a random walk of relations from any starting entity
//! can be extended to arbitrary depth (used up to depth 6 by the Phase 51
//! P3-R corpus) -- see [`RelationKind::of_source`] and [`random_path`].

use std::collections::HashMap;
use std::fmt;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// A typed entity namespace. Six types keep the relation bank small while
/// still supporting chains up to depth 6 (see [`RelationKind`]'s type graph).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    Person,
    City,
    Region,
    Country,
    Org,
    Device,
}

impl EntityType {
    /// Every entity type, in a fixed (arbitrary but stable) order.
    pub const ALL: [EntityType; 6] = [
        EntityType::Person,
        EntityType::City,
        EntityType::Region,
        EntityType::Country,
        EntityType::Org,
        EntityType::Device,
    ];

    /// The single-character label prefix (`p`/`c`/`r`/`k`/`o`/`d`), matching
    /// the Phase 51 P3-R plan's short entity-name convention.
    fn prefix(self) -> char {
        match self {
            EntityType::Person => 'p',
            EntityType::City => 'c',
            EntityType::Region => 'r',
            EntityType::Country => 'k',
            EntityType::Org => 'o',
            EntityType::Device => 'd',
        }
    }

    fn from_prefix(c: char) -> Option<Self> {
        match c {
            'p' => Some(EntityType::Person),
            'c' => Some(EntityType::City),
            'r' => Some(EntityType::Region),
            'k' => Some(EntityType::Country),
            'o' => Some(EntityType::Org),
            'd' => Some(EntityType::Device),
            _ => None,
        }
    }
}

/// A single entity: its type plus a dense index into that type's pool
/// (`0..entities_per_type`). Renders as a short label (`p007`, `c012`, ...)
/// via [`fmt::Display`], and parses back via [`EntityId::parse`] -- the two
/// are exact inverses, which is what lets [`super::replay_trace`] read a
/// model's emitted trace text back into typed entities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityId {
    pub ty: EntityType,
    pub index: u32,
}

impl EntityId {
    /// The rendered label (identical to `to_string()`; provided for callers
    /// that want a `String` without going through `Display`/`ToString`).
    pub fn label(&self) -> String {
        self.to_string()
    }

    /// Parse a label like `"p007"` back into a typed [`EntityId`]. Total and
    /// side-effect-free: returns `None` (never panics) for anything that
    /// isn't `<type-prefix><digits>`, which is exactly the shape
    /// [`super::replay_trace`] needs to treat unrecognised text as a
    /// malformed statement rather than a crash.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        let mut chars = s.chars();
        let prefix = chars.next()?;
        let ty = EntityType::from_prefix(prefix)?;
        let rest: String = chars.collect();
        if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let index: u32 = rest.parse().ok()?;
        Some(EntityId { ty, index })
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{:03}", self.ty.prefix(), self.index)
    }
}

/// One triple in the synthetic KG: `subject <relation> object`. Renders as a
/// prompt line via [`Fact::to_line`] (`"fact <subj> <rel> <obj>"`, per ADR
/// 092).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Fact {
    pub relation: RelationKind,
    pub subject: EntityId,
    pub object: EntityId,
}

impl Fact {
    pub fn to_line(&self) -> String {
        format!(
            "fact {} {} {}",
            self.subject,
            self.relation.word(),
            self.object
        )
    }
}

/// The fixed bank of 15 functional relations. Every [`EntityType`] has at
/// least one relation with that source type (checked by the
/// `every_type_has_an_outgoing_relation` test), which is what guarantees a
/// [`random_path`] walk can be extended to any depth from any starting type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationKind {
    LivesIn,
    WorksFor,
    Manages,
    ReportsTo,
    PartOf,
    Neighbors,
    InCountry,
    CapitalCity,
    BasedIn,
    Supplies,
    RegisteredIn,
    Employs,
    StoredIn,
    DeployedIn,
    OwnedBy,
}

impl RelationKind {
    /// Every relation, in a fixed (arbitrary but stable) order.
    pub const ALL: [RelationKind; 15] = [
        RelationKind::LivesIn,
        RelationKind::WorksFor,
        RelationKind::Manages,
        RelationKind::ReportsTo,
        RelationKind::PartOf,
        RelationKind::Neighbors,
        RelationKind::InCountry,
        RelationKind::CapitalCity,
        RelationKind::BasedIn,
        RelationKind::Supplies,
        RelationKind::RegisteredIn,
        RelationKind::Employs,
        RelationKind::StoredIn,
        RelationKind::DeployedIn,
        RelationKind::OwnedBy,
    ];

    /// The relation's word form, exactly as it appears in `fact`/`recall`/
    /// `check` trace lines (e.g. `"lives_in"`).
    pub fn word(self) -> &'static str {
        match self {
            RelationKind::LivesIn => "lives_in",
            RelationKind::WorksFor => "works_for",
            RelationKind::Manages => "manages",
            RelationKind::ReportsTo => "reports_to",
            RelationKind::PartOf => "part_of",
            RelationKind::Neighbors => "neighbors",
            RelationKind::InCountry => "in_country",
            RelationKind::CapitalCity => "capital_city",
            RelationKind::BasedIn => "based_in",
            RelationKind::Supplies => "supplies",
            RelationKind::RegisteredIn => "registered_in",
            RelationKind::Employs => "employs",
            RelationKind::StoredIn => "stored_in",
            RelationKind::DeployedIn => "deployed_in",
            RelationKind::OwnedBy => "owned_by",
        }
    }

    /// A fixed-width (2-letter) code used to build collision-free composed
    /// query-relation names ([`query_name`]): since every code is exactly 2
    /// bytes and pairwise distinct (checked by the `codes_are_pairwise_distinct`
    /// test), concatenation is uniquely decodable -- two different relation
    /// *sequences* always produce two different query names.
    pub fn code(self) -> &'static str {
        match self {
            RelationKind::LivesIn => "lv",
            RelationKind::WorksFor => "wf",
            RelationKind::Manages => "mg",
            RelationKind::ReportsTo => "rt",
            RelationKind::PartOf => "pf",
            RelationKind::Neighbors => "nb",
            RelationKind::InCountry => "ic",
            RelationKind::CapitalCity => "cc",
            RelationKind::BasedIn => "bi",
            RelationKind::Supplies => "sp",
            RelationKind::RegisteredIn => "rg",
            RelationKind::Employs => "em",
            RelationKind::StoredIn => "si",
            RelationKind::DeployedIn => "dp",
            RelationKind::OwnedBy => "ob",
        }
    }

    pub fn source_type(self) -> EntityType {
        use EntityType::*;
        match self {
            RelationKind::LivesIn
            | RelationKind::WorksFor
            | RelationKind::Manages
            | RelationKind::ReportsTo => Person,
            RelationKind::PartOf | RelationKind::Neighbors => City,
            RelationKind::InCountry => Region,
            RelationKind::CapitalCity => Country,
            RelationKind::BasedIn
            | RelationKind::Supplies
            | RelationKind::RegisteredIn
            | RelationKind::Employs => Org,
            RelationKind::StoredIn | RelationKind::DeployedIn | RelationKind::OwnedBy => Device,
        }
    }

    pub fn target_type(self) -> EntityType {
        use EntityType::*;
        match self {
            RelationKind::LivesIn => City,
            RelationKind::WorksFor => Org,
            RelationKind::Manages => Device,
            RelationKind::ReportsTo => Person,
            RelationKind::PartOf => Region,
            RelationKind::Neighbors => City,
            RelationKind::InCountry => Country,
            RelationKind::CapitalCity => City,
            RelationKind::BasedIn => City,
            RelationKind::Supplies => Org,
            RelationKind::RegisteredIn => Country,
            RelationKind::Employs => Person,
            RelationKind::StoredIn => City,
            RelationKind::DeployedIn => Region,
            RelationKind::OwnedBy => Org,
        }
    }

    /// Parse a relation's word form (the exact inverse of [`RelationKind::word`]).
    /// Total: returns `None` for anything not in the bank.
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|r| r.word() == s)
    }

    /// Every relation whose source type is `ty`, in bank order.
    pub fn of_source(ty: EntityType) -> Vec<RelationKind> {
        Self::ALL
            .iter()
            .copied()
            .filter(|r| r.source_type() == ty)
            .collect()
    }
}

/// A deterministic mixing function turning a base `seed` and a
/// [`RelationKind`] into a distinct per-relation RNG seed, so every
/// relation's fact assignment is an independent stream (no two relations
/// share their random draws, and re-generating the KG at the same `seed`
/// reproduces byte-identical assignments).
fn relation_seed(seed: u64, r: RelationKind) -> u64 {
    seed ^ (r as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xD1B5_4A32_D192_ED03)
}

/// The synthetic KG: for every [`RelationKind`], a total functional map from
/// each entity of the relation's source type to one entity of its target
/// type, assigned deterministically from a seed.
pub struct SyntheticKg {
    entities_per_type: usize,
    maps: HashMap<RelationKind, Vec<u32>>,
}

impl SyntheticKg {
    /// Build the KG: `entities_per_type` entities per [`EntityType`], every
    /// relation's fact assignment drawn from an independent seeded stream.
    pub fn generate(seed: u64, entities_per_type: usize) -> Self {
        let mut maps = HashMap::with_capacity(RelationKind::ALL.len());
        for &r in RelationKind::ALL.iter() {
            let mut rng = StdRng::seed_from_u64(relation_seed(seed, r));
            let objs: Vec<u32> = (0..entities_per_type)
                .map(|_| rng.gen_range(0..entities_per_type.max(1) as u32))
                .collect();
            maps.insert(r, objs);
        }
        Self {
            entities_per_type,
            maps,
        }
    }

    pub fn entities_per_type(&self) -> usize {
        self.entities_per_type
    }

    /// The object `relation` assigns to `subject`, or `None` if `subject`'s
    /// type doesn't match `relation.source_type()` (a type-incompatible
    /// lookup, not an error -- total by design).
    pub fn lookup(&self, relation: RelationKind, subject: EntityId) -> Option<EntityId> {
        if subject.ty != relation.source_type() {
            return None;
        }
        let objs = self.maps.get(&relation)?;
        let obj_idx = *objs.get(subject.index as usize)?;
        Some(EntityId {
            ty: relation.target_type(),
            index: obj_idx,
        })
    }

    /// [`SyntheticKg::lookup`], wrapped as the resulting [`Fact`].
    pub fn fact_for(&self, relation: RelationKind, subject: EntityId) -> Option<Fact> {
        self.lookup(relation, subject).map(|object| Fact {
            relation,
            subject,
            object,
        })
    }

    /// Follow a type-compatible relation path from `start`, returning the
    /// chain of facts hop-by-hop. `None` only if `path` is not type-compatible
    /// with `start` (a path built by [`random_path`] is always compatible).
    pub fn follow_chain(&self, start: EntityId, path: &[RelationKind]) -> Option<Vec<Fact>> {
        let mut cur = start;
        let mut chain = Vec::with_capacity(path.len());
        for &r in path {
            let fact = self.fact_for(r, cur)?;
            cur = fact.object;
            chain.push(fact);
        }
        Some(chain)
    }
}

/// Build a type-compatible relation path of length `depth`, starting from
/// `start`: at each hop, choose uniformly among the relations whose source
/// type matches the current type. Every [`EntityType`] has >= 1 outgoing
/// relation (checked by a test), so this never gets stuck regardless of
/// `depth`.
pub fn random_path(rng: &mut StdRng, start: EntityType, depth: usize) -> Vec<RelationKind> {
    let mut ty = start;
    let mut path = Vec::with_capacity(depth);
    for _ in 0..depth {
        let options = RelationKind::of_source(ty);
        let &r = options
            .choose(rng)
            .expect("every EntityType has >= 1 outgoing relation by construction");
        ty = r.target_type();
        path.push(r);
    }
    path
}

/// The composed query-relation name for a relation path: `"q_"` followed by
/// each relation's 2-letter [`RelationKind::code`], concatenated in order
/// (e.g. `[LivesIn, PartOf]` -> `"q_lvpf"`). Short (token-economical, per ADR
/// 092) and collision-free (fixed-width codes make the concatenation
/// uniquely decodable).
pub fn query_name(path: &[RelationKind]) -> String {
    let mut s = String::with_capacity(2 + path.len() * 2);
    s.push_str("q_");
    for r in path {
        s.push_str(r.code());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_id_label_round_trips_through_parse() {
        let id = EntityId {
            ty: EntityType::Org,
            index: 42,
        };
        assert_eq!(id.label(), "o042");
        assert_eq!(EntityId::parse(&id.label()), Some(id));
    }

    #[test]
    fn entity_id_parse_rejects_garbage() {
        assert_eq!(EntityId::parse(""), None);
        assert_eq!(EntityId::parse("p"), None);
        assert_eq!(EntityId::parse("z007"), None);
        assert_eq!(EntityId::parse("p0a7"), None);
        assert_eq!(EntityId::parse("nonsense"), None);
    }

    #[test]
    fn every_type_has_an_outgoing_relation() {
        for ty in EntityType::ALL {
            assert!(
                !RelationKind::of_source(ty).is_empty(),
                "{ty:?} has no outgoing relation -- random_path would get stuck"
            );
        }
    }

    #[test]
    fn codes_are_pairwise_distinct() {
        let codes: std::collections::HashSet<&str> =
            RelationKind::ALL.iter().map(|r| r.code()).collect();
        assert_eq!(
            codes.len(),
            RelationKind::ALL.len(),
            "relation codes must be pairwise distinct"
        );
        for c in &codes {
            assert_eq!(
                c.len(),
                2,
                "codes must be fixed-width (2 bytes) for collision-free concatenation"
            );
        }
    }

    #[test]
    fn relation_word_round_trips_through_parse() {
        for r in RelationKind::ALL {
            assert_eq!(RelationKind::parse(r.word()), Some(r));
        }
        assert_eq!(RelationKind::parse("not_a_relation"), None);
    }

    #[test]
    fn query_name_is_deterministic_and_distinguishes_different_paths() {
        let a = query_name(&[RelationKind::LivesIn, RelationKind::PartOf]);
        let b = query_name(&[RelationKind::LivesIn, RelationKind::PartOf]);
        assert_eq!(a, b);
        let c = query_name(&[RelationKind::PartOf, RelationKind::LivesIn]);
        assert_ne!(
            a, c,
            "different relation order must produce a different name"
        );
        assert_eq!(a, "q_lvpf");
    }

    #[test]
    fn kg_generation_is_deterministic_for_a_fixed_seed() {
        let kg_a = SyntheticKg::generate(7, 50);
        let kg_b = SyntheticKg::generate(7, 50);
        let subject = EntityId {
            ty: EntityType::Person,
            index: 3,
        };
        assert_eq!(
            kg_a.lookup(RelationKind::LivesIn, subject),
            kg_b.lookup(RelationKind::LivesIn, subject)
        );
    }

    #[test]
    fn lookup_respects_relation_source_type() {
        let kg = SyntheticKg::generate(1, 20);
        // LivesIn is Person -> City; a City subject is type-incompatible.
        let wrong_type_subject = EntityId {
            ty: EntityType::City,
            index: 0,
        };
        assert_eq!(kg.lookup(RelationKind::LivesIn, wrong_type_subject), None);
    }

    #[test]
    fn follow_chain_composes_hop_by_hop() {
        let kg = SyntheticKg::generate(3, 30);
        let mut rng = StdRng::seed_from_u64(9);
        let start = EntityId {
            ty: EntityType::Person,
            index: 5,
        };
        let path = random_path(&mut rng, start.ty, 4);
        let chain = kg.follow_chain(start, &path).unwrap();
        assert_eq!(chain.len(), 4);
        let mut cur = start;
        for fact in &chain {
            assert_eq!(fact.subject, cur);
            assert_eq!(fact.relation.source_type(), cur.ty);
            cur = fact.object;
        }
    }
}
