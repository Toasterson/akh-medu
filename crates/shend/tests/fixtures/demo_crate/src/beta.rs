/// The beta module's `make` — the decoy for the WI-262 test: a resolver that
/// matched on name alone cannot tell this from `alpha::make`.
pub fn make() -> String {
    "beta".to_string()
}
