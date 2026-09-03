/// The alpha module's own `make` — distinguishable from `beta::make` only by
/// binding, not by name. The WI-262 disambiguation test hinges on this.
pub fn make() -> String {
    "alpha".to_string()
}

/// Internal reference inside the defining file (same-file binding is the
/// syntactic-tier case).
pub fn make_twice() -> String {
    let one = make();
    let two = make();
    format!("{one}{two}")
}
