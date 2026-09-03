use crate::alpha::make;

pub fn run() -> String {
    make()
}

/// References a name that exists nowhere in the crate — the WI-262 explicit-
/// unresolved case: it must be recorded unresolved, never guessed or omitted.
pub fn missing() {
    nonexistent_function();
}
