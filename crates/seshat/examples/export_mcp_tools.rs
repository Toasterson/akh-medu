//! Print Seshat's live macro-generated MCP tool descriptors as JSON.
//!
//! No database connection is required: the descriptors come from the same
//! ToolRouter construction used by seshd.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tools = seshat::mcp::SeshatMcpServer::tool_definitions();
    serde_json::to_writer_pretty(std::io::stdout().lock(), &tools)?;
    println!();
    Ok(())
}
