use serde::{Deserialize, Serialize};
use serde_json::json;
use std::process::Command;

use infer::chat_protocol::{ToolCall as ProtocolToolCall, ToolDefinition};

// ============================================================================
// Tool definition
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

impl Tool {
    pub fn to_definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name.clone(),
            self.description.clone(),
            self.parameters.clone(),
        )
    }
}

/// Return the built-in tool definitions.
pub fn builtin_tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "shell".to_string(),
            description: "Execute a shell command and return its stdout and stderr. Use this for file operations, system inspection, running programs, etc.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    }
                },
                "required": ["command"]
            }),
        },
        Tool {
            name: "python".to_string(),
            description: "Execute a Python 3 code snippet and return its stdout and stderr. Use this for calculations, data processing, or any task best done in Python.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python 3 code to execute"
                    }
                },
                "required": ["code"]
            }),
        },
    ]
}

// ============================================================================
// Tool execution
// ============================================================================

/// Execute a tool by name with the given JSON arguments.
/// Returns the tool output as a string.
pub fn execute_tool(name: &str, arguments: &serde_json::Value) -> String {
    match name {
        "shell" => {
            let command = arguments
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            execute_shell(command)
        }
        "python" => {
            let code = arguments.get("code").and_then(|v| v.as_str()).unwrap_or("");
            execute_python(code)
        }
        _ => format!("Error: unknown tool '{name}'"),
    }
}

/// Execute a structured tool call.
pub fn execute_tool_call(call: &ProtocolToolCall) -> String {
    execute_tool(&call.name, &call.arguments)
}

fn execute_shell(command: &str) -> String {
    log::info!("Executing shell: {}", command);
    match Command::new("bash").arg("-c").arg(command).output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let mut result = String::new();
            if !stdout.is_empty() {
                result.push_str(&stdout);
            }
            if !stderr.is_empty() {
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str("[stderr] ");
                result.push_str(&stderr);
            }
            if result.is_empty() {
                result.push_str("(no output)");
            }
            // Truncate very long output
            if result.len() > 8000 {
                result.truncate(8000);
                result.push_str("\n... (output truncated)");
            }
            result
        }
        Err(e) => format!("Error executing command: {e}"),
    }
}

fn execute_python(code: &str) -> String {
    log::info!("Executing python snippet ({} chars)", code.len());
    match Command::new("python3").arg("-c").arg(code).output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let mut result = String::new();
            if !stdout.is_empty() {
                result.push_str(&stdout);
            }
            if !stderr.is_empty() {
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str("[stderr] ");
                result.push_str(&stderr);
            }
            if result.is_empty() {
                result.push_str("(no output)");
            }
            if result.len() > 8000 {
                result.truncate(8000);
                result.push_str("\n... (output truncated)");
            }
            result
        }
        Err(e) => format!("Error executing python: {e}"),
    }
}
