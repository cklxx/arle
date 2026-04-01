use serde::{Deserialize, Serialize};
use serde_json::json;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use infer::chat_protocol::{ToolCall as ProtocolToolCall, ToolDefinition};

static SCRIPT_COUNTER: AtomicU64 = AtomicU64::new(0);

// ============================================================================
// Sandbox configuration
// ============================================================================

/// Lightweight process sandbox using Linux unshare + timeout + prlimit.
/// Provides: PID namespace isolation, time limit, memory limit, clean env.
struct SandboxConfig {
    /// Maximum wall-clock time in seconds before SIGKILL.
    timeout_secs: u64,
    /// Maximum virtual memory in bytes (0 = unlimited).
    max_memory_bytes: u64,
    /// Working directory for the sandboxed process.
    workdir: String,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 30,
            max_memory_bytes: 512 * 1024 * 1024, // 512 MB
            workdir: std::env::temp_dir()
                .to_string_lossy()
                .into_owned(),
        }
    }
}

impl SandboxConfig {
    /// Build a sandboxed Command that wraps the given shell command string.
    ///
    /// Layers (outermost → innermost):
    ///   timeout → prlimit → unshare --user --pid --fork → bash -c <cmd>
    fn wrap_shell(&self, user_cmd: &str) -> Command {
        let mut cmd = Command::new("timeout");
        cmd.arg("--signal=KILL")
            .arg(format!("{}s", self.timeout_secs));

        // prlimit for memory cap
        if self.max_memory_bytes > 0 {
            cmd.arg("prlimit")
                .arg(format!("--as={}", self.max_memory_bytes));
        }

        // PID namespace isolation (user ns required when non-root effective caps are limited)
        cmd.arg("unshare")
            .arg("--user")
            .arg("--pid")
            .arg("--fork");

        cmd.arg("bash").arg("-c").arg(user_cmd);
        cmd.current_dir(&self.workdir);

        // Minimal environment
        cmd.env_clear();
        cmd.env("PATH", "/usr/local/bin:/usr/bin:/bin");
        cmd.env("HOME", &self.workdir);
        cmd.env("TMPDIR", &self.workdir);
        cmd.env("LANG", "C.UTF-8");

        cmd
    }

    /// Build a sandboxed Command for running a Python snippet.
    /// Writes code to a temp file to avoid shell quoting issues.
    fn wrap_python(&self, code: &str) -> std::io::Result<Command> {
        let seq = SCRIPT_COUNTER.fetch_add(1, Ordering::Relaxed);
        let script_path = format!(
            "{}/sandbox_py_{}_{}.py",
            self.workdir,
            std::process::id(),
            seq
        );
        std::fs::write(&script_path, code)?;
        let shell_cmd = format!("python3 -u {} ; rm -f {}", script_path, script_path);
        Ok(self.wrap_shell(&shell_cmd))
    }
}

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

/// Collect stdout + stderr from a process Output into a truncated string.
fn collect_output(output: std::process::Output) -> String {
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

fn execute_shell(command: &str) -> String {
    log::info!("Executing shell (sandboxed): {}", command);
    let sandbox = SandboxConfig::default();
    let mut cmd = sandbox.wrap_shell(command);
    match cmd.output() {
        Ok(output) => {
            if output.status.code() == Some(137) {
                return "Error: command killed (timeout or OOM)".to_string();
            }
            collect_output(output)
        }
        Err(e) => format!("Error executing command: {e}"),
    }
}

fn execute_python(code: &str) -> String {
    log::info!("Executing python snippet (sandboxed, {} chars)", code.len());
    let sandbox = SandboxConfig::default();
    let mut cmd = match sandbox.wrap_python(code) {
        Ok(c) => c,
        Err(e) => return format!("Error preparing python sandbox: {e}"),
    };
    match cmd.output() {
        Ok(output) => {
            if output.status.code() == Some(137) {
                return "Error: python killed (timeout or OOM)".to_string();
            }
            collect_output(output)
        }
        Err(e) => format!("Error executing python: {e}"),
    }
}
