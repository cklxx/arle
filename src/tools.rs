use serde::{Deserialize, Serialize};
use serde_json::json;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use infer::chat_protocol::{ToolCall as ProtocolToolCall, ToolDefinition};

static SCRIPT_COUNTER: AtomicU64 = AtomicU64::new(0);

// ============================================================================
// Sandbox configuration — nsjail backend
// ============================================================================

/// Process sandbox using nsjail.
///
/// Provides:
///   - Mount namespace: rootfs is read-only bind mounts, only /tmp is writable
///   - PID namespace: isolated PID tree
///   - Network isolation: no network access by default
///   - Time limit: SIGKILL after `timeout_secs`
///   - Memory limit: RLIMIT_AS capped at `max_memory_mb`
///   - Minimal environment: only PATH, HOME, TMPDIR, LANG
///
/// Falls back to bare execution if nsjail is not found.
struct SandboxConfig {
    timeout_secs: u64,
    max_memory_mb: u64,
    workdir: String,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 30,
            max_memory_mb: 512,
            workdir: std::env::temp_dir().to_string_lossy().into_owned(),
        }
    }
}

/// Check once whether nsjail is available.
fn nsjail_available() -> bool {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        Command::new("nsjail")
            .arg("--help")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok()
    })
}

impl SandboxConfig {
    /// Build a Command that runs `user_cmd` inside nsjail.
    fn wrap_shell(&self, user_cmd: &str) -> Command {
        if !nsjail_available() {
            log::warn!("nsjail not found — running without sandbox");
            return self.wrap_shell_bare(user_cmd);
        }

        let mut cmd = Command::new("nsjail");
        cmd.arg("--mode").arg("o");
        cmd.arg("--time_limit").arg(self.timeout_secs.to_string());
        cmd.arg("--rlimit_as").arg(self.max_memory_mb.to_string());
        cmd.arg("--quiet");
        cmd.arg("--disable_proc");

        // Read-only bind mounts for system dirs
        for dir in &["/bin", "/lib", "/lib64", "/usr", "/etc", "/dev/null", "/dev/urandom"] {
            if std::path::Path::new(dir).exists() {
                cmd.arg("-R").arg(dir);
            }
        }

        // Read-only bind for /usr/local (python, pip packages, cuda libs)
        if std::path::Path::new("/usr/local").exists() {
            cmd.arg("-R").arg("/usr/local");
        }

        // Writable /tmp
        cmd.arg("-B").arg(&self.workdir);
        // Also bind /tmp if workdir is different
        if self.workdir != "/tmp" {
            cmd.arg("-B").arg("/tmp");
        }

        cmd.arg("--cwd").arg(&self.workdir);

        // Environment
        cmd.arg("--env").arg("PATH=/usr/local/bin:/usr/bin:/bin");
        cmd.arg("--env").arg(format!("HOME={}", self.workdir));
        cmd.arg("--env").arg(format!("TMPDIR={}", self.workdir));
        cmd.arg("--env").arg("LANG=C.UTF-8");
        cmd.arg("--env").arg("PYTHONDONTWRITEBYTECODE=1");

        cmd.arg("--").arg("/bin/bash").arg("-c").arg(user_cmd);
        cmd
    }

    /// Bare fallback when nsjail is unavailable.
    fn wrap_shell_bare(&self, user_cmd: &str) -> Command {
        let mut cmd = Command::new("timeout");
        cmd.arg("--signal=KILL")
            .arg(format!("{}s", self.timeout_secs));
        cmd.arg("bash").arg("-c").arg(user_cmd);
        cmd.current_dir(&self.workdir);
        cmd.env_clear();
        cmd.env("PATH", "/usr/local/bin:/usr/bin:/bin");
        cmd.env("HOME", &self.workdir);
        cmd.env("TMPDIR", &self.workdir);
        cmd.env("LANG", "C.UTF-8");
        cmd
    }

    /// Build a sandboxed Command for running a Python snippet.
    fn wrap_python(&self, code: &str) -> std::io::Result<Command> {
        let seq = SCRIPT_COUNTER.fetch_add(1, Ordering::Relaxed);
        let script_path = format!(
            "{}/sandbox_py_{}_{}.py",
            self.workdir,
            std::process::id(),
            seq
        );
        std::fs::write(&script_path, code)?;
        let shell_cmd = format!(
            "python3 -u {} ; _rc=$?; rm -f {} ; exit $_rc",
            script_path, script_path
        );
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
/// Filters out nsjail's own warning/info lines from stderr.
fn collect_output(output: std::process::Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let raw_stderr = String::from_utf8_lossy(&output.stderr);

    // Filter nsjail's own log lines (e.g. "[W][...] logParams()...")
    let stderr: String = raw_stderr
        .lines()
        .filter(|line| !line.starts_with("[W][") && !line.starts_with("[I][") && !line.starts_with("[D]["))
        .collect::<Vec<_>>()
        .join("\n");

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
    log::info!("Executing shell (nsjail): {}", command);
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
    log::info!("Executing python snippet (nsjail, {} chars)", code.len());
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
