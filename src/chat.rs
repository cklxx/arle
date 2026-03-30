use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::tools::Tool;

// ============================================================================
// Message types
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// For assistant messages that contain tool calls
    pub tool_calls: Vec<ToolCall>,
    /// For tool-result messages: which tool produced this
    #[allow(dead_code)]
    pub tool_name: Option<String>,
}

impl Message {
    pub fn system(content: &str) -> Self {
        Self {
            role: Role::System,
            content: content.to_string(),
            tool_calls: vec![],
            tool_name: None,
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: Role::User,
            content: content.to_string(),
            tool_calls: vec![],
            tool_name: None,
        }
    }

    pub fn assistant(content: &str, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.to_string(),
            tool_calls,
            tool_name: None,
        }
    }

    pub fn tool_result(tool_name: &str, result: &str) -> Self {
        Self {
            role: Role::Tool,
            content: result.to_string(),
            tool_calls: vec![],
            tool_name: Some(tool_name.to_string()),
        }
    }
}

// ============================================================================
// ChatML formatting
// ============================================================================

/// Build the tool definitions block for injection into the system prompt,
/// using the Qwen3 format.
fn format_tool_definitions(tools: &[Tool]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut out = String::from(
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>",
    );

    for tool in tools {
        let schema = serde_json::json!({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        });
        out.push('\n');
        out.push_str(&serde_json::to_string(&schema).expect("tool schema serialization"));
    }

    out.push_str("\n</tools>\n\nFor each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");

    out
}

/// Format a conversation into a ChatML prompt string for Qwen3.
pub fn format_prompt(messages: &[Message], tools: &[Tool]) -> String {
    let mut prompt = String::new();
    let tool_defs = format_tool_definitions(tools);

    for msg in messages {
        match msg.role {
            Role::System => {
                prompt.push_str("<|im_start|>system\n");
                prompt.push_str(&msg.content);
                prompt.push_str(&tool_defs);
                prompt.push_str("<|im_end|>\n");
            }
            Role::User => {
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<|im_end|>\n");
            }
            Role::Assistant => {
                prompt.push_str("<|im_start|>assistant\n");
                prompt.push_str(&msg.content);
                // Render any tool calls inline
                for tc in &msg.tool_calls {
                    prompt.push_str("\n<tool_call>\n");
                    let call_json = serde_json::json!({
                        "name": tc.name,
                        "arguments": tc.arguments,
                    });
                    prompt.push_str(
                        &serde_json::to_string(&call_json).expect("tool call serialization"),
                    );
                    prompt.push_str("\n</tool_call>");
                }
                prompt.push_str("<|im_end|>\n");
            }
            Role::Tool => {
                prompt.push_str("<|im_start|>tool\n<tool_response>\n");
                prompt.push_str(&msg.content);
                prompt.push_str("\n</tool_response><|im_end|>\n");
            }
        }
    }

    // Open the assistant turn for the model to generate into
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

// ============================================================================
// Tool call parsing
// ============================================================================

/// Parse tool calls from model output.
/// Returns (display_text, tool_calls).
/// `display_text` is the content with <think> blocks and <tool_call> blocks removed.
pub fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let tool_re = Regex::new(r"(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>").unwrap();

    let mut tool_calls = Vec::new();
    for cap in tool_re.captures_iter(text) {
        if let Some(json_str) = cap.get(1) {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str.as_str()) {
                let name = parsed
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let arguments = parsed
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                tool_calls.push(ToolCall { name, arguments });
            }
        }
    }

    // Build display text: strip <tool_call>...</tool_call> blocks
    let display = tool_re.replace_all(text, "");
    // Strip <think>...</think> blocks for display
    let think_re = Regex::new(r"(?s)<think>.*?</think>").unwrap();
    let display = think_re.replace_all(&display, "");
    let display = display.trim().to_string();

    (display, tool_calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_calls_basic() {
        let text = r#"Let me check that.
<tool_call>
{"name": "shell", "arguments": {"command": "ls -la"}}
</tool_call>"#;
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "ls -la");
        assert_eq!(content, "Let me check that.");
    }

    #[test]
    fn test_parse_with_think_block() {
        let text = r#"<think>
I should run a command.
</think>
Here is the result.
<tool_call>
{"name": "shell", "arguments": {"command": "echo hello"}}
</tool_call>"#;
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert!(!content.contains("<think>"));
        assert!(content.contains("Here is the result."));
    }

    #[test]
    fn test_no_tool_calls() {
        let text = "Just a plain response with no tools.";
        let (content, calls) = parse_tool_calls(text);
        assert!(calls.is_empty());
        assert_eq!(content, text);
    }
}
