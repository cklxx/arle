//! ChatML message formatting for the `/v1/chat/completions` endpoint.
//!
//! Converts OpenAI-style message arrays into a single ChatML prompt string
//! that the model expects as input.

use serde::{Deserialize, Serialize};

// ============================================================================
// Message types (OpenAI-compatible)
// ============================================================================

/// A single message in a chat conversation.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    /// Text content. `None` when the assistant message contains only tool calls.
    #[serde(default)]
    pub content: Option<String>,
    /// Tool calls emitted by the assistant.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallObject>,
    /// Present on `tool` role messages — the call id being responded to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Present on `tool` role messages — the tool name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// OpenAI-format tool call object embedded in an assistant message.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallObject {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

/// Function name + JSON-encoded arguments.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCall {
    pub name: String,
    /// JSON string (not parsed object) — matches OpenAI wire format.
    pub arguments: String,
}

/// Tool definition passed in a chat completion request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<serde_json::Value>,
}

// ============================================================================
// ChatML formatting
// ============================================================================

/// Convert an OpenAI messages array + optional tool definitions into a
/// ChatML prompt string ready for inference.
///
/// Output format (Qwen / ChatML):
/// ```text
/// <|im_start|>system
/// {content}<|im_end|>
/// <|im_start|>user
/// {content}<|im_end|>
/// <|im_start|>assistant
/// ```
///
/// The trailing `<|im_start|>assistant\n` is the open assistant turn that
/// the model will continue generating into.
pub fn messages_to_prompt(messages: &[ChatMessage], tools: &[ToolDefinition]) -> String {
    let mut prompt = String::new();

    // Inject tool definitions into the system prompt (or prepend a synthetic one).
    let tool_block = build_tool_block(tools);
    let mut system_injected = false;

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                let content = msg.content.as_deref().unwrap_or("");
                prompt.push_str("<|im_start|>system\n");
                prompt.push_str(content);
                if !tool_block.is_empty() {
                    prompt.push_str(&tool_block);
                }
                prompt.push_str("<|im_end|>\n");
                system_injected = true;
            }
            "user" => {
                // If there was no system message yet but we have tools, prepend synthetic system.
                if !system_injected && !tool_block.is_empty() {
                    prompt.push_str("<|im_start|>system\n");
                    prompt.push_str("You are a helpful assistant.");
                    prompt.push_str(&tool_block);
                    prompt.push_str("<|im_end|>\n");
                    system_injected = true;
                }
                let content = msg.content.as_deref().unwrap_or("");
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(content);
                prompt.push_str("<|im_end|>\n");
            }
            "assistant" => {
                prompt.push_str("<|im_start|>assistant\n");
                if let Some(content) = &msg.content {
                    prompt.push_str(content);
                }
                // Re-serialize any tool calls back into <tool_call> XML format.
                for tc in &msg.tool_calls {
                    let call_json = serde_json::json!({
                        "name": tc.function.name,
                        "arguments": serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                            .unwrap_or(serde_json::Value::String(tc.function.arguments.clone())),
                    });
                    prompt.push_str("\n<tool_call>\n");
                    prompt.push_str(&serde_json::to_string(&call_json).unwrap_or_default());
                    prompt.push_str("\n</tool_call>");
                }
                prompt.push_str("<|im_end|>\n");
            }
            "tool" => {
                // Tool result message.
                let content = msg.content.as_deref().unwrap_or("");
                prompt.push_str("<|im_start|>tool\n<tool_response>\n");
                prompt.push_str(content);
                prompt.push_str("\n</tool_response><|im_end|>\n");
            }
            other => {
                // Unknown role — pass through as user turn to avoid dropping context.
                let content = msg.content.as_deref().unwrap_or("");
                prompt.push_str(&format!("<|im_start|>{other}\n"));
                prompt.push_str(content);
                prompt.push_str("<|im_end|>\n");
            }
        }
    }

    // Open the assistant turn for generation.
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Build the tool definitions block in Qwen3 format.
fn build_tool_block(tools: &[ToolDefinition]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut block = String::from(
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
    );

    for tool in tools {
        let schema = serde_json::json!({
            "type": "function",
            "function": {
                "name": tool.function.name,
                "description": tool.function.description.as_deref().unwrap_or(""),
                "parameters": tool.function.parameters.clone().unwrap_or(serde_json::json!({})),
            }
        });
        block.push('\n');
        block.push_str(&serde_json::to_string(&schema).unwrap_or_default());
    }

    block.push_str(
        "\n</tools>\n\nFor each function call, return a JSON object with function name and \
         arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
         {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
    );

    block
}

// ============================================================================
// Tool call parsing (for building structured chat completion responses)
// ============================================================================

/// A parsed tool call extracted from raw model output.
#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Parse `<tool_call>...</tool_call>` blocks from model output.
/// Returns `(visible_content, tool_calls)` where `visible_content` has the
/// tool call blocks and `<think>` blocks stripped.
pub fn parse_tool_calls(text: &str) -> (String, Vec<ParsedToolCall>) {
    let mut tool_calls = Vec::new();
    let mut remaining = text;

    // Simple hand-rolled parser to avoid regex dependency in the library crate.
    let open_tag = "<tool_call>";
    let close_tag = "</tool_call>";
    let mut stripped = String::with_capacity(text.len());

    while let Some(start) = remaining.find(open_tag) {
        stripped.push_str(&remaining[..start]);
        remaining = &remaining[start + open_tag.len()..];
        if let Some(end) = remaining.find(close_tag) {
            let json_str = remaining[..end].trim();
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                let name = val
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let arguments = val
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                tool_calls.push(ParsedToolCall { name, arguments });
            }
            remaining = &remaining[end + close_tag.len()..];
        } else {
            // Unclosed tag — treat rest as content.
            stripped.push_str(remaining);
            remaining = "";
        }
    }
    stripped.push_str(remaining);

    // Strip <think>...</think> blocks from visible content.
    let visible = strip_think_blocks(&stripped);

    (visible.trim().to_string(), tool_calls)
}

fn strip_think_blocks(text: &str) -> String {
    let open = "<think>";
    let close = "</think>";
    let mut out = String::with_capacity(text.len());
    let mut remaining = text;

    while let Some(start) = remaining.find(open) {
        out.push_str(&remaining[..start]);
        remaining = &remaining[start + open.len()..];
        if let Some(end) = remaining.find(close) {
            remaining = &remaining[end + close.len()..];
        } else {
            // Unclosed <think> — drop rest.
            remaining = "";
        }
    }
    out.push_str(remaining);
    out
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_user_message() {
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: Some("hello".into()),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }];
        let prompt = messages_to_prompt(&msgs, &[]);
        assert!(prompt.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn system_and_user() {
        let msgs = vec![
            ChatMessage {
                role: "system".into(),
                content: Some("You are helpful.".into()),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
            },
        ];
        let prompt = messages_to_prompt(&msgs, &[]);
        assert!(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nhi<|im_end|>"));
    }

    #[test]
    fn tool_definition_injected_into_system() {
        let tools = vec![ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: "shell".into(),
                description: Some("Run a shell command".into()),
                parameters: None,
            },
        }];
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: Some("list files".into()),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }];
        let prompt = messages_to_prompt(&msgs, &tools);
        // A synthetic system message should be prepended.
        assert!(prompt.contains("<|im_start|>system\n"));
        assert!(prompt.contains("shell"));
        assert!(prompt.contains("<tools>"));
    }

    #[test]
    fn parse_tool_call_basic() {
        let text = "Sure.\n<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"cmd\":\"ls\"}}\n</tool_call>";
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["cmd"], "ls");
        assert_eq!(content, "Sure.");
    }

    #[test]
    fn parse_strips_think_blocks() {
        let text = "<think>\nI should check.\n</think>\nHere is the answer.";
        let (content, calls) = parse_tool_calls(text);
        assert!(calls.is_empty());
        assert_eq!(content, "Here is the answer.");
    }

    #[test]
    fn multi_turn_conversation() {
        let msgs = vec![
            ChatMessage {
                role: "user".into(),
                content: Some("first".into()),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: Some("response".into()),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "user".into(),
                content: Some("second".into()),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
            },
        ];
        let prompt = messages_to_prompt(&msgs, &[]);
        assert!(prompt.contains("<|im_start|>assistant\nresponse<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }
}
