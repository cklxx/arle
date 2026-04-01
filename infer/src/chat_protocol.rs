//! Shared chat/tool-call protocol helpers used by both the `infer` HTTP layer
//! and the root agent loop.

use serde_json::{Map, Value, json};

/// Structured tool definition used for prompt injection.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    fn prompt_schema(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }
}

/// Structured tool call emitted by the model or embedded in an assistant turn.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }

    fn prompt_payload(&self) -> String {
        serde_json::to_string(&json!({
            "name": self.name,
            "arguments": self.arguments,
        }))
        .expect("tool call serialization")
    }
}

/// Role tags used by the shared ChatML formatter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
    Other(String),
}

impl ChatRole {
    pub fn as_str(&self) -> &str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::Other(role) => role.as_str(),
        }
    }
}

impl From<&str> for ChatRole {
    fn from(role: &str) -> Self {
        match role {
            "system" => Self::System,
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "tool" => Self::Tool,
            other => Self::Other(other.to_string()),
        }
    }
}

/// Shared message shape used for prompt construction.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    pub fn system(content: &str) -> Self {
        Self {
            role: ChatRole::System,
            content: content.to_string(),
            tool_calls: vec![],
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: ChatRole::User,
            content: content.to_string(),
            tool_calls: vec![],
        }
    }

    pub fn assistant(content: &str, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.to_string(),
            tool_calls,
        }
    }

    pub fn tool_result(_tool_name: &str, result: &str) -> Self {
        Self {
            role: ChatRole::Tool,
            content: result.to_string(),
            tool_calls: vec![],
        }
    }
}

/// Parsed assistant output with tool calls stripped from visible content.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedAssistantResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

/// Build the tool definitions block for injection into a system prompt.
pub fn build_tool_block(tools: &[ToolDefinition]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut out = String::from(
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>",
    );

    for tool in tools {
        out.push('\n');
        out.push_str(
            &serde_json::to_string(&tool.prompt_schema()).expect("tool schema serialization"),
        );
    }

    out.push_str("\n</tools>\n\nFor each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
    out
}

/// Convert structured messages + tool definitions into a ChatML prompt.
pub fn messages_to_prompt(messages: &[ChatMessage], tools: &[ToolDefinition]) -> String {
    let mut prompt = String::new();
    let tool_block = build_tool_block(tools);
    let mut system_injected = false;

    for message in messages {
        match &message.role {
            ChatRole::System => {
                prompt.push_str("<|im_start|>system\n");
                prompt.push_str(&message.content);
                if !tool_block.is_empty() {
                    prompt.push_str(&tool_block);
                }
                prompt.push_str("<|im_end|>\n");
                system_injected = true;
            }
            ChatRole::User => {
                if !system_injected && !tool_block.is_empty() {
                    prompt.push_str("<|im_start|>system\n");
                    prompt.push_str("You are a helpful assistant.");
                    prompt.push_str(&tool_block);
                    prompt.push_str("<|im_end|>\n");
                    system_injected = true;
                }
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(&message.content);
                prompt.push_str("<|im_end|>\n");
            }
            ChatRole::Assistant => {
                prompt.push_str("<|im_start|>assistant\n");
                prompt.push_str(&message.content);
                for tool_call in &message.tool_calls {
                    prompt.push_str("\n<tool_call>\n");
                    prompt.push_str(&tool_call.prompt_payload());
                    prompt.push_str("\n</tool_call>");
                }
                prompt.push_str("<|im_end|>\n");
            }
            ChatRole::Tool => {
                prompt.push_str("<|im_start|>tool\n<tool_response>\n");
                prompt.push_str(&message.content);
                prompt.push_str("\n</tool_response><|im_end|>\n");
            }
            ChatRole::Other(role) => {
                prompt.push_str("<|im_start|>");
                prompt.push_str(role);
                prompt.push('\n');
                prompt.push_str(&message.content);
                prompt.push_str("<|im_end|>\n");
            }
        }
    }

    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Parse `<tool_call>...</tool_call>` blocks from raw assistant output.
pub fn parse_tool_calls(text: &str) -> ParsedAssistantResponse {
    let mut tool_calls = Vec::new();
    let mut remaining = text;
    let mut stripped = String::with_capacity(text.len());

    let open_tag = "<tool_call>";
    let close_tag = "</tool_call>";

    while let Some(start) = remaining.find(open_tag) {
        stripped.push_str(&remaining[..start]);
        remaining = &remaining[start + open_tag.len()..];

        if let Some(end) = remaining.find(close_tag) {
            let json_str = remaining[..end].trim();
            if let Ok(value) = serde_json::from_str::<Value>(json_str) {
                let name = value
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let arguments = value
                    .get("arguments")
                    .cloned()
                    .unwrap_or(Value::Object(Map::default()));
                tool_calls.push(ToolCall::new(name, arguments));
            }
            remaining = &remaining[end + close_tag.len()..];
        } else {
            stripped.push_str(remaining);
            remaining = "";
        }
    }

    stripped.push_str(remaining);

    ParsedAssistantResponse {
        content: strip_think_blocks(&stripped).trim().to_string(),
        tool_calls,
    }
}

fn strip_think_blocks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut remaining = text;

    let open_tag = "<think>";
    let close_tag = "</think>";

    while let Some(start) = remaining.find(open_tag) {
        out.push_str(&remaining[..start]);
        remaining = &remaining[start + open_tag.len()..];

        if let Some(end) = remaining.find(close_tag) {
            remaining = &remaining[end + close_tag.len()..];
        } else {
            remaining = "";
        }
    }

    out.push_str(remaining);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_user_message() {
        let prompt = messages_to_prompt(&[ChatMessage::user("hello")], &[]);
        assert!(prompt.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn system_and_user_messages() {
        let prompt = messages_to_prompt(
            &[
                ChatMessage::system("You are helpful."),
                ChatMessage::user("hi"),
            ],
            &[],
        );

        assert!(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nhi<|im_end|>"));
    }

    #[test]
    fn tool_definition_injected_into_system_prompt() {
        let prompt = messages_to_prompt(
            &[ChatMessage::user("list files")],
            &[ToolDefinition::new(
                "shell",
                "Run a shell command",
                json!({}),
            )],
        );

        assert!(prompt.contains("<|im_start|>system\n"));
        assert!(prompt.contains("shell"));
        assert!(prompt.contains("<tools>"));
    }

    #[test]
    fn assistant_tool_calls_render_as_xml_blocks() {
        let prompt = messages_to_prompt(
            &[ChatMessage::assistant(
                "Checking.",
                vec![ToolCall::new("shell", json!({ "command": "pwd" }))],
            )],
            &[],
        );

        assert!(prompt.contains("Checking."));
        assert!(prompt.contains("<tool_call>"));
        assert!(prompt.contains(r#""name":"shell""#));
        assert!(prompt.contains(r#""command":"pwd""#));
    }

    #[test]
    fn parse_tool_call_basic() {
        let parsed = parse_tool_calls(
            "Sure.\n<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"cmd\":\"ls\"}}\n</tool_call>",
        );

        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "shell");
        assert_eq!(parsed.tool_calls[0].arguments["cmd"], "ls");
        assert_eq!(parsed.content, "Sure.");
    }

    #[test]
    fn parse_strips_think_blocks() {
        let parsed = parse_tool_calls("<think>\nI should check.\n</think>\nHere is the answer.");
        assert!(parsed.tool_calls.is_empty());
        assert_eq!(parsed.content, "Here is the answer.");
    }

    #[test]
    fn multi_turn_conversation() {
        let prompt = messages_to_prompt(
            &[
                ChatMessage::user("first"),
                ChatMessage::assistant("response", vec![]),
                ChatMessage::user("second"),
            ],
            &[],
        );

        assert!(prompt.contains("<|im_start|>assistant\nresponse<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }
}
