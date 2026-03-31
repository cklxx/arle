use anyhow::Result;
use log::info;

use infer::sampler::SamplingParams;
use infer::server_engine::{CompleteRequest, ServerEngine};

use crate::chat::{Message, format_prompt, parse_tool_calls};
use crate::tools::{Tool, execute_tool};

const SYSTEM_PROMPT: &str = r#"You are a helpful AI assistant with access to tools. You can use tools to help answer questions, perform calculations, run commands, and more.

When you need to use a tool, output a tool call in the specified format. You may call multiple tools in a single response. After receiving tool results, continue your reasoning and provide a final answer.

Think step by step. If a task requires multiple steps, use tools iteratively. Always verify your work when possible."#;

/// Run the agent loop: generate, parse tool calls, execute, feed results, repeat.
///
/// Returns the final assistant response (after all tool calls are resolved or max_turns hit).
pub fn run_agent(
    engine: &mut dyn ServerEngine,
    user_input: &str,
    tools: &[Tool],
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
) -> Result<String> {
    let mut messages = vec![Message::system(SYSTEM_PROMPT), Message::user(user_input)];

    for turn in 0..max_turns {
        let prompt = format_prompt(&messages, tools);
        info!(
            "Agent turn {}/{}: prompt length = {} chars",
            turn + 1,
            max_turns,
            prompt.len()
        );

        let output = engine.complete(CompleteRequest {
            prompt,
            max_tokens,
            sampling: SamplingParams {
                temperature,
                ..SamplingParams::default()
            },
            stop: Some(vec!["<|im_end|>".to_string()]),
        })?;

        info!(
            "Generated {} chars, finish_reason={:?}",
            output.text.len(),
            output.finish_reason
        );

        let (content, tool_calls) = parse_tool_calls(&output.text);

        if tool_calls.is_empty() {
            // No tool calls -- this is the final response
            // Still add it to messages for completeness
            messages.push(Message::assistant(&output.text, vec![]));
            return Ok(content);
        }

        // Print assistant's reasoning/content if any
        if !content.is_empty() {
            println!("\x1b[2m{}\x1b[0m", content);
        }

        // Show and execute tool calls
        println!();
        for tc in &tool_calls {
            println!(
                "\x1b[33m[tool: {}]\x1b[0m {}",
                tc.name,
                serde_json::to_string(&tc.arguments).unwrap_or_default()
            );

            let result = execute_tool(&tc.name, &tc.arguments);

            // Show truncated result
            let display_result = if result.len() > 500 {
                format!("{}... ({} chars total)", &result[..500], result.len())
            } else {
                result.clone()
            };
            println!("\x1b[36m{}\x1b[0m", display_result);
            println!();

            // Add tool result to conversation
            // We add the assistant message with tool calls first, then each tool result
            // But we need to add them in order, so we collect and add after the loop
            // Actually, we add the assistant message once, then all tool results
            messages.push(Message::tool_result(&tc.name, &result));
        }

        // Insert the assistant message with tool calls before the tool results
        let tool_results_count = tool_calls.len();
        let assistant_msg = Message::assistant(&output.text, tool_calls);
        let insert_pos = messages.len() - tool_results_count;
        messages.insert(insert_pos, assistant_msg);
    }

    // If we exhausted max_turns, return whatever we have
    Ok("(max turns reached - agent stopped)".to_string())
}
