use async_openai::{config::OpenAIConfig, Client};
use clap::Parser;
use serde_json::{json, Value};
use std::{env, fs, path::Path, process::{self, Command}};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let client = Client::with_config(config);

    // Parse tokens from the prompt to extract stacked skills and trailing arguments
    let mut tokens = args.prompt.split_whitespace().peekable();
    let mut expanded_skills = Vec::new();

    while let Some(&token) = tokens.peek() {
        if token.starts_with('/') {
            let candidate_skill = &token[1..];
            let skill_path = format!(".claude/skills/{}/SKILL.md", candidate_skill);

            // If a valid skill file exists, load its body and consume the token
            if Path::new(&skill_path).exists() {
                if let Ok(file_content) = fs::read_to_string(&skill_path) {
                    let body = if file_content.starts_with("---") {
                        file_content
                            .split("---")
                            .nth(2)
                            .unwrap_or(&file_content)
                            .trim()
                            .to_string()
                    } else {
                        file_content.trim().to_string()
                    };
                    expanded_skills.push(body);
                    tokens.next();
                    continue;
                }
            }
        }
        // Stop expansion on the first token that is not a valid skill
        break;
    }

    let remaining_args: Vec<&str> = tokens.collect();
    let full_args_str = remaining_args.join(" ");

    let mut messages: Vec<Value> = Vec::new();

    if !expanded_skills.is_empty() {
        // Substitute arguments into each expanded skill body and push as separate user messages
        for body in expanded_skills {
            let mut substituted_body = body.replace("$ARGUMENTS", &full_args_str);

            for (idx, arg_val) in remaining_args.iter().enumerate() {
                let target_full = format!("$ARGUMENTS[{}]", idx);
                let target_short = format!("${}", idx);
                substituted_body = substituted_body.replace(&target_full, arg_val);
                substituted_body = substituted_body.replace(&target_short, arg_val);
            }

            messages.push(json!({
                "role": "user",
                "content": substituted_body
            }));
        }
    } else {
        messages.push(json!({
            "role": "user",
            "content": args.prompt
        }));
    }

    loop {
        let response: Value = client
            .chat()
            .create_byot(json!({
                "messages": messages,
                "model": "anthropic/claude-haiku-4.5",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "description": "Read and return the contents of a file",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "The path to the file to read"
                                    }
                                },
                                "required": ["file_path"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "Write",
                            "description": "Write content to a file",
                            "parameters": {
                                "type": "object",
                                "required": ["file_path", "content"],
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "The path of the file to write to"
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "The content to write to the file"
                                    }
                                }
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "description": "Execute a shell command",
                            "parameters": {
                                "type": "object",
                                "required": ["command"],
                                "properties": {
                                    "command": {
                                        "type": "string",
                                        "description": "The command to execute"
                                    }
                                }
                            }
                        }
                    }
                ]
            }))
            .await?;

        let message = &response["choices"][0]["message"];

        messages.push(message.clone());

        if let Some(tool_calls) = message["tool_calls"].as_array() {
            if tool_calls.is_empty() {
                if let Some(content) = message["content"].as_str() {
                    println!("{}", content);
                }
                break;
            }

            for tool_call in tool_calls {
                let tool_call_id = tool_call["id"].as_str().unwrap_or_default();
                let func_name = tool_call["function"]["name"].as_str().unwrap_or_default();

                if func_name == "Read" {
                    let raw_args = tool_call["function"]["arguments"].as_str().unwrap_or("{}");
                    let parsed_args: Value = serde_json::from_str(raw_args)?;
                    
                    let file_path = parsed_args["file_path"].as_str().unwrap_or("");
                    let result_content = match fs::read_to_string(file_path) {
                        Ok(contents) => contents,
                        Err(err) => format!("Error reading file: {}", err),
                    };

                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result_content
                    }));
                } else if func_name == "Write" {
                    let raw_args = tool_call["function"]["arguments"].as_str().unwrap_or("{}");
                    let parsed_args: Value = serde_json::from_str(raw_args)?;

                    let file_path = parsed_args["file_path"].as_str().unwrap_or("");
                    let content_to_write = parsed_args["content"].as_str().unwrap_or("");

                    let result_content = match fs::write(file_path, content_to_write) {
                        Ok(_) => "Successfully written to file".to_string(),
                        Err(err) => format!("Error writing file: {}", err),
                    };

                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result_content
                    }));
                } else if func_name == "Bash" {
                    let raw_args = tool_call["function"]["arguments"].as_str().unwrap_or("{}");
                    let parsed_args: Value = serde_json::from_str(raw_args)?;

                    let command_str = parsed_args["command"].as_str().unwrap_or("");

                    let result_content = match Command::new("sh").arg("-c").arg(command_str).output() {
                        Ok(output) => {
                            let stdout = String::from_utf8_lossy(&output.stdout);
                            let stderr = String::from_utf8_lossy(&output.stderr);
                            format!("{}{}", stdout, stderr)
                        }
                        Err(err) => format!("Error executing command: {}", err),
                    };

                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result_content
                    }));
                }
            }
        } else {
            if let Some(content) = message["content"].as_str() {
                println!("{}", content);
            }
            break;
        }
    }

    Ok(())
}