use async_openai::{config::OpenAIConfig, Client};
use clap::Parser;
use serde_json::{json, Value};
use std::{
    env, fs,
    path::Path,
    process::{self, Command},
};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

async fn run_agent_loop(
    client: &Client<OpenAIConfig>,
    mut messages: Vec<Value>,
    system_content: &str,
) -> Result<String, Box<dyn std::error::Error>> {
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
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "Skill",
                            "description": "Load a skill's instructions into the conversation",
                            "parameters": {
                                "type": "object",
                                "required": ["name"],
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "The name of the skill to use"
                                    },
                                    "args": {
                                        "type": "string",
                                        "description": "Optional arguments for the skill"
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
                return Ok(message["content"].as_str().unwrap_or("").to_string());
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

                    let result_content =
                        match Command::new("sh").arg("-c").arg(command_str).output() {
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
                } else if func_name == "Skill" {
                    let raw_args = tool_call["function"]["arguments"].as_str().unwrap_or("{}");
                    let parsed_args: Value = serde_json::from_str(raw_args)?;

                    let skill_name = parsed_args["name"].as_str().unwrap_or("");
                    let skill_args_str = parsed_args["args"].as_str().unwrap_or("");
                    let skill_args: Vec<&str> = skill_args_str.split_whitespace().collect();

                    let skill_path = format!(".claude/skills/{}/SKILL.md", skill_name);

                    let result_content = if let Ok(file_content) = fs::read_to_string(&skill_path)
                    {
                        let frontmatter = if file_content.starts_with("---") {
                            file_content.split("---").nth(1).unwrap_or("")
                        } else {
                            ""
                        };

                        let body = if file_content.starts_with("---") {
                            file_content.split("---").nth(2).unwrap_or(&file_content).trim()
                        } else {
                            file_content.trim()
                        };

                        let mut substituted_body =
                            body.to_string().replace("$ARGUMENTS", skill_args_str);

                        for (idx, arg_val) in skill_args.iter().enumerate() {
                            let target_full = format!("$ARGUMENTS[{}]", idx);
                            let target_short = format!("${}", idx);
                            substituted_body = substituted_body.replace(&target_full, arg_val);
                            substituted_body = substituted_body.replace(&target_short, arg_val);
                        }

                        let is_forked = frontmatter
                            .lines()
                            .any(|line| line.trim() == "context: fork");

                        if is_forked {
                            let subagent_messages = vec![json!({
                                "role": "user",
                                "content": substituted_body
                            })];

                            Box::pin(run_agent_loop(client, subagent_messages, system_content))
                                .await?
                        } else {
                            substituted_body
                        }
                    } else {
                        format!("Skill '{}' not found", skill_name)
                    };

                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result_content
                    }));
                }
            }
        } else {
            return Ok(message["content"].as_str().unwrap_or("").to_string());
        }
    }
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

    let mut skill_list_str = String::new();
    if let Ok(entries) = fs::read_dir(".claude/skills") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let skill_name = path.file_name().unwrap_or_default().to_string_lossy();
                let skill_md = path.join("SKILL.md");
                if let Ok(content) = fs::read_to_string(&skill_md) {
                    let description = if content.contains("---") {
                        content
                            .split("---")
                            .nth(1)
                            .unwrap_or("")
                            .lines()
                            .find(|line| line.trim().starts_with("description:"))
                            .map(|line| {
                                line.trim()
                                    .trim_start_matches("description:")
                                    .trim()
                                    .trim_matches('"')
                                    .trim_matches('\'')
                            })
                            .unwrap_or("")
                            .to_string()
                    } else {
                        content
                            .lines()
                            .find(|line| line.trim().starts_with("description:"))
                            .map(|line| {
                                line.trim()
                                    .trim_start_matches("description:")
                                    .trim()
                                    .trim_matches('"')
                                    .trim_matches('\'')
                            })
                            .unwrap_or("")
                            .to_string()
                    };

                    skill_list_str.push_str(&format!("- {}: {}\n", skill_name, description));
                }
            }
        }
    }

    let system_content = format!(
        "You have access to the following skills:\n\n{}\nIf a skill matches the user's request, call the Skill tool with its name and follow the instructions it returns.",
        skill_list_str.trim_end()
    );

    let mut tokens = args.prompt.split_whitespace().peekable();
    let mut expanded_skills = Vec::new();

    while let Some(&token) = tokens.peek() {
        if token.starts_with('/') {
            let candidate_skill = &token[1..];
            let skill_path = format!(".claude/skills/{}/SKILL.md", candidate_skill);

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
        break;
    }

    let remaining_args: Vec<&str> = tokens.collect();
    let full_args_str = remaining_args.join(" ");

    let mut messages: Vec<Value> = vec![json!({
        "role": "system",
        "content": system_content
    })];

    if !expanded_skills.is_empty() {
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

    let result = run_agent_loop(&client, messages, &system_content).await?;
    println!("{}", result);

    Ok(())
}