use async_openai::{config::OpenAIConfig, Client};
use clap::Parser;
use serde_json::{json, Value};
use std::{env, fs, process};

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

    // CHANGED: Maintain conversation history in a persistent vector
    let mut messages = vec![json!({
        "role": "user",
        "content": args.prompt
    })];

    // CHANGED: Enter the agent loop
    loop {
        let response: Value = client
            .chat()
            .create_byot(json!({
                "messages": messages, // CHANGED: Pass the full conversation history
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
                    }
                ]
            }))
            .await?;

        let message = &response["choices"][0]["message"];

        // CHANGED: Append the assistant's response to conversation history
        messages.push(message.clone());

        // CHANGED: Check for tool calls
        if let Some(tool_calls) = message["tool_calls"].as_array() {
            if tool_calls.is_empty() {
                // If tool_calls array is empty, print content and break
                if let Some(content) = message["content"].as_str() {
                    println!("{}", content);
                }
                break;
            }

            // Iterate through each tool call and process execution
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

                    // CHANGED: Append tool execution result back into conversation history
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result_content
                    }));
                }else if func_name == "Write" {
                    let raw_args = tool_call["function"]["arguments"].as_str().unwrap_or("{}");
                    let parsed_args: Value = serde_json::from_str(raw_args)?;

                    let file_path = parsed_args["file_path"].as_str().unwrap_or("");
                    let content_to_write = parsed_args["content"].as_str().unwrap_or("");

                    // NEW: Write content to specified file path (creates or overwrites)
                    let result_content = match fs::write(file_path, content_to_write) {
                        Ok(_) => "Successfully written to file".to_string(),
                        Err(err) => format!("Error writing file: {}", err),
                    };

                    // NEW: Append write tool execution result to message history
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result_content
                    }));
                }
            }
        } else {
            // CHANGED: Terminal state — print final response content and exit loop
            if let Some(content) = message["content"].as_str() {
                println!("{}", content);
            }
            break;
        }
    }

    Ok(())
}