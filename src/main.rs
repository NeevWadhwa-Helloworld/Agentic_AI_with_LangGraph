use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde_json::{Value, json};
use std::{env, process};

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

    #[allow(unused_variables)]
    let response: Value = client
        .chat()
        .create_byot(json!({
            "messages": [
                {
                    "role": "user",
                    "content": args.prompt
                }
            ],
            "model": "anthropic/claude-haiku-4.5",
            "tools":[
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
                }
            ]
        }))
        .await?;

    let message = &response["choices"][0]["message"];

    if let Some(tool_calls) = message["tool_calls"].as_array() {
        if let Some(first_call) = tool_calls.first() {
            let func_name = first_call["function"]["name"].as_str().unwrap_or_default();
            if func_name == "Read" {
                let raw_args = first_call["function"]["arguments"].as_str().unwrap_or("{}");
                let parsed_args: Value = serde_json::from_str(raw_args)?;
                if let Some(file_path) = parsed_args["file_path"].as_str() {
                    let contents = fs::read_to_string(file_path)?;
                    print!("{}", contents);
                }
            }
        }
    } else if let Some(content) = message["content"].as_str() {
        println!("{}", content);
    }

    Ok(())
}
