use async_openai::{config::AzureConfig, error::OpenAIError, types::CreateChatCompletionRequestArgs};
use async_openai::types::ChatCompletionRequestUserMessageArgs;
use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use tabby_inference::{CompletionOptions, CompletionStream};
use tracing::warn;

pub struct AzureEngine {
    client: async_openai::Client<AzureConfig>,
    model_name: String,
}

impl AzureEngine {
    pub fn create(api_endpoint: &str, api_version: &str, deployment_id: &str, model_name: &str, api_key: Option<String>) -> Self {

        let config = AzureConfig::default()
            .with_api_base(api_endpoint)
            .with_api_version(api_version)
            .with_deployment_id(deployment_id)
            .with_api_key(api_key.unwrap_or_default());

        let client = async_openai::Client::with_config(config);

        Self {
            client,
            model_name: model_name.to_owned(),
        }
    }
}

#[async_trait]
impl CompletionStream for AzureEngine {
    async fn generate(&self, prompt: &str, options: CompletionOptions) -> BoxStream<String> {
        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model_name)
            .messages([ChatCompletionRequestUserMessageArgs::default()
                .content(prompt)
                .build()
                .unwrap()
                .into(),
            ])
            .temperature(options.sampling_temperature)
            .max_tokens(options.max_decoding_tokens as u16)
            // .stream(true)
            .build();

        println!("{:?}", request);

        let s = stream! {
            let request = match request {
                Ok(x) => x,
                Err(e) => {
                    warn!("Failed to build completion request {:?}", e);
                    return;
                }
            };

            let s = match self.client.chat().create(request).await {
                Ok(x) => x,
                Err(e) => {
                    warn!("Failed to create completion request {:?}", e);
                    return;
                }
            };

            // for await x in s {
            //     match x {
            //         Ok(x) => {
            //             println!("----azure response---- {:?}", x);
            //             if x.choices.len() == 0 {
            //                 break;
            //             }
            //             yield x.choices[0].delta.content.clone().unwrap();
            //         },
            //         Err(OpenAIError::StreamError(_)) => {
            //             warn!("Stream error");
            //             break;
            //         },
            //         Err(e) => {
            //             warn!("Failed to stream response: {}", e);
            //             break;
            //         }
            //     };
            // }

            println!("----azure response---- {:?}", s);
            if s.choices.len() == 0 {
                warn!("Empty choice from Azure {:#?}", s);
                return;
            }
            yield s.choices[0].message.content.clone().unwrap();
        };

        Box::pin(s)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::io::{stdout, Write};
    use super::*;
    use async_openai::types::{ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs};
    use futures::StreamExt;

    #[tokio::test]
    async fn test_azure_ai_engine() {
        let config = AzureConfig::default()
            .with_api_base("https://centific-jp-gpt.openai.azure.com")
            .with_api_version("2024-02-15-preview")
            .with_deployment_id("danone-gpt4-32k")
            // get the API key from the environment variable AZURE_OPENAI_KEY
            .with_api_key(std::env::var("AZURE_OPENAI_KEY").unwrap_or_default());
        let client = async_openai::Client::with_config(config);

        let request = CreateChatCompletionRequestArgs::default()
            .model("danone-gpt4-32k")
            .messages([ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant.")
                .build()
                .unwrap()
                .into(),
                ChatCompletionRequestUserMessageArgs::default()
                    .content("How does large language model work?")
                    .build()
                    .unwrap()
                    .into(),
            ])
            .max_tokens(40_u16)
            .build()
            .unwrap();

        let response = client.chat().create(request).await.unwrap();
        println!("\nResponse:\n");
        for choice in response.choices {
            println!(
                "{}: Role: {}  Content: {:?}",
                choice.index, choice.message.role, choice.message.content
            );
        }
    }

    #[tokio::test]
    async fn test_stream() -> Result<(), Box<dyn Error>> {
        let config = AzureConfig::default()
            .with_api_base("https://centific-jp-gpt.openai.azure.com")
            .with_api_version("2024-02-15-preview")
            .with_deployment_id("danone-gpt4-32k")
            // get the API key from the environment variable AZURE_OPENAI_KEY
            .with_api_key(std::env::var("AZURE_OPENAI_KEY").unwrap_or_default());
        let client = async_openai::Client::with_config(config);

        let request = CreateChatCompletionRequestArgs::default()
            .model("danone-gpt4-32k")
            .messages([ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant.")
                .build()
                .unwrap()
                .into(),
                ChatCompletionRequestUserMessageArgs::default()
                    .content("How does large language model work?")
                    .build()
                    .unwrap()
                    .into(),
            ])
            .temperature(0.1)
            .max_tokens(40_u16)
            .build()
            .unwrap();

        println!("{:?}", request);

        let mut stream = client.chat().create_stream(request).await?;

        let mut lock = stdout().lock();
        while let Some(result) = stream.next().await {
            match result {
                Ok(response) => {
                    response.choices.iter().for_each(|chat_choice| {
                        if let Some(ref content) = chat_choice.delta.content {
                            write!(lock, "{}", content).unwrap();
                        }
                    });
                }
                Err(err) => {
                    writeln!(lock, "error: {err}").unwrap();
                }
            }
            stdout().flush()?;
        }

        Ok(())
    }
}
