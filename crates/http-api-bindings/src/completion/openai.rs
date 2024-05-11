use async_openai::{config::OpenAIConfig, error::OpenAIError, types::CreateCompletionRequestArgs};
use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use tabby_inference::{CompletionOptions, CompletionStream};
use tracing::warn;

pub struct OpenAIEngine {
    client: async_openai::Client<OpenAIConfig>,
    model_name: String,
}

impl OpenAIEngine {
    pub fn create(api_endpoint: &str, model_name: &str, api_key: Option<String>) -> Self {
        let config = OpenAIConfig::default()
            .with_api_base(api_endpoint)
            .with_api_key(api_key.unwrap_or_default());

        let client = async_openai::Client::with_config(config);

        Self {
            client,
            model_name: model_name.to_owned(),
        }
    }
}

#[async_trait]
impl CompletionStream for OpenAIEngine {
    async fn generate(&self, prompt: &str, options: CompletionOptions) -> BoxStream<String> {
        let request = CreateCompletionRequestArgs::default()
            .model(&self.model_name)
            .temperature(options.sampling_temperature)
            .max_tokens(options.max_decoding_tokens as u16)
            .stream(true)
            .prompt(prompt)
            .build();

        let s = stream! {
            let request = match request {
                Ok(x) => x,
                Err(e) => {
                    warn!("Failed to build completion request {:?}", e);
                    return;
                }
            };

            let s = match self.client.completions().create_stream(request).await {
                Ok(x) => x,
                Err(e) => {
                    warn!("Failed to create completion request {:?}", e);
                    return;
                }
            };

            for await x in s {
                match x {
                    Ok(x) => {
                        yield x.choices[0].text.clone();
                    },
                    Err(OpenAIError::StreamError(_)) => break,
                    Err(e) => {
                        warn!("Failed to stream response: {}", e);
                        break;
                    }
                };
            }
        };

        Box::pin(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::CreateCompletionRequestArgs;

    #[tokio::test]
    async fn test_openai_engine() {
        let config = OpenAIConfig::default()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key(std::env::var("OPENAI_API_KEY").unwrap());
        let client = async_openai::Client::with_config(config);

        let request = CreateCompletionRequestArgs::default()
            .model("gpt-3.5-turbo-instruct")
            .prompt("Tell me the recipe of alfredo pasta")
            .max_tokens(40_u16)
            .build()
            .unwrap();

        let response = client.completions().create(request).await.unwrap();
        println!("{:?}", response.choices.first().unwrap().text);
    }
}
