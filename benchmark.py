import os
import time
import argparse
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import tiktoken
from openai import OpenAI
from datetime import datetime
import json

# Default prompt used for benchmarking. Long enough to hopefully get a good sense of prompt processing speed, and generate enough response tokens to get a reasonable measure there too.
DEFAULT_PROMPT = ("Imagine you are planning a week-long vacation to a place you've never visited before. "
                  "Describe the destination, including its main attractions and cultural highlights. "
                  "What activities would you prioritize during your visit? Additionally, explain how you would prepare for the trip, "
                  "including any specific items you would pack and any research you would conduct beforehand. "
                  "Finally, discuss how you would balance relaxation and adventure during your vacation.")

# Function to measure time to first token and response time
def benchmark_model(client, model_name, prompt):
    start_time = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    # Initialize variables to measure time and gather response
    time_to_first_token = None
    response_time_start = None
    response_time_end = None
    full_response = ""
    num_chunks = 0
    chunk_times = []
    chunk_tokens = []

    # Stream the response
    for chunk in response:
        if time_to_first_token is None:
            time_to_first_token = time.time() - start_time
            response_time_start = time.time()
            previous_chunk_time = response_time_start
            # Skip the first chunk's content
            continue
        current_chunk_time = time.time()
        full_response += chunk.choices[0].delta.content or ""
        chunk_duration = current_chunk_time - previous_chunk_time
        chunk_times.append(chunk_duration)
        chunk_tokens.append(len(chunk.choices[0].delta.content or ""))
        response_time_end = current_chunk_time
        previous_chunk_time = current_chunk_time
        num_chunks += 1

    # Calculate response time
    response_time = response_time_end - response_time_start

    # Tokenize the full response using tiktoken
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except:
        # TODO: Tokenization for non-OpenAI models is only approximated by the gpt-4 tokenizer for now.
        #   Ideally, we should detect the type of model, and choose the correct tokenizer for full accuracy.
        encoding = tiktoken.encoding_for_model('gpt-4')
    
    num_tokens = len(encoding.encode(full_response))
    prompt_tokens = len(encoding.encode(prompt))

    # Calculate tokens per second
    tokens_per_second = num_tokens / response_time if response_time > 0 else float('inf')
    avg_tokens_per_chunk = sum(chunk_tokens) / num_chunks if num_chunks > 0 else float('inf')
    avg_time_between_chunks = sum(chunk_times) / len(chunk_times) if len(chunk_times) > 0 else float('inf')
    # prompt_tokens_per_second unfortunately includes the time to generate the first chunk of output tokens, but this seems unavoidable.
    # The longer the input prompt, the more accurate this number should be.
    prompt_tokens_per_second = prompt_tokens / time_to_first_token if time_to_first_token > 0 else float('inf')

    # Return the benchmark results
    return {
        "time_to_first_token": time_to_first_token,
        "prompt_tokens_per_second": prompt_tokens_per_second,
        "tokens_per_second": tokens_per_second,
        "num_response_tokens": num_tokens,
        "avg_tokens_per_chunk": avg_tokens_per_chunk,
        "avg_time_between_chunks": avg_time_between_chunks
    }

def write_results(model_name, results, output_dir):
    csv_file = os.path.join(output_dir, "output.csv")
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Model Name", "Time To First Token", "Prompt Tok/s", "Response Tok/s",
                "Num Response Tokens", "Avg Tokens per Chunk", "Avg Time Between Chunks"
            ])
        for result in results:
            writer.writerow([
                model_name,
                f"{result['time_to_first_token']:.2f}",
                f"{result['prompt_tokens_per_second']:.2f}",
                f"{result['tokens_per_second']:.2f}",
                result['num_response_tokens'],
                f"{result['avg_tokens_per_chunk']:.2f}",
                f"{result['avg_time_between_chunks']:.2f}"
            ])

def calculate_model_ranks(df):
    medians = df.groupby('Model Name').median().reset_index()
    sorted_medians = medians.sort_values(by='Response Tok/s', ascending=True)
    return sorted_medians['Model Name'].tolist()

def generate_plots(csv_file, output_dir):
    df = pd.read_csv(csv_file)
    
    # Calculate the ranks and order the models
    model_order = calculate_model_ranks(df)
    
    # Boxplot for Time To First Token
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Model Name", y="Time To First Token", data=df, order=model_order)
    plt.title("Time To First Token by Model")
    plt.xticks(rotation=15)
    plt.savefig(os.path.join(output_dir, "time_to_first_token_boxplot.png"))
    plt.close()

    # Boxplot for Prompt Tok/s
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Model Name", y="Prompt Tok/s", data=df, order=model_order)
    plt.title("Prompt Tokens per Second by Model")
    plt.xticks(rotation=15)
    plt.savefig(os.path.join(output_dir, "prompt_tokens_per_second_boxplot.png"))
    plt.close()

    # Boxplot for Response Tok/s
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Model Name", y="Response Tok/s", data=df, order=model_order)
    plt.title("Response Tokens per Second by Model")
    plt.xticks(rotation=15)
    plt.savefig(os.path.join(output_dir, "response_tokens_per_second_boxplot.png"))
    plt.close()

    # Boxplot for Number of Response Tokens
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Model Name", y="Num Response Tokens", data=df, order=model_order)
    plt.title("Number of Response Tokens by Model")
    plt.xticks(rotation=15)
    plt.savefig(os.path.join(output_dir, "num_response_tokens_boxplot.png"))
    plt.close()

    # Boxplot for Average Tokens per Chunk
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Model Name", y="Avg Tokens per Chunk", data=df, order=model_order)
    plt.title("Average Tokens per Chunk by Model")
    plt.xticks(rotation=15)
    plt.savefig(os.path.join(output_dir, "avg_tokens_per_chunk_boxplot.png"))
    plt.close()

    # Boxplot for Average Time Between Chunks
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Model Name", y="Avg Time Between Chunks", data=df, order=model_order)
    plt.title("Average Time Between Chunks by Model")
    plt.xticks(rotation=15)
    plt.savefig(os.path.join(output_dir, "avg_time_between_chunks_boxplot.png"))
    plt.close()

def generate_markdown_summary(csv_file, output_dir):
    df = pd.read_csv(csv_file)
    
    # Calculate medians and IQR for each model and variable
    summary = df.groupby('Model Name').agg(
        time_to_first_token_median=pd.NamedAgg(column='Time To First Token', aggfunc='median'),
        time_to_first_token_iqr=pd.NamedAgg(column='Time To First Token', aggfunc=lambda x: x.quantile(0.75) - x.quantile(0.25)),
        prompt_tok_s_median=pd.NamedAgg(column='Prompt Tok/s', aggfunc='median'),
        prompt_tok_s_iqr=pd.NamedAgg(column='Prompt Tok/s', aggfunc=lambda x: x.quantile(0.75) - x.quantile(0.25)),
        response_tok_s_median=pd.NamedAgg(column='Response Tok/s', aggfunc='median'),
        response_tok_s_iqr=pd.NamedAgg(column='Response Tok/s', aggfunc=lambda x: x.quantile(0.75) - x.quantile(0.25)),
        num_response_tokens_median=pd.NamedAgg(column='Num Response Tokens', aggfunc='median'),
        num_response_tokens_iqr=pd.NamedAgg(column='Num Response Tokens', aggfunc=lambda x: x.quantile(0.75) - x.quantile(0.25)),
        avg_tokens_per_chunk_median=pd.NamedAgg(column='Avg Tokens per Chunk', aggfunc='median'),
        avg_tokens_per_chunk_iqr=pd.NamedAgg(column='Avg Tokens per Chunk', aggfunc=lambda x: x.quantile(0.75) - x.quantile(0.25)),
        avg_time_between_chunks_median=pd.NamedAgg(column='Avg Time Between Chunks', aggfunc='median'),
        avg_time_between_chunks_iqr=pd.NamedAgg(column='Avg Time Between Chunks', aggfunc=lambda x: x.quantile(0.75) - x.quantile(0.25))
    ).reset_index()
    
    # Sort the summary by the same order as the plots
    model_order = calculate_model_ranks(df)
    summary['Model Name'] = pd.Categorical(summary['Model Name'], categories=model_order, ordered=True)
    summary = summary.sort_values('Model Name')
    
    md_file_path = os.path.join(output_dir, "output.md")
    with open(md_file_path, "w") as md_file:
        md_file.write("# Model Performance Summary\n\n")
        md_file.write("| Model | Time To First Token | Prompt Tok/s | Response Tok/s | Num Response Tokens | Avg Tokens per Chunk | Avg Time Between Chunks |\n")
        md_file.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        
        for _, row in summary.iterrows():
            md_file.write(f"| {row['Model Name']} | {row['time_to_first_token_median']:.2f} +/- {row['time_to_first_token_iqr']:.2f} | "
                          f"{row['prompt_tok_s_median']:.2f} +/- {row['prompt_tok_s_iqr']:.2f} | "
                          f"{row['response_tok_s_median']:.2f} +/- {row['response_tok_s_iqr']:.2f} | "
                          f"{row['num_response_tokens_median']:.2f} +/- {row['num_response_tokens_iqr']:.2f} | "
                          f"{row['avg_tokens_per_chunk_median']:.2f} +/- {row['avg_tokens_per_chunk_iqr']:.2f} | "
                          f"{row['avg_time_between_chunks_median']:.2f} +/- {row['avg_time_between_chunks_iqr']:.2f} |\n")
        
        md_file.write("\n*Values are presented as median +/- IQR (Interquartile Range). Tokenization of non-OpenAI models is approximate.*\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark OpenAI-compatible GPT models.")
    parser.add_argument("--models", "-m", type=str, help="Comma-separated list of model names to benchmark.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="The prompt to send to the models.")
    parser.add_argument("--number", "-n", type=int, default=10, help="Number of times to run the benchmark (default 10).")
    parser.add_argument("--plot", type=str, choices=["yes", "no", "only"], default="yes", help="Generate plots: 'yes' (default), 'no', or 'only'.")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var).")
    parser.add_argument("--base-url", type=str, help="Base URL for OpenAI-compatible API (or set OPENAI_BASE_URL env var).")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory for results (default: results_YYYYMMDD_HHMMSS).")
    args = parser.parse_args()

    # Create output directory with timestamp
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)

    # Get API credentials from args or environment
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")

    # Initialize the OpenAI client
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    
    client = OpenAI(**client_kwargs)

    # Save parameters to a JSON file
    params = {
        "timestamp": datetime.now().isoformat(),
        "models": args.models,
        "prompt": args.prompt,
        "number": args.number,
        "api_key": "***" if api_key else None,
        "base_url": base_url,
        "output_dir": output_dir
    }
    
    params_file = os.path.join(output_dir, "parameters.json")
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)

    if args.plot != "only":
        # Calculate total number of steps for the overall progress bar
        models = args.models.split(",")
        total_steps = len(models) * args.number

        # Run the benchmark for each model
        with tqdm(total=total_steps, desc="Overall Progress") as overall_progress:
            for model_name in models:
                results = []
                for _ in tqdm(range(args.number), desc=f"Benchmarking {model_name.strip()}", leave=False):
                    result = benchmark_model(client, model_name.strip(), args.prompt)
                    results.append(result)
                    overall_progress.update(1)

                # Write the results to the CSV file
                write_results(model_name.strip(), results, output_dir)

        print(f"Results have been written to {os.path.join(output_dir, 'output.csv')}")
        print(f"Parameters have been saved to {params_file}")

    if args.plot != "no":
        csv_file = os.path.join(output_dir, "output.csv")
        # Generate the plots
        generate_plots(csv_file, output_dir)
        print(f"Plots have been saved to {output_dir}")
        # Generate the markdown summary
        generate_markdown_summary(csv_file, output_dir)
        print(f"Markdown summary has been saved to {os.path.join(output_dir, 'output.md')}")
