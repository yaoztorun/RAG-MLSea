import json
import matplotlib.pyplot as plt
import os

def generate_scatter_plot(json_path, output_path):
    print(f"Reading data from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rouge_scores = []
    sas_scores = []
    colors = []
    
    for q in data.get("per_question", []):
        metrics = q.get("metrics", {})
        
        rouge = metrics.get("rag_rougeL", 0.0)
        sas = metrics.get("rag_sas", 0.0)
        judge = metrics.get("llm_judge_score", 0)
        
        rouge_scores.append(rouge)
        sas_scores.append(sas)
        
        # Green if Judge = 1, Red if Judge = 0
        if judge == 1:
            colors.append('green')
        else:
            colors.append('red')
            
    print(f"Loaded {len(rouge_scores)} data points.")

    plt.figure(figsize=(10, 8))
    
    # Plot data points
    # Separate them for legend
    x_green = [r for r, c in zip(rouge_scores, colors) if c == 'green']
    y_green = [s for s, c in zip(sas_scores, colors) if c == 'green']
    x_red = [r for r, c in zip(rouge_scores, colors) if c == 'red']
    y_red = [s for s, c in zip(sas_scores, colors) if c == 'red']
    
    plt.scatter(x_green, y_green, color='green', alpha=0.7, label='Judge = 1 (Correct)', s=100, edgecolors='black')
    plt.scatter(x_red, y_red, color='red', alpha=0.7, label='Judge = 0 (Incorrect)', s=100, edgecolors='black')
    
    plt.title('Correlation: ROUGE-L vs SAS vs LLM Judge', fontsize=16)
    plt.xlabel('ROUGE-L Score', fontsize=14)
    plt.ylabel('SAS (Semantic Answer Similarity)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='lower right')
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {output_path}")

if __name__ == "__main__":
    json_path = r"C:\Users\esati\thesis-rag-llm-project\RAG-MLSea-github\data\intermediate\post_retrieval\generation_evaluation_n50.json"
    output_path = r"C:\Users\esati\thesis-rag-llm-project\RAG-MLSea-github\docs\metric_correlation_scatter.png"
    
    generate_scatter_plot(json_path, output_path)
