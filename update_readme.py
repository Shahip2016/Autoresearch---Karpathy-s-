import os
import re

README_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.tsv')

def get_leaderboard():
    """Parses results.tsv and returns a Markdown table of the top 5 experiments."""
    if not os.path.exists(RESULTS_FILE):
        return None
    
    results = []
    with open(RESULTS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) >= 3:
                results.append({
                    'timestamp': parts[0],
                    'val_loss': float(parts[1]),
                    'val_bpb': float(parts[2]), 
                    'params': parts[3] if len(parts) > 3 else "N/A",
                    'iter': parts[4] if len(parts) > 4 else "N/A"
                })
    
    if not results:
        return None
    
    # Sort by val_bpb (lower is better)
    results.sort(key=lambda x: x['val_bpb'])
    top_5 = results[:5]
    
    md = "## Experiment Leaderboard (Top 5)\n\n"
    md += "| Rank | Iteration | Val BPB | Timestamp | Parameters |\n"
    md += "|---|---|---|---|---|\n"
    for i, r in enumerate(top_5, 1):
        md += f"| {i} | {r['iter']} | **{r['val_bpb']:.6f}** | {r['timestamp']} | {r['params']} |\n"
    return md

def update_readme():
    """Injects the leaderboard into README.md using placeholder tags."""
    leaderboard_md = get_leaderboard()
    if not leaderboard_md:
        print("[!] No results found in results.tsv. Skipping README update.")
        return

    try:
        if os.path.exists(README_PATH):
            with open(README_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content = "# AutoResearch Results\n"

        tag_start = "<!-- LEADERBOARD_START -->"
        tag_end = "<!-- LEADERBOARD_END -->"
        
        full_section = f"{tag_start}\n\n{leaderboard_md}\n{tag_end}"
        
        if tag_start in content and tag_end in content:
            # Replace existing section
            new_content = re.sub(f"{tag_start}.*?{tag_end}", full_section, content, flags=re.DOTALL)
            print("[*] Updating existing leaderboard in README.md.")
        else:
            # Append to the end of file
            new_content = content.strip() + "\n\n" + full_section + "\n"
            print("[*] Adding new leaderboard section to README.md.")
        
        with open(README_PATH, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("[+] README.md updated successfully.")
    except Exception as e:
        print(f"[-] Error updating README.md: {e}")

if __name__ == "__main__":
    update_readme()
