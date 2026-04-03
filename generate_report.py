import os
import json
import argparse
from datetime import datetime

def generate_html_report(results_file, out_file):
    if not os.path.exists(results_file):
        print(f"File {results_file} not found.")
        return
        
    print(f"Reading results from {results_file}...")
    
    # Try parsing as JSON list
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return
        
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Training Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Training Report - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</h1>
        <table>
            <tr>
                <th>Iteration</th>
                <th>Train Loss</th>
                <th>Val Loss</th>
                <th>BPB</th>
                <th>Perplexity</th>
                <th>Acc 1%</th>
            </tr>
    """
    
    for row in data:
        html += f"""
            <tr>
                <td>{row.get('iter', 'N/A')}</td>
                <td>{row.get('train_loss', 0):.4f}</td>
                <td>{row.get('val_loss', 0):.4f}</td>
                <td>{row.get('val_bpb', 0):.4f}</td>
                <td>{row.get('val_ppl', 0):.2f}</td>
                <td>{row.get('val_acc1', 0):.2f}%</td>
            </tr>
        """
        
    html += """
        </table>
    </body>
    </html>
    """
    
    with open(out_file, 'w') as f:
        f.write(html)
        
    print(f"Report generated successfully at {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate an HTML report of training progress.")
    parser.add_argument('--input', type=str, default='out/loss_history.json', help="Path to loss_history.json")
    parser.add_argument('--output', type=str, default='training_report.html', help="Path to output HTML report")
    args = parser.parse_args()
    
    generate_html_report(args.input, args.output)
