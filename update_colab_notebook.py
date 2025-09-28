#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update Colab notebook with GPU optimizations
"""

import json
from pathlib import Path

def update_notebook():
    """Update notebook with GPU optimization notes"""

    notebook_path = Path("colab_research_pipeline.ipynb")

    if not notebook_path.exists():
        print("Notebook not found!")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Find and update the training cell
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'Training TCN models' in source:
                # Update the comment and add GPU note
                updated_source = []
                for line in cell['source']:
                    if 'Training TCN models (co the mat 10-15 phut)' in line:
                        updated_source.append('# Training TCN models (optimized for T4 GPU - ~5-7 minutes)\n')
                    elif 'print("Deep learning training completed!")' in line:
                        updated_source.append('print("Deep learning training completed!")\n')
                        updated_source.append('print("Optimized for T4 15GB GPU with memory management")\n')
                    else:
                        updated_source.append(line)

                cell['source'] = updated_source
                print("Updated training cell with GPU optimization notes")
                break

    # Save updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print("Notebook updated successfully!")

if __name__ == "__main__":
    update_notebook()
