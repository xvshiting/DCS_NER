import os
import json
from flask import Flask, render_template, request, jsonify
from pathlib import Path

app = Flask(__name__)

# 数据集目录 - 指向父目录的dataset文件夹
DATASET_DIR = Path(__file__).parent.parent / "dataset"

def get_jsonl_files():
    """获取dataset目录下所有的jsonl文件"""
    jsonl_files = []
    if DATASET_DIR.exists():
        for file in DATASET_DIR.glob("*.jsonl"):
            jsonl_files.append({
                "name": file.name,
                "path": str(file),
                "size": file.stat().st_size
            })
    return sorted(jsonl_files, key=lambda x: x["name"])

def count_lines(filepath):
    """快速计算文件行数"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def read_jsonl_page(filepath, page=1, per_page=10):
    """读取jsonl文件的指定页数据"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        total_pages = (total_lines + per_page - 1) // per_page
        
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        data = []
        for i, line in enumerate(lines[start_idx:end_idx], start=start_idx):
            try:
                item = json.loads(line.strip())
                item['_line_number'] = i + 1  # 添加行号
                data.append(item)
            except json.JSONDecodeError:
                continue
        
        return {
            "data": data,
            "page": page,
            "per_page": per_page,
            "total_lines": total_lines,
            "total_pages": total_pages
        }
    except Exception as e:
        return {
            "error": str(e),
            "data": [],
            "page": page,
            "per_page": per_page,
            "total_lines": 0,
            "total_pages": 0
        }

@app.route('/')
def index():
    """主页"""
    jsonl_files = get_jsonl_files()
    return render_template('index.html', jsonl_files=jsonl_files)

@app.route('/api/files')
def api_files():
    """API: 获取所有jsonl文件列表"""
    jsonl_files = get_jsonl_files()
    return jsonify(jsonl_files)

@app.route('/api/data')
def api_data():
    """API: 获取指定文件的数据"""
    filename = request.args.get('file')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    if not filename:
        return jsonify({"error": "文件名不能为空"}), 400
    
    filepath = DATASET_DIR / filename
    if not filepath.exists() or not filepath.suffix == '.jsonl':
        return jsonify({"error": "文件不存在或不是jsonl文件"}), 404
    
    result = read_jsonl_page(str(filepath), page, per_page)
    return jsonify(result)

@app.route('/api/file_info')
def api_file_info():
    """API: 获取文件信息（总行数等）"""
    filename = request.args.get('file')
    
    if not filename:
        return jsonify({"error": "文件名不能为空"}), 400
    
    filepath = DATASET_DIR / filename
    if not filepath.exists() or not filepath.suffix == '.jsonl':
        return jsonify({"error": "文件不存在或不是jsonl文件"}), 404
    
    total_lines = count_lines(str(filepath))
    file_size = filepath.stat().st_size
    
    return jsonify({
        "filename": filename,
        "total_lines": total_lines,
        "file_size": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
