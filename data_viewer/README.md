# JSONL 数据查看器

一个美观的 Flask Web 应用，用于方便地浏览和查看 dataset 目录下的 JSONL 文件。

## 功能特性

- 📁 **文件选择**: 自动扫描并列出 dataset 目录下的所有 JSONL 文件
- 📊 **数据展示**: 美观的界面展示每条数据，包括：
  - 文本内容
  - 实体列表（带类型标签）
  - 类型列表
  - 数据集信息
  - 完整 JSON 视图
- 📄 **分页功能**: 支持分页浏览，可自定义每页显示数量（5/10/20/50）
- 🔍 **快速跳转**: 支持首页、上一页、下一页、末页，以及直接输入页码跳转
- 📈 **文件信息**: 显示文件总行数和文件大小

## 安装

1. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 确保 dataset 目录在项目根目录下（与 data_viewer 同级）

2. 运行应用：
```bash
cd data_viewer
python app.py
```

3. 在浏览器中打开：
```
http://localhost:5000
```

## 目录结构

```
data_viewer/
├── app.py              # Flask 主应用
├── templates/
│   └── index.html     # 前端页面模板
├── requirements.txt    # Python 依赖
└── README.md          # 说明文档
```

## 注意事项

- 应用会自动读取父目录下的 `dataset` 文件夹中的 JSONL 文件
- 大文件加载可能需要一些时间，建议使用分页功能
- 默认端口为 5000，可在 `app.py` 中修改
