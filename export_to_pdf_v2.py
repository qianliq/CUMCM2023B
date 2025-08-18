#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将Jupyter Notebook的所有输出导出为PDF格式
使用html转pdf的方式，确保图片也能正确显示
"""

import json
import os
import base64
import tempfile
import subprocess
from pathlib import Path

def extract_notebook_outputs_with_images(notebook_path):
    """提取notebook的所有输出，包括图片"""
    content_items = []
    image_files = []
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        nb_data = json.loads(content)
        
        cell_num = 1
        for cell in nb_data.get('cells', []):
            if cell.get('cell_type') == 'code':
                cell_outputs = cell.get('outputs', [])
                if cell_outputs:
                    content_items.append({
                        'type': 'header',
                        'content': f"Cell {cell_num} 输出"
                    })
                    
                    for output_idx, output in enumerate(cell_outputs):
                        # 处理标准输出
                        if output.get('output_type') == 'stream' and output.get('name') == 'stdout':
                            text_lines = output.get('text', [])
                            text_content = ''.join(text_lines) if isinstance(text_lines, list) else text_lines
                            content_items.append({
                                'type': 'text',
                                'content': text_content
                            })
                        
                        # 处理错误输出
                        elif output.get('output_type') == 'stream' and output.get('name') == 'stderr':
                            text_lines = output.get('text', [])
                            text_content = ''.join(text_lines) if isinstance(text_lines, list) else text_lines
                            content_items.append({
                                'type': 'error',
                                'content': text_content
                            })
                        
                        # 处理执行结果
                        elif output.get('output_type') == 'execute_result':
                            data = output.get('data', {})
                            if 'text/plain' in data:
                                result = data['text/plain']
                                text_content = ''.join(result) if isinstance(result, list) else result
                                content_items.append({
                                    'type': 'result',
                                    'content': text_content
                                })
                        
                        # 处理显示数据（包括图片）
                        elif output.get('output_type') == 'display_data':
                            data = output.get('data', {})
                            
                            # 处理文本数据
                            if 'text/plain' in data:
                                result = data['text/plain']
                                text_content = ''.join(result) if isinstance(result, list) else result
                                content_items.append({
                                    'type': 'display',
                                    'content': text_content
                                })
                            
                            # 处理图片数据
                            if 'image/png' in data:
                                png_data = data['image/png']
                                # 保存图片到永久文件
                                image_filename = f"cell_{cell_num}_output_{output_idx}.png"
                                permanent_image_filename = f"image_cell_{cell_num}_output_{output_idx}.png"
                                try:
                                    image_data = base64.b64decode(png_data)
                                    # 临时文件用于HTML显示
                                    with open(image_filename, 'wb') as img_file:
                                        img_file.write(image_data)
                                    # 永久文件保存在当前目录
                                    with open(permanent_image_filename, 'wb') as img_file:
                                        img_file.write(image_data)
                                    
                                    image_files.append({
                                        'temp_file': image_filename,
                                        'permanent_file': permanent_image_filename,
                                        'cell_num': cell_num,
                                        'output_idx': output_idx
                                    })
                                    content_items.append({
                                        'type': 'image',
                                        'content': image_filename,
                                        'permanent_file': permanent_image_filename,
                                        'format': 'png'
                                    })
                                except Exception as e:
                                    content_items.append({
                                        'type': 'error',
                                        'content': f"图片处理错误: {str(e)}"
                                    })
                            
                            if 'image/jpeg' in data:
                                jpeg_data = data['image/jpeg']
                                image_filename = f"cell_{cell_num}_output_{output_idx}.jpg"
                                permanent_image_filename = f"image_cell_{cell_num}_output_{output_idx}.jpg"
                                try:
                                    image_data = base64.b64decode(jpeg_data)
                                    # 临时文件用于HTML显示
                                    with open(image_filename, 'wb') as img_file:
                                        img_file.write(image_data)
                                    # 永久文件保存在当前目录
                                    with open(permanent_image_filename, 'wb') as img_file:
                                        img_file.write(image_data)
                                    
                                    image_files.append({
                                        'temp_file': image_filename,
                                        'permanent_file': permanent_image_filename,
                                        'cell_num': cell_num,
                                        'output_idx': output_idx
                                    })
                                    content_items.append({
                                        'type': 'image',
                                        'content': image_filename,
                                        'permanent_file': permanent_image_filename,
                                        'format': 'jpeg'
                                    })
                                except Exception as e:
                                    content_items.append({
                                        'type': 'error',
                                        'content': f"图片处理错误: {str(e)}"
                                    })
                
                cell_num += 1
    
    except Exception as e:
        content_items.append({
            'type': 'error',
            'content': f"读取notebook出错: {str(e)}"
        })
    
    return content_items, image_files

def create_html_with_images(content_items, title):
    """创建包含图片的HTML内容"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
            body {{ 
                font-family: 'Arial', sans-serif; 
                margin: 40px; 
                line-height: 1.6;
                color: #333;
            }}
            h1 {{ 
                color: #2c3e50; 
                text-align: center; 
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{ 
                color: #34495e; 
                border-bottom: 1px solid #bdc3c7; 
                padding-bottom: 5px;
                margin-top: 30px;
            }}
            .text {{ 
                margin: 15px 0; 
                white-space: pre-wrap; 
                background: #f8f9fa;
                padding: 10px;
                border-left: 4px solid #3498db;
                font-family: 'Courier New', monospace;
            }}
            .result {{
                margin: 15px 0; 
                white-space: pre-wrap; 
                background: #e8f5e8;
                padding: 10px;
                border-left: 4px solid #27ae60;
                font-family: 'Courier New', monospace;
            }}
            .error {{ 
                color: #e74c3c; 
                background: #fdf2f2;
                padding: 10px;
                border-left: 4px solid #e74c3c;
                margin: 15px 0;
                font-family: 'Courier New', monospace;
            }}
            .image-container {{ 
                text-align: center; 
                margin: 20px 0; 
                padding: 10px;
                background: #fff;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .image-container img {{ 
                max-width: 100%; 
                height: auto; 
                border: 1px solid #ddd;
                border-radius: 3px;
            }}
            .cell-header {{
                background: #3498db;
                color: white;
                padding: 8px 15px;
                margin: 20px 0 10px 0;
                border-radius: 5px;
                font-weight: bold;
            }}
            @media print {{
                body {{ margin: 20px; }}
                .image-container img {{ max-width: 90%; }}
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
    """
    
    for item in content_items:
        if item['type'] == 'header':
            html_content += f"<div class='cell-header'>{item['content']}</div>\n"
        elif item['type'] == 'text' or item['type'] == 'display':
            content = item['content'].replace('<', '&lt;').replace('>', '&gt;')
            html_content += f"<div class='text'>{content}</div>\n"
        elif item['type'] == 'result':
            content = item['content'].replace('<', '&lt;').replace('>', '&gt;')
            html_content += f"<div class='result'>{content}</div>\n"
        elif item['type'] == 'error':
            content = item['content'].replace('<', '&lt;').replace('>', '&gt;')
            html_content += f"<div class='error'>错误: {content}</div>\n"
        elif item['type'] == 'image':
            html_content += f"""
            <div class='image-container'>
                <img src='{item['content']}' alt='Notebook输出图片' />
                <p><em>图片已保存为: {item.get('permanent_file', item['content'])}</em></p>
            </div>\n"""
    
    html_content += "</body></html>"
    return html_content

def convert_html_to_pdf(html_file, pdf_file):
    """使用系统工具将HTML转换为PDF"""
    
    # 方法1: 尝试使用wkhtmltopdf
    try:
        result = subprocess.run([
            'wkhtmltopdf', 
            '--page-size', 'A4',
            '--orientation', 'Portrait',
            '--margin-top', '20mm',
            '--margin-bottom', '20mm',
            '--margin-left', '20mm',
            '--margin-right', '20mm',
            '--enable-local-file-access',
            html_file, 
            pdf_file
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return True, "wkhtmltopdf"
        else:
            print(f"wkhtmltopdf错误: {result.stderr}")
    except FileNotFoundError:
        pass
    
    # 方法2: 尝试使用Chrome/Chromium的无头模式 (Windows友好)
    try:
        # 尝试多个可能的Chrome路径
        chrome_paths = [
            r'C:\Program Files\Google\Chrome\Application\chrome.exe',
            r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
            r'C:\Users\%s\AppData\Local\Google\Chrome\Application\chrome.exe' % os.getenv('USERNAME', ''),
            'chrome',  # 如果在PATH中
            'chromium',
            'chromium-browser'
        ]
        
        for chrome_path in chrome_paths:
            if chrome_path.startswith('C:') and not os.path.exists(chrome_path):
                continue
                
            try:
                html_full_path = os.path.abspath(html_file)
                result = subprocess.run([
                    chrome_path,
                    '--headless',
                    '--disable-gpu',
                    '--print-to-pdf=' + os.path.abspath(pdf_file),
                    '--print-to-pdf-no-header',
                    '--no-margins',
                    'file:///' + html_full_path.replace('\\', '/')
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and os.path.exists(pdf_file):
                    return True, "Chrome"
                else:
                    if result.stderr:
                        print(f"Chrome错误 ({chrome_path}): {result.stderr}")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
                
    except Exception as e:
        print(f"Chrome转换出错: {str(e)}")
    
    # 方法3: 尝试使用weasyprint (可能在Windows上有问题)
    try:
        result = subprocess.run([
            'weasyprint', 
            html_file, 
            pdf_file
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return True, "weasyprint"
        else:
            print(f"weasyprint错误: {result.stderr}")
    except FileNotFoundError:
        pass
    
    # 方法4: 尝试使用 pdfkit (需要先安装 pip install pdfkit)
    try:
        import pdfkit
        
        options = {
            'page-size': 'A4',
            'orientation': 'Portrait',
            'margin-top': '20mm',
            'margin-right': '20mm',
            'margin-bottom': '20mm',
            'margin-left': '20mm',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None
        }
        
        pdfkit.from_file(html_file, pdf_file, options=options)
        if os.path.exists(pdf_file):
            return True, "pdfkit"
    except (ImportError, Exception) as e:
        pass
    
    return False, None

def cleanup_temp_files(image_files):
    """清理临时图片文件，但保留永久图片文件"""
    for img_info in image_files:
        try:
            if isinstance(img_info, dict):
                temp_file = img_info.get('temp_file')
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
            else:
                # 兼容旧版本格式
                if os.path.exists(img_info):
                    os.remove(img_info)
        except Exception as e:
            temp_file = img_info.get('temp_file', img_info) if isinstance(img_info, dict) else img_info
            print(f"清理临时文件 {temp_file} 时出错: {str(e)}")

def main():
    # 处理每个问题的notebook
    for q in ['q1', 'q2', 'q3', 'q4']:
        notebook_path = f'{q}/{q}.ipynb'
        if os.path.exists(notebook_path):
            print(f'处理 {notebook_path}...')
            
            # 切换到问题目录，以便图片路径正确
            original_dir = os.getcwd()
            os.chdir(q)
            
            try:
                content_items, image_files = extract_notebook_outputs_with_images(f'{q}.ipynb')
                
                # 生成HTML和PDF文件
                title = f'{q.upper()} Jupyter Notebook 输出结果'
                html_content = create_html_with_images(content_items, title)
                
                # 保存HTML文件
                html_file = f'{q}result.html'
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                # 尝试转换为PDF
                pdf_file = f'{q}result.pdf'
                success, tool_used = convert_html_to_pdf(html_file, pdf_file)
                
                if success:
                    print(f'  ✓ PDF已生成: {q}/{pdf_file} (使用 {tool_used})')
                else:
                    print(f'  ⚠ PDF转换失败，HTML文件已保存: {q}/{html_file}')
                    print(f'  提示: 你可以打开HTML文件并使用浏览器打印为PDF')
                
                # 统计信息
                text_items = len([i for i in content_items if i['type'] in ['text', 'result', 'display']])
                image_items = len([i for i in content_items if i['type'] == 'image'])
                permanent_images = [img_info.get('permanent_file') for img_info in image_files if isinstance(img_info, dict)]
                
                print(f'  - 文本输出项: {text_items}')
                print(f'  - 图片输出项: {image_items}')
                if permanent_images:
                    print(f'  - 永久保存的图片文件:')
                    for img_file in permanent_images:
                        print(f'    * {img_file}')
                
                # 清理临时图片文件（保留永久文件）
                cleanup_temp_files(image_files)
                
            finally:
                os.chdir(original_dir)
        else:
            print(f'未找到notebook文件: {notebook_path}')
    
    print("\n导出完成!")
    print("如果PDF转换失败，有以下解决方案:")
    print("  Windows推荐方案:")
    print("    1. 使用Chrome浏览器打开HTML文件，按Ctrl+P打印为PDF")
    print("    2. 安装wkhtmltopdf: https://wkhtmltopdf.org/downloads.html")
    print("    3. 安装pdfkit: pip install pdfkit")
    print("  其他系统:")
    print("    - macOS: brew install wkhtmltopdf")
    print("    - Linux: sudo apt-get install wkhtmltopdf")
    print("    - weasyprint: pip install weasyprint (Linux/macOS更稳定)")
    print("\n  提示: HTML文件已生成，您可以直接在浏览器中查看或打印为PDF")

if __name__ == "__main__":
    main()
