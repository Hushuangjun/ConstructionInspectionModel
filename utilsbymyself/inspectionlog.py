import os
from datetime import datetime

def append_to_markdown_table(md_file_path, text, image_path):

    row = f"| {text} | ![]({image_path}) |\n"
    current_date = datetime.now()
    date_str = str(current_date.year) + "年" + str(current_date.month) + "月" + str(current_date.day) + "日"

    # 如果文件不存在，则添加表头
    if not os.path.exists(md_file_path):
        with open(md_file_path, 'w', encoding='utf-8') as md_file:
            md_file.write("# **施工巡检日志**\n")
            md_file.write(f"巡检日期：{date_str}\n")
            md_file.write("## 巡检内容\n")
            md_file.write("| 隐患 | 现场图片 |\n")
            md_file.write("| --- | --- |\n")
            md_file.write(row)
    else:
        # 文件存在，直接追加内容
        with open(md_file_path, 'a', encoding='utf-8') as md_file:
            md_file.write(row)



if __name__ == "__main___":
    text_list = ["第一行文字", "第二行文字", "第三行文字"]
    image_folder_path = r"D:\Desktop\test" 
    image_list = ["20.jpg", "21.jpg", "22.jpg"] 

    output_md_file = r"D:\Desktop\test\output.md"  

    for text, image_name in zip(text_list, image_list):
        append_to_markdown_table(output_md_file, text, image_folder_path, image_name)
