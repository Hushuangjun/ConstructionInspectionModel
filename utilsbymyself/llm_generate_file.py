import requests
import json
from datetime import datetime


def get_access_token():

    # 此处需要大模型提供的API key，具体可参考官方提供的链接形式。
    url = " "
    
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


def generate_mdfile_byLLM(llm_content,md_file_path):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-speed-pro-128k?access_token=" + get_access_token()

    # 获取当前日期时间
    current_datetime = datetime.now()

    # 使用 strftime 方法将年月日拼接成字符串
    current_date = current_datetime.strftime("%Y-%m-%d")

    # 巡检项目
    current_location = "同济大学图书馆一期项目"


    content = f"""
        作为施工现场巡检人员，本次巡检，共发现如下问题：{llm_content}。
        请你完善一个施工巡检日志，首先总的概括发现的问题，然后结合项目位置：{current_location}阐述可能的后果，最后给出可能的解决方法。
        施工日志已存在的部分：
        '# **施工巡检日志**\n
        巡检日期：date\n
        ## 安全隐患\n
        ...'
        所以请你回答时，从二级标题开始，当然，你的回答不要包括我上面已经存在的部分。
        你回答标题应遵循如下：
        ## 巡检内容总结
        这部分先描述项目，再总结发现的问题，不要讲解决方法。
        ## 隐患可能导致的后果
        这部分讲所发现的问题可能导致的后果。
        ## 整改措施
        这部分讲解决方法。
        ## 结语
    """

    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }


    response = requests.request("POST", url, headers=headers, data=payload)
    
    try:
        result = response.json().get("result", "No result found.")
        with open(md_file_path, "w", encoding="utf-8") as file:
            file.write(result)
        

    except json.JSONDecodeError:
        print("Failed to decode response as JSON:", response.text)


if __name__ == "__main__":
    llm_content = "框架主梁底拆模后出现蜂窝麻面"
    log_path = r"d:\desktop\test\output.md"
    generate_mdfile_byLLM(llm_content,log_path)