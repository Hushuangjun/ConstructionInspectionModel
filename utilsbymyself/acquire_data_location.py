from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
import requests 
import cv2


def get_exif_data(image_path):
    # 打开图片文件
    image = Image.open(image_path)
    
    # 获取图片的Exif数据
    exif_data = image._getexif()
    if exif_data is None:
        return None
    
    # 将Exif标签映射到可读的字段名
    exif_info = {}
    for tag, value in exif_data.items():
        tag_name = TAGS.get(tag, tag)
        exif_info[tag_name] = value
    
    return exif_info

def get_gps_info(exif_info):
    gps_info = {}
    if 'GPSInfo' in exif_info:
        for tag, value in exif_info['GPSInfo'].items():
            tag_name = GPSTAGS.get(tag, tag)
            gps_info[tag_name] = value
    return gps_info

def convert_to_degrees(value):
    """将GPS坐标从分秒制转换为度数制"""
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)

def get_shooting_info(image_path):
    exif_info = get_exif_data(image_path)
    
    if not exif_info:
        return None, None, None  # 没有Exif数据
    
    # 获取拍摄时间
    date_time = exif_info.get('DateTime', '未知时间')
    if date_time != '未知时间':
        date_time = datetime.strptime(date_time, '%Y:%m:%d %H:%M:%S')
    
    # 获取经纬度信息
    gps_info = get_gps_info(exif_info)
    latitude = None
    longitude = None
    
    if gps_info:
        lat = gps_info.get('GPSLatitude')
        lat_ref = gps_info.get('GPSLatitudeRef')
        lon = gps_info.get('GPSLongitude')
        lon_ref = gps_info.get('GPSLongitudeRef')
        
        if lat and lon and lat_ref and lon_ref:
            latitude = convert_to_degrees(lat)
            longitude = convert_to_degrees(lon)
            # 根据方向（N/S，E/W）调整经纬度符号
            if lat_ref != 'N':
                latitude = -latitude
            if lon_ref != 'E':
                longitude = -longitude
    
    return date_time, latitude, longitude




def convert_to_intuitive_location(lat,lng):
    url = "https://api.map.baidu.com/reverse_geocoding/v3"

    # 此处填写你在控制台-应用管理-创建应用后获取的AK
    ak = " "
    location = f"{lat},{lng}"

    params = {
        "ak":       ak,
        "output":    "json",
        "coordtype":    "wgs84ll",
        "extensions_poi":    "0",
        "location":    location,
    }

    response = requests.get(url=url, params=params)
    if response:
        # 解析响应的JSON数据
        response_data = response.json()
        
        # 获取详细地址和省份
        if response_data.get('status') == 0:
            formatted_address = response_data.get('result', {}).get('formatted_address', 'Address not found')
            return formatted_address
        else:
            return  "Error: Unable to retrieve location information"

def main_date_location(image_path):
    date_time, lat, lng = get_shooting_info(image_path)
    accurate_location = convert_to_intuitive_location(lat,lng)
    return date_time, lat, lng, accurate_location




if __name__ == "__main__":
    
    video_path = r"D:\Desktop\test\test1.mp4"
    cap = cv2.VideoCapture(video_path)  
    count = 10
    while count:
        ret,frame = cap.read()
        if not ret:
            break
        cv2.imwrite(r'D:\temp\temp_frame.jpg', frame)
        image_path = r'D:\temp\temp_frame.jpg'
        date_time, latitude, longitude, accurate_location  = main_date_location(image_path)

        if date_time:
            print(f"拍摄时间: {date_time}")
        else:
            print("未能提取到拍摄时间")

        if latitude is not None and longitude is not None:
            print(f"拍摄时的经纬度: 纬度 {latitude:.5f}, 经度 {longitude:.5f}")
        else:
            print("未能提取到经纬度信息")
        if accurate_location is not None:
            print(f"具体位置为:{accurate_location}")
        else:
            print("未能提取具体位置！")
        count -= 1


    test_image = "d:\desktop\date_location.jpg"
    date_time, latitude, longitude, accurate_location  = main_date_location(test_image)
    if date_time:
        print(f"拍摄时间: {date_time}")
    else:
        print("未能提取到拍摄时间")

    if latitude is not None and longitude is not None:
        print(f"拍摄时的经纬度: 纬度 {latitude:.5f}, 经度 {longitude:.5f}")
    else:
        print("未能提取到经纬度信息")
    if accurate_location is not None:
        print(f"具体位置为:{accurate_location}")
    else:
        print("未能提取具体位置！")

