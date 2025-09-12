import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import json
import os


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')

    # 处理3位缩写格式（如#f00 → #ff0000）
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])

    # 验证长度
    if len(hex_color) != 6:
        raise ValueError(f"输入格式不正确: {hex_color}")

    # 将十六进制颜色转换为RGB格式
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return [r, g, b]


def xml_to_geojson(xml_path, output_path):
    """
    将ImageScope XML标注文件转换为GeoJSON格式

    参数:
        xml_path (str): XML文件路径
        output_path (str): 输出GeoJSON文件路径
    """
    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 创建GeoJSON结构
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    # 遍历所有Annotation
    for i, annotation in enumerate(root.findall('.//Annotation')):
        # 获取Annotation属性
        name = annotation.get('Name', f'Annotation_{i + 1}')
        annotation_type = annotation.get('Type', 'Polygon')
        part_of_group = annotation.get('PartOfGroup', 'None')
        color = annotation.get('Color', '#000000')
        color = hex_to_rgb(color)

        # 获取Coordinates
        coordinates_elem = annotation.find('.//Coordinates')
        if coordinates_elem is None:
            continue

        # 提取所有Coordinate点
        points = []
        for coord in coordinates_elem.findall('.//Coordinate'):
            x = float(coord.get('X'))
            y = float(coord.get('Y'))
            points.append([x, y])

        # 确保多边形闭合（首尾点相同）
        if len(points) > 0 and points[0] != points[-1]:
            points.append(points[0])

        # 创建GeoJSON Feature
        feature = {
            "type": "Feature",
            "properties": {
                "name": name,
                "type": annotation_type,
                "group": part_of_group,
                "color": color,
                "id": i + 1
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [points]  # 注意：GeoJSON多边形需要数组的数组
            }
        }

        geojson['features'].append(feature)

    # 保存为GeoJSON文件
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)


parser = argparse.ArgumentParser(description='xml标注转geojson标注')
parser.add_argument('--xml_dir', type=str, default='/NAS4/llb/协和CRC标注数据')
parser.add_argument('--geojson_dir', type=str, default='/NAS3/lbliao/Data/CRC/geojson/tumor')
if __name__ == "__main__":
    args = parser.parse_args()
    geojson_dir = args.geojson_dir
    os.makedirs(geojson_dir, exist_ok=True)

    xml_dir = Path(args.xml_dir)
    xml_paths = list(xml_dir.rglob('*.xml'))
    geojson_dir = Path(args.geojson_dir)
    for xml_path in xml_paths:
        base = xml_path.stem
        pattern = r'\d{4}-\d{2}-\d{2} \d{2}.\d{2}.\d{2}'  # 匹配 YYYY-MM-DD HH:MM:SS
        cleaned_name = re.sub(pattern, '', base)
        cleaned_name = cleaned_name.replace('-', ' ')
        cleaned_name = cleaned_name.strip().replace('  ', ' ') + '.geojson'
        if ' ' not in cleaned_name:
            cleaned_name = cleaned_name[:-10] + ' ' + cleaned_name[-10:]
        geojson_path = os.path.join(xml_path.parent, cleaned_name)
        xml_to_geojson(str(xml_path), geojson_path)
        print(f'{base} 转 geojson 成功，保存为 {cleaned_name}')
