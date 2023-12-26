import yaml

def getConfig(config_path):
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)  # 读取配置项的值
    return config
