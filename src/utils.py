import yaml


def visible_print(*msg):
    print()
    print("=" * 120)
    print(*msg)
    print("=" * 120)


def get_config_yaml():
    with open("./config.yaml") as yaml_data:
        exp_config = yaml.load(yaml_data, Loader=yaml.FullLoader)

    return exp_config