from ruamel.yaml import YAML

def over_write_args_from_file(args, yml):
    """
    overwrite arguments according to config file
    """
    if yml == '':
        return
    yaml = YAML(typ='rt') # rt is (round-trip) mode
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read())
        for k in dic:
            setattr(args, k, dic[k])

