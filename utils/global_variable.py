def global_variable_init():
    """在主模块初始化"""
    global GLOBALS_DICT
    GLOBALS_DICT = {}
    print("Global variable init successfully.")
 
 
def global_variable_set_dict(name, value) -> bool:
    """设置"""
    try:
        GLOBALS_DICT[name] = value
        return True
    except KeyError:
        return False
 
 
def global_variable_get(name):
    """取值"""
    try:
        return GLOBALS_DICT[name]
    except KeyError:
        return "Not Found"