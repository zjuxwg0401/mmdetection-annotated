import inspect

import mmcv
import ipdb


class Registry(object):
    
    def __init__(self, name):#此处的self是一个对象（object），是当前类的实例，name即为刚传进来的‘detector'值
        self._name = name
        self._module_dict = dict() #定义的属性，是一个字典

    def __repr__(self):
        #__repr__方法是python的一种特征方法，由于object类已经提供了该方法，后面都是继承方法，print方法之所以能够打印，正是由于每个类都有继承自
        #object类的__repr__方法，__repr__方法默认返回的是该对象实现类的“类名+object at+内存地址“值，代码中可以重写此方法。
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property #把方法转变为属性，能过 self.name就能访问name的值
    def name(self):
        return self._name
    #因为没有定义其setter方法，所以是个只读属性，不能通过self.name = name进行修改
    
    @property #同样，函数属性化，只读
    def module_dict(self):
        return self._module_dict

    # 普通方法
    def get(self, key):
        return self._module_dict.get(key, None)
    # key     -- 字典中要查找的键。
    # default -- 如果指定键的值不存在时，返回该默认值。


    def _register_module(self, module_class):
        # 关键的一个方法，作用就是register a module
        # 在model文件夹中的py文件中，里面的class定义上面都会出现 @DETECTORS.register_module,意思就是将类当作形参
        #将类送入了方法register_module()中执行，@的用法 看后面解释。
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):#判断是否为类
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__#获取类名
        if module_name in self._module_dict:#重注册检测
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class#在module中dict新增key和value，key为类名，value为类对象

    def register_module(self, cls):#高阶函数，函数嵌套，对上面的方法，修改了名字，添加了返回值，返回值为类本身
        self._register_module(cls)
        return cls


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg #cfg是否是个字典，这个字典至少须要包含键“type”
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()#args相当于中间变量 temp，这个字典
    obj_type = args.pop('type')  #   字典 pop() 方法删除字典给定键 key 及对应的值，返回值为被删除的值。key 值必须给出。 否则，返回 default 值。
    if mmcv.is_str(obj_type):
        # 这里的registry的get返回的_module_dict属性中包含的是detector下的模型type
        # 索引key得到相应的class
        obj_type = registry.get(obj_type)
        if obj_type is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif not inspect.isclass(obj_type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():    #items()返回字典的键值对用于遍历
            args.setdefault(name, value)            #将default_args的键值对加入到args中，将模型和训练配置进行整合送入类中
    # 注意：无论训练/检测，都会build DETECTORS，；
    # **args是将字典unpack得到各个元素，分别与形参匹配送入函数中；
    return obj_type(**args)
