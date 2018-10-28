"""
pyomo model 的代理方法
"""
from pyomo.environ import *
from strgen import StringGenerator as SG
# from pyomo.util.modeling import unique_component_name
from pyomotools.tools import unique_component_name


class ModelProxy:
    def __init__(self, model: Model):
        self.model = model
        self._anonymous = None

    @property
    def anonymous(self):
        return self._anonymous

    @anonymous.setter
    def anonymous(self, value):
        random_name = SG("[\w]{3}").render()
        random_name = unique_component_name(self.model, random_name)
        setattr(self.model, random_name, value)
        self._anonymous = value

    @staticmethod
    def get_model(m):
        if isinstance(m, ModelProxy):
            return m.model
        return m

    def __getattr__(self, item):
        """
        只有自身没有这个属性的时候才会调用这个属性
        :param item:
        :return:
        """
        return getattr(self.model, item)

    def __setattr__(self, key, value):
        if key in ["model", "_anonymous", "anonymous"]:
            super(ModelProxy, self).__setattr__(key, value)
        else:
            setattr(self.model, key, value)


