'''
Author: WangXiang
Date: 2024-04-14 02:52:39
LastEditTime: 2024-04-14 02:55:56
'''


class ModuleManager:

    def __init__(self, modules) -> None:
        self.module_functions = {}
        for m in modules:
            self.register_module(m)
        
    def register_module(self, m):
        attr_names = dir(m)
        for attr_name in attr_names:
            attr = getattr(m, attr_name)
            if not callable(m, attr):
                continue
            if attr_name in self.module_functions.keys():
                print(f"[Warning]: {attr_name} ahs been already registered (source: {self.module_functions[attr_name]['__name__']}), the function with the same name for module {m.__name__} will be ignored.")
            else:
                self.module_functions[attr_name] = {
                    '__name__': m.__name__,
                    '__file__': m.__file__,
                    'function': attr
                }
    
    def pickle_modules(self):
        return {k: v['function'] for k, v in self.module_functions.items()}
    
    def __call__(self):
        return self.pickle_modules()