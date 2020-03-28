__author__ = 'Ehaschia'

"""
Alphabet maps objects to integer ids. It provides two way mapping from the index to the objects.
"""
import json
import os
from io_module.logger import get_logger


class Alphabet(object):
    def __init__(self, name, keep_growing=True, singleton=False):
        self.__name = name

        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing
        self.singletons = set() if singleton else None

        self.logger = get_logger('Alphabet')

    def add(self, instance):
        if instance not in self.instance2index:
            self.instance2index[instance] = len(self.instances)
            self.instances.append(instance)

    def add_singleton(self, id):
        if self.singletons is None:
            raise RuntimeError('Alphabet %s does not have singleton.' % self.__name)
        else:
            self.singletons.add(id)

    def add_singletons(self, ids):
        if self.singletons is None:
            raise RuntimeError('Alphabet %s does not have singleton.' % self.__name)
        else:
            self.singletons.update(ids)

    def is_singleton(self, id):
        if self.singletons is None:
            raise RuntimeError('Alphabet %s does not have singleton.' % self.__name)
        else:
            return id in self.singletons

    def get_index(self, instance):
        # try:
        #     return self.instance2index[instance]
        # except KeyError:
        #     if self.keep_growing:
        #         index = len(self.instances)
        #         self.add(instance)
        #         return index
        #     else:
        #         raise KeyError("instance not found: %s" % instance)
        if instance in self.instance2index:
            return self.instance2index[instance]
        elif self.keep_growing:
            index = len(self.instances)
            self.add(instance)
            return index
        else:
            return -1

    def is_known(self, instance):
        return instance in self.instance2index

    def get_instance(self, index):
        try:
            return self.instances[index]
        except IndexError:
            raise IndexError('unknown index: %d' % index)

    def size(self):
        return len(self.instances)

    def singleton_size(self):
        return len(self.singletons)

    def items(self):
        return self.instance2index.items()

    def enumerate_items(self, start):
        if start >= self.size():
            raise IndexError("Enumerate is allowed between [0 : size of the alphabet)")
        return zip(range(start, len(self.instances)), self.instances[start:])

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def get_content(self):
        if self.singletons is None:
            return {'instance2index': self.instance2index, 'instances': self.instances}
        else:
            return {'instance2index': self.instance2index, 'instances': self.instances,
                    'singletions': list(self.singletons)}

    def __from_json(self, data):
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]
        if 'singletions' in data:
            self.singletons = set(data['singletions'])
        else:
            self.singletons = None

    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        saving_name = name if name else self.__name
        try:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            json.dump(self.get_content(),
                      open(os.path.join(output_directory, saving_name + ".json"), 'w'), indent=4)
        except Exception as e:
            self.logger.warn("Alphabet is not saved: %s" % repr(e))

    def load(self, input_directory, name=None):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        loading_name = name if name else self.__name
        self.__from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))
        self.keep_growing = False
