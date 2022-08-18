import argparse


class DatasetsManager:
    _datasets = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def export(cls, name):
        def export_helper(plugin):
            cls._datasets[name] = plugin
            return plugin

        return export_helper

    @classmethod
    def add_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")

        parser.add_argument("-d", "--dataset", help="Dasaset to be loaded")

        args, _ = parser.parse_known_args()
        assert args.dataset in cls._datasets, f"Dataset {args.dataset} is not defined"
        for dataset, c in cls._datasets.items():
            if dataset != args.dataset:
                continue
            add_args = getattr(c, "add_args", None)

            if callable(add_args):
                parser = add_args(parser)
        return parser

    @classmethod
    def tunning_scopes(cls, args):

        for dataset, c in cls._datasets.items():
            if dataset != args.dataset:
                continue
            scopes = {}
            tunning_scopes = getattr(c, "tunning_scopes", None)
            if callable(tunning_scopes):
                scopes = tunning_scopes()
        return scopes

    def list_datasets(self):
        return self._datasets

    def build_dataset(self, name, **kwargs):

        assert name in self._datasets, f"Dataset {name} is unknown"
        return self._datasets[name](**kwargs)
